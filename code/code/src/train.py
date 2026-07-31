import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import add_relative_market_features
from utils import engineer_features_39, engineer_features_158plus39
from utils import (
    MARKET_PRESSURE_FEATURES,
    RELATIVE_MARKET_FEATURES,
    RELATIVE_MARKET_FEATURE_SET,
    RISK_MARKET_FEATURES,
    RISK_MARKET_FEATURE_SET,
    INDUSTRY_RESIDUAL_FEATURE_SET,
    INDUSTRY_RESIDUAL_FEATURES,
    INDUSTRY_ASOF_COLUMN,
    INDUSTRY_NEUTRAL_TARGET,
)
from utils import add_industry_residual_features, add_industry_neutral_label
from utils import create_ranking_dataset_vectorized
from utils import extract_selection_risk_context
from utils import align_oof_prediction_records, calibrate_ensemble_policy
from utils import evaluate_ensemble_policy
from utils import attach_label_end_dates, forward_fit_module_gated_policy
from utils import maximum_drawdown, percentile_ranks
import joblib
import os
import json
import multiprocessing as mp
import random
import hashlib
import subprocess
import sys
import tempfile


def policy_only_enabled():
    return os.environ.get('POLICY_ONLY', '0').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def lockbox_eval_enabled():
    return os.environ.get('LOCKBOX_EVAL', '0').strip().lower() in {
        '1', 'true', 'yes', 'on',
    }


def stress_eval_enabled():
    return os.environ.get('STRESS_EVAL', '0').strip().lower() in {
        '1', 'true', 'yes', 'on',
    }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic = config.get('deterministic_training', True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_accelerator(device):
    """启用不改变训练目标的 CUDA 数值与吞吐优化。"""
    if device.type != 'cuda':
        return
    if config.get('tf32_enabled', True):
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def use_amp(device):
    return device.type == 'cuda' and config.get('amp_enabled', True)


def create_grad_scaler(device):
    return torch.amp.GradScaler('cuda', enabled=use_amp(device))


def move_batch_tensor(tensor, device):
    return tensor.to(
        device,
        non_blocking=(
            device.type == 'cuda'
            and config.get('non_blocking_transfer', True)
        ),
    )


def cast_auxiliary_outputs_to_float(auxiliary_outputs):
    """让 AMP 生成的辅助头输出以 FP32 参与数值敏感的损失计算。"""
    return {
        name: value.float() if isinstance(value, torch.Tensor) else value
        for name, value in auxiliary_outputs.items()
    }


def forward_model_batch(model, batch, sequences, stock_indices, masks, device):
    """优先使用冻结主干缓存；排名阶段仍按原始输入完整前向。"""
    if 'cached_ranking_features' not in batch:
        return model(sequences, stock_indices, masks, return_aux=True)
    return model.forward_from_cached(
        move_batch_tensor(batch['cached_ranking_features'], device),
        regime_sequence=move_batch_tensor(
            batch['cached_regime_sequence'], device,
        ),
        market_sequence=move_batch_tensor(
            batch['cached_market_sequence'], device,
        ),
        stock_mask=masks,
        return_aux=True,
    )


def dense_stage_loss(
    criterion, model, outputs, relevance, return_outputs, targets,
    allocation_outputs, exposures, auxiliary_outputs, risk_1d_targets,
    risk_3d_targets, tail_5d_targets, path_loss_5d_targets,
    industry_neutral_targets, regime_targets, stage, return_components=False,
):
    """无 padding 的完整 batch 一次计算辅助阶段损失，避免逐日期 Python 循环。"""
    return criterion(
        outputs,
        relevance,
        return_outputs,
        targets,
        allocation_outputs,
        exposures,
        identity_gate=model.identity_gate_value(),
        risk_1d_logits=auxiliary_outputs['risk_1d_logits'],
        risk_3d_logits=auxiliary_outputs['risk_3d_logits'],
        risk_5d_logits=auxiliary_outputs['risk_5d_logits'],
        tail_5d_logits=auxiliary_outputs['tail_5d_logits'],
        path_loss_5d_outputs=auxiliary_outputs['path_loss_5d_output'],
        tail_5d_targets=tail_5d_targets,
        path_loss_5d_targets=path_loss_5d_targets,
        industry_residual_outputs=auxiliary_outputs['industry_residual_returns'],
        industry_neutral_targets=industry_neutral_targets,
        regime_gate=auxiliary_outputs['regime_gate'],
        risk_1d_targets=risk_1d_targets,
        risk_3d_targets=risk_3d_targets,
        regime_targets=regime_targets,
        stage=stage,
        return_components=return_components,
    )


def optimizer_parameters_with_grad(optimizer):
    """只返回当前阶段优化器实际管理且已经产生梯度的参数。"""
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group['params']
        if parameter.grad is not None
    ]


def resume_training_enabled():
    return os.environ.get('RESUME_TRAINING', '0') == '1'


TRAINING_STAGES = ('ranking', 'risk', 'allocation', 'exposure')


def apply_v17_profile():
    """在进程启动时选择可复跑的 V17 基线或候选路径。

    配置文件可以保持 candidate 的 205 维默认值；基线由环境变量覆盖，
    从而保证两者共用同一原始数据、锁箱和三折边界。
    """
    profile = os.environ.get('V17_PROFILE', 'candidate').strip().lower()
    if profile not in {'baseline', 'candidate'}:
        raise ValueError('V17_PROFILE 只能为 baseline 或 candidate')
    if profile == 'baseline':
        config.update({
            'feature_num': RISK_MARKET_FEATURE_SET,
            'industry_residual_head_enabled': False,
            'industry_residual_weight': 0.0,
            'path_loss_5d_head_enabled': False,
            'path_loss_5d_weight': 0.0,
            'tail_5d_target_mode': 'endpoint_return',
            'lgbm_enabled': False,
            # 保守基线：更高的最低仓位及不启用新风险融合模块。
            'min_exposure': max(float(config.get('min_exposure', .80)), .80),
        })
    else:
        config['feature_num'] = INDUSTRY_RESIDUAL_FEATURE_SET
    config['v17_profile'] = profile
    config['output_dir'] = (
        f"./model/{config['sequence_length']}_{config['feature_num']}_"
        f"{config.get('experiment_name', 'v17')}_{profile}"
    )
    if os.environ.get('V17_INCLUDE_LOCKBOX', '0') == '1':
        # 最终部署训练是一次性例外；其目录永远不能覆盖策略重放或开发期 OOF。
        config['output_dir'] = config['full_deployment_output_dir']
    return profile


def split_v17_lockbox(df):
    """冻结最后 N 个自然月；开发期之外的数据不参与训练或选型。"""
    dated = df.copy()
    dated['日期'] = pd.to_datetime(dated['日期'])
    if os.environ.get('V17_INCLUDE_LOCKBOX', '0') == '1':
        if os.environ.get('LOCKBOX_ACCEPTED', '0') != '1':
            raise ValueError('仅在锁箱验收后设置 LOCKBOX_ACCEPTED=1 才可纳入最终重训')
        print('锁箱验收已确认：仅用于固定配置的最终部署重训')
        return dated, None
    if not config.get('lockbox_enabled', False):
        return dated, None
    last_date = dated['日期'].max()
    months = int(config.get('lockbox_months', 2))
    if months < 1 or pd.isna(last_date):
        raise ValueError('lockbox_months 必须大于0，且训练数据必须包含日期')
    start = (last_date.to_period('M') - (months - 1)).start_time
    development = dated.loc[dated['日期'] < start].copy()
    if development.empty or dated.loc[dated['日期'] >= start].empty:
        raise ValueError('锁箱切分得到空数据集')
    print(f'锁箱已冻结: {start.date()} 至 {last_date.date()}，不参与开发期训练')
    return development, pd.Timestamp(start)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def lockbox_return_stats(returns):
    """汇总非重叠五日持有收益，并保留可审计的最大回撤定义。"""
    values = np.asarray(returns, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('锁箱收益必须是非空有限数列')
    wealth = np.cumprod(1.0 + values)
    max_drawdown = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
    return {
        'mean_return': float(values.mean()),
        'p10_return': float(np.quantile(values, 0.10)),
        'max_drawdown': max_drawdown,
        'positive_rate': float(np.mean(values > 0.0)),
    }


def lockbox_realized_return(result_path, raw, entry_date, exit_date):
    """以 t+1 开盘买入、t+5 收盘卖出评估单日锁箱持仓。"""
    holdings = pd.read_csv(result_path, dtype={'stock_id': str})
    required = {'stock_id', 'weight'}
    if not required.issubset(holdings):
        raise ValueError(f'锁箱预测缺少列 {required}: {result_path}')
    holdings['stock_id'] = holdings['stock_id'].astype(str).str.zfill(6)
    weights = holdings['weight'].to_numpy(dtype=np.float64)
    if (
        len(holdings) > 5 or holdings['stock_id'].duplicated().any()
        or not np.isfinite(weights).all() or (weights < -1e-12).any()
        or weights.sum() > 1.0 + 1e-9
    ):
        raise ValueError(f'锁箱预测违反 Top-5/资金约束: {result_path}')
    entry = raw.loc[raw['日期'] == entry_date, ['股票代码', '开盘']].copy()
    exit_prices = raw.loc[raw['日期'] == exit_date, ['股票代码', '收盘']].copy()
    prices = entry.merge(exit_prices, on='股票代码', how='inner').set_index('股票代码')
    prices = prices.reindex(holdings['stock_id'])
    if prices.isna().any().any() or (prices['开盘'] <= 0).any():
        raise ValueError(
            f'锁箱缺少 {entry_date.date()} 开盘或 {exit_date.date()} 收盘价格'
        )
    realized = prices['收盘'].to_numpy(dtype=np.float64) / prices['开盘'].to_numpy(dtype=np.float64) - 1.0
    return float(np.dot(weights, realized)), {
        'stocks': holdings['stock_id'].tolist(),
        'weights': [float(weight) for weight in weights],
        'weight_sum': float(weights.sum()),
        'cash_weight': float(1.0 - weights.sum()),
    }


def atomic_write_lockbox_report(path, report):
    """只创建一次锁箱报告；硬链接发布确保已有报告绝不被覆盖。"""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f'锁箱报告已存在，拒绝覆盖: {path}')
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.lockbox_report.', suffix='.json', dir=directory,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f'锁箱报告已存在，拒绝覆盖: {path}') from exc
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def run_lockbox_eval(output_dir, stress=False):
    """一次性回放新锁箱，或只诊断已知压力期；两者均不训练模型。"""
    if os.environ.get('V17_INCLUDE_LOCKBOX', '0') == '1':
        raise ValueError('评估回放与最终部署重训互斥')
    report_path = os.path.join(
        output_dir,
        'observed_stress_report.json' if stress else 'lockbox_report.json',
    )
    if os.path.exists(report_path):
        raise FileExistsError(f'锁箱报告已存在，拒绝第二次运行: {report_path}')
    policy_path = os.path.join(output_dir, 'ensemble_policy.json')
    manifest_path = os.path.join(output_dir, 'artifact_manifest.json')
    for path, label in ((policy_path, 'v1.20 策略'), (manifest_path, 'v1.20 manifest')):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'LOCKBOX_EVAL 需要{label}: {path}')
    with open(policy_path, encoding='utf-8') as handle:
        policy = json.load(handle)
    promotion = policy.get('promotion_criteria', {})
    if not promotion.get('passed', False):
        raise ValueError('v1.20 开发期未晋级，评估回放被拒绝')
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    lockbox_start_raw = manifest.get('lockbox_start')
    if not lockbox_start_raw:
        raise ValueError('v1.20 manifest 未冻结 lockbox_start')
    known_stress_end = pd.Timestamp(config['known_stress_end'])
    if stress:
        lockbox_start = pd.Timestamp(config['known_stress_start'])
        evaluation_end = known_stress_end
    else:
        lockbox_start = pd.Timestamp(lockbox_start_raw)
        if lockbox_start <= known_stress_end:
            raise ValueError(
                'LOCKBOX_EVAL 只允许 2026-07-29 之后新增数据形成的新锁箱；'
                '当前末两月已被 v1.19 消费，请使用 STRESS_EVAL=1 做已知压力诊断'
            )
        evaluation_end = None
    baseline_dir = os.path.abspath(config['baseline_source_dir'])
    baseline_policy = os.path.join(baseline_dir, 'ensemble_policy.json')
    if not os.path.isfile(baseline_policy):
        raise FileNotFoundError(f'缺少冻结 v1.17 baseline 策略: {baseline_policy}')

    data_file = os.path.join(config['data_path'], 'train.csv')
    raw = pd.read_csv(data_file, dtype={'股票代码': str})
    raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
    raw['日期'] = pd.to_datetime(raw['日期'])
    dates = np.array(sorted(raw['日期'].unique()), dtype='datetime64[ns]')
    first = int(np.searchsorted(dates, lockbox_start.to_datetime64(), side='left'))
    last_anchor_exclusive = len(dates) - 5
    if evaluation_end is not None:
        last_anchor_exclusive = min(
            last_anchor_exclusive,
            int(np.searchsorted(
                dates, evaluation_end.to_datetime64(), side='right',
            )) - 5,
        )
    anchors = [pd.Timestamp(dates[index]) for index in range(
        first, max(first, last_anchor_exclusive), int(config['evaluation_stride']),
    )]
    if not anchors:
        raise ValueError('锁箱没有可完成 t+1 至 t+5 持有期的交易日')
    # train.py 位于 code/src；脚本路径从本文件定位，工作目录单独固定为 code/。
    source_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(source_dir))
    predict_script = os.path.join(source_dir, 'predict.py')
    prediction_root = os.path.join(
        output_dir, '_stress_predictions' if stress else '_lockbox_predictions',
    )
    candidates = {
        'v1.17_baseline': baseline_dir,
        'v1.20_candidate': os.path.abspath(output_dir),
    }
    records = {name: [] for name in candidates}
    total_steps = len(anchors) * len(candidates)
    print(
        f"阶段 {'Stress' if stress else 'Lockbox'}：{lockbox_start.date()} 起 "
        f'{len(anchors)} 个非重叠锚点，'
        '仅加载冻结工件推理'
    )
    with tqdm(total=total_steps, desc='Lockbox 回放', unit='模型日', dynamic_ncols=True) as progress:
        for prediction_date in anchors:
            date_index = int(np.searchsorted(dates, prediction_date.to_datetime64(), side='left'))
            entry_date = pd.Timestamp(dates[date_index + 1])
            exit_date = pd.Timestamp(dates[date_index + 5])
            for name, model_dir in candidates.items():
                destination = os.path.join(
                    prediction_root, name, prediction_date.strftime('%Y-%m-%d'),
                )
                result_path = os.path.join(destination, 'result.csv')
                if not os.path.isfile(result_path):
                    environment = os.environ.copy()
                    environment.update({
                        'MODEL_OUTPUT_DIR': model_dir,
                        'PREDICTION_OUTPUT_DIR': destination,
                        'PREDICTION_DATE': prediction_date.strftime('%Y-%m-%d'),
                    })
                    completed = subprocess.run(
                        [sys.executable, predict_script],
                        cwd=project_dir, env=environment, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f'Lockbox {name} {prediction_date.date()} 推理失败:\n'
                            f'{completed.stdout[-4000:]}'
                        )
                realized, holding = lockbox_realized_return(
                    result_path, raw, entry_date, exit_date,
                )
                records[name].append({
                    'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': exit_date.strftime('%Y-%m-%d'),
                    'return': realized,
                    **holding,
                })
                progress.update(1)
                progress.set_postfix_str(f'{name} {prediction_date:%Y-%m-%d}')
    results = {
        name: {**lockbox_return_stats([row['return'] for row in rows]), 'daily': rows}
        for name, rows in records.items()
    }
    baseline_stats = results['v1.17_baseline']
    candidate_stats = results['v1.20_candidate']
    mean_passed = (
        candidate_stats['mean_return'] >= baseline_stats['mean_return']
        if stress else candidate_stats['mean_return'] > baseline_stats['mean_return']
    )
    accepted = bool(
        mean_passed
        and candidate_stats['p10_return'] >= baseline_stats['p10_return']
        and candidate_stats['max_drawdown'] >= baseline_stats['max_drawdown']
    )
    report = {
        'protocol': (
            'v1.20_observed_stress_t_plus_1_open_to_t_plus_5_close_stride_5'
            if stress else 'v1.20_fresh_lockbox_t_plus_1_open_to_t_plus_5_close_stride_5'
        ),
        'lockbox_start': lockbox_start.strftime('%Y-%m-%d'),
        'lockbox_end': (
            None if evaluation_end is None else evaluation_end.strftime('%Y-%m-%d')
        ),
        'fresh_lockbox': not stress,
        'model_dirs': candidates,
        'selected_candidate': policy.get('candidate_name'),
        'hashes': {
            'train_csv': sha256_file(data_file),
            'v1.20_policy': sha256_file(policy_path),
            'v1.20_manifest': sha256_file(manifest_path),
            'v1.17_policy': sha256_file(baseline_policy),
        },
        'results': results,
        'candidate_minus_baseline': {
            key: float(candidate_stats[key] - baseline_stats[key])
            for key in ('mean_return', 'p10_return', 'max_drawdown')
        },
        'accepted_for_final_deployment': bool(accepted and not stress),
        'stress_gate_passed': bool(accepted) if stress else None,
    }
    atomic_write_lockbox_report(report_path, report)
    print(
        f"{'Stress' if stress else 'Lockbox'} 报告已写入: {report_path}; "
        f"最终部署许可: {'通过' if accepted and not stress else '未通过'}"
    )
    return candidate_stats['mean_return']


def require_accepted_lockbox_for_deployment():
    """最终五年重训只能读取一次性锁箱报告，环境变量本身不是验收证据。"""
    candidate_dir = (
        f"./model/{config['sequence_length']}_{config['feature_num']}_"
        f"{config['experiment_name']}_candidate"
    )
    report_path = os.path.join(candidate_dir, 'lockbox_report.json')
    if not os.path.isfile(report_path):
        raise FileNotFoundError(
            f'最终部署需要已冻结的 lockbox_report.json: {report_path}'
        )
    with open(report_path, encoding='utf-8') as handle:
        report = json.load(handle)
    if not report.get('accepted_for_final_deployment', False):
        raise ValueError('锁箱未通过，禁止 v1.20 最终全历史部署重训')
    if not report.get('fresh_lockbox', False) or pd.Timestamp(
        report.get('lockbox_start', '1900-01-01')
    ) <= pd.Timestamp(config['known_stress_end']):
        raise ValueError('最终部署只接受 2026-07-29 之后的新锁箱报告')


def run_frozen_final_deployment(output_dir):
    """用新锁箱前已冻结的策略和轮数作全历史重训，绝不再做 OOF 选型。"""
    candidate_dir = (
        f"./model/{config['sequence_length']}_{config['feature_num']}_"
        f"{config['experiment_name']}_candidate"
    )
    source_summary_path = os.path.join(candidate_dir, 'cross_validation_summary.json')
    source_policy_path = os.path.join(candidate_dir, 'ensemble_policy.json')
    source_manifest_path = os.path.join(candidate_dir, 'artifact_manifest.json')
    for path, label in (
        (source_summary_path, '冻结开发期报告'),
        (source_policy_path, '冻结部署策略'),
        (source_manifest_path, '冻结 manifest'),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'最终部署缺少{label}: {path}')
    with open(source_summary_path, encoding='utf-8') as handle:
        source_summary = json.load(handle)
    with open(source_policy_path, encoding='utf-8') as handle:
        policy = json.load(handle)
    with open(source_manifest_path, encoding='utf-8') as handle:
        source_manifest = json.load(handle)
    stage_epochs = source_summary.get('full_training', {}).get('stage_epochs')
    if set(stage_epochs or ()) != set(TRAINING_STAGES):
        raise ValueError('冻结开发期报告缺少四阶段最终训练轮数')
    source_lgbm_folds = []
    for fold in source_manifest.get('folds', []):
        fold_id = int(fold['fold'])
        path = os.path.join(candidate_dir, 'lgbm', f'fold_{fold_id}.joblib')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'冻结开发期缺少 LightGBM 折模型: {path}')
        source_lgbm_folds.append({
            'best_iteration': int(joblib.load(path).booster_.current_iteration()),
        })
    if not source_lgbm_folds:
        raise ValueError('冻结开发期缺少 LightGBM 折迭代数')
    if os.path.exists(os.path.join(output_dir, 'ensemble_policy.json')) and not resume_training_enabled():
        raise FileExistsError(f'最终部署目录已存在，拒绝覆盖: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    stockid2idx = {
        stock_id: index + 2
        for index, stock_id in enumerate(sorted(full_df['股票代码'].unique()))
    }
    with open(os.path.join(output_dir, 'stockid2idx.json'), 'w', encoding='utf-8') as handle:
        json.dump(stockid2idx, handle, indent=2, ensure_ascii=False)
    print('阶段 最终部署：全五年特征、Scaler 与冻结配置重训（不执行 OOF 策略标定）')
    full_data, features = preprocess_data(full_df, is_train=True, stockid2idx=stockid2idx)
    full_data['日期'] = pd.to_datetime(full_data['日期'])
    frozen_folds = [
        {key: (int(value) if key == 'fold' else pd.Timestamp(value)) for key, value in fold.items()}
        for fold in source_manifest.get('folds', [])
    ]
    write_v17_manifest(output_dir, features, frozen_folds, None, len(stockid2idx))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configure_accelerator(device)
    set_seed(int(config['seed']))
    dataset, train_end = prepare_full_training_dataset(full_data, features, output_dir)
    final_dir = os.path.join(output_dir, f"seed_{int(config['seed'])}")
    final_training = train_final_model(
        dataset, train_end, features, len(stockid2idx), device, final_dir,
        {stage: int(stage_epochs[stage]) for stage in TRAINING_STAGES},
        int(config['seed']),
    )
    lgbm_model_path = fit_lgbm_final(
        full_data, features, source_lgbm_folds, output_dir,
    )
    policy.update({
        'policy_role': 'frozen_v1.20_policy_full_history_retrained',
        'model_paths': [os.path.join(f"seed_{int(config['seed'])}", 'best_model.pth')],
        'ensemble_enabled': False,
        'ensemble_seeds': [int(config['seed'])],
        'lgbm_model_path': lgbm_model_path,
        'config_path': 'config.json',
        'scaler_path': 'scaler.pkl',
        'final_deployment_source_dir': os.path.relpath(candidate_dir, output_dir),
        'final_deployment_recalibrated': False,
    })
    with open(os.path.join(output_dir, 'ensemble_policy.json'), 'w', encoding='utf-8') as handle:
        json.dump(policy, handle, indent=2, ensure_ascii=False)
    summary = dict(source_summary)
    summary.update({
        'training_mode': 'full_history_deployment_frozen_policy',
        'full_training': {'models': [final_training], 'stage_epochs': stage_epochs},
        'final_deployment': {
            'source_dir': os.path.relpath(candidate_dir, output_dir),
            'source_policy_sha256': sha256_file(source_policy_path),
            'source_summary_sha256': sha256_file(source_summary_path),
            'lockbox_report_sha256': sha256_file(os.path.join(candidate_dir, 'lockbox_report.json')),
            'recalibrated_on_lockbox': False,
        },
    })
    with open(os.path.join(output_dir, 'cross_validation_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f'最终部署重训完成，工件已隔离写入: {output_dir}')
    return float(summary['mean_weighted_portfolio_return'])


def write_v17_manifest(output_dir, features, folds, lockbox_start, stock_count):
    """记录特征、架构与锁箱边界，供推理拒绝不兼容工件。"""
    manifest = {
        'schema_version': 1,
        'feature_num': config['feature_num'], 'feature_count': len(features),
        'feature_sha256': hashlib.sha256('\n'.join(features).encode()).hexdigest(),
        'sequence_length': config['sequence_length'], 'd_model': config['d_model'],
        'nhead': config['nhead'], 'num_layers': config['num_layers'],
        'dim_feedforward': config['dim_feedforward'],
        'stock_embedding_dim': config['stock_embedding_dim'],
        'score_head_variant': config.get('score_head_variant', 'mlp_v1'),
        'rankglu_bottleneck': config.get('rankglu_bottleneck'),
        'rankglu_gamma_max': config.get('rankglu_gamma_max'),
        'stock_mapping_size': int(stock_count),
        'lockbox_start': None if lockbox_start is None else lockbox_start.strftime('%Y-%m-%d'),
        'folds': [{key: int(value) if key == 'fold' else pd.Timestamp(value).strftime('%Y-%m-%d')
                   for key, value in fold.items()} for fold in folds],
    }
    with open(os.path.join(output_dir, 'artifact_manifest.json'), 'w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def lgbm_relevance_labels(frame):
    """将每日行业中性收益排序映射为固定档位的日期内 relevance。"""
    levels = int(config.get('lgbm_relevance_levels', 31))
    if levels < 2:
        raise ValueError('lgbm_relevance_levels 必须至少为2')
    ranks = frame.groupby('日期', sort=False)[INDUSTRY_NEUTRAL_TARGET].rank(
        method='first', ascending=True,
    ).sub(1)
    group_sizes = frame.groupby('日期', sort=False)[INDUSTRY_NEUTRAL_TARGET].transform('size')
    return np.floor(ranks * levels / group_sizes).clip(
        0, levels - 1,
    ).astype(np.int32)


def lgbm_progress_callback(description, total_iterations):
    """返回树迭代进度条及与 LightGBM 兼容的回调。"""
    progress = tqdm(
        total=int(total_iterations), desc=description, unit='树',
        dynamic_ncols=True, leave=True,
    )

    def update_progress(environment):
        completed = int(environment.iteration) + 1
        progress.update(max(0, completed - progress.n))
        for result in environment.evaluation_result_list or ():
            if result[0].startswith('valid') and result[1] == 'ndcg@5':
                progress.set_postfix_str(f'ndcg@5={result[2]:.4f}')
                break

    update_progress.order = 20
    update_progress.before_iteration = False
    return progress, update_progress


def validate_lgbm_checkpoint(model, features, checkpoint_path, expected_iterations=None):
    """拒绝复用特征或训练轮数不一致的 LightGBM 工件。"""
    if (
        int(getattr(model, 'n_features_in_', -1)) != len(features)
        or list(getattr(model, 'feature_name_', [])) != list(features)
    ):
        raise ValueError(f'LightGBM checkpoint 特征不匹配: {checkpoint_path}')
    if expected_iterations is not None:
        actual_iterations = int(model.booster_.current_iteration())
        if actual_iterations != int(expected_iterations):
            raise ValueError(f'LightGBM checkpoint 迭代数不匹配: {checkpoint_path}')


def save_joblib_checkpoint(value, checkpoint_path):
    """原子写入 checkpoint，进程中断时保留上一个完整版本。"""
    temporary_path = f'{checkpoint_path}.tmp'
    joblib.dump(value, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def _oof_score_signature(days, fold_id, features, model):
    """将缓存绑定到来源 OOF 的日期、股票集合、折和树特征顺序。"""
    digest = hashlib.sha256('\n'.join(features).encode())
    for day in sorted(
        (item for item in days if int(item['fold']) == int(fold_id)),
        key=lambda item: item['prediction_date'],
    ):
        digest.update(day['prediction_date'].encode())
        digest.update(np.asarray(day['stock_indices'], dtype=np.int64).tobytes())
    digest.update(str(int(model.booster_.current_iteration())).encode())
    return digest.hexdigest()


def load_or_build_lgbm_oof_scores(
    source_dir, output_dir, source_config, folds, ensemble_days,
):
    """从已保存折树模型恢复 OOF 分数，绝不重新训练任何模型。

    缓存写在 v1.18 策略目录而非 v1.17 来源目录，避免策略重放修改候选工件。
    """
    cache_dir = os.path.join(output_dir, 'lgbm_oof_cache')
    os.makedirs(cache_dir, exist_ok=True)
    models = {}
    features = None
    for fold in folds:
        fold_id = int(fold['fold'])
        path = os.path.join(source_dir, 'lgbm', f'fold_{fold_id}.joblib')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'缺少 v1.17 LightGBM 折模型: {path}')
        model = joblib.load(path)
        model_features = list(getattr(model, 'feature_name_', []))
        if not model_features:
            raise ValueError(f'LightGBM 折模型缺少特征名: {path}')
        if features is None:
            features = model_features
        elif model_features != features:
            raise ValueError('LightGBM 折模型特征顺序不一致')
        validate_lgbm_checkpoint(model, features, path)
        models[fold_id] = model

    cached = {}
    missing_folds = []
    for fold in folds:
        fold_id = int(fold['fold'])
        cache_path = os.path.join(cache_dir, f'fold_{fold_id}.joblib')
        signature = _oof_score_signature(
            ensemble_days, fold_id, features, models[fold_id],
        )
        if os.path.isfile(cache_path):
            payload = joblib.load(cache_path)
            if payload.get('signature') == signature:
                cached[fold_id] = payload['scores']
                print(f'复用 LightGBM OOF 分数缓存：Fold {fold_id}')
                continue
        missing_folds.append((fold, signature, cache_path))

    if missing_folds:
        print('阶段 LightGBM OOF 缓存：仅用已保存折模型重建缺失分数，不训练模型')
        data_file = os.path.join(source_config.get('data_path', config['data_path']), 'train.csv')
        raw = pd.read_csv(data_file, dtype={'股票代码': str})
        raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
        manifest_path = os.path.join(source_dir, 'artifact_manifest.json')
        with open(manifest_path, encoding='utf-8') as file:
            manifest = json.load(file)
        lockbox_start = manifest.get('lockbox_start')
        if lockbox_start:
            raw['日期'] = pd.to_datetime(raw['日期'])
            raw = raw.loc[raw['日期'] < pd.Timestamp(lockbox_start)].copy()
        with open(os.path.join(source_dir, 'stockid2idx.json'), encoding='utf-8') as file:
            stockid2idx = json.load(file)
        processed, processed_features = preprocess_data(
            raw, is_train=True, stockid2idx=stockid2idx,
        )
        if list(processed_features) != list(features):
            raise ValueError('LightGBM OOF 缓存特征与 v1.17 205维来源不匹配')
        processed['日期'] = pd.to_datetime(processed['日期'])
        for fold, signature, cache_path in tqdm(
            missing_folds, desc='LightGBM OOF 缓存', unit='折', dynamic_ncols=True,
        ):
            fold_id = int(fold['fold'])
            valid = processed.loc[
                (processed['日期'] >= pd.Timestamp(fold['val_start']))
                & (processed['日期'] <= pd.Timestamp(fold['val_end']))
            ]
            by_date = {
                pd.Timestamp(date).strftime('%Y-%m-%d'): rows.set_index('instrument')
                for date, rows in valid.groupby('日期', sort=False)
            }
            rows = {}
            for day in (item for item in ensemble_days if int(item['fold']) == fold_id):
                date_rows = by_date.get(day['prediction_date'])
                if date_rows is None:
                    raise ValueError(f"LightGBM OOF 缓存缺少 Fold {fold_id} 日期 {day['prediction_date']}")
                ordered = date_rows.reindex(
                    np.asarray(day['stock_indices'], dtype=np.int64),
                )
                if ordered[features].isna().any().any():
                    raise ValueError('LightGBM OOF 缓存与 Transformer 股票集合不一致')
                rows[day['prediction_date']] = models[fold_id].predict(
                    ordered[features], num_iteration=models[fold_id].best_iteration_,
                )
            cached[fold_id] = rows
            save_joblib_checkpoint(
                {'signature': signature, 'scores': rows}, cache_path,
            )
            print(f'LightGBM OOF 分数缓存已原子写入：Fold {fold_id}')

    for day in ensemble_days:
        score = cached[int(day['fold'])].get(day['prediction_date'])
        if score is None or len(score) != len(day['stock_indices']):
            raise ValueError('LightGBM OOF 缓存日期或股票边界校验失败')
        day['lgbm_scores'] = np.asarray(score, dtype=np.float64)
    return [{'fold': int(fold['fold']), 'cached': True} for fold in folds]


def recent_lgbm_training_frame(data, train_end, window_days):
    """按实际交易日截取树模型最近训练窗口，保留同日全部股票。"""
    if int(window_days) < 1:
        raise ValueError('lgbm_train_window_days 必须为正整数')
    dates = np.sort(data.loc[data['日期'] <= train_end, '日期'].unique())
    if len(dates) < int(window_days):
        raise ValueError(
            f'LightGBM 仅有 {len(dates)} 个训练日，不足近期窗口 {window_days}'
        )
    selected_dates = dates[-int(window_days):]
    frame = data.loc[data['日期'].isin(selected_dates)].sort_values(
        ['日期', 'instrument'], kind='mergesort',
    ).copy()
    return frame, selected_dates


def fit_lgbm_oof_scores(data, features, folds, oof_records, output_dir):
    """同一训练文件内的表格排序补充模型；只以预测日连续输入训练。"""
    if not config.get('lgbm_enabled', False):
        return []
    try:
        from lightgbm import LGBMRanker, early_stopping
    except ImportError as exc:
        raise RuntimeError('v1.20 需要 lightgbm；请在 code 目录运行 uv sync') from exc
    if INDUSTRY_NEUTRAL_TARGET not in data:
        raise ValueError('LightGBM 需要行业中性标签')
    results = []
    lgbm_dir = os.path.join(output_dir, 'lgbm')
    os.makedirs(lgbm_dir, exist_ok=True)
    window_days = int(config.get('lgbm_train_window_days', 0))
    if window_days < 1:
        raise ValueError('lgbm_train_window_days 必须为正整数')
    for fold in folds:
        try:
            train, train_dates = recent_lgbm_training_frame(
                data, fold['train_end'], window_days,
            )
        except ValueError as exc:
            raise ValueError(f"LightGBM Fold {fold['fold']}：{exc}") from exc
        valid = data.loc[
            (data['日期'] >= fold['val_start']) & (data['日期'] <= fold['val_end'])
        ].sort_values(['日期', 'instrument'], kind='mergesort').copy()
        groups = train.groupby('日期', sort=False).size().to_numpy()
        valid_groups = valid.groupby('日期', sort=False).size().to_numpy()
        if train.empty or valid.empty or groups.min() < 2 or valid_groups.min() < 2:
            raise ValueError(f"LightGBM Fold {fold['fold']} 的日期分组不足")
        description = f"LightGBM Fold {fold['fold']}"
        checkpoint_path = os.path.join(lgbm_dir, f"fold_{fold['fold']}.joblib")
        if resume_training_enabled() and os.path.isfile(checkpoint_path):
            model = joblib.load(checkpoint_path)
            validate_lgbm_checkpoint(model, features, checkpoint_path)
            print(f"恢复 LightGBM Fold {fold['fold']} checkpoint")
        else:
            train_labels = lgbm_relevance_labels(train)
            valid_labels = lgbm_relevance_labels(valid)
            print(
                f"阶段 LightGBM：Fold {fold['fold']}，近期 {window_days} 日 "
                f"({pd.Timestamp(train_dates[0]).date()} ~ {pd.Timestamp(train_dates[-1]).date()})，"
                f"训练 {len(train):,} / 验证 {len(valid):,} 行"
            )
            model = LGBMRanker(
                objective=config.get('lgbm_objective', 'lambdarank'), metric='ndcg',
                learning_rate=config['lgbm_learning_rate'], n_estimators=config['lgbm_n_estimators'],
                num_leaves=config['lgbm_num_leaves'], min_child_samples=config['lgbm_min_child_samples'],
                colsample_bytree=config['lgbm_feature_fraction'], subsample=config['lgbm_bagging_fraction'],
                reg_lambda=config['lgbm_lambda_l2'], random_state=config['seed'],
                n_jobs=config['lgbm_n_jobs'], verbosity=-1,
            )
            progress, update_progress = lgbm_progress_callback(
                description, config['lgbm_n_estimators'],
            )
            try:
                model.fit(
                    train[features], train_labels, group=groups,
                    eval_X=valid[features], eval_y=valid_labels,
                    eval_group=[valid_groups], eval_at=[5],
                    callbacks=[
                        update_progress,
                        early_stopping(config['lgbm_early_stopping_rounds'], verbose=False),
                    ],
                )
            finally:
                progress.close()
            print(f"阶段 LightGBM：Fold {fold['fold']} 完成，最佳 {model.best_iteration_} 棵树")
            save_joblib_checkpoint(model, checkpoint_path)
        by_date = {pd.Timestamp(date).strftime('%Y-%m-%d'): rows.set_index('instrument')
                   for date, rows in valid.groupby('日期', sort=False)}
        for records in oof_records[int(fold['fold'])].values():
            for record in records:
                rows = by_date.get(record['prediction_date'])
                if rows is None:
                    continue
                ordered = rows.reindex(np.asarray(record['stock_indices'], dtype=np.int64))
                if ordered[features].isna().any().any():
                    raise ValueError('LightGBM OOF 与 Transformer 股票集合不一致')
                record['lgbm_scores'] = model.predict(ordered[features], num_iteration=model.best_iteration_)
        results.append({
            'fold': int(fold['fold']),
            'best_iteration': int(model.best_iteration_ or config['lgbm_n_estimators']),
            'train_window_days': window_days,
            'train_start': pd.Timestamp(train_dates[0]).strftime('%Y-%m-%d'),
            'train_end': pd.Timestamp(train_dates[-1]).strftime('%Y-%m-%d'),
        })
    return results


def fit_lgbm_final(data, features, fold_results, output_dir):
    """以折中位最佳轮数重训部署树模型；锁箱仍不进入数据。"""
    if not fold_results:
        return None
    from lightgbm import LGBMRanker
    iterations = int(np.median([row['best_iteration'] for row in fold_results]))
    path = os.path.join(output_dir, 'lgbm_ranker.joblib')
    if resume_training_enabled() and os.path.isfile(path):
        model = joblib.load(path)
        validate_lgbm_checkpoint(model, features, path, expected_iterations=iterations)
        print('恢复 LightGBM 全量重训 checkpoint')
        return os.path.basename(path)
    window_days = int(config.get('lgbm_train_window_days', 0))
    data, window_dates = recent_lgbm_training_frame(
        data, data['日期'].max(), window_days,
    )
    groups = data.groupby('日期', sort=False).size().to_numpy()
    labels = lgbm_relevance_labels(data)
    model = LGBMRanker(
        objective=config.get('lgbm_objective', 'lambdarank'), learning_rate=config['lgbm_learning_rate'],
        n_estimators=iterations, num_leaves=config['lgbm_num_leaves'],
        min_child_samples=config['lgbm_min_child_samples'],
        colsample_bytree=config['lgbm_feature_fraction'], subsample=config['lgbm_bagging_fraction'],
        reg_lambda=config['lgbm_lambda_l2'], random_state=config['seed'],
        n_jobs=config['lgbm_n_jobs'], verbosity=-1,
    )
    print(
        f'阶段 LightGBM：全量开发期近期 {window_days} 日重训 '
        f'({pd.Timestamp(window_dates[0]).date()} ~ {pd.Timestamp(window_dates[-1]).date()})，'
        f'{len(data):,} 行，迭代 {iterations}'
    )
    progress, update_progress = lgbm_progress_callback(
        'LightGBM 全量重训', iterations,
    )
    try:
        model.fit(data[features], labels, group=groups, callbacks=[update_progress])
    finally:
        progress.close()
    save_joblib_checkpoint(model, path)
    return os.path.basename(path)


def attach_oof_strategy_metadata(ensemble_days, data):
    """把预测日已知的行业快照和市场状态接到 OOF，绝不写入模型输入。"""
    required = [INDUSTRY_ASOF_COLUMN, 'market_return_20', 'market_downside_vol_20']
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f'OOF 策略元数据缺少列: {missing}')
    lookup = data.set_index(['日期', 'instrument'])[required]
    for day in ensemble_days:
        index = pd.MultiIndex.from_arrays([
            np.repeat(pd.Timestamp(day['prediction_date']), len(day['stock_indices'])),
            np.asarray(day['stock_indices'], dtype=np.int64),
        ], names=['日期', 'instrument'])
        rows = lookup.reindex(index)
        if rows[['market_return_20', 'market_downside_vol_20']].isna().any().any():
            raise ValueError(f"OOF {day['prediction_date']} 缺少市场状态")
        day['industry_labels'] = rows[INDUSTRY_ASOF_COLUMN].to_numpy(dtype=object)
        day['market_return_20'] = float(rows['market_return_20'].median())
        day['market_downside_vol_20'] = float(rows['market_downside_vol_20'].median())
    return ensemble_days


def market_state_diagnostics(metrics, data, folds):
    """按各折训练期中位数给 OOF 打四个因果市场状态标签，仅报告不选型。"""
    daily = metrics.get('daily', [])
    medians = {}
    for fold in folds:
        train = data.loc[data['日期'] <= fold['train_end']]
        medians[int(fold['fold'])] = {
            name: float(train[name].median())
            for name in ('market_return_20', 'market_downside_vol_20')
        }
    states = {}
    for row in daily:
        fold = int(row['fold'])
        median = medians[fold]
        high_return = float(row.get('market_return_20', 0.0)) >= median['market_return_20']
        high_downside = float(row.get('market_downside_vol_20', 0.0)) >= median['market_downside_vol_20']
        name = (
            f"market_return_20_{'high' if high_return else 'low'}__"
            f"market_downside_vol_20_{'high' if high_downside else 'low'}"
        )
        states.setdefault(name, []).append(row['weighted_portfolio_return'])
    return {
        'training_medians_by_fold': medians,
        'states': {
            name: {
                'count': len(values),
                'mean_weighted_portfolio_return': float(np.mean(values)),
                'p10_weighted_portfolio_return': float(np.quantile(values, 0.10)),
            }
            for name, values in sorted(states.items())
        },
        'selection_use': 'diagnostic_only',
    }


def fuse_lgbm_scores(days, weight):
    """按日百分位融合；Allocation、Exposure、风险头保持 Transformer 来源。"""
    fused = []
    for day in days:
        copied = dict(day)
        if weight == 0 or 'lgbm_scores' not in day:
            fused.append(copied)
            continue
        score = ((1.0 - weight) * percentile_ranks(np.mean(day['scores'], axis=0))
                 + weight * percentile_ranks(day['lgbm_scores']))
        copied['scores'] = score.reshape(1, -1)
        for key in ('allocation_logits', 'risk_1d_probabilities', 'risk_3d_probabilities',
                    'risk_5d_probabilities', 'tail_5d_probabilities'):
            copied[key] = np.mean(day[key], axis=0, keepdims=True)
        copied['exposures'] = np.asarray([np.median(day['exposures'])])
        copied['regime_gates'] = np.asarray([np.median(day['regime_gates'])])
        fused.append(copied)
    return fused


def fuse_lgbm_scores_by_fold(days, lgbm_forward):
    """按各自前向折权重融合，禁止把后续折决定回填到早期 OOF。"""
    weights = {
        int(row['fold']): float(row['lgbm_weight'])
        for row in lgbm_forward
    }
    folds = {int(day['fold']) for day in days}
    if folds != set(weights):
        raise ValueError('逐折 LightGBM 融合缺少 OOF 折权重')
    fused = []
    for fold_id in sorted(folds):
        fused.extend(fuse_lgbm_scores(
            [day for day in days if int(day['fold']) == fold_id],
            weights[fold_id],
        ))
    return fused


def calibrate_v17_oof_strategy(ensemble_days, folds, calibration_kwargs, lgbm_folds):
    """完成逐折严格前向 LightGBM 融合与策略校准。"""
    # LightGBM 权重选择只需要旧版完整策略校准接口支持的字段；其余字段
    # 留给后续严格模块门控，避免把新模块参数传给兼容接口。
    lgbm_calibration_kwargs = {
        key: value for key, value in calibration_kwargs.items()
        if key not in {
            'cluster_max_raw_rank', 'cluster_cap_grid',
            'minimum_allocation_blend', 'minimum_exposure_blend',
            'policy_simplicity_tolerance', 'module_min_positive_fold_fraction',
        }
    }
    lgbm_forward = []
    if lgbm_folds:
        for fold in folds:
            fold_id = int(fold['fold'])
            if fold_id <= 2:
                weight, source = 0.0, 'fallback_insufficient_earlier_folds'
            else:
                earlier = [
                    day for day in ensemble_days
                    if int(day['fold']) < fold_id
                    and pd.Timestamp(day['label_end_date']) < fold['val_start']
                ]
                if len({int(day['fold']) for day in earlier}) < 2:
                    raise ValueError(f'Fold {fold_id} 没有两折已实现的更早 OOF 标签')
                candidates = [
                    (
                        calibrate_ensemble_policy(
                            fuse_lgbm_scores(earlier, float(candidate)),
                            **lgbm_calibration_kwargs,
                        )['oof_metrics']['mean_weighted_portfolio_return'],
                        float(candidate),
                    )
                    for candidate in config['lgbm_blend_grid']
                ]
                _, weight = max(candidates, key=lambda row: (row[0], -row[1]))
                source = 'strict_earlier_oof'
            lgbm_forward.append({
                'fold': fold_id, 'lgbm_weight': weight, 'source': source,
            })
        # 部署只能沿用最后一折当时可得的选择；绝不能中位数回填早期 OOF。
        deployment_row = lgbm_forward[-1]
        deployment_lgbm_weight = float(deployment_row['lgbm_weight'])
        ensemble_days = fuse_lgbm_scores_by_fold(ensemble_days, lgbm_forward)
    else:
        deployment_lgbm_weight = 0.0
    all_oof_policy = calibrate_ensemble_policy(
        ensemble_days, **lgbm_calibration_kwargs,
    )
    if not config.get('nested_oof_enabled', False):
        return {
            'lgbm_forward': lgbm_forward,
            'deployment_lgbm_weight': deployment_lgbm_weight,
            'policy': all_oof_policy,
            'cross_fitted_policy': {
                'method': 'disabled', 'metrics': all_oof_policy['oof_metrics'],
                'fold_policies': [], 'policy_stability': {},
            },
            'ensemble_metrics': all_oof_policy['oof_metrics'],
        }
    cross_fitted_policy = forward_fit_module_gated_policy(
        ensemble_days,
        forward_module_max_fold_loss=float(config.get(
            'forward_module_max_fold_loss', 0.0025,
        )),
        forward_module_max_p10_loss=float(config.get(
            'forward_module_max_p10_loss', 0.005,
        )),
        **calibration_kwargs,
    )
    cross_fitted_policy['deployment_policy_differences'] = {
        field: [
            float(all_oof_policy[field]) - float(row['policy'][field])
            for row in cross_fitted_policy['fold_policies']
        ]
        for field in (
            'allocation_blend', 'disagreement_gamma', 'selection_risk_gamma',
            'risk_score_penalty', 'correlation_exposure_gamma', 'exposure_head_blend',
        )
    }
    return {
        'lgbm_forward': lgbm_forward,
        'deployment_lgbm_weight': deployment_lgbm_weight,
        'deployment_lgbm_weight_source': {
            'fold': int(deployment_row['fold']) if lgbm_folds else None,
            'source': deployment_row['source'] if lgbm_folds else 'transformer_only',
        },
        # 复制部署策略，避免写入 cross_fitted_oof 时形成循环引用。
        'policy': dict(cross_fitted_policy['robust_deployment_policy']),
        'cross_fitted_policy': cross_fitted_policy,
        'ensemble_metrics': cross_fitted_policy['metrics'],
    }


def promotion_against_baseline(metrics, baseline_metrics):
    """v1.20 的唯一晋级门槛：收益增益和逐日/回撤 10bp 护栏。"""
    tolerance = 1e-12
    def max_drawdown(source):
        if 'max_drawdown' in source:
            return float(source['max_drawdown'])
        daily = sorted(source.get('daily', []), key=lambda row: row['prediction_date'])
        if not daily:
            raise ValueError('晋级判断缺少计算最大回撤所需的逐日收益')
        returns = np.asarray([
            row['weighted_portfolio_return'] for row in daily
        ], dtype=np.float64)
        return maximum_drawdown(returns)
    candidate_folds = {int(row['fold']): row for row in metrics['folds']}
    baseline_folds = {
        int(row['fold']): row for row in baseline_metrics['folds']
    }
    fold_deltas = [
        {
            'fold': fold_id,
            'weighted_return_delta': float(
                candidate_folds[fold_id]['mean_weighted_portfolio_return']
                - baseline_folds[fold_id]['mean_weighted_portfolio_return']
            ),
            'rank_ic_delta': float(
                candidate_folds[fold_id]['mean_rank_ic']
                - baseline_folds[fold_id]['mean_rank_ic']
            ),
        }
        for fold_id in sorted(candidate_folds)
    ]
    deltas = {
        key: float(metrics[key] - baseline_metrics[key])
        for key in (
            'mean_weighted_portfolio_return',
            'p10_weighted_portfolio_return',
            'worst_fold_weighted_portfolio_return',
            'mean_rank_ic',
        )
    }
    deltas['worst_weighted_portfolio_return'] = float(
        metrics['worst_weighted_portfolio_return']
        - baseline_metrics['worst_weighted_portfolio_return']
    )
    deltas['max_drawdown'] = max_drawdown(metrics) - max_drawdown(baseline_metrics)
    checks = {
        'mean_weighted_return_gain_at_least_10bp': (
            deltas['mean_weighted_portfolio_return'] >= 0.001 - tolerance
        ),
        'at_least_two_positive_fold_gains': (
            sum(row['weighted_return_delta'] > 0.0 for row in fold_deltas) >= 2
        ),
        'p10_not_worse_than_10bp': (
            deltas['p10_weighted_portfolio_return'] >= -0.001 - tolerance
        ),
        'worst_fold_not_worse_than_10bp': (
            deltas['worst_fold_weighted_portfolio_return'] >= -0.001 - tolerance
        ),
        'worst_day_not_worse_than_10bp': (
            deltas['worst_weighted_portfolio_return'] >= -0.001 - tolerance
        ),
        'max_drawdown_not_worse_than_10bp': (
            deltas['max_drawdown'] >= -0.001 - tolerance
        ),
        'rank_ic_not_worse_than_0_005': (
            deltas['mean_rank_ic'] >= -0.005 - tolerance
        ),
    }
    return {
        'applicable': True, **checks, 'passed': bool(all(checks.values())),
        'metric_deltas': deltas, 'fold_deltas': fold_deltas,
    }


def calibrate_v20_recent_lgbm_candidate(ensemble_days, calibration_kwargs):
    """v1.20 的唯一预注册选股：近期 504 日 rank_xendcg，不做模型融合。"""
    replay = forward_fit_module_gated_policy(
        fuse_lgbm_scores(ensemble_days, 1.0),
        forward_module_max_fold_loss=float(config.get(
            'forward_module_max_fold_loss', 0.0025,
        )),
        forward_module_max_p10_loss=float(config.get(
            'forward_module_max_p10_loss', 0.005,
        )),
        **calibration_kwargs,
    )
    policy = dict(replay['robust_deployment_policy'])
    policy.update({
        'candidate_name': 'recent504_rankxendcg_lgbm_indcap',
        'lgbm_weight': 1.0,
        'pre_registered': True,
    })
    return {
        'lgbm_weight': 1.0,
        'cross_fitted_policy': replay,
        'metrics': replay['metrics'],
        'policy': policy,
    }


def oof_strategy_checkpoint_signature(features, folds, lgbm_folds, ensemble_seeds):
    """为恢复检查绑定特征、折边界、树模型迭代数和完整训练配置。"""
    payload = {
        'feature_sha256': hashlib.sha256('\n'.join(features).encode()).hexdigest(),
        'folds': [{
            key: int(value) if key == 'fold' else pd.Timestamp(value).strftime('%Y-%m-%d')
            for key, value in fold.items()
        } for fold in folds],
        'lgbm_folds': lgbm_folds,
        'ensemble_seeds': list(ensemble_seeds),
        'config': config,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def recover_completed_stage_results(fold_dir, steps_per_epoch):
    """从 TensorBoard 恢复已完整结束、但尚未来得及汇总的四阶段结果。"""
    checkpoint_path = os.path.join(fold_dir, 'best_model.pth')
    event_dir = os.path.join(fold_dir, 'log')
    if not os.path.isfile(checkpoint_path) or not os.path.isdir(event_dir):
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        accumulator = EventAccumulator(event_dir)
        accumulator.Reload()
    except (ImportError, KeyError, OSError, ValueError):
        return None

    scalar_tags = set(accumulator.Tags().get('scalars', []))
    recovered = {}
    for stage in TRAINING_STAGES:
        settings = stage_settings(stage)
        learning_rate_tag = f'{stage}/learning_rate'
        eval_loss_tag = f'{stage}/eval/loss'
        required_tags = {learning_rate_tag, eval_loss_tag}
        if not required_tags.issubset(scalar_tags):
            return None
        learning_rate_events = accumulator.Scalars(learning_rate_tag)
        eval_loss_events = accumulator.Scalars(eval_loss_tag)
        if not learning_rate_events or not eval_loss_events:
            return None
        epochs_ran = max(event.step for event in learning_rate_events) + 1
        if max(event.step for event in eval_loss_events) >= epochs_ran:
            return None

        checkpoint_metric = settings['checkpoint_metric']
        if checkpoint_metric == 'negative_eval_loss':
            score_by_step = {
                event.step: -float(event.value)
                for event in eval_loss_events
            }
        else:
            metric_tags = {
                'top5_return_plus_rank_ic': (
                    'top5_return',
                    'rank_ic',
                ),
                'weighted_portfolio_return_plus_rank_ic': (
                    'weighted_portfolio_return',
                    'rank_ic',
                ),
                'allocation_contribution': (
                    'allocation_contribution',
                ),
                'weighted_portfolio_risk_adjusted': (
                    'weighted_portfolio_return',
                    'weighted_portfolio_downside_deviation',
                ),
            }.get(checkpoint_metric, (checkpoint_metric,))
            events_by_metric = {}
            for metric in metric_tags:
                tag = f'{stage}/eval/{metric}'
                if tag not in scalar_tags:
                    return None
                events_by_metric[metric] = {
                    event.step: float(event.value)
                    for event in accumulator.Scalars(tag)
                }
            score_by_step = {}
            for event in eval_loss_events:
                step_metrics = {
                    metric: values[event.step]
                    for metric, values in events_by_metric.items()
                    if event.step in values
                }
                if len(step_metrics) != len(events_by_metric):
                    return None
                score_by_step[event.step] = calculate_checkpoint_score(
                    step_metrics,
                    checkpoint_metric,
                )

        best_score = -float('inf')
        best_step = None
        epochs_without_improvement = 0
        for step in sorted(score_by_step):
            current_score = score_by_step[step]
            if current_score > best_score:
                best_score = current_score
                best_step = step
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        completed_by_limit = epochs_ran == settings['max_epochs']
        completed_by_patience = (
            epochs_without_improvement >= settings['patience']
            and max(score_by_step) == epochs_ran - 1
        )
        if not (completed_by_limit or completed_by_patience):
            return None
        recovered[stage] = {
            'stage': stage,
            'best_epoch': int(best_step + 1),
            'epochs_ran': int(epochs_ran),
            'checkpoint_metric': checkpoint_metric,
            'checkpoint_score': float(best_score),
            'steps_per_epoch': int(steps_per_epoch),
            'recovered_from_tensorboard': True,
        }
    return recovered


feature_cloums_map = {
    '39': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39_reduced20': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39_reduced20': engineer_features_158plus39
}

# 这五列均可由同一时点已保留列线性恢复；本次消融不改变特征工程本身。
LINEAR_REDUNDANT_FEATURES = {
    'high_low_spread', 'open_close_spread', 'high_close_spread',
    'low_close_spread', 'kdj_j',
}
feature_cloums_map['158+39_reduced25'] = [
    name for name in feature_cloums_map['158+39_reduced20']
    if name not in LINEAR_REDUNDANT_FEATURES
]
feature_engineer_func_map['158+39_reduced25'] = engineer_features_158plus39
feature_cloums_map[RELATIVE_MARKET_FEATURE_SET] = [
    *feature_cloums_map['158+39_reduced25'],
    *RELATIVE_MARKET_FEATURES,
]
feature_engineer_func_map[RELATIVE_MARKET_FEATURE_SET] = engineer_features_158plus39
feature_cloums_map[RISK_MARKET_FEATURE_SET] = [
    *feature_cloums_map[RELATIVE_MARKET_FEATURE_SET],
    *RISK_MARKET_FEATURES,
]
feature_engineer_func_map[RISK_MARKET_FEATURE_SET] = engineer_features_158plus39
feature_cloums_map[INDUSTRY_RESIDUAL_FEATURE_SET] = [
    *feature_cloums_map[RISK_MARKET_FEATURE_SET],
    *INDUSTRY_RESIDUAL_FEATURES,
]
feature_engineer_func_map[INDUSTRY_RESIDUAL_FEATURE_SET] = (
    engineer_features_158plus39
)
assert len(feature_cloums_map['158+39_reduced20']) == 171
assert len(feature_cloums_map['158+39_reduced25']) == 166
assert len(feature_cloums_map[RELATIVE_MARKET_FEATURE_SET]) == 178
assert len(feature_cloums_map[RISK_MARKET_FEATURE_SET]) == 193
assert len(feature_cloums_map[INDUSTRY_RESIDUAL_FEATURE_SET]) == 205


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t2'] = processed.groupby('股票代码')['开盘'].shift(-2)
    processed['open_t4'] = processed.groupby('股票代码')['开盘'].shift(-4)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    path_loss_enabled = bool(config.get('path_loss_5d_head_enabled', False))
    tail_target_mode = config.get('tail_5d_target_mode', 'endpoint_return')
    if tail_target_mode not in {'endpoint_return', 'holding_path_min'}:
        raise ValueError('tail_5d_target_mode 只能为 endpoint_return 或 holding_path_min')
    if path_loss_enabled or tail_target_mode == 'holding_path_min':
        lows = processed.groupby('股票代码')['最低']
        for offset in range(1, 6):
            processed[f'low_t{offset}'] = lows.shift(-offset)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed.loc[processed['open_t1'] > 1e-4].copy()

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['return_1d_target'] = (
        (processed['open_t2'] - processed['open_t1'])
        / (processed['open_t1'] + 1e-12)
    )
    processed['return_3d_target'] = (
        (processed['open_t4'] - processed['open_t1'])
        / (processed['open_t1'] + 1e-12)
    )
    if path_loss_enabled or tail_target_mode == 'holding_path_min':
        future_low = processed[[f'low_t{offset}' for offset in range(1, 6)]].min(axis=1)
        processed['path_loss_5d_target'] = (
            future_low - processed['open_t1']
        ) / (processed['open_t1'] + 1e-12)
    else:
        # 使旧数据集构造接口也始终有此列；该值不会进入损失。
        processed['path_loss_5d_target'] = processed['label']
    risk_1d_temperature = float(config.get('risk_1d_target_temperature', 0.01))
    risk_3d_temperature = float(config.get('risk_3d_target_temperature', 0.02))
    processed['risk_1d_target'] = 1.0 / (
        1.0 + np.exp(np.clip(
            processed['return_1d_target'] / risk_1d_temperature,
            -30.0,
            30.0,
        ))
    )
    processed['risk_3d_target'] = 1.0 / (
        1.0 + np.exp(np.clip(
            processed['return_3d_target'] / risk_3d_temperature,
            -30.0,
            30.0,
        ))
    )
    required_labels = [
        'label',
        'risk_1d_target',
        'risk_3d_target',
    ]
    if path_loss_enabled:
        required_labels.append('path_loss_5d_target')
    processed = processed.dropna(subset=required_labels)

    dates = processed['日期']
    tail_threshold = float(config.get('tail_5d_threshold', -0.03))
    market_future_return = processed['label'].groupby(dates).transform('mean')
    tail_return = (
        processed['path_loss_5d_target']
        if tail_target_mode == 'holding_path_min' else processed['label']
    )
    processed['tail_5d_target'] = tail_return.le(tail_threshold).astype(np.float32)
    market_tail_share = processed['tail_5d_target'].groupby(dates).transform('mean')
    market_return_temperature = float(
        config.get('regime_market_return_temperature', 0.01)
    )
    tail_share_baseline = float(
        config.get('regime_tail_share_baseline', 0.20)
    )
    tail_share_temperature = float(
        config.get('regime_tail_share_temperature', 0.10)
    )
    if market_return_temperature <= 0 or tail_share_temperature <= 0:
        raise ValueError('Regime目标 temperature 必须大于0')
    regime_logit = (
        0.60 * (-market_future_return / market_return_temperature)
        + 0.40 * (
            (market_tail_share - tail_share_baseline)
            / tail_share_temperature
        )
    )
    processed['regime_target'] = 1.0 / (
        1.0 + np.exp(np.clip(-regime_logit, -30.0, 30.0))
    )

    temporary_columns = ['open_t1', 'open_t2', 'open_t4', 'open_t5']
    temporary_columns.extend(
        f'low_t{offset}' for offset in range(1, 6)
        if f'low_t{offset}' in processed.columns
    )
    processed.drop(
        columns=temporary_columns,
        inplace=True,
    )
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    if config['feature_num'] in {
        RELATIVE_MARKET_FEATURE_SET,
        RISK_MARKET_FEATURE_SET,
        INDUSTRY_RESIDUAL_FEATURE_SET,
    }:
        processed = add_relative_market_features(processed)
    if config['feature_num'] == INDUSTRY_RESIDUAL_FEATURE_SET:
        industry_history_path = config.get('industry_history_path')
        if not industry_history_path:
            raise ValueError('205维行业残差特征需要 industry_history_path')
        print('阶段 2/4：按 as-of 行业快照计算 12 个行业残差特征')
        processed = add_industry_residual_features(
            processed,
            industry_history_path,
            min_industry_size=int(config.get('minimum_industry_size', 3)),
            industry_column=INDUSTRY_ASOF_COLUMN,
        )

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    if config['feature_num'] == INDUSTRY_RESIDUAL_FEATURE_SET:
        print('阶段 3/4：构建同期行业中性五日收益标签')
        processed = add_industry_neutral_label(
            processed,
            target_column='label',
            industry_column=INDUSTRY_ASOF_COLUMN,
            output_column=INDUSTRY_NEUTRAL_TARGET,
            min_industry_size=int(config.get('minimum_industry_size', 3)),
        )
        processed = processed.dropna(subset=[INDUSTRY_NEUTRAL_TARGET])
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """
    def __init__(self, listwise_temperature=0.2, listwise_weight=0.2, k=5,
                 weight_factor=2.0, pairwise_weight=1,
                 base_weight=1.0, regression_weight=0.2, regression_beta=0.02,
                 ic_weight=0.2, allocation_weight=0.1, exposure_weight=1.0,
                 allocation_temperature=1.0, allocation_target_temperature=0.02,
                 allocation_candidate_k=20, allocation_return_clip=0.10,
                 exposure_target_temperature=0.02, min_exposure=0.80,
                 max_exposure=0.999999,
                 exposure_selected_return_weight=0.70,
                 exposure_market_return_weight=0.30,
                 exposure_downside_weight=0.25,
                 id_gate_regularization=0.0,
                 lambdarank_candidate_k=20,
                 lambdarank_hard_negative_k=20,
                 lambdarank_return_gap_scale=0.02,
                 risk_1d_weight=0.0,
                 risk_3d_weight=0.0,
                 risk_5d_weight=0.0,
                 risk_5d_target_temperature=0.03,
                 tail_5d_weight=0.0,
                 tail_5d_threshold=-0.03,
                 regime_weight=0.0,
                 industry_residual_weight=0.0,
                 industry_residual_beta=0.02,
                 path_loss_5d_weight=0.0,
                 path_loss_5d_beta=0.02):
        super(WeightedRankingLoss, self).__init__()
        if listwise_temperature <= 0:
            raise ValueError('listwise_temperature 必须大于 0')
        self.listwise_temperature = listwise_temperature
        self.listwise_weight = listwise_weight
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.regression_weight = regression_weight
        self.regression_beta = regression_beta
        self.ic_weight = ic_weight
        self.allocation_weight = allocation_weight
        self.exposure_weight = exposure_weight
        self.allocation_temperature = allocation_temperature
        self.allocation_target_temperature = allocation_target_temperature
        self.allocation_candidate_k = int(allocation_candidate_k)
        self.allocation_return_clip = float(allocation_return_clip)
        self.exposure_target_temperature = exposure_target_temperature
        self.exposure_selected_return_weight = float(
            exposure_selected_return_weight
        )
        self.exposure_market_return_weight = float(
            exposure_market_return_weight
        )
        self.exposure_downside_weight = float(exposure_downside_weight)
        self.id_gate_regularization = float(id_gate_regularization)
        self.lambdarank_candidate_k = int(lambdarank_candidate_k)
        self.lambdarank_hard_negative_k = int(
            lambdarank_hard_negative_k
        )
        self.lambdarank_return_gap_scale = float(
            lambdarank_return_gap_scale
        )
        self.risk_1d_weight = float(risk_1d_weight)
        self.risk_3d_weight = float(risk_3d_weight)
        self.risk_5d_weight = float(risk_5d_weight)
        self.risk_5d_target_temperature = float(
            risk_5d_target_temperature
        )
        self.tail_5d_weight = float(tail_5d_weight)
        self.tail_5d_threshold = float(tail_5d_threshold)
        self.regime_weight = float(regime_weight)
        self.industry_residual_weight = float(industry_residual_weight)
        self.industry_residual_beta = float(industry_residual_beta)
        self.path_loss_5d_weight = float(path_loss_5d_weight)
        self.path_loss_5d_beta = float(path_loss_5d_beta)
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        if min(
            allocation_temperature,
            allocation_target_temperature,
            exposure_target_temperature,
        ) <= 0:
            raise ValueError('仓位损失的 temperature 必须大于 0')
        if not 0.0 <= min_exposure < max_exposure < 1.0:
            raise ValueError('仓位范围必须满足 0 <= min_exposure < max_exposure < 1')
        if self.allocation_candidate_k < self.k:
            raise ValueError('allocation_candidate_k 不能小于 Top-k')
        if self.allocation_return_clip <= 0:
            raise ValueError('allocation_return_clip 必须大于 0')
        if min(
            self.exposure_selected_return_weight,
            self.exposure_market_return_weight,
            self.exposure_downside_weight,
            self.id_gate_regularization,
        ) < 0:
            raise ValueError('Exposure 与 ID gate 损失权重不能为负')
        if (
            self.lambdarank_candidate_k < self.k
            or self.lambdarank_hard_negative_k < 0
            or self.lambdarank_return_gap_scale <= 0
        ):
            raise ValueError('LambdaRank候选数量或收益差尺度不合法')
        if min(
            self.risk_1d_weight,
            self.risk_3d_weight,
            self.risk_5d_weight,
            self.tail_5d_weight,
            self.regime_weight,
            self.industry_residual_weight,
            self.path_loss_5d_weight,
        ) < 0:
            raise ValueError('风险头和状态门控损失权重不能为负')
        if self.risk_5d_target_temperature <= 0:
            raise ValueError('5日风险目标 temperature 必须大于0')
        if self.tail_5d_threshold >= 0:
            raise ValueError('5日尾部风险阈值必须小于0')
        if self.industry_residual_beta <= 0 or self.path_loss_5d_beta <= 0:
            raise ValueError('辅助回归的 beta 必须大于0')

    def listwise_loss(self, y_pred, y_true, weights):
        """基于排名百分位构造平滑目标，再计算加权 Listwise Cross Entropy。"""
        log_pred_probs = F.log_softmax(y_pred, dim=1)
        rank_min = y_true.min(dim=1, keepdim=True).values
        rank_range = (
            y_true.max(dim=1, keepdim=True).values - rank_min
        ).clamp(min=1e-12)
        rank_percentiles = (y_true - rank_min) / rank_range
        target_probs = F.softmax(
            rank_percentiles / self.listwise_temperature,
            dim=1,
        )

        # 权重作用于目标概率后重新归一化，避免用约 N 个股票的权重和
        # 去除以总概率为 1 的交叉熵，导致 Listwise 项随股票数缩小。
        weighted_target_probs = target_probs * weights
        weighted_target_probs = weighted_target_probs / (
            weighted_target_probs.sum(dim=1, keepdim=True) + 1e-12
        )
        return -(weighted_target_probs * log_pred_probs).sum(dim=1).mean()

    def lambda_rank_loss(self, y_pred, y_true, raw_returns, weights):
        """以收益差和交换后的 ΔNDCG@5 加权困难候选股票对。"""
        day_losses = []
        for batch_index in range(y_pred.size(0)):
            scores = y_pred[batch_index]
            relevance = y_true[batch_index]
            returns = raw_returns[batch_index]
            num_items = scores.numel()
            candidate_k = min(self.lambdarank_candidate_k, num_items)
            hard_negative_k = min(
                self.lambdarank_hard_negative_k,
                max(0, num_items - candidate_k),
            )
            predicted_order = torch.argsort(
                scores.detach(),
                descending=True,
                stable=True,
            )
            true_order = torch.argsort(
                relevance.detach(),
                descending=True,
                stable=True,
            )
            candidate_indices = torch.unique(torch.cat([
                predicted_order[:candidate_k],
                true_order[:candidate_k],
                predicted_order[
                    candidate_k:candidate_k + hard_negative_k
                ],
            ]))
            if candidate_indices.numel() < 2:
                day_losses.append(scores.sum() * 0.0)
                continue

            predicted_positions = torch.empty(
                num_items,
                dtype=torch.long,
                device=scores.device,
            )
            predicted_positions[predicted_order] = torch.arange(
                num_items,
                device=scores.device,
            )
            relevance_min = relevance.min()
            relevance_range = (
                relevance.max() - relevance_min
            ).clamp(min=1e-12)
            normalized_relevance = (
                relevance - relevance_min
            ) / relevance_range
            gains = torch.pow(2.0, normalized_relevance) - 1.0
            cutoff = min(self.k, num_items)
            ideal_discounts = 1.0 / torch.log2(
                torch.arange(
                    cutoff,
                    device=scores.device,
                    dtype=scores.dtype,
                ) + 2.0
            )
            ideal_dcg = (
                gains[true_order[:cutoff]] * ideal_discounts
            ).sum().clamp(min=1e-12)

            local_scores = scores[candidate_indices]
            local_returns = returns[candidate_indices]
            local_gains = gains[candidate_indices]
            local_positions = predicted_positions[candidate_indices]
            local_discounts = torch.where(
                local_positions < cutoff,
                1.0 / torch.log2(
                    local_positions.to(scores.dtype) + 2.0
                ),
                torch.zeros_like(local_positions, dtype=scores.dtype),
            )
            score_difference = (
                local_scores[:, None] - local_scores[None, :]
            )
            return_difference = (
                local_returns[:, None] - local_returns[None, :]
            )
            delta_ndcg = (
                (local_gains[:, None] - local_gains[None, :]).abs()
                * (
                    local_discounts[:, None]
                    - local_discounts[None, :]
                ).abs()
                / ideal_dcg
            )
            return_gap_weight = (
                return_difference.abs()
                / self.lambdarank_return_gap_scale
            ).clamp(min=0.25, max=4.0)
            local_top_weight = weights[
                batch_index,
                candidate_indices,
            ]
            pair_weight = (
                delta_ndcg
                * return_gap_weight
                * (
                    local_top_weight[:, None]
                    + local_top_weight[None, :]
                )
                * 0.5
            )
            valid_pairs = torch.triu(
                return_difference.ne(0),
                diagonal=1,
            )
            pair_weight = pair_weight * valid_pairs
            weight_sum = pair_weight.sum()
            if weight_sum <= 1e-12:
                day_losses.append(scores.sum() * 0.0)
                continue
            pair_loss = F.softplus(
                -score_difference * torch.sign(return_difference)
            )
            day_losses.append(
                (pair_loss * pair_weight).sum() / weight_sum
            )
        return torch.stack(day_losses).mean()

    def rank_ic_loss(self, y_pred, y_true):
        """用预测分数与真实排名的 Pearson 相关性近似优化 Spearman Rank IC。"""
        pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        target_centered = y_true - y_true.mean(dim=1, keepdim=True)
        numerator = (pred_centered * target_centered).sum(dim=1)
        denominator = torch.sqrt(
            pred_centered.square().sum(dim=1)
            * target_centered.square().sum(dim=1)
        ).clamp(min=1e-12)
        correlation = numerator / denominator
        return (1.0 - correlation).mean()

    def allocation_and_exposure_loss(
        self,
        y_pred,
        allocation_logits,
        exposure,
        raw_returns,
    ):
        """监督预测 Top-k 内的相对权重，并根据该组合真实收益监督总仓位。"""
        allocation_k = min(self.allocation_candidate_k, y_pred.size(1))
        allocation_indices = torch.topk(
            y_pred.detach(),
            allocation_k,
            dim=1,
        ).indices
        selected_allocation_logits = allocation_logits.gather(
            1,
            allocation_indices,
        )
        allocation_returns = raw_returns.gather(1, allocation_indices).clamp(
            min=-self.allocation_return_clip,
            max=self.allocation_return_clip,
        )

        target_relative_weights = F.softmax(
            allocation_returns / self.allocation_target_temperature,
            dim=1,
        )
        predicted_log_weights = F.log_softmax(
            selected_allocation_logits / self.allocation_temperature,
            dim=1,
        )
        allocation_loss = -(
            target_relative_weights * predicted_log_weights
        ).sum(dim=1).mean()

        exposure_k = min(self.k, y_pred.size(1))
        exposure_indices = torch.topk(
            y_pred.detach(),
            exposure_k,
            dim=1,
        ).indices
        selected_returns = raw_returns.gather(1, exposure_indices)
        selected_mean_return = selected_returns.mean(dim=1)
        market_mean_return = raw_returns.mean(dim=1)
        selected_downside = torch.sqrt(
            torch.relu(-selected_returns).square().mean(dim=1) + 1e-12
        )
        exposure_signal = (
            self.exposure_selected_return_weight * selected_mean_return
            + self.exposure_market_return_weight * market_mean_return
            - self.exposure_downside_weight * selected_downside
        )
        target_exposure_probability = torch.sigmoid(
            exposure_signal / self.exposure_target_temperature
        )
        predicted_exposure_probability = (
            (exposure.reshape(-1) - self.min_exposure)
            / (self.max_exposure - self.min_exposure)
        ).clamp(min=1e-6, max=1.0 - 1e-6)
        exposure_loss = F.binary_cross_entropy(
            predicted_exposure_probability,
            target_exposure_probability,
        )
        return allocation_loss, exposure_loss
        
    def forward(
        self,
        y_pred,
        y_true,
        predicted_returns,
        raw_returns,
        allocation_logits,
        exposure,
        identity_gate=None,
        risk_1d_logits=None,
        risk_3d_logits=None,
        risk_5d_logits=None,
        tail_5d_logits=None,
        path_loss_5d_outputs=None,
        tail_5d_targets=None,
        path_loss_5d_targets=None,
        industry_residual_outputs=None,
        industry_neutral_targets=None,
        regime_gate=None,
        risk_1d_targets=None,
        risk_3d_targets=None,
        regime_targets=None,
        stage='joint',
        return_components=False,
    ):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        _, top_indices = torch.topk(y_true, k, dim=1)
        
        # 2. 创建权重向量
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
            
        components = {}
        if stage in {'ranking', 'joint'}:
            listwise = self.listwise_loss(y_pred, y_true, weights)
            lambdarank = self.lambda_rank_loss(
                y_pred,
                y_true,
                raw_returns,
                weights,
            )
            rank_ic = self.rank_ic_loss(y_pred, y_true)
            regression = F.smooth_l1_loss(
                predicted_returns,
                raw_returns,
                beta=self.regression_beta,
            )
            components.update({
                'listwise_loss': self.listwise_weight * listwise,
                'lambdarank_loss': self.pairwise_weight * lambdarank,
                'ic_loss': self.ic_weight * rank_ic,
                'regression_loss': self.regression_weight * regression,
            })
            if (
                industry_residual_outputs is not None
                and industry_neutral_targets is not None
                and self.industry_residual_weight > 0
            ):
                components['industry_residual_loss'] = (
                    self.industry_residual_weight * F.smooth_l1_loss(
                        industry_residual_outputs,
                        industry_neutral_targets,
                        beta=self.industry_residual_beta,
                    )
                )
        if stage in {'risk', 'joint'}:
            if (
                risk_1d_logits is not None
                and risk_1d_targets is not None
                and self.risk_1d_weight > 0
            ):
                components['risk_1d_loss'] = (
                    self.risk_1d_weight
                    * F.binary_cross_entropy_with_logits(
                        risk_1d_logits,
                        risk_1d_targets,
                    )
                )
            if (
                risk_3d_logits is not None
                and risk_3d_targets is not None
                and self.risk_3d_weight > 0
            ):
                components['risk_3d_loss'] = (
                    self.risk_3d_weight
                    * F.binary_cross_entropy_with_logits(
                        risk_3d_logits,
                        risk_3d_targets,
                    )
                )
            if risk_5d_logits is not None and self.risk_5d_weight > 0:
                risk_5d_targets = torch.sigmoid(
                    -raw_returns / self.risk_5d_target_temperature
                )
                components['risk_5d_loss'] = (
                    self.risk_5d_weight
                    * F.binary_cross_entropy_with_logits(
                        risk_5d_logits,
                        risk_5d_targets,
                    )
                )
            if (
                tail_5d_logits is not None
                and tail_5d_targets is not None
                and self.tail_5d_weight > 0
            ):
                components['tail_5d_loss'] = (
                    self.tail_5d_weight
                    * F.binary_cross_entropy_with_logits(
                        tail_5d_logits,
                        tail_5d_targets,
                    )
                )
            if (
                path_loss_5d_outputs is not None
                and path_loss_5d_targets is not None
                and self.path_loss_5d_weight > 0
            ):
                components['path_loss_5d_loss'] = (
                    self.path_loss_5d_weight * F.smooth_l1_loss(
                        path_loss_5d_outputs,
                        path_loss_5d_targets,
                        beta=self.path_loss_5d_beta,
                    )
                )
            if (
                regime_gate is not None
                and regime_targets is not None
                and self.regime_weight > 0
            ):
                components['regime_loss'] = (
                    self.regime_weight
                    * F.binary_cross_entropy(
                        regime_gate.reshape(-1),
                        regime_targets.mean(dim=1),
                    )
                )
        if stage in {'allocation', 'exposure', 'joint'}:
            allocation, exposure_loss = self.allocation_and_exposure_loss(
                y_pred,
                allocation_logits,
                exposure,
                raw_returns,
            )
            if stage in {'allocation', 'joint'}:
                components['allocation_loss'] = (
                    self.allocation_weight * allocation
                )
            if stage in {'exposure', 'joint'}:
                components['exposure_loss'] = (
                    self.exposure_weight * exposure_loss
                )
        if (
            stage in {'ranking', 'joint'}
            and identity_gate is not None
            and self.id_gate_regularization > 0
        ):
            components['id_gate_regularization'] = (
                self.id_gate_regularization * identity_gate.square()
            )
        if not components:
            raise ValueError(f'训练阶段没有启用损失项: {stage}')
        total_loss = sum(components.values())
        if return_components:
            return total_loss, components
        return total_loss

def calculate_ranking_metrics(
    y_pred,
    y_true,
    masks,
    predicted_returns=None,
    allocation_logits=None,
    exposures=None,
    allocation_temperature=1.0,
    k=5,
    return_records=False,
):
    """计算逐日收益分解；可同时返回逐日记录以避免 batch 均值偏差。"""
    batch_size = y_pred.size(0)
    records = []
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]
        return_mae = 0.0
        if predicted_returns is not None:
            valid_return_pred = predicted_returns[i][valid_indices]
            return_mae = torch.mean(
                torch.abs(valid_return_pred - valid_true)
            ).item()
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        top5_return = pred_return_sum / k
        allocation_only_return = top5_return
        equal_weight_at_exposure_return = top5_return
        weighted_portfolio_return = top5_return
        gross_exposure = 1.0
        max_position = 1.0 / k
        if allocation_logits is not None and exposures is not None:
            valid_allocation_logits = allocation_logits[i][valid_indices]
            selected_allocation_logits = valid_allocation_logits[pred_indices]
            relative_weights = F.softmax(
                selected_allocation_logits / allocation_temperature,
                dim=0,
            )
            gross_exposure = float(
                exposures[i].clamp(min=0.0, max=1.0).item()
            )
            positions = relative_weights * gross_exposure
            allocation_only_return = torch.sum(
                relative_weights * pred_top_returns
            ).item()
            equal_weight_at_exposure_return = (
                top5_return * gross_exposure
            )
            weighted_portfolio_return = torch.sum(
                positions * pred_top_returns
            ).item()
            max_position = positions.max().item()

        rank_ic = spearmanr(
            valid_pred.detach().cpu().numpy(),
            valid_true.detach().cpu().numpy(),
        ).statistic
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        random_return_sum = k * valid_true.mean().item()
        ratio_pred = (
            pred_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9 else 0.0
        )
        ratio_random = (
            random_return_sum / (max_return_sum + 1e-12)
            if abs(max_return_sum) > 1e-9 else 0.0
        )
        denominator = max_return_sum - random_return_sum
        final_score = (
            (pred_return_sum - random_return_sum) / (denominator + 1e-12)
            if abs(denominator) > 1e-6 else 0.0
        )
        records.append({
            'pred_return_sum': pred_return_sum,
            'top5_return': top5_return,
            'rank_ic': float(rank_ic) if np.isfinite(rank_ic) else 0.0,
            'return_mae': return_mae,
            'max_return_sum': max_return_sum,
            'random_return_sum': random_return_sum,
            'allocation_only_return': allocation_only_return,
            'equal_weight_at_exposure_return': (
                equal_weight_at_exposure_return
            ),
            'weighted_portfolio_return': weighted_portfolio_return,
            'allocation_contribution': (
                allocation_only_return - top5_return
            ),
            'exposure_contribution': (
                weighted_portfolio_return - allocation_only_return
            ),
            'gross_exposure': gross_exposure,
            'cash_weight': 1.0 - gross_exposure,
            'max_position': max_position,
            'ratio_pred': ratio_pred,
            'ratio_random': ratio_random,
            'final_score': final_score,
        })

    metrics = summarize_ranking_metric_records(records)
    if return_records:
        return metrics, records
    return metrics


def summarize_ranking_metric_records(records):
    if not records:
        return {
            'pred_return_sum': 0.0,
            'top5_return': 0.0,
            'rank_ic': 0.0,
            'return_mae': 0.0,
            'max_return_sum': 0.0,
            'random_return_sum': 0.0,
            'allocation_only_return': 0.0,
            'equal_weight_at_exposure_return': 0.0,
            'weighted_portfolio_return': 0.0,
            'allocation_contribution': 0.0,
            'exposure_contribution': 0.0,
            'gross_exposure': 0.0,
            'cash_weight': 0.0,
            'max_position': 0.0,
            'ratio_pred': 0.0,
            'ratio_random': 0.0,
            'final_score': 0.0,
            'weighted_portfolio_return_std': 0.0,
            'weighted_portfolio_return_p10': 0.0,
            'weighted_portfolio_return_worst': 0.0,
            'weighted_portfolio_positive_rate': 0.0,
            'weighted_portfolio_downside_deviation': 0.0,
            'num_evaluation_dates': 0,
        }
    keys = records[0].keys()
    metrics = {
        key: float(np.mean([record[key] for record in records]))
        for key in keys
    }
    weighted_returns = np.asarray([
        record['weighted_portfolio_return'] for record in records
    ], dtype=np.float64)
    negative_returns = np.minimum(weighted_returns, 0.0)
    metrics.update({
        'weighted_portfolio_return_std': float(weighted_returns.std()),
        'weighted_portfolio_return_p10': float(
            np.quantile(weighted_returns, 0.10)
        ),
        'weighted_portfolio_return_worst': float(weighted_returns.min()),
        'weighted_portfolio_positive_rate': float(
            np.mean(weighted_returns > 0.0)
        ),
        'weighted_portfolio_downside_deviation': float(
            np.sqrt(np.mean(negative_returns ** 2))
        ),
        'num_evaluation_dates': len(records),
    })
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(
        self,
        sequences,
        targets,
        relevance_scores,
        stock_indices,
        prediction_dates,
        risk_1d_targets=None,
        risk_3d_targets=None,
        regime_targets=None,
        tail_5d_targets=None,
        path_loss_5d_targets=None,
        industry_neutral_targets=None,
    ):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.prediction_dates = prediction_dates
        self.risk_1d_targets = (
            risk_1d_targets
            if risk_1d_targets is not None
            else [np.full_like(target, 0.5) for target in targets]
        )
        self.risk_3d_targets = (
            risk_3d_targets
            if risk_3d_targets is not None
            else [np.full_like(target, 0.5) for target in targets]
        )
        self.regime_targets = (
            regime_targets
            if regime_targets is not None
            else [np.full_like(target, 0.5) for target in targets]
        )
        self.tail_5d_targets = (
            tail_5d_targets if tail_5d_targets is not None
            else [np.zeros_like(target) for target in targets]
        )
        self.path_loss_5d_targets = (
            path_loss_5d_targets if path_loss_5d_targets is not None
            else [np.array(target, dtype=np.float32, copy=True) for target in targets]
        )
        self.industry_neutral_targets = (
            industry_neutral_targets if industry_neutral_targets is not None
            else [np.zeros_like(target) for target in targets]
        )
        lengths = {
            len(sequences),
            len(targets),
            len(relevance_scores),
            len(stock_indices),
            len(prediction_dates),
            len(self.risk_1d_targets),
            len(self.risk_3d_targets),
            len(self.regime_targets),
            len(self.tail_5d_targets),
            len(self.path_loss_5d_targets),
            len(self.industry_neutral_targets),
        }
        if len(lengths) != 1:
            raise ValueError('排序数据集各字段长度不一致')
        half_life = float(config.get(
            'ranking_recency_half_life_days',
            0.0,
        ))
        if half_life < 0:
            raise ValueError('ranking_recency_half_life_days 不能为负')
        if half_life == 0 or not prediction_dates:
            self.recency_weights = np.ones(
                len(prediction_dates),
                dtype=np.float32,
            )
        else:
            parsed_dates = pd.DatetimeIndex(pd.to_datetime(prediction_dates))
            trading_dates = pd.DatetimeIndex(sorted(parsed_dates.unique()))
            date_positions = {
                date: position
                for position, date in enumerate(trading_dates)
            }
            latest_position = len(trading_dates) - 1
            ages = np.asarray([
                latest_position - date_positions[date]
                for date in parsed_dates
            ], dtype=np.float64)
            self.recency_weights = np.power(
                0.5,
                ages / half_life,
            ).astype(np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.from_numpy(
                np.array(self.sequences[idx], dtype=np.float32, copy=True)
            ),
            'targets': torch.from_numpy(
                np.array(self.targets[idx], dtype=np.float32, copy=True)
            ),
            'relevance': torch.from_numpy(
                np.array(
                    self.relevance_scores[idx],
                    dtype=np.int64,
                    copy=True,
                )
            ),
            'stock_indices': torch.as_tensor(
                self.stock_indices[idx],
                dtype=torch.long,
            ),
            'prediction_date': self.prediction_dates[idx],
            'risk_1d_targets': torch.from_numpy(
                np.array(
                    self.risk_1d_targets[idx],
                    dtype=np.float32,
                    copy=True,
                )
            ),
            'risk_3d_targets': torch.from_numpy(
                np.array(
                    self.risk_3d_targets[idx],
                    dtype=np.float32,
                    copy=True,
                )
            ),
            'regime_targets': torch.from_numpy(
                np.array(
                    self.regime_targets[idx],
                    dtype=np.float32,
                    copy=True,
                )
            ),
            'tail_5d_targets': torch.from_numpy(
                np.array(self.tail_5d_targets[idx], dtype=np.float32, copy=True)
            ),
            'path_loss_5d_targets': torch.from_numpy(
                np.array(self.path_loss_5d_targets[idx], dtype=np.float32, copy=True)
            ),
            'industry_neutral_targets': torch.from_numpy(
                np.array(self.industry_neutral_targets[idx], dtype=np.float32, copy=True)
            ),
            'recency_weight': torch.tensor(
                self.recency_weights[idx],
                dtype=torch.float32,
            ),
        }


class FrozenBackboneDataset(torch.utils.data.Dataset):
    """将冻结的 Ranking 主干表示与原始样本并列保存的轻量数据集。"""
    def __init__(self, base_dataset, cached_samples):
        if len(base_dataset) != len(cached_samples):
            raise ValueError('冻结主干缓存与数据集长度不一致')
        self.base_dataset = base_dataset
        self.cached_samples = cached_samples

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = dict(self.base_dataset[index])
        item.update(self.cached_samples[index])
        return item


def build_ranking_dataset(dataset_parts, source_data):
    """把 vectorized 样本按预测日/股票 ID 对齐到 V17 的额外监督标签。"""
    (
        sequences, targets, relevance_scores, stock_indices, prediction_dates,
        risk_1d_targets, risk_3d_targets, regime_targets,
    ) = dataset_parts
    required = {'日期', 'instrument', 'tail_5d_target', 'path_loss_5d_target'}
    missing = required.difference(source_data.columns)
    if missing:
        raise ValueError(f'构造 V17 排序数据集缺少列: {sorted(missing)}')
    lookup_columns = [
        '日期', 'instrument', 'tail_5d_target', 'path_loss_5d_target',
    ]
    if INDUSTRY_NEUTRAL_TARGET in source_data:
        lookup_columns.append(INDUSTRY_NEUTRAL_TARGET)
    lookup = source_data.loc[:, lookup_columns].copy()
    lookup['日期'] = pd.to_datetime(lookup['日期'])
    lookup['instrument'] = lookup['instrument'].astype(np.int64)
    if lookup.duplicated(['日期', 'instrument']).any():
        raise ValueError('V17 标签键 日期/instrument 不唯一')
    lookup = lookup.set_index(['日期', 'instrument'])

    def aligned(column, default=0.0):
        rows = []
        for date, instruments in zip(prediction_dates, stock_indices):
            index = pd.MultiIndex.from_arrays([
                np.repeat(pd.Timestamp(date), len(instruments)),
                np.asarray(instruments, dtype=np.int64),
            ])
            if column in lookup:
                values = lookup.reindex(index)[column].to_numpy(dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f'V17 标签 {column} 与排序样本无法完整对齐')
            else:
                values = np.full(len(instruments), default, dtype=np.float32)
            rows.append(values)
        return rows

    return RankingDataset(
        sequences, targets, relevance_scores, stock_indices, prediction_dates,
        risk_1d_targets, risk_3d_targets, regime_targets,
        tail_5d_targets=aligned('tail_5d_target'),
        path_loss_5d_targets=aligned('path_loss_5d_target'),
        industry_neutral_targets=aligned(INDUSTRY_NEUTRAL_TARGET),
    )


def cache_frozen_backbone_dataset(model, dataset, device):
    """一次性缓存已冻结的主干输出，避免后三个阶段重复 Transformer 前向。"""
    if not config.get('cache_frozen_backbone', True):
        return dataset
    cache_loader = build_data_loader(dataset, False, device)
    cached_samples = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(cache_loader, desc='Caching frozen ranking backbone'):
            sequences = move_batch_tensor(batch['sequences'], device)
            stock_indices = move_batch_tensor(batch['stock_indices'], device)
            masks = move_batch_tensor(batch['masks'], device)
            ranking_features, regime_sequence, market_sequence = (
                model.encode_backbone(sequences, stock_indices, masks)
            )
            for index, stock_count in enumerate(
                masks.sum(dim=1).to(torch.long).tolist()
            ):
                cached_samples.append({
                    'cached_ranking_features': (
                        ranking_features[index, :stock_count].float().cpu()
                    ),
                    'cached_regime_sequence': (
                        regime_sequence[index].float().cpu()
                        if regime_sequence is not None
                        else torch.empty(sequences.size(2), 0)
                    ),
                    'cached_market_sequence': (
                        market_sequence[index].float().cpu()
                        if market_sequence is not None
                        else torch.empty(sequences.size(2), 0)
                    ),
                })
    return FrozenBackboneDataset(dataset, cached_samples)

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    risk_1d_targets = [item['risk_1d_targets'] for item in batch]
    risk_3d_targets = [item['risk_3d_targets'] for item in batch]
    regime_targets = [item['regime_targets'] for item in batch]
    tail_5d_targets = [item['tail_5d_targets'] for item in batch]
    path_loss_5d_targets = [item['path_loss_5d_targets'] for item in batch]
    industry_neutral_targets = [item['industry_neutral_targets'] for item in batch]
    prediction_dates = [item['prediction_date'] for item in batch]
    recency_weights = torch.stack([
        item['recency_weight'] for item in batch
    ])
    has_cached_backbone = 'cached_ranking_features' in batch[0]
    if has_cached_backbone:
        cached_features = [item['cached_ranking_features'] for item in batch]
        cached_regime_sequences = [
            item['cached_regime_sequence'] for item in batch
        ]
        cached_market_sequences = [
            item['cached_market_sequence'] for item in batch
        ]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    padded_risk_1d_targets = []
    padded_risk_3d_targets = []
    padded_regime_targets = []
    padded_tail_5d_targets = []
    padded_path_loss_5d_targets = []
    padded_industry_neutral_targets = []
    masks = []
    padded_cached_features = []
    
    for (
        seq,
        tgt,
        rel,
        stock_idx,
        risk_1d,
        risk_3d,
        regime,
        tail_5d,
        path_loss_5d,
        industry_neutral,
    ) in zip(
        sequences,
        targets,
        relevance,
        stock_indices,
        risk_1d_targets,
        risk_3d_targets,
        regime_targets,
        tail_5d_targets,
        path_loss_5d_targets,
        industry_neutral_targets,
    ):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            risk_target_pad = torch.full((pad_size,), 0.5)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
            risk_1d = torch.cat([risk_1d, risk_target_pad], dim=0)
            risk_3d = torch.cat([risk_3d, risk_target_pad], dim=0)
            regime = torch.cat([regime, risk_target_pad], dim=0)
            tail_5d = torch.cat([tail_5d, torch.zeros(pad_size)], dim=0)
            path_loss_5d = torch.cat([path_loss_5d, torch.zeros(pad_size)], dim=0)
            industry_neutral = torch.cat([industry_neutral, torch.zeros(pad_size)], dim=0)

        if has_cached_backbone:
            cached_feature = cached_features[len(padded_cached_features)]
            if num_stocks < max_stocks:
                cached_feature = torch.cat([
                    cached_feature,
                    torch.zeros(
                        max_stocks - num_stocks,
                        cached_feature.size(1),
                        dtype=cached_feature.dtype,
                    ),
                ], dim=0)
            padded_cached_features.append(cached_feature)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        padded_risk_1d_targets.append(risk_1d)
        padded_risk_3d_targets.append(risk_3d)
        padded_regime_targets.append(regime)
        padded_tail_5d_targets.append(tail_5d)
        padded_path_loss_5d_targets.append(path_loss_5d)
        padded_industry_neutral_targets.append(industry_neutral)
        masks.append(mask)
    
    result = {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks),                     # [batch, max_stocks]
        'risk_1d_targets': torch.stack(padded_risk_1d_targets),
        'risk_3d_targets': torch.stack(padded_risk_3d_targets),
        'regime_targets': torch.stack(padded_regime_targets),
        'tail_5d_targets': torch.stack(padded_tail_5d_targets),
        'path_loss_5d_targets': torch.stack(padded_path_loss_5d_targets),
        'industry_neutral_targets': torch.stack(padded_industry_neutral_targets),
        'prediction_dates': prediction_dates,
        'recency_weights': recency_weights,
    }
    if has_cached_backbone:
        result.update({
            'cached_ranking_features': torch.stack(padded_cached_features),
            'cached_regime_sequence': torch.stack(cached_regime_sequences),
            'cached_market_sequence': torch.stack(cached_market_sequences),
        })
    return result


def non_overlapping_subset(dataset, stride):
    """保留最后一个有效日期，并从末尾每隔 stride 个交易日向前抽样。"""
    if stride < 1:
        raise ValueError('evaluation_stride 必须大于等于 1')
    if len(dataset) == 0:
        return Subset(dataset, [])
    indices = list(range(len(dataset) - 1, -1, -stride))
    indices.reverse()
    selected_dates = [dataset.prediction_dates[index] for index in indices]
    parsed_dates = pd.DatetimeIndex(selected_dates)
    if len(parsed_dates) > 1:
        full_dates = pd.DatetimeIndex(dataset.prediction_dates)
        full_positions = [full_dates.get_loc(date) for date in parsed_dates]
        if any(
            right - left < stride
            for left, right in zip(full_positions, full_positions[1:])
        ):
            raise ValueError('非重叠验证日期间隔小于 evaluation_stride')
    return Subset(dataset, indices)


def build_data_loader(dataset, shuffle, device):
    num_workers = int(config.get('num_workers', 0))
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(
            device.type == 'cuda' and config.get('pin_memory', True)
        ),
        persistent_workers=num_workers > 0,
    )

# 排序训练函数
def train_ranking_model(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    grad_scaler,
    stage='ranking',
):
    set_model_stage_mode(model, stage, training=True)
    total_loss = 0
    total_loss_components = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = (
            move_batch_tensor(batch['sequences'], device)
            if 'cached_ranking_features' not in batch else None
        )
        targets = move_batch_tensor(batch['targets'], device)
        relevance = move_batch_tensor(batch['relevance'], device)
        stock_indices = move_batch_tensor(batch['stock_indices'], device)
        masks = move_batch_tensor(batch['masks'], device)
        risk_1d_targets = move_batch_tensor(
            batch['risk_1d_targets'],
            device,
        )
        risk_3d_targets = move_batch_tensor(
            batch['risk_3d_targets'],
            device,
        )
        regime_targets = move_batch_tensor(
            batch['regime_targets'],
            device,
        )
        tail_5d_targets = move_batch_tensor(batch['tail_5d_targets'], device)
        path_loss_5d_targets = move_batch_tensor(
            batch['path_loss_5d_targets'], device,
        )
        industry_neutral_targets = move_batch_tensor(
            batch['industry_neutral_targets'], device,
        )
        recency_weights = move_batch_tensor(
            batch['recency_weights'],
            device,
        )
        
        optimizer.zero_grad(set_to_none=True)
        
        # Transformer 前向使用 AMP；排序、相关性与仓位损失保持 FP32。
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp(device),
        ):
            (
                outputs,
                return_outputs,
                allocation_outputs,
                exposures,
                auxiliary_outputs,
            ) = forward_model_batch(
                model, batch, sequences, stock_indices, masks, device,
            )
        outputs = outputs.float()
        return_outputs = return_outputs.float()
        allocation_outputs = allocation_outputs.float()
        exposures = exposures.float()
        auxiliary_outputs = cast_auxiliary_outputs_to_float(auxiliary_outputs)
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_return_outputs = return_outputs * masks
        masked_allocation_outputs = allocation_outputs * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 辅助阶段的样本通常都拥有完整 300 股票池；将这些损失一次批量
        # 计算，保留 ranking 的逐日衰减加权与变长 batch 回退路径。
        batch_loss = None
        batch_loss_components = {}
        batch_weight_total = None
        batch_size = targets.size(0)
        if stage != 'ranking' and bool(masks.bool().all()):
            batch_loss, batch_loss_components = dense_stage_loss(
                criterion, model, outputs, masked_relevance,
                return_outputs, targets, allocation_outputs, exposures,
                auxiliary_outputs, risk_1d_targets, risk_3d_targets,
                tail_5d_targets, path_loss_5d_targets,
                industry_neutral_targets, regime_targets, stage,
                return_components=True,
            )
            batch_weight_total = batch_loss.new_tensor(1.0)
        else:
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
            
                if valid_indices.numel() == 0:
                    continue
                
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
            
                # 获取有效股票的预测值和预处理好的相关性得分
                valid_pred = masked_outputs[i][valid_indices]
                valid_relevance = masked_relevance[i][valid_indices]
                valid_return_pred = masked_return_outputs[i][valid_indices]
                valid_allocation_logits = masked_allocation_outputs[i][valid_indices]
                valid_raw_return = masked_targets[i][valid_indices]
                valid_risk_1d_targets = risk_1d_targets[i][valid_indices]
                valid_risk_3d_targets = risk_3d_targets[i][valid_indices]
                valid_regime_targets = regime_targets[i][valid_indices]
                valid_tail_5d_targets = tail_5d_targets[i][valid_indices]
                valid_path_loss_5d_targets = path_loss_5d_targets[i][valid_indices]
                valid_industry_neutral_targets = industry_neutral_targets[i][valid_indices]
            
                if len(valid_pred) > 1:
                    # 直接使用预处理好的相关性得分，无需重新计算
                    loss, loss_components = criterion(
                    valid_pred.unsqueeze(0),
                    valid_relevance.unsqueeze(0),
                    valid_return_pred.unsqueeze(0),
                    valid_raw_return.unsqueeze(0),
                    valid_allocation_logits.unsqueeze(0),
                    exposures[i].reshape(1),
                    identity_gate=model.identity_gate_value(),
                    risk_1d_logits=(
                        auxiliary_outputs['risk_1d_logits'][
                            i,
                            valid_indices,
                        ].unsqueeze(0)
                        if auxiliary_outputs['risk_1d_logits'] is not None
                        else None
                    ),
                    risk_3d_logits=(
                        auxiliary_outputs['risk_3d_logits'][
                            i,
                            valid_indices,
                        ].unsqueeze(0)
                        if auxiliary_outputs['risk_3d_logits'] is not None
                        else None
                    ),
                    risk_5d_logits=(
                        auxiliary_outputs['risk_5d_logits'][
                            i,
                            valid_indices,
                        ].unsqueeze(0)
                        if auxiliary_outputs['risk_5d_logits'] is not None
                        else None
                    ),
                    tail_5d_logits=(
                        auxiliary_outputs['tail_5d_logits'][
                            i,
                            valid_indices,
                        ].unsqueeze(0)
                        if auxiliary_outputs['tail_5d_logits'] is not None
                        else None
                    ),
                    path_loss_5d_outputs=(
                        auxiliary_outputs['path_loss_5d_output'][i, valid_indices].unsqueeze(0)
                        if auxiliary_outputs['path_loss_5d_output'] is not None else None
                    ),
                    tail_5d_targets=valid_tail_5d_targets.unsqueeze(0),
                    path_loss_5d_targets=valid_path_loss_5d_targets.unsqueeze(0),
                    industry_residual_outputs=(
                        auxiliary_outputs['industry_residual_returns'][i, valid_indices].unsqueeze(0)
                        if auxiliary_outputs['industry_residual_returns'] is not None else None
                    ),
                    industry_neutral_targets=valid_industry_neutral_targets.unsqueeze(0),
                    regime_gate=auxiliary_outputs['regime_gate'][
                        i
                    ].reshape(1),
                    risk_1d_targets=valid_risk_1d_targets.unsqueeze(0),
                    risk_3d_targets=valid_risk_3d_targets.unsqueeze(0),
                    regime_targets=valid_regime_targets.unsqueeze(0),
                    stage=stage,
                    return_components=True,
                    )
                    sample_weight = (
                    recency_weights[i]
                    if stage == 'ranking'
                    else recency_weights.new_tensor(1.0)
                    )
                    weighted_loss = sample_weight * loss
                    batch_loss = (
                    batch_loss + weighted_loss
                    if isinstance(batch_loss, torch.Tensor)
                    else weighted_loss
                    )
                    batch_weight_total = (
                    batch_weight_total + sample_weight
                    if isinstance(batch_weight_total, torch.Tensor)
                    else sample_weight
                    )
                    for name, value in loss_components.items():
                        batch_loss_components[name] = (
                        batch_loss_components.get(name, 0.0)
                        + sample_weight * value
                        )
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_weight_total.clamp(min=1e-12)
            batch_loss_components = {
                name: value / batch_weight_total.clamp(min=1e-12)
                for name, value in batch_loss_components.items()
            }
            grad_scaler.scale(batch_loss).backward()
            if config.get('grad_clip', True):
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters_with_grad(optimizer),
                    config['max_grad_norm'],
                )
                if writer:
                    writer.add_scalar(
                        f'{stage}/train/grad_norm',
                        grad_norm,
                        global_step=epoch * len(dataloader) + local_step,
                    )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            total_loss += batch_loss.item()
            for name, value in batch_loss_components.items():
                total_loss_components[name] = total_loss_components.get(name, 0.0) + value.item()
            
            local_step += 1
            if writer:
                writer.add_scalar(
                    f'{stage}/train/loss',
                    batch_loss.item(),
                    global_step=epoch * len(dataloader) + local_step,
                )
                for name, value in batch_loss_components.items():
                    writer.add_scalar(
                        f'{stage}/train/{name}',
                        value.item(),
                        global_step=epoch*len(dataloader)+local_step,
                    )
    
    # 训练阶段只记录损失项。完整收益/Rank IC 指标在每个 epoch 的验证阶段计算，
    # 避免 batch 级 SciPy Spearman 与 GPU→CPU 同步阻塞训练吞吐。
    if local_step > 0:
        for name, value in total_loss_components.items():
            total_loss_components[name] = value / local_step
    
    return (
        total_loss / len(dataloader) if len(dataloader) > 0 else 0,
        total_loss_components,
    )

def evaluate_ranking_model(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    return_predictions=False,
    selection_risk_feature_names=None,
    selection_risk_scaler=None,
    stage='ranking',
):
    set_model_stage_mode(model, stage, training=False)
    total_loss = 0
    num_batches = 0
    metric_records = []
    prediction_records = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = (
                move_batch_tensor(batch['sequences'], device)
                if 'cached_ranking_features' not in batch else None
            )
            targets = move_batch_tensor(batch['targets'], device)
            stock_indices = move_batch_tensor(batch['stock_indices'], device)
            masks = move_batch_tensor(batch['masks'], device)
            risk_1d_targets = move_batch_tensor(
                batch['risk_1d_targets'],
                device,
            )
            risk_3d_targets = move_batch_tensor(
                batch['risk_3d_targets'],
                device,
            )
            regime_targets = move_batch_tensor(
                batch['regime_targets'],
                device,
            )
            tail_5d_targets = move_batch_tensor(batch['tail_5d_targets'], device)
            path_loss_5d_targets = move_batch_tensor(
                batch['path_loss_5d_targets'], device,
            )
            industry_neutral_targets = move_batch_tensor(
                batch['industry_neutral_targets'], device,
            )
            
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp(device),
            ):
                (
                    outputs,
                    return_outputs,
                    allocation_outputs,
                    exposures,
                    auxiliary_outputs,
                ) = forward_model_batch(
                    model, batch, sequences, stock_indices, masks, device,
                )
            outputs = outputs.float()
            return_outputs = return_outputs.float()
            allocation_outputs = allocation_outputs.float()
            exposures = exposures.float()
            auxiliary_outputs = cast_auxiliary_outputs_to_float(
                auxiliary_outputs,
            )
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            masked_return_outputs = return_outputs * masks
            masked_allocation_outputs = allocation_outputs * masks
            
            # 计算损失
            batch_loss = None
            batch_size = targets.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                valid_return_pred = masked_return_outputs[i][valid_indices]
                valid_allocation_logits = masked_allocation_outputs[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    
                    loss = criterion(
                        valid_pred.unsqueeze(0),
                        relevance_scores.unsqueeze(0),
                        valid_return_pred.unsqueeze(0),
                        valid_true.unsqueeze(0),
                        valid_allocation_logits.unsqueeze(0),
                        exposures[i].reshape(1),
                        identity_gate=model.identity_gate_value(),
                        risk_1d_logits=(
                            auxiliary_outputs['risk_1d_logits'][
                                i,
                                valid_indices,
                            ].unsqueeze(0)
                            if auxiliary_outputs[
                                'risk_1d_logits'
                            ] is not None
                            else None
                        ),
                        risk_3d_logits=(
                            auxiliary_outputs['risk_3d_logits'][
                                i,
                                valid_indices,
                            ].unsqueeze(0)
                            if auxiliary_outputs[
                                'risk_3d_logits'
                            ] is not None
                            else None
                        ),
                        risk_5d_logits=(
                            auxiliary_outputs['risk_5d_logits'][
                                i,
                                valid_indices,
                            ].unsqueeze(0)
                            if auxiliary_outputs[
                                'risk_5d_logits'
                            ] is not None
                            else None
                        ),
                        tail_5d_logits=(
                            auxiliary_outputs['tail_5d_logits'][
                                i,
                                valid_indices,
                            ].unsqueeze(0)
                            if auxiliary_outputs[
                                'tail_5d_logits'
                            ] is not None
                            else None
                        ),
                        path_loss_5d_outputs=(
                            auxiliary_outputs['path_loss_5d_output'][i, valid_indices].unsqueeze(0)
                            if auxiliary_outputs['path_loss_5d_output'] is not None else None
                        ),
                        tail_5d_targets=tail_5d_targets[i, valid_indices].unsqueeze(0),
                        path_loss_5d_targets=path_loss_5d_targets[i, valid_indices].unsqueeze(0),
                        industry_residual_outputs=(
                            auxiliary_outputs['industry_residual_returns'][i, valid_indices].unsqueeze(0)
                            if auxiliary_outputs['industry_residual_returns'] is not None else None
                        ),
                        industry_neutral_targets=industry_neutral_targets[i, valid_indices].unsqueeze(0),
                        regime_gate=auxiliary_outputs['regime_gate'][
                            i
                        ].reshape(1),
                        risk_1d_targets=risk_1d_targets[
                            i,
                            valid_indices,
                        ].unsqueeze(0),
                        risk_3d_targets=risk_3d_targets[
                            i,
                            valid_indices,
                        ].unsqueeze(0),
                        regime_targets=regime_targets[
                            i,
                            valid_indices,
                        ].unsqueeze(0),
                        stage=stage,
                    )
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            _, batch_metric_records = calculate_ranking_metrics(
                masked_outputs,
                masked_targets,
                masks,
                predicted_returns=masked_return_outputs,
                allocation_logits=masked_allocation_outputs,
                exposures=exposures,
                allocation_temperature=config.get('allocation_temperature', 1.0),
                k=5,
                return_records=True,
            )
            metric_records.extend(batch_metric_records)
            if return_predictions:
                for i, prediction_date in enumerate(
                    batch['prediction_dates']
                ):
                    valid_indices = masks[i].nonzero(
                        as_tuple=False
                    ).flatten()
                    if valid_indices.numel() < 5:
                        continue
                    prediction_record = {
                        'prediction_date': prediction_date,
                        'stock_indices': (
                            stock_indices[i][valid_indices]
                            .detach().cpu().numpy()
                        ),
                        'targets': (
                            masked_targets[i][valid_indices]
                            .detach().cpu().numpy()
                        ),
                        'scores': (
                            masked_outputs[i][valid_indices]
                            .detach().cpu().numpy()
                        ),
                        'allocation_logits': (
                            masked_allocation_outputs[i][valid_indices]
                            .detach().cpu().numpy()
                        ),
                        'exposure': float(exposures[i].item()),
                        'regime_gate': float(
                            auxiliary_outputs['regime_gate'][i].item()
                        ),
                        'risk_1d_probabilities': (
                            torch.sigmoid(
                                auxiliary_outputs['risk_1d_logits'][
                                    i,
                                    valid_indices,
                                ]
                            ).detach().cpu().numpy()
                            if auxiliary_outputs[
                                'risk_1d_logits'
                            ] is not None
                            else np.full(valid_indices.numel(), 0.5)
                        ),
                        'risk_3d_probabilities': (
                            torch.sigmoid(
                                auxiliary_outputs['risk_3d_logits'][
                                    i,
                                    valid_indices,
                                ]
                            ).detach().cpu().numpy()
                            if auxiliary_outputs[
                                'risk_3d_logits'
                            ] is not None
                            else np.full(valid_indices.numel(), 0.5)
                        ),
                        'risk_5d_probabilities': (
                            torch.sigmoid(
                                auxiliary_outputs['risk_5d_logits'][
                                    i,
                                    valid_indices,
                                ]
                            ).detach().cpu().numpy()
                            if auxiliary_outputs[
                                'risk_5d_logits'
                            ] is not None
                            else np.full(valid_indices.numel(), 0.5)
                        ),
                        'tail_5d_probabilities': (
                            torch.sigmoid(
                                auxiliary_outputs['tail_5d_logits'][
                                    i,
                                    valid_indices,
                                ]
                            ).detach().cpu().numpy()
                            if auxiliary_outputs[
                                'tail_5d_logits'
                            ] is not None
                            else np.full(valid_indices.numel(), 0.5)
                        ),
                        'risk_1d_targets': risk_1d_targets[
                            i,
                            valid_indices,
                        ].detach().cpu().numpy(),
                        'risk_3d_targets': risk_3d_targets[
                            i,
                            valid_indices,
                        ].detach().cpu().numpy(),
                        'risk_5d_targets': torch.sigmoid(
                            -masked_targets[i][valid_indices]
                            / float(config.get(
                                'risk_5d_target_temperature',
                                0.03,
                            ))
                        ).detach().cpu().numpy(),
                        'tail_5d_targets': tail_5d_targets[
                            i, valid_indices
                        ].detach().cpu().numpy(),
                        'regime_target': float(
                            regime_targets[
                                i,
                                valid_indices,
                            ].mean().item()
                        ),
                    }
                    if (
                        selection_risk_feature_names is not None
                        and selection_risk_scaler is not None
                    ):
                        if sequences is None:
                            raise ValueError(
                                'OOF风险上下文需要原始验证序列；'
                                '请使用未缓存的验证 DataLoader'
                            )
                        risk_context = extract_selection_risk_context(
                            sequences[i][valid_indices]
                            .detach().float().cpu().numpy(),
                            selection_risk_feature_names,
                            selection_risk_scaler,
                            lookback=int(config.get(
                                'selection_risk_lookback',
                                20,
                            )),
                        )
                        prediction_record.update(risk_context)
                    prediction_records.append(prediction_record)
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    total_metrics = summarize_ranking_metric_records(metric_records)
    
    if writer:
        writer.add_scalar(
            f'{stage}/eval/loss',
            avg_loss,
            global_step=epoch,
        )
        for k, v in total_metrics.items():
            writer.add_scalar(
                f'{stage}/eval/{k}',
                v,
                global_step=epoch,
            )
    
    if return_predictions:
        return avg_loss, total_metrics, prediction_records
    return avg_loss, total_metrics


def evaluate_identity_sensitivity(
    model,
    dataloader,
    device,
    permutation_seed,
    top_k=5,
):
    """仅对最佳checkpoint执行ID消融，避免把测试周用于模型选择。"""
    model.eval()
    real_top5_returns = []
    comparisons = {
        'all_unk_vs_real': [],
        'permuted_vs_real': [],
    }
    real_exposures = []
    alternative_exposures = {
        'all_unk_vs_real': [],
        'permuted_vs_real': [],
    }
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(permutation_seed))
    identity_permutation = torch.arange(model.num_stocks + 2)
    known_identity_indices = torch.arange(2, model.num_stocks + 2)
    identity_permutation[known_identity_indices] = known_identity_indices[
        torch.randperm(model.num_stocks, generator=generator)
    ]

    with torch.inference_mode():
        for batch in dataloader:
            sequences = move_batch_tensor(batch['sequences'], device)
            targets = move_batch_tensor(batch['targets'], device)
            stock_indices = move_batch_tensor(batch['stock_indices'], device)
            masks = move_batch_tensor(batch['masks'], device)
            all_unk_indices = stock_indices.masked_fill(masks.bool(), 1)
            permuted_indices = identity_permutation.to(device)[stock_indices]

            outputs_by_mode = {}
            for mode, indices in (
                ('real', stock_indices),
                ('all_unk_vs_real', all_unk_indices),
                ('permuted_vs_real', permuted_indices),
            ):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp(device),
                ):
                    scores, _, _, exposures = model(
                        sequences,
                        indices,
                        masks,
                    )
                outputs_by_mode[mode] = (
                    scores.float(),
                    exposures.float(),
                )

            for row_index in range(stock_indices.size(0)):
                valid_indices = masks[row_index].nonzero(
                    as_tuple=False
                ).flatten()
                if valid_indices.numel() < top_k:
                    continue
                real_scores = outputs_by_mode['real'][0][
                    row_index, valid_indices
                ].detach().cpu().numpy()
                valid_targets = targets[
                    row_index, valid_indices
                ].detach().cpu().numpy()
                real_order = np.argsort(
                    real_scores,
                    kind='stable',
                )[::-1]
                real_top = set(real_order[:top_k].tolist())
                real_return = float(valid_targets[real_order[:top_k]].mean())
                real_top5_returns.append(real_return)
                real_exposures.append(float(
                    outputs_by_mode['real'][1][row_index].item()
                ))

                for mode in comparisons:
                    alternative_scores = outputs_by_mode[mode][0][
                        row_index, valid_indices
                    ].detach().cpu().numpy()
                    correlation = spearmanr(
                        real_scores,
                        alternative_scores,
                    ).statistic
                    alternative_order = np.argsort(
                        alternative_scores,
                        kind='stable',
                    )[::-1]
                    alternative_top = set(
                        alternative_order[:top_k].tolist()
                    )
                    alternative_return = float(
                        valid_targets[alternative_order[:top_k]].mean()
                    )
                    comparisons[mode].append({
                        'score_spearman': float(
                            correlation if np.isfinite(correlation) else 0.0
                        ),
                        'top5_overlap': len(
                            real_top.intersection(alternative_top)
                        ) / top_k,
                        'top5_return': alternative_return,
                        'top5_return_delta_vs_real': (
                            alternative_return - real_return
                        ),
                    })
                    alternative_exposures[mode].append(float(
                        outputs_by_mode[mode][1][row_index].item()
                    ))

    if not real_top5_returns:
        raise ValueError('ID敏感性评估没有有效日期')
    result = {
        'num_evaluation_dates': len(real_top5_returns),
        'identity_gate': float(
            model.identity_gate_value().detach().cpu().item()
        ),
        'real': {
            'mean_top5_return': float(np.mean(real_top5_returns)),
            'mean_exposure': float(np.mean(real_exposures)),
        },
    }
    for mode, rows in comparisons.items():
        result[mode] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        result[mode]['mean_exposure'] = float(np.mean(
            alternative_exposures[mode]
        ))
    return result


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    stock_indices = torch.LongTensor(day_stock_indices).unsqueeze(0).to(device)
    stock_mask = torch.ones_like(stock_indices, dtype=torch.float32)
    
    with torch.no_grad():
        # 模型预测
        outputs, _, allocation_logits, exposures = model(
            sequences,
            stock_indices,
            stock_mask,
        )
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        allocation_scores = allocation_logits.squeeze(0)
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_index_tensor = torch.as_tensor(top_indices, device=device)
        relative_weights = F.softmax(
            allocation_scores[top_index_tensor]
            / config.get('allocation_temperature', 1.0),
            dim=0,
        )
        position_weights = (relative_weights * exposures[0]).cpu().numpy()
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'weight': float(position_weights[len(top_stocks)]),
                'exposure': float(exposures[0].item()),
                'cash_weight': float(1.0 - exposures[0].item()),
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score'],
            '权重': stock['weight'],
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def build_walk_forward_folds(df, num_folds, validation_months, purge_days):
    """基于实际交易日期构造扩展窗口验证折，并显式隔离标签未来期。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    trading_dates = pd.DatetimeIndex(sorted(df['日期'].dropna().unique()))
    if len(trading_dates) == 0:
        raise ValueError('没有可用于切分的交易日期')


    folds = []
    last_date = trading_dates[-1]
    for reverse_offset in reversed(range(num_folds)):
        end_target = last_date - pd.DateOffset(months=reverse_offset * validation_months)
        start_target = last_date - pd.DateOffset(months=(reverse_offset + 1) * validation_months)
        val_start_idx = trading_dates.searchsorted(start_target, side='right')
        val_end_idx = trading_dates.searchsorted(end_target, side='right') - 1
        train_end_idx = val_start_idx - purge_days - 1

        if train_end_idx < 0 or val_start_idx > val_end_idx:
            raise ValueError('历史长度不足，无法构造请求的 walk-forward 折数')

        folds.append({
            'fold': len(folds) + 1,
            'train_end': trading_dates[train_end_idx],
            'purge_start': trading_dates[train_end_idx + 1],
            'purge_end': trading_dates[val_start_idx - 1],
            'val_start': trading_dates[val_start_idx],
            'val_end': trading_dates[val_end_idx],
        })

    return folds

def configure_model_for_stage(model, stage):
    """冻结非当前阶段参数，并避免冻结主干的 dropout 改变候选集合。"""
    valid_stages = set(TRAINING_STAGES)
    if stage not in valid_stages:
        raise ValueError(f'未知训练阶段: {stage}')
    allocation_prefixes = ('allocation_head.',)
    exposure_prefixes = (
        'exposure_market_encoder.',
        'exposure_head.',
        'exposure_regime_penalty_raw',
        'exposure_risk_penalty_raw',
    )
    risk_prefixes = (
        'risk_1d_head.',
        'risk_3d_head.',
        'risk_5d_head.',
        'tail_5d_head.',
        'path_loss_5d_head.',
        'regime_market_encoder.',
        'regime_gate_head.',
    )
    for name, parameter in model.named_parameters():
        is_allocation = name.startswith(allocation_prefixes)
        is_exposure = name.startswith(exposure_prefixes)
        is_risk = name.startswith(risk_prefixes)
        if stage == 'ranking':
            parameter.requires_grad = not (
                is_allocation or is_exposure or is_risk
            )
        elif stage == 'risk':
            parameter.requires_grad = is_risk
        elif stage == 'allocation':
            parameter.requires_grad = is_allocation
        else:
            parameter.requires_grad = is_exposure
    # optimizer.zero_grad() 只清理当前优化器参数；阶段切换时必须主动清除
    # 上一阶段残留梯度，防止其参与后续诊断或裁剪。
    model.zero_grad(set_to_none=True)


def set_model_stage_mode(model, stage, training):
    if not training:
        model.eval()
        return
    if stage == 'ranking':
        model.train()
        model.allocation_head.eval()
        model.exposure_head.eval()
        if hasattr(model, 'risk_1d_head'):
            model.risk_1d_head.eval()
            model.risk_3d_head.eval()
            if hasattr(model, 'risk_5d_head'):
                model.risk_5d_head.eval()
            if hasattr(model, 'tail_5d_head'):
                model.tail_5d_head.eval()
            if hasattr(model, 'path_loss_5d_head'):
                model.path_loss_5d_head.eval()
        if hasattr(model, 'regime_market_encoder'):
            model.regime_market_encoder.eval()
            model.regime_gate_head.eval()
        if hasattr(model, 'exposure_market_encoder'):
            model.exposure_market_encoder.eval()
    elif stage == 'risk':
        model.eval()
        model.risk_1d_head.train()
        model.risk_3d_head.train()
        if hasattr(model, 'risk_5d_head'):
            model.risk_5d_head.train()
        if hasattr(model, 'tail_5d_head'):
            model.tail_5d_head.train()
        if hasattr(model, 'path_loss_5d_head'):
            model.path_loss_5d_head.train()
        model.regime_market_encoder.train()
        model.regime_gate_head.train()
    elif stage == 'allocation':
        model.eval()
        model.allocation_head.train()
    elif stage == 'exposure':
        model.eval()
        model.exposure_head.train()
        if hasattr(model, 'exposure_market_encoder'):
            model.exposure_market_encoder.train()
    else:
        raise ValueError(f'未知训练阶段: {stage}')


def build_training_components(model, stage='ranking'):
    configure_model_for_stage(model, stage)
    criterion = WeightedRankingLoss(
        k=5,
        listwise_temperature=config.get('listwise_temperature', 0.2),
        listwise_weight=config.get('listwise_weight', 0.2),
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0),
        regression_weight=config['regression_weight'],
        regression_beta=config['regression_beta'],
        ic_weight=config.get('ic_weight', 0.2),
        allocation_weight=config.get('allocation_weight', 0.1),
        exposure_weight=config.get('exposure_weight', 1.0),
        allocation_temperature=config.get('allocation_temperature', 1.0),
        allocation_target_temperature=config.get('allocation_target_temperature', 0.02),
        allocation_candidate_k=config.get('allocation_candidate_k', 20),
        allocation_return_clip=config.get('allocation_return_clip', 0.10),
        exposure_target_temperature=config.get('exposure_target_temperature', 0.02),
        exposure_selected_return_weight=config.get(
            'exposure_selected_return_weight',
            0.70,
        ),
        exposure_market_return_weight=config.get(
            'exposure_market_return_weight',
            0.30,
        ),
        exposure_downside_weight=config.get('exposure_downside_weight', 0.25),
        id_gate_regularization=config.get('id_gate_regularization', 0.0),
        lambdarank_candidate_k=config.get('lambdarank_candidate_k', 20),
        lambdarank_hard_negative_k=config.get(
            'lambdarank_hard_negative_k',
            20,
        ),
        lambdarank_return_gap_scale=config.get(
            'lambdarank_return_gap_scale',
            0.02,
        ),
        risk_1d_weight=config.get('risk_1d_weight', 0.0),
        risk_3d_weight=config.get('risk_3d_weight', 0.0),
        risk_5d_weight=config.get('risk_5d_weight', 0.0),
        risk_5d_target_temperature=config.get(
            'risk_5d_target_temperature',
            0.03,
        ),
        tail_5d_weight=config.get('tail_5d_weight', 0.0),
        tail_5d_threshold=config.get('tail_5d_threshold', -0.03),
        regime_weight=config.get('regime_weight', 0.0),
        industry_residual_weight=config.get('industry_residual_weight', 0.0),
        industry_residual_beta=config.get('industry_residual_beta', 0.02),
        path_loss_5d_weight=config.get('path_loss_5d_weight', 0.0),
        path_loss_5d_beta=config.get('path_loss_5d_beta', 0.02),
        min_exposure=config.get('min_exposure', 0.80),
        max_exposure=config.get('max_exposure', 0.999999),
    )
    regular_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == 'identity_gate_logit':
            no_decay_parameters.append(parameter)
        else:
            regular_parameters.append(parameter)
    optimizer_parameters = [{
        'params': regular_parameters,
        'weight_decay': config['weight_decay'],
    }]
    if no_decay_parameters:
        optimizer_parameters.append({
            'params': no_decay_parameters,
            'weight_decay': 0.0,
        })
    optimizer_kwargs = {
        'lr': float(config.get(
            f'{stage}_learning_rate',
            config['learning_rate'],
        )),
    }
    if (
        next(model.parameters()).is_cuda
        and config.get('fused_optimizer', True)
    ):
        optimizer_kwargs['fused'] = True
    try:
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            **optimizer_kwargs,
        )
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop('fused', None)
        optimizer = torch.optim.AdamW(
            optimizer_parameters,
            **optimizer_kwargs,
        )
    return criterion, optimizer


def calculate_checkpoint_score(metrics, checkpoint_metric):
    """计算单折 checkpoint 分数，支持 Top-5 与 Rank IC 的组合目标。"""
    if checkpoint_metric == 'top5_return_plus_rank_ic':
        return (
            metrics.get('top5_return', 0.0)
            + config.get('checkpoint_rank_ic_weight', 0.2) * metrics.get('rank_ic', 0.0)
        )
    if checkpoint_metric == 'weighted_portfolio_return_plus_rank_ic':
        return (
            metrics.get('weighted_portfolio_return', 0.0)
            + config.get('checkpoint_rank_ic_weight', 0.2) * metrics.get('rank_ic', 0.0)
        )
    if checkpoint_metric == 'allocation_contribution':
        return metrics.get('allocation_contribution', 0.0)
    if checkpoint_metric == 'weighted_portfolio_risk_adjusted':
        return (
            metrics.get('weighted_portfolio_return', 0.0)
            - config.get('ensemble_downside_weight', 0.5)
            * metrics.get('weighted_portfolio_downside_deviation', 0.0)
        )
    return metrics.get(checkpoint_metric, 0.0)


def stage_settings(stage):
    defaults = {
        'ranking': {
            'max_epochs': config['max_epochs'],
            'patience': config['patience'],
            'checkpoint_metric': 'top5_return_plus_rank_ic',
        },
        'risk': {
            'max_epochs': 12,
            'patience': 4,
            'checkpoint_metric': 'negative_eval_loss',
        },
        'allocation': {
            'max_epochs': 12,
            'patience': 4,
            'checkpoint_metric': 'allocation_contribution',
        },
        'exposure': {
            'max_epochs': 12,
            'patience': 4,
            'checkpoint_metric': 'weighted_portfolio_risk_adjusted',
        },
    }
    settings = defaults[stage]
    return {
        'max_epochs': int(config.get(
            f'{stage}_max_epochs',
            settings['max_epochs'],
        )),
        'patience': int(config.get(
            f'{stage}_patience',
            settings['patience'],
        )),
        'checkpoint_metric': config.get(
            f'{stage}_checkpoint_metric',
            settings['checkpoint_metric'],
        ),
    }


def fit_training_stage(
    model,
    train_loader,
    val_loader,
    device,
    writer,
    stage,
    checkpoint_path,
    fold_number=None,
):
    """训练单一阶段并恢复该阶段最佳完整模型状态。"""
    settings = stage_settings(stage)
    criterion, optimizer = build_training_components(model, stage=stage)
    grad_scaler = create_grad_scaler(device)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=settings['max_epochs'],
    )
    best_score = -float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_ran = 0
    eval_interval = (
        1 if stage == 'ranking'
        else max(1, int(config.get('auxiliary_eval_interval', 1)))
    )
    for epoch in range(settings['max_epochs']):
        epochs_ran = epoch + 1
        train_loss, train_metrics = train_ranking_model(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            grad_scaler,
            stage=stage,
        )
        scheduler.step()
        if writer:
            writer.add_scalar(
                f'{stage}/learning_rate',
                scheduler.get_last_lr()[0],
                global_step=epoch,
            )
        should_evaluate = (
            epoch == 0
            or (epoch + 1) % eval_interval == 0
            or epoch + 1 == settings['max_epochs']
        )
        if not should_evaluate:
            print(
                f"{stage} Epoch {epoch + 1}: train_loss={train_loss:.4f}; "
                f"验证按每 {eval_interval} 轮一次执行"
            )
            continue
        eval_loss, eval_metrics = evaluate_ranking_model(
            model,
            val_loader,
            criterion,
            device,
            writer,
            epoch,
            stage=stage,
        )
        if settings['checkpoint_metric'] == 'negative_eval_loss':
            current_score = -float(eval_loss)
        else:
            current_score = calculate_checkpoint_score(
                eval_metrics,
                settings['checkpoint_metric'],
            )
        location = (
            f'Fold {fold_number} '
            if fold_number is not None
            else 'Full train '
        )
        print(
            f"{location}{stage} Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, "
            f"top5={eval_metrics.get('top5_return', 0.0):.6f}, "
            f"rank_ic={eval_metrics.get('rank_ic', 0.0):.4f}, "
            f"allocation_gain={eval_metrics.get('allocation_contribution', 0.0):.6f}, "
            f"weighted={eval_metrics.get('weighted_portfolio_return', 0.0):.6f}, "
            f"checkpoint={current_score:.6f}"
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings['patience']:
                break
    if best_epoch < 1:
        raise RuntimeError(f'{stage}阶段没有产生有效checkpoint')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return {
        'stage': stage,
        'best_epoch': best_epoch,
        'epochs_ran': epochs_ran,
        'checkpoint_metric': settings['checkpoint_metric'],
        'checkpoint_score': float(best_score),
        'steps_per_epoch': len(train_loader),
    }


def summarize_identity_sensitivity(fold_results):
    """汇总各折最佳checkpoint的ID消融结果。"""
    sensitivity_rows = [
        result['id_sensitivity']
        for result in fold_results
        if 'id_sensitivity' in result
    ]
    if not sensitivity_rows:
        return {}
    summary = {
        'num_folds': len(sensitivity_rows),
        'mean_identity_gate': float(np.mean([
            row['identity_gate'] for row in sensitivity_rows
        ])),
    }
    for mode in ('all_unk_vs_real', 'permuted_vs_real'):
        keys = sensitivity_rows[0][mode].keys()
        summary[mode] = {
            key: float(np.mean([
                row[mode][key] for row in sensitivity_rows
            ]))
            for key in keys
        }
    return summary


def train_one_fold(
    full_data,
    features,
    fold,
    num_stocks,
    device,
    output_dir,
    base_seed,
):
    """训练单个 walk-forward 折，并用最佳 checkpoint 统一评估训练/验证集。"""
    fold_number = fold['fold']
    set_seed(base_seed + fold_number)
    fold_dir = os.path.join(output_dir, f'fold_{fold_number}')
    os.makedirs(fold_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(fold_dir, 'log'))

    train_data = full_data[full_data['日期'] <= fold['train_end']].copy()
    validation_context = full_data[full_data['日期'] <= fold['val_end']].copy()
    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    validation_context[features] = validation_context[features].replace([np.inf, -np.inf], np.nan)
    train_data = train_data.dropna(subset=features)
    validation_context = validation_context.dropna(subset=features)

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    validation_context[features] = scaler.transform(validation_context[features])
    joblib.dump(scaler, os.path.join(fold_dir, 'scaler.pkl'))

    train_parts = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        max_window_end_date=fold['train_end'],
    )
    val_parts = create_ranking_dataset_vectorized(
        validation_context,
        features,
        config['sequence_length'],
        min_window_end_date=fold['val_start'],
        max_window_end_date=fold['val_end'],
    )
    train_dataset = build_ranking_dataset(train_parts, train_data)
    val_dataset = build_ranking_dataset(val_parts, validation_context)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"第 {fold_number} 折没有可用的训练或验证样本")

    evaluation_stride = int(config.get('evaluation_stride', 5))
    train_eval_dataset = non_overlapping_subset(
        train_dataset,
        evaluation_stride,
    )
    val_eval_dataset = non_overlapping_subset(
        val_dataset,
        evaluation_stride,
    )
    train_loader = build_data_loader(train_dataset, True, device)
    train_eval_loader = build_data_loader(train_eval_dataset, False, device)
    val_loader = build_data_loader(val_eval_dataset, False, device)
    raw_val_loader = val_loader

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks).to(device)
    print(
        f"\n========== Seed {base_seed} "
        f"Fold {fold_number}/{config['num_folds']} =========="
    )
    print(
        f"边界: {fold}; 训练样本: {len(train_dataset)}; "
        f"验证样本: {len(val_dataset)}; "
        f"非重叠验证日期: {len(val_eval_dataset)}"
    )
    stage_results = {}
    checkpoint_path = os.path.join(fold_dir, 'best_model.pth')
    if resume_training_enabled():
        stage_results = recover_completed_stage_results(
            fold_dir,
            steps_per_epoch=len(train_loader),
        )
    if stage_results:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(
            f'Fold {fold_number}: 从 TensorBoard 与最终checkpoint恢复'
            '已完成的四阶段训练，仅补做 OOF 汇总'
        )
    else:
        stage_results = {}
        for stage in TRAINING_STAGES:
            stage_results[stage] = fit_training_stage(
                model,
                train_loader,
                val_loader,
                device,
                writer,
                stage,
                checkpoint_path,
                fold_number=fold_number,
            )
            if stage == 'ranking' and config.get(
                'cache_frozen_backbone',
                True,
            ):
                print('Ranking 阶段完成：缓存冻结主干表示供后续辅助阶段复用')
                train_loader = build_data_loader(
                    cache_frozen_backbone_dataset(
                        model, train_dataset, device,
                    ),
                    True,
                    device,
                )
                train_eval_loader = build_data_loader(
                    cache_frozen_backbone_dataset(
                        model, train_eval_dataset, device,
                    ),
                    False,
                    device,
                )
                val_loader = build_data_loader(
                    cache_frozen_backbone_dataset(
                        model, val_eval_dataset, device,
                    ),
                    False,
                    device,
                )

    criterion, _ = build_training_components(model, stage='exposure')
    best_epoch = stage_results['ranking']['best_epoch']
    _, train_eval_metrics = evaluate_ranking_model(
        model,
        train_eval_loader,
        criterion,
        device,
        None,
        best_epoch - 1,
        stage='exposure',
    )
    _, val_eval_metrics, oof_predictions = evaluate_ranking_model(
        model,
        raw_val_loader,
        criterion,
        device,
        None,
        best_epoch - 1,
        return_predictions=True,
        selection_risk_feature_names=features,
        selection_risk_scaler=scaler,
        stage='exposure',
    )
    id_sensitivity = evaluate_identity_sensitivity(
        model,
        raw_val_loader,
        device,
        permutation_seed=(
            int(config.get('identity_sensitivity_seed', 20260728))
            + int(base_seed) * 100
            + int(fold_number)
        ),
        top_k=5,
    )
    train_eval_metrics = {key: float(value) for key, value in train_eval_metrics.items()}
    val_eval_metrics = {key: float(value) for key, value in val_eval_metrics.items()}
    result = {
        'base_seed': int(base_seed),
        'fold': fold_number,
        'train_end': fold['train_end'].strftime('%Y-%m-%d'),
        'purge_start': fold['purge_start'].strftime('%Y-%m-%d'),
        'purge_end': fold['purge_end'].strftime('%Y-%m-%d'),
        'val_start': fold['val_start'].strftime('%Y-%m-%d'),
        'val_end': fold['val_end'].strftime('%Y-%m-%d'),
        'best_epoch': best_epoch,
        'epochs_ran': stage_results['ranking']['epochs_ran'],
        'checkpoint_metric': stage_results['ranking']['checkpoint_metric'],
        'checkpoint_score': stage_results['ranking']['checkpoint_score'],
        'stage_training': stage_results,
        'evaluation_stride': evaluation_stride,
        'num_daily_validation_samples': len(val_dataset),
        'num_non_overlapping_validation_samples': len(val_eval_dataset),
        'train_metrics': train_eval_metrics,
        'val_metrics': val_eval_metrics,
        'id_sensitivity': id_sensitivity,
        'top5_gap': train_eval_metrics.get('top5_return', 0.0) - val_eval_metrics.get('top5_return', 0.0),
        'weighted_portfolio_return_gap': (
            train_eval_metrics.get('weighted_portfolio_return', 0.0)
            - val_eval_metrics.get('weighted_portfolio_return', 0.0)
        ),
        'rank_ic_gap': train_eval_metrics.get('rank_ic', 0.0) - val_eval_metrics.get('rank_ic', 0.0),
    }
    with open(os.path.join(fold_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    joblib.dump(
        oof_predictions,
        os.path.join(fold_dir, 'oof_predictions.joblib'),
        compress=3,
    )
    writer.close()
    return result, oof_predictions


def load_completed_fold_artifacts(output_dir, fold, base_seed):
    """加载并校验已完整写盘的单折结果；文件不全时返回 None。"""
    fold_number = int(fold['fold'])
    fold_dir = os.path.join(output_dir, f'fold_{fold_number}')
    artifact_paths = {
        'metrics': os.path.join(fold_dir, 'metrics.json'),
        'predictions': os.path.join(fold_dir, 'oof_predictions.joblib'),
        'model': os.path.join(fold_dir, 'best_model.pth'),
        'scaler': os.path.join(fold_dir, 'scaler.pkl'),
    }
    if not all(os.path.isfile(path) for path in artifact_paths.values()):
        return None

    with open(artifact_paths['metrics'], encoding='utf-8') as file:
        result = json.load(file)
    expected_metadata = {
        'base_seed': int(base_seed),
        'fold': fold_number,
        'train_end': fold['train_end'].strftime('%Y-%m-%d'),
        'purge_start': fold['purge_start'].strftime('%Y-%m-%d'),
        'purge_end': fold['purge_end'].strftime('%Y-%m-%d'),
        'val_start': fold['val_start'].strftime('%Y-%m-%d'),
        'val_end': fold['val_end'].strftime('%Y-%m-%d'),
    }
    mismatches = {
        name: (result.get(name), expected)
        for name, expected in expected_metadata.items()
        if result.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            f'第 {fold_number} 折恢复产物与当前切分不一致: {mismatches}'
        )
    if not all(
        stage in result.get('stage_training', {})
        for stage in TRAINING_STAGES
    ):
        raise ValueError(f'第 {fold_number} 折恢复产物缺少阶段训练结果')
    predictions = joblib.load(artifact_paths['predictions'])
    return result, predictions


def prepare_full_training_dataset(full_data, features, output_dir):
    """全量 scaler 与排序数据集只构建一次，供已启用的随机种子共享。"""
    train_data = full_data.copy()
    train_data[features] = train_data[features].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    train_data = train_data.dropna(subset=features)
    if train_data.empty:
        raise ValueError('全量重训没有可用样本')

    scaler = StandardScaler()
    train_data[features] = scaler.fit_transform(train_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    train_parts = create_ranking_dataset_vectorized(
        train_data,
        features,
        config['sequence_length'],
        max_window_end_date=train_data['日期'].max(),
    )
    train_dataset = build_ranking_dataset(train_parts, train_data)
    if len(train_dataset) == 0:
        raise ValueError('全量重训无法构造排序样本')
    return train_dataset, train_data['日期'].max()


def train_final_model(
    train_dataset,
    train_end,
    features,
    num_stocks,
    device,
    output_dir,
    stage_epochs,
    base_seed,
):
    """按各折最佳更新步数依次完成四阶段全量重训。"""
    set_seed(base_seed + 1000)
    final_dir = os.path.join(output_dir, 'full_train')
    os.makedirs(final_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(final_dir, 'log'))
    train_loader = build_data_loader(train_dataset, True, device)
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks).to(device)
    stages = TRAINING_STAGES
    first_stage_index = 0
    if resume_training_enabled():
        for stage_index, stage in reversed(list(enumerate(stages))):
            stage_checkpoint_path = os.path.join(
                final_dir,
                f'checkpoint_after_{stage}.pth',
            )
            if not os.path.isfile(stage_checkpoint_path):
                continue
            checkpoint = torch.load(
                stage_checkpoint_path,
                map_location=device,
            )
            expected_stage_epochs = {
                name: int(stage_epochs[name])
                for name in stages
            }
            if (
                checkpoint.get('base_seed') != int(base_seed)
                or checkpoint.get('completed_stage') != stage
                or checkpoint.get('stage_epochs') != expected_stage_epochs
                or checkpoint.get('config') != config
            ):
                raise ValueError(
                    f'全量训练恢复点与当前配置不一致: {stage_checkpoint_path}'
                )
            model.load_state_dict(checkpoint['model_state_dict'])
            first_stage_index = stage_index + 1
            print(f'恢复全量训练阶段完成点: {stage_checkpoint_path}')
            break

    print(
        f"\n========== Seed {base_seed} full-data retraining: "
        f"{stage_epochs} =========="
    )
    for stage in stages[first_stage_index:]:
        if (
            stage != 'ranking'
            and config.get('cache_frozen_backbone', True)
            and not isinstance(train_loader.dataset, FrozenBackboneDataset)
        ):
            print('全量训练：缓存冻结 Ranking 主干表示')
            train_loader = build_data_loader(
                cache_frozen_backbone_dataset(model, train_dataset, device),
                True,
                device,
            )
        num_epochs = int(stage_epochs[stage])
        criterion, optimizer = build_training_components(
            model,
            stage=stage,
        )
        grad_scaler = create_grad_scaler(device)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.2,
            total_iters=num_epochs,
        )
        for epoch in range(num_epochs):
            train_loss, train_metrics = train_ranking_model(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                writer,
                grad_scaler,
                stage=stage,
            )
            scheduler.step()
            writer.add_scalar(
                f'{stage}/learning_rate',
                scheduler.get_last_lr()[0],
                global_step=epoch,
            )
            print(
                f"Full train {stage} Epoch {epoch + 1}/{num_epochs}: "
                f"loss={train_loss:.4f}, "
                f"id_gate={model.identity_gate_value().detach().item():.4f}"
            )
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'base_seed': int(base_seed),
                'completed_stage': stage,
                'stage_epochs': {
                    name: int(stage_epochs[name])
                    for name in stages
                },
                'config': config,
            },
            os.path.join(final_dir, f'checkpoint_after_{stage}.pth'),
        )
        if stage == 'ranking' and config.get('cache_frozen_backbone', True):
            print('全量训练 Ranking 阶段完成：缓存主干表示')
            train_loader = build_data_loader(
                cache_frozen_backbone_dataset(model, train_dataset, device),
                True,
                device,
            )

    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    metadata = {
        'base_seed': int(base_seed),
        'stage_epochs': {
            stage: int(stage_epochs[stage])
            for stage in TRAINING_STAGES
        },
        'epoch_selection': 'per_stage_median_optimizer_updates',
        'train_end': pd.Timestamp(train_end).strftime('%Y-%m-%d'),
        'ranking_samples': len(train_dataset),
        'identity_gate': float(
            model.identity_gate_value().detach().cpu().item()
        ),
    }
    with open(os.path.join(output_dir, 'final_training.json'), 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    writer.close()
    return metadata


def build_policy_calibration_kwargs(runtime_config, ensemble_enabled):
    """集中生成策略校准参数，确保训练后校准与策略重放一致。"""
    return dict(
        min_exposure=runtime_config['min_exposure'],
        max_exposure=runtime_config['max_exposure'],
        allocation_temperature=runtime_config.get(
            'allocation_temperature',
            1.0,
        ),
        allocation_blend_grid=runtime_config.get(
            'allocation_blend_grid',
            [0.25, 0.5, 0.75, 1.0],
        ),
        disagreement_gamma_grid=(
            runtime_config.get(
                'disagreement_gamma_grid',
                [0.0, 2.0, 4.0, 8.0],
            )
            if ensemble_enabled else [0.0]
        ),
        selection_risk_gamma_grid=runtime_config.get(
            'selection_risk_gamma_grid',
            [0.0],
        ),
        risk_score_penalty_grid=runtime_config.get(
            'risk_score_penalty_grid',
            [0.0],
        ),
        risk_1d_blend=float(runtime_config.get('risk_1d_blend', 0.40)),
        risk_3d_blend=float(runtime_config.get('risk_3d_blend', 0.60)),
        risk_5d_blend=float(runtime_config.get('risk_5d_blend', 0.0)),
        tail_5d_blend=float(runtime_config.get('tail_5d_blend', 0.0)),
        correlation_exposure_gamma_grid=runtime_config.get(
            'correlation_exposure_gamma_grid',
            [0.0],
        ),
        exposure_head_blend_grid=runtime_config.get(
            'exposure_head_blend_grid',
            [0.25, 0.50, 0.75, 1.0],
        ),
        selection_candidate_k=int(runtime_config.get(
            'selection_candidate_k',
            20,
        )),
        correlation_lookbacks=runtime_config.get(
            'selection_correlation_lookbacks',
            [20],
        ),
        cluster_correlation_threshold=float(runtime_config.get(
            'cluster_correlation_threshold',
            0.60,
        )),
        max_stocks_per_cluster=int(runtime_config.get(
            'max_stocks_per_cluster',
            2,
        )),
        cluster_max_raw_rank=int(runtime_config.get(
            'cluster_max_raw_rank',
            10,
        )),
        tail_5d_threshold=float(runtime_config.get(
            'tail_5d_threshold',
            -0.03,
        )),
        fixed_exposure_baseline=float(runtime_config.get(
            'fixed_exposure_baseline',
            0.6231689453125,
        )),
        downside_weight=float(runtime_config.get(
            'ensemble_downside_weight',
            0.5,
        )),
        top_k=5,
        policy_simplicity_tolerance=float(runtime_config.get(
            'policy_simplicity_tolerance',
            0.001,
        )),
        module_min_positive_fold_fraction=float(runtime_config.get(
            'module_min_positive_fold_fraction',
            2 / 3,
        )),
        cluster_cap_grid=runtime_config.get(
            'cluster_cap_grid',
            [False, True],
        ),
        minimum_allocation_blend=float(runtime_config.get(
            'minimum_allocation_deployment_blend',
            0.25,
        )),
        minimum_exposure_blend=float(runtime_config.get(
            'minimum_exposure_deployment_blend',
            0.25,
        )),
    )


def _compact_policy(policy, include_metrics=True):
    fields = (
        'allocation_blend',
        'disagreement_gamma',
        'selection_risk_gamma',
        'risk_score_penalty',
        'risk_1d_blend',
        'risk_3d_blend',
        'risk_5d_blend',
        'tail_5d_blend',
        'correlation_exposure_gamma',
        'exposure_head_blend',
        'selection_candidate_k',
        'correlation_lookbacks',
        'cluster_cap_enabled',
        'cluster_correlation_threshold',
        'max_stocks_per_cluster',
        'cluster_max_raw_rank',
        'tail_5d_threshold',
        'fixed_exposure_baseline',
        'min_exposure',
        'max_exposure',
        'allocation_temperature',
        'top_k',
        'downside_weight',
        'selection_metric',
    )
    compact = {
        field: policy[field] for field in fields if field in policy
    }
    if include_metrics and 'oof_metrics' in policy:
        compact['oof_metrics'] = {
            key: value
            for key, value in policy['oof_metrics'].items()
            if key != 'daily'
        }
    return compact


def run_policy_only():
    """复用既有 OOF 与模型产物，仅重放并部署策略层。"""
    source_dir = os.path.abspath(config['policy_only_source_dir'])
    output_dir = os.path.abspath(config['policy_output_dir'])
    if source_dir == output_dir:
        raise ValueError('策略输出目录必须与模型产物来源目录不同')
    required_source_files = (
        'config.json',
        'ensemble_policy.json',
        'cross_validation_summary.json',
        'scaler.pkl',
        'stockid2idx.json',
    )
    missing = [
        name for name in required_source_files
        if not os.path.isfile(os.path.join(source_dir, name))
    ]
    if missing:
        raise FileNotFoundError(f'策略来源目录缺少产物: {missing}')
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(source_dir, 'config.json'),
        encoding='utf-8',
    ) as file:
        source_config = json.load(file)
    with open(
        os.path.join(source_dir, 'ensemble_policy.json'),
        encoding='utf-8',
    ) as file:
        source_policy = json.load(file)
    with open(
        os.path.join(source_dir, 'cross_validation_summary.json'),
        encoding='utf-8',
    ) as file:
        source_summary = json.load(file)
    ensemble_seeds = [
        int(seed) for seed in source_policy.get('ensemble_seeds', [42])
    ]
    ensemble_enabled = bool(source_policy.get(
        'ensemble_enabled',
        len(ensemble_seeds) > 1,
    ))
    fold_ids = sorted({
        int(row['fold']) for row in source_summary.get('folds', [])
    })
    if len(fold_ids) < 3:
        raise ValueError('策略重放需要来源目录中完整的至少三折 OOF')
    folds = [
        {
            'fold': int(row['fold']),
            'train_end': pd.Timestamp(row['train_end']),
            'val_start': pd.Timestamp(row['val_start']),
            'val_end': pd.Timestamp(row['val_end']),
        }
        for row in source_summary['folds']
        if int(row['fold']) in fold_ids
    ]
    source_data_path = source_config.get('data_path', './data_5y')
    source_data_file = os.path.join(source_data_path, 'train.csv')
    if not os.path.isfile(source_data_file):
        raise FileNotFoundError(
            f'无法从训练快照加载交易日历: {source_data_file}'
        )
    trading_dates = pd.read_csv(
        source_data_file,
        usecols=['日期'],
    )['日期'].dropna().unique()
    print(
        'POLICY_ONLY=1: 跳过特征工程、三折训练和全量重训；'
        f'从 {source_dir} 加载 {len(fold_ids)} 折 OOF'
    )
    ensemble_days = []
    for fold in tqdm(
        fold_ids,
        desc='加载并对齐 OOF',
        unit='折',
        dynamic_ncols=True,
    ):
        records_by_model = []
        for seed in ensemble_seeds:
            prediction_path = os.path.join(
                source_dir,
                f'seed_{seed}',
                f'fold_{fold}',
                'oof_predictions.joblib',
            )
            if not os.path.isfile(prediction_path):
                raise FileNotFoundError(
                    f'缺少 Seed {seed} Fold {fold} OOF: '
                    f'{prediction_path}'
                )
            records_by_model.append(attach_label_end_dates(
                joblib.load(prediction_path),
                trading_dates,
                horizon=int(source_config.get('purge_days', 5)),
            ))
        ensemble_days.extend(align_oof_prediction_records(
            records_by_model,
            fold,
        ))
    lgbm_folds = load_or_build_lgbm_oof_scores(
        source_dir, output_dir, source_config, folds, ensemble_days,
    )
    calibration_kwargs = build_policy_calibration_kwargs(
        config,
        ensemble_enabled,
    )
    calibration = calibrate_v17_oof_strategy(
        ensemble_days,
        folds,
        calibration_kwargs,
        lgbm_folds,
    )
    replay = calibration['cross_fitted_policy']
    cross_metrics = replay['metrics']
    candidate_policy = replay['all_oof_candidate_policy']
    robust_policy = replay['robust_deployment_policy']
    artifact_source_dir = os.path.relpath(source_dir, output_dir)
    policy = dict(robust_policy)
    policy.update({
        'ensemble_enabled': ensemble_enabled,
        'mode': source_policy.get('mode', 'single_model'),
        'policy_role': 'robust_module_gated_deployment_policy',
        'promotion_metric_source': 'cross_fitted_oof',
        'ensemble_seeds': ensemble_seeds,
        'model_paths': source_policy['model_paths'],
        'scaler_path': source_policy.get('scaler_path', 'scaler.pkl'),
        'config_path': source_policy.get('config_path', 'config.json'),
        'selection_risk_lookback': int(source_policy.get(
            'selection_risk_lookback',
            source_config.get('selection_risk_lookback', 20),
        )),
        'artifact_source_dir': artifact_source_dir,
        'lgbm_forward': calibration['lgbm_forward'],
        'deployment_lgbm_weight': calibration['deployment_lgbm_weight'],
        'deployment_lgbm_weight_source': calibration[
            'deployment_lgbm_weight_source'
        ],
        # predict.py 使用这两个字段加载来源目录中的全量开发期树模型。
        'lgbm_weight': calibration['deployment_lgbm_weight'],
        'lgbm_model_path': 'lgbm_ranker.joblib',
        'module_eligibility': replay['module_eligibility'],
        'module_alternative_reports': candidate_policy.get(
            'module_alternative_reports',
            {},
        ),
        'module_fallbacks': {
            'risk_score': 0.0,
            'reversal': 0.0,
            'correlation_cluster': False,
            'allocation': 0.25,
            'exposure_head': 0.25,
            'correlation_exposure': 0.0,
        },
        'cross_fitted_policy': {
            key: value for key, value in replay.items()
            if key not in (
                'all_oof_candidate_policy',
                'robust_deployment_policy',
            )
        },
        'all_oof_candidate_policy': _compact_policy(candidate_policy),
        'robust_deployment_policy': _compact_policy(robust_policy),
    })
    baseline_path = os.path.abspath(config['baseline_source_dir'])
    with open(
        os.path.join(baseline_path, 'cross_validation_summary.json'),
        encoding='utf-8',
    ) as file:
        baseline_summary = json.load(file)
    baseline_metrics = baseline_summary['cross_fitted_oof']['metrics']
    baseline_by_fold = {
        int(row['fold']): row
        for row in baseline_metrics['folds']
    }
    candidate_by_fold = {
        int(row['fold']): row for row in cross_metrics['folds']
    }
    fold_deltas = [
        {
            'fold': fold_id,
            'weighted_return_delta': float(
                candidate_by_fold[fold_id]['mean_weighted_portfolio_return']
                - baseline_by_fold[fold_id]['mean_weighted_portfolio_return']
            ),
            'rank_ic_delta': float(
                candidate_by_fold[fold_id]['mean_rank_ic']
                - baseline_by_fold[fold_id]['mean_rank_ic']
            ),
        }
        for fold_id in fold_ids
    ]
    metric_deltas = {
        'mean_weighted_portfolio_return': float(
            cross_metrics['mean_weighted_portfolio_return']
            - baseline_metrics['mean_weighted_portfolio_return']
        ),
        'p10_weighted_portfolio_return': float(
            cross_metrics['p10_weighted_portfolio_return']
            - baseline_metrics['p10_weighted_portfolio_return']
        ),
        'worst_fold_weighted_portfolio_return': float(
            cross_metrics['worst_fold_weighted_portfolio_return']
            - baseline_metrics['worst_fold_weighted_portfolio_return']
        ),
        'mean_rank_ic': float(
            cross_metrics['mean_rank_ic'] - baseline_metrics['mean_rank_ic']
        ),
    }
    promotion_criteria = {
        'applicable': True,
        'mean_weighted_return_gain_at_least_10bp': bool(
            metric_deltas['mean_weighted_portfolio_return'] >= 0.001
        ),
        'at_least_two_positive_fold_gains': bool(sum(
            row['weighted_return_delta'] > 0.0 for row in fold_deltas
        ) >= 2),
        'p10_not_worse_than_10bp': bool(
            metric_deltas['p10_weighted_portfolio_return'] >= -0.001
        ),
        'worst_fold_not_worse_than_10bp': bool(
            metric_deltas['worst_fold_weighted_portfolio_return'] >= -0.001
        ),
        'rank_ic_not_worse_than_0_005': bool(
            metric_deltas['mean_rank_ic'] >= -0.005
        ),
        'allocation_minimum_retained': bool(
            policy['allocation_blend'] >= 0.25
        ),
        'exposure_minimum_retained': bool(
            policy['exposure_head_blend'] >= 0.25
        ),
    }
    promotion_criteria['passed'] = all(
        value for key, value in promotion_criteria.items()
        if key != 'applicable'
    )
    policy['promotion_criteria'] = promotion_criteria
    runtime_config = dict(config)
    runtime_config['artifact_source_dir'] = artifact_source_dir
    runtime_config['source_training_config'] = os.path.join(
        artifact_source_dir,
        source_policy.get('config_path', 'config.json'),
    )
    with open(
        os.path.join(output_dir, 'config.json'),
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(runtime_config, file, indent=2, ensure_ascii=False)
    with open(
        os.path.join(output_dir, 'ensemble_policy.json'),
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(policy, file, indent=2, ensure_ascii=False)

    robust_metrics = robust_policy['oof_metrics']
    summary = {
        'training_mode': 'policy_only_module_gated_replay',
        'artifact_source_dir': artifact_source_dir,
        'num_folds': len(fold_ids),
        'evaluation_stride': int(source_config.get(
            'evaluation_stride',
            5,
        )),
        'ensemble_seeds': ensemble_seeds,
        'allocation_blend': float(policy['allocation_blend']),
        'selection_risk_gamma': float(policy['selection_risk_gamma']),
        'risk_score_penalty': float(policy['risk_score_penalty']),
        'cluster_cap_enabled': bool(policy['cluster_cap_enabled']),
        'correlation_exposure_gamma': float(
            policy['correlation_exposure_gamma']
        ),
        'exposure_head_blend': float(policy['exposure_head_blend']),
        'lgbm_forward': calibration['lgbm_forward'],
        'deployment_lgbm_weight': calibration['deployment_lgbm_weight'],
        'deployment_lgbm_weight_source': calibration[
            'deployment_lgbm_weight_source'
        ],
        **{
            key: cross_metrics[key] for key in (
                'mean_top5_return',
                'worst_fold_top5_return',
                'mean_weighted_portfolio_return',
                'worst_fold_weighted_portfolio_return',
                'p10_weighted_portfolio_return',
                'std_weighted_portfolio_return',
                'positive_rate',
                'mean_gross_exposure',
                'mean_cash_weight',
                'mean_rank_ic',
                'worst_daily_rank_ic',
                'worst_rank_ic',
                'worst_fold_mean_rank_ic',
                'mean_model_disagreement',
                'mean_allocation_contribution',
                'mean_allocation_at_exposure_contribution',
                'mean_exposure_contribution',
                'mean_exposure_policy_contribution',
                'exposure_policy_objective_delta',
                'policy_objective',
                'fixed_exposure_policy_objective',
                'exposure_std',
                'regime_gate_std',
                'mean_positive_correlation',
                'raw_mean_positive_correlation',
                'mean_diversification_return_contribution',
                'cluster_constraint_application_rate',
                'cluster_constraint_skip_rate',
                'max_selected_raw_rank',
            )
        },
        'weighted_portfolio_positive_rate': cross_metrics['positive_rate'],
        'original_ranking_baseline': candidate_policy.get(
            'stage_reports',
            {},
        ).get(
            'ranking',
            {},
        ).get(
            'baseline_metrics',
            {},
        ),
        'module_eligibility': replay['module_eligibility'],
        'module_alternative_reports': candidate_policy.get(
            'module_alternative_reports',
            {},
        ),
        'cross_fitted_oof': {
            key: value for key, value in replay.items()
            if key not in (
                'all_oof_candidate_policy',
                'robust_deployment_policy',
            )
        },
        'ensemble_oof': {
            key: value for key, value in cross_metrics.items()
            if key != 'daily'
        },
        'all_oof_candidate_policy': _compact_policy(candidate_policy),
        'robust_deployment_policy': _compact_policy(robust_policy),
        'deployment_oof': {
            key: value for key, value in robust_metrics.items()
            if key != 'daily'
        },
        'deployment_policy': _compact_policy(robust_policy, False),
        'promotion_criteria': promotion_criteria,
        'baseline_comparison': {
            'baseline_source_dir': os.path.relpath(baseline_path, output_dir),
            'metric_deltas': metric_deltas,
            'fold_deltas': fold_deltas,
        },
        'original_v17_candidate': {
            'artifact_source_dir': artifact_source_dir,
            'reported_cross_fitted_metrics': source_summary.get(
                'cross_fitted_oof', {}
            ).get('metrics', {}),
            'warning': (
                'v1.17 的全局 LightGBM 部署权重曾回填早期 OOF；'
                '仅作历史参考，不作为 v1.18 晋级依据。'
            ),
        },
        'disabled_strategy_modules': {
            'risk_penalty': True,
            'reversal_penalty': True,
            'correlation_exposure': True,
            'correlation_cluster_replacement': True,
            'exposure_mode': '25% Exposure Head + 75% 0.999999 fallback',
        },
        'folds': cross_metrics['folds'],
        'source_training_folds': source_summary.get('folds', []),
        'full_training': {
            'reused': True,
            'artifact_source_dir': artifact_source_dir,
            'source_full_training': source_summary.get('full_training', {}),
        },
    }
    for optional_key in (
        'mean_tail_5d_brier',
        'mean_tail_5d_baseline_brier',
        'mean_tail_5d_brier_skill',
        'mean_tail_5d_roc_auc',
        'mean_tail_5d_pr_auc',
        'mean_tail_5d_event_rate',
        'mean_risk_1d_baseline_brier',
        'mean_risk_1d_brier_skill',
        'mean_risk_1d_event_rate',
        'mean_risk_3d_baseline_brier',
        'mean_risk_3d_brier_skill',
        'mean_risk_3d_event_rate',
        'mean_risk_5d_baseline_brier',
        'mean_risk_5d_brier_skill',
        'mean_risk_5d_event_rate',
        'mean_selected_tail_5d',
        'combined_risk_return_spearman',
        'regime_return_spearman',
        'regime_market_return_spearman',
        'regime_tail_share_spearman',
        'mean_effective_candidate_k',
        'max_effective_candidate_k',
        'candidate_pool_expansion_rate',
        'mean_reversal_risk',
    ):
        if optional_key in cross_metrics:
            summary[optional_key] = cross_metrics[optional_key]
    with open(
        os.path.join(output_dir, 'cross_validation_summary.json'),
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print('\n========== POLICY_ONLY strategy replay summary ==========')
    print(json.dumps({
        'mean_top5_return': summary['mean_top5_return'],
        'mean_weighted_portfolio_return': (
            summary['mean_weighted_portfolio_return']
        ),
        'worst_fold_weighted_portfolio_return': (
            summary['worst_fold_weighted_portfolio_return']
        ),
        'mean_rank_ic': summary['mean_rank_ic'],
        'robust_deployment_policy': summary['deployment_policy'],
        'promotion_criteria': promotion_criteria,
    }, indent=2, ensure_ascii=False))
    return summary['mean_weighted_portfolio_return']


def main():
    if stress_eval_enabled():
        if lockbox_eval_enabled():
            raise ValueError('STRESS_EVAL 与 LOCKBOX_EVAL 不能同时启用')
        apply_v17_profile()
        return run_lockbox_eval(config['output_dir'], stress=True)
    if lockbox_eval_enabled():
        apply_v17_profile()
        return run_lockbox_eval(config['output_dir'])
    if policy_only_enabled():
        return run_policy_only()
    profile = apply_v17_profile()
    final_deployment = (
        os.environ.get('V17_INCLUDE_LOCKBOX', '0') == '1'
        and os.environ.get('LOCKBOX_ACCEPTED', '0') == '1'
    )
    if final_deployment:
        require_accepted_lockbox_for_deployment()
        return run_frozen_final_deployment(config['output_dir'])
    if config.get('policy_only_experiment', False) and not final_deployment:
        raise ValueError(
            '当前配置是策略重放实验，不允许重新训练；'
            '请使用 POLICY_ONLY=1 ./train.sh；锁箱验收通过后才可设置 '
            'V17_INCLUDE_LOCKBOX=1 LOCKBOX_ACCEPTED=1 做独立最终部署重训'
        )
    configured_ensemble_seeds = [int(seed) for seed in config.get(
        'ensemble_seeds',
        [42, 142, 242],
    )]
    ensemble_enabled = bool(config.get('ensemble_enabled', False))
    if ensemble_enabled:
        ensemble_seeds = configured_ensemble_seeds
        if len(ensemble_seeds) < 2:
            raise ValueError('启用 ensemble 时至少需要两个随机种子')
    else:
        ensemble_seeds = [int(config.get('seed', 42))]
    if not ensemble_seeds or len(set(ensemble_seeds)) != len(ensemble_seeds):
        raise ValueError('训练随机种子必须非空且互不重复')
    set_seed(ensemble_seeds[0])
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'config.json')
    if resume_training_enabled() and os.path.isfile(config_path):
        with open(config_path, encoding='utf-8') as file:
            existing_config = json.load(file)
        if existing_config != config:
            raise ValueError(
                'RESUME_TRAINING=1 但训练目录中的 config.json 与当前配置不同'
            )
    with open(config_path, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4, ensure_ascii=False)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    configure_accelerator(device)
    print(
        f"训练模式: {profile}/{'多种子 ensemble' if ensemble_enabled else '单种子三折'}; "
        f"seeds={ensemble_seeds}; 设备: {device}; AMP={use_amp(device)}; "
        f"TF32={device.type == 'cuda' and config.get('tf32_enabled', True)}; "
        f"batch_size={config['batch_size']}"
    )

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df, lockbox_start = split_v17_lockbox(full_df)
    folds = build_walk_forward_folds(
        full_df,
        num_folds=config['num_folds'],
        validation_months=config['validation_months'],
        purge_days=config['purge_days'],
    )

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx + 2 for idx, sid in enumerate(sorted(all_stock_ids))}
    with open(os.path.join(output_dir, 'stockid2idx.json'), 'w', encoding='utf-8') as file:
        json.dump(stockid2idx, file, indent=2, ensure_ascii=False)

    full_data, features = preprocess_data(full_df, is_train=True, stockid2idx=stockid2idx)
    full_data['日期'] = pd.to_datetime(full_data['日期'])
    write_v17_manifest(
        output_dir, features, folds, lockbox_start, len(stockid2idx),
    )
    if config.get('exposure_market_encoder_enabled', False):
        market_feature_names = [
            *RELATIVE_MARKET_FEATURES[-5:],
            *MARKET_PRESSURE_FEATURES,
        ]
        expected_market_indices = [
            features.index(name)
            for name in market_feature_names
        ]
        configured_market_indices = [
            int(index)
            for index in config.get('market_state_feature_indices', [])
        ]
        if configured_market_indices != expected_market_indices:
            raise ValueError(
                'market_state_feature_indices 与市场状态特征位置不一致: '
                f'{configured_market_indices} != {expected_market_indices}'
            )
        configured_regime_indices = [
            int(index)
            for index in config.get(
                'regime_market_feature_indices',
                configured_market_indices,
            )
        ]
        if configured_regime_indices != expected_market_indices:
            raise ValueError(
                'regime_market_feature_indices 与市场压力特征位置不一致: '
                f'{configured_regime_indices} != {expected_market_indices}'
            )

    fold_results = []
    oof_records = {
        int(fold['fold']): {seed: [] for seed in ensemble_seeds}
        for fold in folds
    }
    for base_seed in ensemble_seeds:
        seed_dir = os.path.join(output_dir, f'seed_{base_seed}')
        os.makedirs(seed_dir, exist_ok=True)
        for fold in folds:
            completed_fold = None
            if resume_training_enabled():
                completed_fold = load_completed_fold_artifacts(
                    seed_dir,
                    fold,
                    base_seed,
                )
            if completed_fold is None:
                result, predictions = train_one_fold(
                    full_data=full_data,
                    features=features,
                    fold=fold,
                    num_stocks=len(stockid2idx),
                    device=device,
                    output_dir=seed_dir,
                    base_seed=base_seed,
                )
            else:
                result, predictions = completed_fold
                print(
                    f"复用 Seed {base_seed} Fold {fold['fold']} "
                    '已完成产物'
                )
            fold_results.append(result)
            oof_records[int(fold['fold'])][base_seed] = predictions

    lgbm_folds = fit_lgbm_oof_scores(
        full_data, features, folds, oof_records, output_dir,
    )

    ensemble_days = []
    single_seed_days = {seed: [] for seed in ensemble_seeds}
    # full_data 会移除标签不完整的末端日期；OOF 标签结束日仍须在开发期原始日历上推导。
    full_trading_dates = full_df['日期'].dropna().unique()
    for fold in folds:
        fold_number = int(fold['fold'])
        aligned_days = align_oof_prediction_records(
            [
                attach_label_end_dates(
                    oof_records[fold_number][seed],
                    full_trading_dates,
                    horizon=int(config.get('purge_days', 5)),
                )
                for seed in ensemble_seeds
            ],
            fold=fold_number,
        )
        if lgbm_folds:
            lgbm_by_date = {
                record['prediction_date']: record['lgbm_scores']
                for record in oof_records[fold_number][ensemble_seeds[0]]
                if 'lgbm_scores' in record
            }
            for day in aligned_days:
                day['lgbm_scores'] = np.asarray(
                    lgbm_by_date[day['prediction_date']], dtype=np.float64,
                )
        ensemble_days.extend(aligned_days)
        for seed in ensemble_seeds:
            seed_days = align_oof_prediction_records(
                [attach_label_end_dates(
                    oof_records[fold_number][seed],
                    full_trading_dates,
                    horizon=int(config.get('purge_days', 5)),
                )],
                fold=fold_number,
            )
            if lgbm_folds:
                for day in seed_days:
                    day['lgbm_scores'] = np.asarray(
                        lgbm_by_date[day['prediction_date']], dtype=np.float64,
                    )
            single_seed_days[seed].extend(seed_days)

    attach_oof_strategy_metadata(ensemble_days, full_data)
    for days in single_seed_days.values():
        attach_oof_strategy_metadata(days, full_data)

    policy_calibration_kwargs = dict(
        min_exposure=config['min_exposure'],
        max_exposure=config['max_exposure'],
        allocation_temperature=config.get('allocation_temperature', 1.0),
        allocation_blend_grid=config.get(
            'allocation_blend_grid',
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ),
        disagreement_gamma_grid=(
            config.get(
                'disagreement_gamma_grid',
                [0.0, 2.0, 4.0, 8.0],
            )
            if ensemble_enabled else [0.0]
        ),
        selection_risk_gamma_grid=config.get(
            'selection_risk_gamma_grid',
            [0.0],
        ),
        risk_score_penalty_grid=config.get(
            'risk_score_penalty_grid',
            [0.0],
        ),
        risk_1d_blend=float(config.get('risk_1d_blend', 0.40)),
        risk_3d_blend=float(config.get('risk_3d_blend', 0.60)),
        risk_5d_blend=float(config.get('risk_5d_blend', 0.0)),
        tail_5d_blend=float(config.get('tail_5d_blend', 0.0)),
        correlation_exposure_gamma_grid=config.get(
            'correlation_exposure_gamma_grid',
            [0.0],
        ),
        exposure_head_blend_grid=config.get(
            'exposure_head_blend_grid',
            [1.0],
        ),
        selection_candidate_k=int(config.get(
            'selection_candidate_k',
            20,
        )),
        correlation_lookbacks=config.get(
            'selection_correlation_lookbacks',
            [20],
        ),
        cluster_cap_enabled=bool(config.get(
            'cluster_cap_enabled',
            False,
        )),
        cluster_correlation_threshold=float(config.get(
            'cluster_correlation_threshold',
            0.60,
        )),
        max_stocks_per_cluster=int(config.get(
            'max_stocks_per_cluster',
            2,
        )),
        tail_5d_threshold=float(config.get(
            'tail_5d_threshold',
            -0.03,
        )),
        fixed_exposure_baseline=float(config.get(
            'fixed_exposure_baseline',
            0.6231689453125,
        )),
        max_stocks_per_industry=(
            int(config['max_stocks_per_industry'])
            if config.get('industry_cap_enabled', False) else None
        ),
        industry_candidate_k=int(config.get('industry_candidate_k', 10)),
        downside_weight=config.get('ensemble_downside_weight', 0.5),
        top_k=5,
    )
    checkpoint_path = os.path.join(output_dir, 'oof_strategy_calibration.joblib')
    signature = oof_strategy_checkpoint_signature(
        features, folds, lgbm_folds, ensemble_seeds,
    )
    calibration = None
    if resume_training_enabled() and os.path.isfile(checkpoint_path):
        cached = joblib.load(checkpoint_path)
        if cached.get('signature') == signature:
            calibration = cached['calibration']
            print('恢复严格 OOF 策略校准 checkpoint')
        else:
            print('OOF 策略校准 checkpoint 与当前工件不匹配，将重新校准')
    if calibration is None:
        print('阶段 OOF 策略校准：固定近期 504 日纯 LightGBM + 行业最多两只')
        candidate = calibrate_v20_recent_lgbm_candidate(
            ensemble_days, policy_calibration_kwargs,
        )
        calibration = {'candidate': candidate}
        save_joblib_checkpoint(
            {
                'signature': signature, 'calibration': calibration,
            }, checkpoint_path,
        )
        print('OOF 策略校准 checkpoint 已保存')
    candidate = calibration['candidate']
    baseline_path = os.path.abspath(config['baseline_source_dir'])
    with open(
        os.path.join(baseline_path, 'cross_validation_summary.json'),
        encoding='utf-8',
    ) as file:
        baseline_summary = json.load(file)
    baseline_metrics = baseline_summary['cross_fitted_oof']['metrics']
    candidate['promotion_criteria'] = promotion_against_baseline(
        candidate['metrics'], baseline_metrics,
    )
    selected = candidate
    selected_name = selected['policy']['candidate_name']
    deployment_lgbm_weight = float(selected['lgbm_weight'])
    policy = dict(selected['policy'])
    cross_fitted_policy = selected['cross_fitted_policy']
    ensemble_metrics = selected['metrics']
    promotion_criteria = selected['promotion_criteria']
    market_states = market_state_diagnostics(ensemble_metrics, full_data, folds)
    single_seed_summaries = {}
    for seed in ensemble_seeds:
        single_seed_summaries[str(seed)] = evaluate_ensemble_policy(
            fuse_lgbm_scores(single_seed_days[seed], deployment_lgbm_weight),
            policy,
        )
    mean_single_return = float(np.mean([
        metrics['mean_weighted_portfolio_return']
        for metrics in single_seed_summaries.values()
    ]))
    identity_sensitivity = summarize_identity_sensitivity(fold_results)

    full_train_dataset, full_train_end = prepare_full_training_dataset(
        full_data=full_data,
        features=features,
        output_dir=output_dir,
    )
    full_steps_per_epoch = max(
        1,
        int(np.ceil(len(full_train_dataset) / config['batch_size'])),
    )
    stage_epochs = {}
    stage_epoch_selection = {}
    for stage in TRAINING_STAGES:
        fold_best_updates = [
            int(result['stage_training'][stage]['best_epoch'])
            * int(result['stage_training'][stage]['steps_per_epoch'])
            for result in fold_results
        ]
        median_updates = int(np.median(fold_best_updates))
        update_matched_epochs = max(
            1,
            int(np.ceil(median_updates / full_steps_per_epoch)),
        )
        minimum_epoch = int(config.get(
            f'{stage}_min_final_epochs',
            1,
        ))
        maximum_epoch = stage_settings(stage)['max_epochs']
        if not 1 <= minimum_epoch <= maximum_epoch:
            raise ValueError(f'{stage}_min_final_epochs 超出阶段训练范围')
        stage_epochs[stage] = min(
            maximum_epoch,
            max(minimum_epoch, update_matched_epochs),
        )
        stage_epoch_selection[stage] = {
            'fold_best_updates': fold_best_updates,
            'median_best_updates': median_updates,
            'full_steps_per_epoch': full_steps_per_epoch,
            'update_matched_epochs': update_matched_epochs,
            'min_final_epochs': minimum_epoch,
            'final_epochs': stage_epochs[stage],
        }
    full_training = []
    model_paths = []
    for base_seed in ensemble_seeds:
        seed_dir = os.path.join(output_dir, f'seed_{base_seed}')
        full_training.append(train_final_model(
            train_dataset=full_train_dataset,
            train_end=full_train_end,
            features=features,
            num_stocks=len(stockid2idx),
            device=device,
            output_dir=seed_dir,
            stage_epochs=stage_epochs,
            base_seed=base_seed,
        ))
        model_paths.append(
            os.path.join(f'seed_{base_seed}', 'best_model.pth')
        )

    lgbm_model_path = fit_lgbm_final(
        full_data, features, lgbm_folds, output_dir,
    )

    policy.update({
        'ensemble_enabled': ensemble_enabled,
        'mode': 'rank_ensemble' if ensemble_enabled else 'single_model',
        'policy_role': 'deployment_policy_calibrated_on_all_oof',
        'promotion_metric_source': (
            'cross_fitted_oof'
            if config.get('nested_oof_enabled', False)
            else 'all_oof'
        ),
        'ensemble_seeds': ensemble_seeds,
        'model_paths': model_paths,
        'lgbm_model_path': lgbm_model_path,
        'lgbm_weight': deployment_lgbm_weight,
        'lgbm_train_window_days': int(config['lgbm_train_window_days']),
        'max_stocks_per_industry': int(config['max_stocks_per_industry']),
        'industry_candidate_k': int(config['industry_candidate_k']),
        'pre_registered_candidates': {
            selected_name: {
                'lgbm_weight': candidate['lgbm_weight'],
                'promotion_criteria': candidate['promotion_criteria'],
            },
        },
        'scaler_path': 'scaler.pkl',
        'config_path': 'config.json',
        'selection_risk_lookback': int(config.get(
            'selection_risk_lookback',
            20,
        )),
        'cross_fitted_oof': cross_fitted_policy,
        'promotion_criteria': promotion_criteria,
    })
    with open(
        os.path.join(output_dir, 'ensemble_policy.json'),
        'w',
        encoding='utf-8',
    ) as file:
        json.dump(policy, file, indent=2, ensure_ascii=False)

    summary = {
        'training_mode': (
            'rank_ensemble' if ensemble_enabled else 'single_seed_3fold'
        ),
        'num_folds': len(folds),
        'evaluation_stride': int(config.get('evaluation_stride', 5)),
        'ensemble_seeds': ensemble_seeds,
        'allocation_blend': float(policy['allocation_blend']),
        'selection_risk_gamma': float(policy['selection_risk_gamma']),
        'risk_score_penalty': float(policy.get(
            'risk_score_penalty',
            0.0,
        )),
        'correlation_exposure_gamma': float(policy.get(
            'correlation_exposure_gamma',
            0.0,
        )),
        'exposure_head_blend': float(policy.get(
            'exposure_head_blend',
            1.0,
        )),
        'mean_top5_return': ensemble_metrics['mean_top5_return'],
        'worst_fold_top5_return': ensemble_metrics[
            'worst_fold_top5_return'
        ],
        'mean_weighted_portfolio_return': ensemble_metrics[
            'mean_weighted_portfolio_return'
        ],
        'worst_fold_weighted_portfolio_return': ensemble_metrics[
            'worst_fold_weighted_portfolio_return'
        ],
        'p10_weighted_portfolio_return': ensemble_metrics[
            'p10_weighted_portfolio_return'
        ],
        'worst_weighted_portfolio_return': ensemble_metrics[
            'worst_weighted_portfolio_return'
        ],
        'max_drawdown': ensemble_metrics['max_drawdown'],
        'std_weighted_portfolio_return': ensemble_metrics[
            'std_weighted_portfolio_return'
        ],
        'weighted_portfolio_positive_rate': ensemble_metrics[
            'positive_rate'
        ],
        'mean_gross_exposure': ensemble_metrics['mean_gross_exposure'],
        'mean_head_gross_exposure': ensemble_metrics[
            'mean_head_gross_exposure'
        ],
        'mean_cash_weight': ensemble_metrics['mean_cash_weight'],
        'mean_rank_ic': ensemble_metrics['mean_rank_ic'],
        'worst_rank_ic': ensemble_metrics['worst_rank_ic'],
        'mean_model_disagreement': ensemble_metrics[
            'mean_model_disagreement'
        ],
        'mean_regime_gate': ensemble_metrics['mean_regime_gate'],
        'regime_gate_std': ensemble_metrics['regime_gate_std'],
        'mean_selected_risk_1d': ensemble_metrics[
            'mean_selected_risk_1d'
        ],
        'mean_selected_risk_3d': ensemble_metrics[
            'mean_selected_risk_3d'
        ],
        'mean_selected_risk_5d': ensemble_metrics[
            'mean_selected_risk_5d'
        ],
        'mean_selected_tail_5d': ensemble_metrics[
            'mean_selected_tail_5d'
        ],
        'mean_selected_combined_risk': ensemble_metrics[
            'mean_selected_combined_risk'
        ],
        'mean_risk_1d_brier': ensemble_metrics['mean_risk_1d_brier'],
        'mean_risk_3d_brier': ensemble_metrics['mean_risk_3d_brier'],
        'mean_risk_5d_brier': ensemble_metrics['mean_risk_5d_brier'],
        'mean_tail_5d_brier': ensemble_metrics['mean_tail_5d_brier'],
        'mean_regime_brier': ensemble_metrics['mean_regime_brier'],
        'regime_return_spearman': ensemble_metrics[
            'regime_return_spearman'
        ],
        'regime_market_return_spearman': ensemble_metrics[
            'regime_market_return_spearman'
        ],
        'regime_tail_share_spearman': ensemble_metrics[
            'regime_tail_share_spearman'
        ],
        'tail_risk_return_spearman': ensemble_metrics[
            'tail_risk_return_spearman'
        ],
        'combined_risk_return_spearman': ensemble_metrics[
            'combined_risk_return_spearman'
        ],
        'mean_allocation_contribution': ensemble_metrics[
            'mean_allocation_contribution'
        ],
        'mean_allocation_at_exposure_contribution': ensemble_metrics[
            'mean_allocation_at_exposure_contribution'
        ],
        'mean_exposure_contribution': ensemble_metrics[
            'mean_exposure_contribution'
        ],
        'mean_positive_correlation': ensemble_metrics[
            'mean_positive_correlation'
        ],
        'raw_mean_positive_correlation': ensemble_metrics[
            'raw_mean_positive_correlation'
        ],
        'mean_raw_top5_return': ensemble_metrics[
            'mean_raw_top5_return'
        ],
        'mean_diversification_return_contribution': ensemble_metrics[
            'mean_diversification_return_contribution'
        ],
        'industry_constraint_application_rate': ensemble_metrics[
            'industry_constraint_application_rate'
        ],
        'industry_constraint_fallback_rate': ensemble_metrics[
            'industry_constraint_fallback_rate'
        ],
        'mean_industry_count': ensemble_metrics['mean_industry_count'],
        'mean_industry_hhi': ensemble_metrics['mean_industry_hhi'],
        'mean_max_industry_weight': ensemble_metrics['mean_max_industry_weight'],
        'market_state_diagnostics': market_states,
        'max_selected_cluster_count': ensemble_metrics[
            'max_selected_cluster_count'
        ],
        'mean_effective_candidate_k': ensemble_metrics[
            'mean_effective_candidate_k'
        ],
        'max_effective_candidate_k': ensemble_metrics[
            'max_effective_candidate_k'
        ],
        'candidate_pool_expansion_rate': ensemble_metrics[
            'candidate_pool_expansion_rate'
        ],
        'mean_reversal_risk': ensemble_metrics['mean_reversal_risk'],
        'exposure_std': ensemble_metrics['exposure_std'],
        'exposure_return_spearman': ensemble_metrics[
            'exposure_return_spearman'
        ],
        'fixed_exposure_policy_objective': ensemble_metrics[
            'fixed_exposure_policy_objective'
        ],
        'identity_sensitivity': identity_sensitivity,
        'ensemble_oof': ensemble_metrics,
        'cross_fitted_oof': cross_fitted_policy,
        'deployment_oof': policy['oof_metrics'],
        'deployment_policy': {
            key: policy[key] for key in (
                'allocation_blend',
                'disagreement_gamma',
                'selection_risk_gamma',
                'risk_score_penalty',
                'correlation_exposure_gamma',
                'exposure_head_blend',
            )
        },
        'single_seed_oof': single_seed_summaries,
        'single_seed_mean_weighted_return': mean_single_return,
        'promotion_criteria': promotion_criteria,
        'selected_candidate': selected_name,
        'pre_registered_candidates': {
            selected_name: {
                'lgbm_weight': candidate['lgbm_weight'],
                'metrics': {
                    key: value for key, value in candidate['metrics'].items()
                    if key != 'daily'
                },
                'promotion_criteria': candidate['promotion_criteria'],
            },
        },
        'baseline_comparison': {
            'baseline_source_dir': os.path.relpath(baseline_path, output_dir),
            'selected_metric_deltas': promotion_criteria['metric_deltas'],
            'selected_fold_deltas': promotion_criteria['fold_deltas'],
        },
        'folds': fold_results,
        'full_training': {
            'stage_epochs': stage_epochs,
            'epoch_selection': (
                'per_stage_median_optimizer_updates_across_folds'
            ),
            'num_fold_checkpoints': len(fold_results),
            'stage_epoch_selection': stage_epoch_selection,
            'models': full_training,
        },
    }
    with open(os.path.join(output_dir, 'cross_validation_summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n========== Cross-validation summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary['mean_weighted_portfolio_return']

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    completion_label = (
        '策略重放完成'
        if policy_only_enabled()
        else '训练完成'
    )
    print(
        f"\n########## {completion_label}！OOF 平均组合收益: "
        f"{best_score:.6f} ##########"
    )

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
)
from utils import create_ranking_dataset_vectorized
from utils import extract_selection_risk_context
from utils import align_oof_prediction_records, calibrate_ensemble_policy
from utils import cross_fit_ensemble_policy, evaluate_ensemble_policy
from utils import cross_fit_module_gated_policy
import joblib
import os
import json
import multiprocessing as mp
import random


def policy_only_enabled():
    return os.environ.get('POLICY_ONLY', '0').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
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
    risk_3d_targets, regime_targets, stage, return_components=False,
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
assert len(feature_cloums_map['158+39_reduced20']) == 171
assert len(feature_cloums_map['158+39_reduced25']) == 166
assert len(feature_cloums_map[RELATIVE_MARKET_FEATURE_SET]) == 178
assert len(feature_cloums_map[RISK_MARKET_FEATURE_SET]) == 193


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t2'] = processed.groupby('股票代码')['开盘'].shift(-2)
    processed['open_t4'] = processed.groupby('股票代码')['开盘'].shift(-4)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['return_1d_target'] = (
        (processed['open_t2'] - processed['open_t1'])
        / (processed['open_t1'] + 1e-12)
    )
    processed['return_3d_target'] = (
        (processed['open_t4'] - processed['open_t1'])
        / (processed['open_t1'] + 1e-12)
    )
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
    processed = processed.dropna(subset=[
        'label',
        'risk_1d_target',
        'risk_3d_target',
    ])

    dates = processed['日期']
    tail_threshold = float(config.get('tail_5d_threshold', -0.03))
    market_future_return = processed['label'].groupby(dates).transform('mean')
    market_tail_share = (
        processed['label'].le(tail_threshold).groupby(dates).transform('mean')
    )
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

    processed.drop(
        columns=['open_t1', 'open_t2', 'open_t4', 'open_t5'],
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
    }:
        processed = add_relative_market_features(processed)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
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
                 regime_weight=0.0):
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
        ) < 0:
            raise ValueError('风险头和状态门控损失权重不能为负')
        if self.risk_5d_target_temperature <= 0:
            raise ValueError('5日风险目标 temperature 必须大于0')
        if self.tail_5d_threshold >= 0:
            raise ValueError('5日尾部风险阈值必须小于0')

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
            if tail_5d_logits is not None and self.tail_5d_weight > 0:
                tail_5d_targets = (
                    raw_returns <= self.tail_5d_threshold
                ).to(raw_returns.dtype)
                components['tail_5d_loss'] = (
                    self.tail_5d_weight
                    * F.binary_cross_entropy_with_logits(
                        tail_5d_logits,
                        tail_5d_targets,
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
        lengths = {
            len(sequences),
            len(targets),
            len(relevance_scores),
            len(stock_indices),
            len(prediction_dates),
            len(self.risk_1d_targets),
            len(self.risk_3d_targets),
            len(self.regime_targets),
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
    ) in zip(
        sequences,
        targets,
        relevance,
        stock_indices,
        risk_1d_targets,
        risk_3d_targets,
        regime_targets,
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
                regime_targets, stage, return_components=True,
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
                        'tail_5d_targets': (
                            masked_targets[i][valid_indices]
                            <= float(config.get(
                                'tail_5d_threshold',
                                -0.03,
                            ))
                        ).float().detach().cpu().numpy(),
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
    train_dataset = RankingDataset(*train_parts)
    val_dataset = RankingDataset(*val_parts)
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
    train_dataset = RankingDataset(*train_parts)
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
            records_by_model.append(joblib.load(prediction_path))
        ensemble_days.extend(align_oof_prediction_records(
            records_by_model,
            fold,
        ))
    calibration_kwargs = build_policy_calibration_kwargs(
        config,
        ensemble_enabled,
    )
    replay = cross_fit_module_gated_policy(
        ensemble_days,
        **calibration_kwargs,
    )
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
    enabled_modules_valid = all(
        details.get('eligible', False)
        for module, details in replay['module_eligibility'].items()
        if policy.get({
            'risk_score': 'risk_score_penalty',
            'reversal': 'selection_risk_gamma',
            'correlation_cluster': 'cluster_cap_enabled',
            'allocation': 'allocation_blend',
            'exposure_head': 'exposure_head_blend',
            'correlation_exposure': 'correlation_exposure_gamma',
        }[module]) != policy['module_fallbacks'][module]
    )
    cluster_report = replay['module_eligibility']['correlation_cluster']
    promotion_criteria = {
        'applicable': True,
        'mean_weighted_return': bool(
            cross_metrics['mean_weighted_portfolio_return'] > 0.003115
        ),
        'worst_fold_weighted_return': bool(
            cross_metrics['worst_fold_weighted_portfolio_return'] > 0.001785
        ),
        'mean_top5_return': bool(
            cross_metrics['mean_top5_return'] > 0.006746
        ),
        'mean_rank_ic': bool(cross_metrics['mean_rank_ic'] >= 0.0514),
        'enabled_modules_pass_gates': bool(enabled_modules_valid),
        'cluster_selection_not_harmful': bool(
            not policy['cluster_cap_enabled']
            or cluster_report.get('mean_paired_contribution', 0.0) >= 0.0
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
                'worst_rank_ic',
                'mean_model_disagreement',
                'mean_allocation_contribution',
                'mean_exposure_contribution',
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
        'folds': source_summary.get('folds', []),
        'full_training': {
            'reused': True,
            'artifact_source_dir': artifact_source_dir,
            'source_full_training': source_summary.get('full_training', {}),
        },
    }
    for optional_key in (
        'mean_tail_5d_brier',
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
    if policy_only_enabled():
        return run_policy_only()
    if config.get('policy_only_experiment', False):
        raise ValueError(
            '当前配置是策略重放实验，不允许重新训练；'
            '请使用 POLICY_ONLY=1 ./train.sh'
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
        f"训练模式: {'多种子 ensemble' if ensemble_enabled else '单种子三折'}; "
        f"seeds={ensemble_seeds}; 设备: {device}; AMP={use_amp(device)}; "
        f"TF32={device.type == 'cuda' and config.get('tf32_enabled', True)}; "
        f"batch_size={config['batch_size']}"
    )

    data_file = os.path.join(config['data_path'], 'train.csv')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
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

    ensemble_days = []
    single_seed_days = {seed: [] for seed in ensemble_seeds}
    for fold in folds:
        fold_number = int(fold['fold'])
        ensemble_days.extend(align_oof_prediction_records(
            [
                oof_records[fold_number][seed]
                for seed in ensemble_seeds
            ],
            fold=fold_number,
        ))
        for seed in ensemble_seeds:
            single_seed_days[seed].extend(align_oof_prediction_records(
                [oof_records[fold_number][seed]],
                fold=fold_number,
            ))

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
        downside_weight=config.get('ensemble_downside_weight', 0.5),
        top_k=5,
    )
    policy = calibrate_ensemble_policy(
        ensemble_days,
        **policy_calibration_kwargs,
    )
    if config.get('nested_oof_enabled', False):
        cross_fitted_policy = cross_fit_ensemble_policy(
            ensemble_days,
            **policy_calibration_kwargs,
        )
        cross_fitted_policy['deployment_policy_differences'] = {
            field: [
                float(policy[field]) - float(
                    row['policy'][field]
                )
                for row in cross_fitted_policy['fold_policies']
            ]
            for field in (
                'allocation_blend',
                'disagreement_gamma',
                'selection_risk_gamma',
                'risk_score_penalty',
                'correlation_exposure_gamma',
                'exposure_head_blend',
            )
        }
        ensemble_metrics = cross_fitted_policy['metrics']
    else:
        cross_fitted_policy = {
            'method': 'disabled',
            'metrics': policy['oof_metrics'],
            'fold_policies': [],
            'policy_stability': {},
        }
        ensemble_metrics = policy['oof_metrics']
    single_seed_summaries = {}
    for seed in ensemble_seeds:
        single_seed_summaries[str(seed)] = evaluate_ensemble_policy(
            single_seed_days[seed],
            policy,
        )
    mean_single_return = float(np.mean([
        metrics['mean_weighted_portfolio_return']
        for metrics in single_seed_summaries.values()
    ]))
    identity_sensitivity = summarize_identity_sensitivity(fold_results)
    unk_sensitivity = identity_sensitivity.get('all_unk_vs_real', {})
    promotion_criteria = {
        'applicable': True,
        'mean_weighted_return': bool(
            ensemble_metrics['mean_weighted_portfolio_return']
            >= config.get('promotion_mean_weighted_return', 0.019902)
        ),
        'worst_fold_weighted_return': bool(
            ensemble_metrics['worst_fold_weighted_portfolio_return']
            >= config.get(
                'promotion_worst_fold_weighted_return',
                0.012523,
            )
        ),
        'p10_weighted_return': bool(
            ensemble_metrics['p10_weighted_portfolio_return']
            > config.get('promotion_p10_weighted_return', -0.025672)
        ),
        'mean_rank_ic': bool(
            ensemble_metrics['mean_rank_ic']
            >= config.get('promotion_mean_rank_ic', 0.0514)
        ),
        'allocation_at_exposure_positive': bool(
            ensemble_metrics[
                'mean_allocation_at_exposure_contribution'
            ] > 1e-6
        ),
        'selection_correlation_improved': bool(
            ensemble_metrics['mean_positive_correlation']
            < ensemble_metrics['raw_mean_positive_correlation']
        ),
        'exposure_objective_improved': bool(
            ensemble_metrics['policy_objective']
            > ensemble_metrics['fixed_exposure_policy_objective']
        ),
        'exposure_not_constant': bool(
            ensemble_metrics['exposure_std']
            >= config.get('promotion_min_exposure_std', 0.01)
        ),
        'regime_gate_not_constant': bool(
            ensemble_metrics['regime_gate_std']
            >= config.get('promotion_min_regime_gate_std', 0.01)
        ),
        'regime_gate_direction': bool(
            ensemble_metrics['regime_return_spearman'] < 0.0
        ),
        'cluster_cap_respected': bool(
            ensemble_metrics['max_selected_cluster_count']
            <= config.get('max_stocks_per_cluster', 2)
        ),
        'id_score_correlation': bool(
            unk_sensitivity.get('score_spearman', 0.0)
            >= config.get('promotion_id_score_correlation', 0.90)
        ),
        'id_top5_overlap': bool(
            unk_sensitivity.get('top5_overlap', 0.0)
            >= config.get('promotion_id_top5_overlap', 0.40)
        ),
    }
    promotion_criteria['passed'] = all(
        value for key, value in promotion_criteria.items()
        if key != 'applicable'
    )

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

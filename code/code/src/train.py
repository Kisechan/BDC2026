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
from utils import RELATIVE_MARKET_FEATURES, RELATIVE_MARKET_FEATURE_SET
from utils import create_ranking_dataset_vectorized
from utils import extract_selection_risk_context
from utils import align_oof_prediction_records, calibrate_ensemble_policy
from utils import summarize_ensemble_days
import joblib
import os
import json
import multiprocessing as mp
import random
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
assert len(feature_cloums_map['158+39_reduced20']) == 171
assert len(feature_cloums_map['158+39_reduced25']) == 166
assert len(feature_cloums_map[RELATIVE_MARKET_FEATURE_SET]) == 178


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
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
    if config['feature_num'] == RELATIVE_MARKET_FEATURE_SET:
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
                 id_gate_regularization=0.0):
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

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()
        
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        
        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()
        
        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)
        
        pairwise_loss = F.softplus(-pred_diff * torch.sign(true_diff))
        
        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix
        
        # 按有效权重和归一化，使 Top-k 权重只改变相对重要性而不改变整体尺度。
        weight_sum = (mask * weight_matrix).sum(dim=[1, 2]).clamp(min=1e-12)
        loss = (weighted_loss.sum(dim=[1, 2]) / weight_sum).mean()
        
        return loss

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
            
        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        rank_ic = self.rank_ic_loss(y_pred, y_true)
        regression = F.smooth_l1_loss(
            predicted_returns,
            raw_returns,
            beta=self.regression_beta,
        )
        allocation, exposure_loss = self.allocation_and_exposure_loss(
            y_pred,
            allocation_logits,
            exposure,
            raw_returns,
        )
        
        # 排序为主任务，原始收益回归为辅助任务。
        components = {
            'listwise_loss': self.listwise_weight * listwise,
            'pairwise_loss': self.pairwise_weight * pairwise,
            'ic_loss': self.ic_weight * rank_ic,
            'regression_loss': self.regression_weight * regression,
            'allocation_loss': self.allocation_weight * allocation,
            'exposure_loss': self.exposure_weight * exposure_loss,
        }
        if identity_gate is not None and self.id_gate_regularization > 0:
            components['id_gate_regularization'] = (
                self.id_gate_regularization * identity_gate.square()
            )
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
    ):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
        self.prediction_dates = prediction_dates
        lengths = {
            len(sequences),
            len(targets),
            len(relevance_scores),
            len(stock_indices),
            len(prediction_dates),
        }
        if len(lengths) != 1:
            raise ValueError('排序数据集各字段长度不一致')
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.from_numpy(
                np.asarray(self.sequences[idx], dtype=np.float32)
            ),
            'targets': torch.from_numpy(
                np.asarray(self.targets[idx], dtype=np.float32)
            ),
            'relevance': torch.from_numpy(
                np.asarray(self.relevance_scores[idx], dtype=np.int64)
            ),
            'stock_indices': torch.as_tensor(
                self.stock_indices[idx],
                dtype=torch.long,
            ),
            'prediction_date': self.prediction_dates[idx],
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    prediction_dates = [item['prediction_date'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
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
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks),                     # [batch, max_stocks]
        'prediction_dates': prediction_dates,
    }


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
):
    model.train()
    total_loss = 0
    total_loss_components = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = move_batch_tensor(batch['sequences'], device)
        targets = move_batch_tensor(batch['targets'], device)
        relevance = move_batch_tensor(batch['relevance'], device)
        stock_indices = move_batch_tensor(batch['stock_indices'], device)
        masks = move_batch_tensor(batch['masks'], device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Transformer 前向使用 AMP；排序、相关性与仓位损失保持 FP32。
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp(device),
        ):
            outputs, return_outputs, allocation_outputs, exposures = model(
                sequences,
                stock_indices,
                masks,
            )
        outputs = outputs.float()
        return_outputs = return_outputs.float()
        allocation_outputs = allocation_outputs.float()
        exposures = exposures.float()
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_return_outputs = return_outputs * masks
        masked_allocation_outputs = allocation_outputs * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_loss_components = {}
        batch_size = sequences.size(0)
        
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
                    return_components=True,
                )
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
                for name, value in loss_components.items():
                    batch_loss_components[name] = batch_loss_components.get(name, 0.0) + value
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss_components = {
                name: value / batch_size for name, value in batch_loss_components.items()
            }
            grad_scaler.scale(batch_loss).backward()
            if config.get('grad_clip', True):
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            
            total_loss += batch_loss.item()
            for name, value in batch_loss_components.items():
                total_loss_components[name] = total_loss_components.get(name, 0.0) + value.item()
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for name, value in batch_loss_components.items():
                    writer.add_scalar(
                        f'train/{name}',
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
):
    model.eval()
    total_loss = 0
    num_batches = 0
    metric_records = []
    prediction_records = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = move_batch_tensor(batch['sequences'], device)
            targets = move_batch_tensor(batch['targets'], device)
            stock_indices = move_batch_tensor(batch['stock_indices'], device)
            masks = move_batch_tensor(batch['masks'], device)
            
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp(device),
            ):
                outputs, return_outputs, allocation_outputs, exposures = model(
                    sequences,
                    stock_indices,
                    masks,
                )
            outputs = outputs.float()
            return_outputs = return_outputs.float()
            allocation_outputs = allocation_outputs.float()
            exposures = exposures.float()
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            masked_return_outputs = return_outputs * masks
            masked_allocation_outputs = allocation_outputs * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
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
                    }
                    if (
                        selection_risk_feature_names is not None
                        and selection_risk_scaler is not None
                    ):
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
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
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

def build_training_components(model):
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
        min_exposure=config.get('min_exposure', 0.80),
        max_exposure=config.get('max_exposure', 0.999999),
    )
    regular_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
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
        'lr': config['learning_rate'],
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
    return metrics.get(checkpoint_metric, 0.0)


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

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks).to(device)
    criterion, optimizer = build_training_components(model)
    grad_scaler = create_grad_scaler(device)
    max_epochs = config['max_epochs']
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=max_epochs,
    )

    checkpoint_metric = config.get('checkpoint_metric', 'top5_return')
    best_score = -float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    epochs_ran = 0
    checkpoint_path = os.path.join(fold_dir, 'best_model.pth')
    print(
        f"\n========== Seed {base_seed} "
        f"Fold {fold_number}/{config['num_folds']} =========="
    )
    print(
        f"边界: {fold}; 训练样本: {len(train_dataset)}; "
        f"验证样本: {len(val_dataset)}; "
        f"非重叠验证日期: {len(val_eval_dataset)}"
    )

    for epoch in range(max_epochs):
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
        )
        eval_loss, eval_metrics = evaluate_ranking_model(
            model, val_loader, criterion, device, writer, epoch
        )
        scheduler.step()
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
        current_score = calculate_checkpoint_score(eval_metrics, checkpoint_metric)
        print(
            f"Fold {fold_number} Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}, "
            f"val_top5={eval_metrics.get('top5_return', 0.0):.6f}, "
            f"val_rank_ic={eval_metrics.get('rank_ic', 0.0):.4f}, "
            f"val_weighted_return={eval_metrics.get('weighted_portfolio_return', 0.0):.6f}, "
            f"val_exposure={eval_metrics.get('gross_exposure', 0.0):.4f}, "
            f"id_gate={model.identity_gate_value().detach().item():.4f}, "
            f"checkpoint_score={current_score:.6f}"
        )

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config['patience']:
                print(
                    f"Fold {fold_number} early stopping: "
                    f"{config['patience']} epochs without {checkpoint_metric} improvement"
                )
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    _, train_eval_metrics = evaluate_ranking_model(
        model, train_eval_loader, criterion, device, None, best_epoch - 1
    )
    _, val_eval_metrics, oof_predictions = evaluate_ranking_model(
        model,
        val_loader,
        criterion,
        device,
        None,
        best_epoch - 1,
        return_predictions=True,
        selection_risk_feature_names=features,
        selection_risk_scaler=scaler,
    )
    id_sensitivity = evaluate_identity_sensitivity(
        model,
        val_loader,
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
        'epochs_ran': epochs_ran,
        'checkpoint_metric': checkpoint_metric,
        'checkpoint_score': float(best_score),
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
    num_epochs,
    median_best_epoch,
    base_seed,
):
    """按统一 CV epoch 用一个随机种子完成全量重训。"""
    set_seed(base_seed + 1000)
    final_dir = os.path.join(output_dir, 'full_train')
    os.makedirs(final_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(final_dir, 'log'))
    train_loader = build_data_loader(train_dataset, True, device)
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks).to(device)
    criterion, optimizer = build_training_components(model)
    grad_scaler = create_grad_scaler(device)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=num_epochs,
    )

    print(
        f"\n========== Seed {base_seed} full-data retraining: "
        f"{num_epochs} epochs =========="
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
        )
        scheduler.step()
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
        print(
            f"Full train Epoch {epoch + 1}/{num_epochs}: "
            f"loss={train_loss:.4f}, "
            f"listwise={train_metrics.get('listwise_loss', 0.0):.4f}, "
            f"pairwise={train_metrics.get('pairwise_loss', 0.0):.4f}, "
            f"ic_loss={train_metrics.get('ic_loss', 0.0):.4f}, "
            f"regression={train_metrics.get('regression_loss', 0.0):.4f}, "
            f"allocation={train_metrics.get('allocation_loss', 0.0):.4f}, "
            f"exposure_loss={train_metrics.get('exposure_loss', 0.0):.4f}, "
            f"id_gate={model.identity_gate_value().detach().item():.4f}"
        )

    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    metadata = {
        'base_seed': int(base_seed),
        'epochs': num_epochs,
        'epoch_selection': 'median_across_all_fold_checkpoints',
        'median_best_epoch': median_best_epoch,
        'min_final_epochs': config['min_final_epochs'],
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


def main():
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
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as file:
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
        expected_market_indices = [
            features.index(name)
            for name in RELATIVE_MARKET_FEATURES[-5:]
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

    fold_results = []
    oof_records = {
        int(fold['fold']): {seed: [] for seed in ensemble_seeds}
        for fold in folds
    }
    for base_seed in ensemble_seeds:
        seed_dir = os.path.join(output_dir, f'seed_{base_seed}')
        os.makedirs(seed_dir, exist_ok=True)
        for fold in folds:
            result, predictions = train_one_fold(
                full_data=full_data,
                features=features,
                fold=fold,
                num_stocks=len(stockid2idx),
                device=device,
                output_dir=seed_dir,
                base_seed=base_seed,
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

    policy = calibrate_ensemble_policy(
        ensemble_days,
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
        selection_candidate_k=int(config.get(
            'selection_candidate_k',
            20,
        )),
        fixed_exposure_baseline=float(config.get(
            'fixed_exposure_baseline',
            0.6231689453125,
        )),
        downside_weight=config.get('ensemble_downside_weight', 0.5),
        top_k=5,
    )
    single_seed_summaries = {}
    for seed in ensemble_seeds:
        single_seed_summaries[str(seed)] = summarize_ensemble_days(
            single_seed_days[seed],
            min_exposure=config['min_exposure'],
            max_exposure=config['max_exposure'],
            allocation_temperature=config.get(
                'allocation_temperature',
                1.0,
            ),
            allocation_blend=1.0,
            disagreement_gamma=0.0,
            selection_risk_gamma=0.0,
            selection_candidate_k=int(config.get(
                'selection_candidate_k',
                20,
            )),
            fixed_exposure_baseline=float(config.get(
                'fixed_exposure_baseline',
                0.6231689453125,
            )),
            downside_weight=config.get('ensemble_downside_weight', 0.5),
            top_k=5,
        )
    mean_single_return = float(np.mean([
        metrics['mean_weighted_portfolio_return']
        for metrics in single_seed_summaries.values()
    ]))
    ensemble_metrics = policy['oof_metrics']
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
            ] > 0.0
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

    # 所有已运行折 checkpoint 统一决定全量模型训练轮数。
    median_best_epoch = int(np.median([result['best_epoch'] for result in fold_results]))
    min_final_epochs = config.get('min_final_epochs', 1)
    if not 1 <= min_final_epochs <= config['max_epochs']:
        raise ValueError('min_final_epochs 必须位于 [1, max_epochs] 范围内')
    final_epochs = min(
        config['max_epochs'],
        max(min_final_epochs, median_best_epoch),
    )
    full_train_dataset, full_train_end = prepare_full_training_dataset(
        full_data=full_data,
        features=features,
        output_dir=output_dir,
    )
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
            num_epochs=final_epochs,
            median_best_epoch=median_best_epoch,
            base_seed=base_seed,
        ))
        model_paths.append(
            os.path.join(f'seed_{base_seed}', 'best_model.pth')
        )

    policy.update({
        'ensemble_enabled': ensemble_enabled,
        'mode': 'rank_ensemble' if ensemble_enabled else 'single_model',
        'ensemble_seeds': ensemble_seeds,
        'model_paths': model_paths,
        'scaler_path': 'scaler.pkl',
        'config_path': 'config.json',
        'selection_risk_lookback': int(config.get(
            'selection_risk_lookback',
            20,
        )),
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
        'mean_cash_weight': ensemble_metrics['mean_cash_weight'],
        'mean_rank_ic': ensemble_metrics['mean_rank_ic'],
        'worst_rank_ic': ensemble_metrics['worst_rank_ic'],
        'mean_model_disagreement': ensemble_metrics[
            'mean_model_disagreement'
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
        'single_seed_oof': single_seed_summaries,
        'single_seed_mean_weighted_return': mean_single_return,
        'promotion_criteria': promotion_criteria,
        'folds': fold_results,
        'full_training': {
            'epochs': final_epochs,
            'epoch_selection': 'median_across_all_fold_checkpoints',
            'num_fold_checkpoints': len(fold_results),
            'median_best_epoch': median_best_epoch,
            'min_final_epochs': min_final_epochs,
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
    print(
        f"\n########## 训练完成！OOF 平均组合收益: "
        f"{best_score:.6f} ##########"
    )

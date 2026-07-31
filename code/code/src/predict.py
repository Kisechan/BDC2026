import json
import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr

from config import config
from model import StockTransformer
from utils import add_relative_market_features
from utils import add_industry_residual_features
from utils import build_ensemble_portfolio
from utils import engineer_features_39, engineer_features_158plus39
from utils import extract_selection_risk_context
from utils import (
	MARKET_PRESSURE_FEATURES,
	RELATIVE_MARKET_FEATURES,
	RELATIVE_MARKET_FEATURE_SET,
	RISK_MARKET_FEATURES,
	RISK_MARKET_FEATURE_SET,
)


feature_cloums_map = {
	'39': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39_reduced20': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60',
		'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39_reduced20': engineer_features_158plus39,
}

# 与训练阶段保持完全相同的 166 维消融配置；名称保留 158+39 特征族血缘。
LINEAR_REDUNDANT_FEATURES = {
	'high_low_spread', 'open_close_spread', 'high_close_spread',
	'low_close_spread', 'kdj_j',
}
INDUSTRY_RESIDUAL_FEATURE_SET = (
	'158+39_reduced25_relmarket12_risk15_indresid12'
)
INDUSTRY_RESIDUAL_FEATURES = (
	'indresid_market_return_1',
	'indresid_market_return_3',
	'indresid_market_return_5',
	'indresid_industry_return_1',
	'indresid_industry_return_3',
	'indresid_industry_return_5',
	'indresid_market_return_1_pct',
	'indresid_market_return_3_pct',
	'indresid_market_return_5_pct',
	'indresid_industry_return_1_pct',
	'indresid_industry_return_3_pct',
	'indresid_industry_return_5_pct',
)
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
feature_engineer_func_map[INDUSTRY_RESIDUAL_FEATURE_SET] = engineer_features_158plus39
assert len(feature_cloums_map['158+39_reduced20']) == 171
assert len(feature_cloums_map['158+39_reduced25']) == 166
assert len(feature_cloums_map[RELATIVE_MARKET_FEATURE_SET]) == 178
assert len(feature_cloums_map[RISK_MARKET_FEATURE_SET]) == 193
assert len(feature_cloums_map[INDUSTRY_RESIDUAL_FEATURE_SET]) == 205


def validate_artifact_compatibility(
	trained_config,
	policy,
	artifact_source_dir,
	feature_columns,
	stockid2idx,
	scaler,
	model_paths,
):
	"""以同一份 manifest、scaler 和 checkpoint 防止 193/205 维工件混用。"""
	feature_count = len(feature_columns)
	manifest_path = policy.get('manifest_path', 'artifact_manifest.json')
	if not os.path.isabs(manifest_path):
		manifest_path = os.path.join(artifact_source_dir, manifest_path)
	require_manifest = bool(
		feature_count == 205
		or policy.get('require_artifact_manifest', False)
		or trained_config.get('require_artifact_manifest', False)
	)
	manifest = None
	if os.path.exists(manifest_path):
		with open(manifest_path, 'r', encoding='utf-8') as handle:
			manifest = json.load(handle)
		if not isinstance(manifest, dict):
			raise ValueError(f'artifact_manifest.json 必须是 JSON 对象: {manifest_path}')
	elif require_manifest:
		raise FileNotFoundError(
			f'205维工件必须包含 artifact_manifest.json: {manifest_path}'
		)

	if manifest is not None:
		for key, expected in {
			'feature_num': trained_config['feature_num'],
			'feature_count': feature_count,
			'sequence_length': trained_config['sequence_length'],
			'stock_mapping_size': len(stockid2idx),
		}.items():
			if key not in manifest:
				raise ValueError(f'artifact_manifest.json 缺少 {key}: {manifest_path}')
			if manifest[key] != expected:
				raise ValueError(
					f'artifact_manifest.json 的 {key} 与推理工件不一致: '
					f'{manifest[key]} != {expected}'
				)
		architecture = manifest.get('architecture', manifest)
		for key in ('d_model', 'nhead', 'num_layers', 'dim_feedforward'):
			if key in architecture and architecture[key] != trained_config[key]:
				raise ValueError(
					f'artifact_manifest.json 的 {key} 与训练配置不一致: '
					f'{architecture[key]} != {trained_config[key]}'
				)
		if trained_config.get('score_head_variant', 'mlp_v1') != 'mlp_v1':
			for key in ('score_head_variant', 'rankglu_bottleneck', 'rankglu_gamma_max'):
				if manifest.get(key) != trained_config.get(key):
					raise ValueError(
						f'artifact_manifest.json 的 {key} 与 RankGLU 训练配置不一致'
					)

	scaler_feature_count = getattr(scaler, 'n_features_in_', len(scaler.mean_))
	if int(scaler_feature_count) != feature_count:
		raise ValueError(
			'Scaler 与当前特征表维度不一致，禁止混用工件: '
			f'{scaler_feature_count} != {feature_count}'
		)
	checkpoint_dimensions = []
	for model_path in model_paths:
		state_dict = torch.load(
			model_path,
			map_location='cpu',
			weights_only=True,
		)
		if 'model_state_dict' in state_dict:
			state_dict = state_dict['model_state_dict']
		input_projection = state_dict.get('input_proj.weight')
		stock_embedding = state_dict.get('stock_embedding.weight')
		if input_projection is None or input_projection.ndim != 2:
			raise ValueError(f'模型缺少 input_proj.weight: {model_path}')
		if int(input_projection.shape[1]) != feature_count:
			raise ValueError(
				'模型输入维度与当前特征表不一致，禁止混用工件: '
				f'{input_projection.shape[1]} != {feature_count} ({model_path})'
			)
		if (
			stock_embedding is None
			or stock_embedding.ndim != 2
			or int(stock_embedding.shape[0]) != len(stockid2idx) + 2
		):
			raise ValueError(
				f'股票映射与模型 embedding 不兼容: {model_path}'
			)
		checkpoint_dimensions.append(int(input_projection.shape[1]))
	return {
		'status': 'validated' if manifest is not None else 'legacy_no_manifest',
		'manifest_path': manifest_path if manifest is not None else None,
		'feature_num': trained_config['feature_num'],
		'feature_count': feature_count,
		'scaler_feature_count': int(scaler_feature_count),
		'stock_mapping_size': len(stockid2idx),
		'checkpoint_input_dimensions': checkpoint_dimensions,
	}


def cross_sectional_percentiles(scores):
	"""将单日分数标准化为 [0, 1] 百分位，供异构排序器融合。"""
	scores = np.asarray(scores, dtype=np.float64)
	if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
		raise ValueError('排序融合分数必须是非空且有限的一维数组')
	if scores.size == 1:
		return np.ones(1, dtype=np.float64)
	return (
		pd.Series(scores).rank(method='average').to_numpy(dtype=np.float64) - 1.0
	) / (scores.size - 1.0)


def resolve_lgbm_ranker(policy, artifact_source_dir):
	"""解析可选 LightGBM 排序器；零权重时不要求本地工件。"""
	weight = float(policy.get('lgbm_weight', 0.0))
	if not 0.0 <= weight <= 1.0:
		raise ValueError('lgbm_weight 必须位于 [0, 1]')
	configured_path = policy.get('lgbm_model_path')
	if weight == 0.0:
		return None, weight, None
	if not configured_path:
		raise ValueError('lgbm_weight 大于 0 时必须设置 lgbm_model_path')
	model_path = configured_path
	if not os.path.isabs(model_path):
		model_path = os.path.join(artifact_source_dir, model_path)
	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到 LightGBM 排序器: {model_path}')
	return joblib.load(model_path), weight, model_path


def validate_lgbm_feature_count(rank_model, feature_count):
	"""LightGBM 同样必须使用当前完整特征表，避免 193/205 维混接。"""
	model_feature_count = getattr(rank_model, 'n_features_in_', None)
	if model_feature_count is None and hasattr(rank_model, 'booster_'):
		model_feature_count = rank_model.booster_.num_feature()
	if model_feature_count is not None and int(model_feature_count) != feature_count:
		raise ValueError(
			'LightGBM 输入维度与当前特征表不一致，禁止混用工件: '
			f'{model_feature_count} != {feature_count}'
		)
	return int(model_feature_count) if model_feature_count is not None else None


def preprocess_predict_data(df, stockid2idx, runtime_config=None):
	runtime_config = runtime_config or config
	feature_num = runtime_config['feature_num']
	assert feature_num in feature_engineer_func_map, f"Unsupported feature_num: {feature_num}"
	feature_engineer = feature_engineer_func_map[feature_num]
	feature_columns = feature_cloums_map[feature_num]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	if feature_num in {
		RELATIVE_MARKET_FEATURE_SET,
		RISK_MARKET_FEATURE_SET,
		INDUSTRY_RESIDUAL_FEATURE_SET,
	}:
		processed = add_relative_market_features(processed)
	if feature_num == INDUSTRY_RESIDUAL_FEATURE_SET:
		processed = add_industry_residual_features(
			processed,
			runtime_config['industry_history_path'],
		)
	processed['instrument'] = processed['股票代码'].map(stockid2idx).fillna(1)
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date, stockid2idx):
	sequences, sequence_stock_ids, sequence_stock_indices = [], [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_indices.append(stockid2idx.get(stock_id, 1))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids, sequence_stock_indices


def build_bounded_positions(
	scores,
	allocation_logits,
	exposure,
	top_k=5,
	runtime_config=None,
):
	"""按 ranking score 选股，并将相对权重严格缩放到 Exposure Head 总仓位。"""
	runtime_config = runtime_config or config
	min_exposure = float(runtime_config.get('min_exposure', 0.80))
	max_exposure = float(runtime_config.get('max_exposure', 0.999999))
	temperature = float(runtime_config.get('allocation_temperature', 1.0))
	if not 0.0 <= min_exposure < max_exposure < 1.0:
		raise ValueError('仓位范围必须满足 0 <= min_exposure < max_exposure < 1')
	if temperature <= 0:
		raise ValueError('allocation_temperature 必须大于 0')

	scores = np.asarray(scores, dtype=np.float64)
	allocation_logits = np.asarray(allocation_logits, dtype=np.float64)
	if scores.ndim != 1 or allocation_logits.shape != scores.shape:
		raise ValueError('ranking score 与 allocation logits 必须是一维同形数组')
	if len(scores) < top_k:
		raise ValueError(f'可预测股票不足{top_k}只，当前仅有 {len(scores)} 只')
	if not np.isfinite(scores).all() or not np.isfinite(allocation_logits).all():
		raise ValueError('模型输出包含 NaN 或无穷值')
	if not np.isfinite(exposure):
		raise ValueError('Exposure Head 输出包含 NaN 或无穷值')

	serialization_margin = 1e-10
	safe_min_exposure = min_exposure + serialization_margin
	safe_max_exposure = max_exposure - serialization_margin
	if safe_min_exposure >= safe_max_exposure:
		raise ValueError('仓位上下界过窄，无法保留序列化安全边际')
	bounded_exposure = float(
		np.clip(exposure, safe_min_exposure, safe_max_exposure)
	)
	top_indices = np.argsort(scores)[::-1][:top_k]
	selected_logits = allocation_logits[top_indices] / temperature
	selected_logits = selected_logits - selected_logits.max()
	relative_weights = np.exp(selected_logits)
	relative_weights /= relative_weights.sum(dtype=np.float64)
	positions = relative_weights * bounded_exposure

	# 消除浮点累计误差，使写盘前的股票仓位和精确等于 bounded_exposure。
	positions[np.argmax(positions)] += bounded_exposure - positions.sum(dtype=np.float64)
	if (positions < 0).any() or positions.sum(dtype=np.float64) > 1.0:
		raise ValueError('生成的股票仓位违反非负或总仓位不超过1的约束')
	return top_indices, positions, bounded_exposure


def validate_result(output_df, candidate_stock_ids, runtime_config=None):
	"""在写盘前后使用赛事约束校验预测结果。"""
	runtime_config = runtime_config or config
	required_columns = ['stock_id', 'weight']
	if output_df.columns.tolist() != required_columns:
		raise ValueError(f'结果列必须严格为 {required_columns}')
	if not 1 <= len(output_df) <= 5:
		raise ValueError(f'结果必须包含1至5只股票，当前为 {len(output_df)} 只')
	if output_df['stock_id'].duplicated().any():
		raise ValueError('结果包含重复股票代码')

	stock_ids = output_df['stock_id'].astype(str).str.zfill(6)
	unknown_ids = sorted(set(stock_ids) - set(candidate_stock_ids))
	if unknown_ids:
		raise ValueError(f'结果包含候选范围外股票: {unknown_ids}')

	weights = pd.to_numeric(output_df['weight'], errors='coerce').to_numpy(dtype=np.float64)
	if not np.isfinite(weights).all():
		raise ValueError('结果权重包含 NaN 或无穷值')
	if (weights < 0).any() or (weights > 1).any():
		raise ValueError('每只股票的权重必须在 [0, 1] 范围内')
	weight_sum = float(weights.sum(dtype=np.float64))
	if weight_sum > 1.0:
		raise ValueError(f'结果权重和不能超过1，当前为 {weight_sum:.12f}')
	min_exposure = float(runtime_config.get('min_exposure', 0.80))
	max_exposure = float(runtime_config.get('max_exposure', 0.999999))
	if weight_sum < min_exposure or weight_sum > max_exposure:
		raise ValueError(
			f'结果总仓位必须位于 [{min_exposure}, {max_exposure}]，'
			f'当前为 {weight_sum:.12f}'
		)

	return weight_sum


def make_serialization_safe_positions(portfolio, runtime_config):
	"""把 ensemble 仓位移到合法区间内部，避免 CSV 浮点往返触碰边界。"""
	min_exposure = float(runtime_config['min_exposure'])
	max_exposure = float(runtime_config['max_exposure'])
	margin = 1e-10
	safe_exposure = float(np.clip(
		portfolio['exposure'],
		min_exposure + margin,
		max_exposure - margin,
	))
	positions = np.asarray(portfolio['positions'], dtype=np.float64).copy()
	positions *= safe_exposure / positions.sum(dtype=np.float64)
	positions[np.argmax(positions)] += (
		safe_exposure - positions.sum(dtype=np.float64)
	)
	if (positions < 0).any() or positions.sum(dtype=np.float64) > 1.0:
		raise ValueError('序列化安全调整后仓位不合法')
	return positions, safe_exposure


def build_industry_concentration(stock_ids, weights, latest_date, history_path):
	"""按预测日期 as-of 连接行业快照；行业文件缺失不影响旧工件推理。"""
	if not history_path or not os.path.exists(history_path):
		return {'available': False, 'reason': f'未找到行业历史: {history_path}'}
	try:
		history = pd.read_csv(history_path, dtype={'stock_id': str})
		required = {'effective_date', 'stock_id', 'industry'}
		if not required.issubset(history.columns):
			return {
				'available': False,
				'reason': f'行业历史缺少字段: {sorted(required - set(history.columns))}',
			}
		history['effective_date'] = pd.to_datetime(history['effective_date'])
		history['stock_id'] = history['stock_id'].astype(str).str.zfill(6)
		history = history.loc[history['effective_date'] <= pd.Timestamp(latest_date)]
		history = history.sort_values(['stock_id', 'effective_date'])
		industry_by_stock = history.groupby('stock_id', sort=False)['industry'].last()
	except (OSError, ValueError, pd.errors.ParserError) as error:
		return {'available': False, 'reason': f'行业历史读取失败: {error}'}

	selected = pd.DataFrame({
		'stock_id': pd.Series(stock_ids, dtype=str).str.zfill(6),
		'weight': np.asarray(weights, dtype=np.float64),
	})
	selected['industry'] = selected['stock_id'].map(industry_by_stock).fillna('UNCLASSIFIED')
	by_industry = selected.groupby('industry', sort=True).agg(
		weight=('weight', 'sum'),
		stock_count=('stock_id', 'size'),
		stock_ids=('stock_id', list),
	).reset_index().sort_values(['weight', 'industry'], ascending=[False, True])
	industry_weights = by_industry['weight'].to_numpy(dtype=np.float64)
	return {
		'available': True,
		'as_of_date': pd.Timestamp(latest_date).strftime('%Y-%m-%d'),
		'history_path': history_path,
		'industry_count': int(len(by_industry)),
		'covered_stocks': int(selected['industry'].ne('UNCLASSIFIED').sum()),
		'unclassified_stocks': int(selected['industry'].eq('UNCLASSIFIED').sum()),
		'hhi': float(np.square(industry_weights).sum()),
		'max_industry_weight': float(industry_weights.max()),
		'weights': by_industry.to_dict(orient='records'),
	}


def build_prediction_day_diagnostics(
	top5,
	weights,
	exposure,
	portfolio,
	latest_date,
	trained_config,
	model_regime_gates,
	combined_risk,
):
	"""为报告提供提交日的状态、资金约束与风险字段。"""
	weights = np.asarray(weights, dtype=np.float64)
	return {
		'date': pd.Timestamp(latest_date).strftime('%Y-%m-%d'),
		'stock_ids': list(top5),
		'portfolio_state': {
			'raw_top5_changed': bool(np.any(
				portfolio['top_indices'] != portfolio['raw_top_indices']
			)),
			'cluster_constraint_applied': bool(
				portfolio['cluster_constraint_applied']
			),
			'head_base_exposure': float(portfolio['head_base_exposure']),
			'base_exposure': float(portfolio['base_exposure']),
		},
		'funding_constraints': {
			'gross_exposure': float(exposure),
			'cash_weight': float(1.0 - exposure),
			'max_position': float(weights.max()),
			'weights_sum': float(weights.sum(dtype=np.float64)),
			'within_exposure_bounds': bool(
				float(trained_config['min_exposure']) - 1e-9 <= exposure
				<= float(trained_config['max_exposure']) + 1e-9
			),
			'non_negative': bool((weights >= 0).all()),
		},
		'risk_state': {
			'regime_gate': float(np.median(model_regime_gates)),
			'combined_risk_mean': float(np.mean(combined_risk)),
			'combined_risk_max': float(np.max(combined_risk)),
			'mean_positive_correlation': float(
				portfolio['mean_positive_correlation']
			),
		},
		'industry_concentration': build_industry_concentration(
			top5,
			weights,
			latest_date,
			trained_config.get('industry_history_path'),
		),
	}


def compare_runtime_configs(saved_config, live_config):
	"""报告源码配置与训练快照漂移；推理始终以训练快照为准。"""
	ignored_keys = {
		'output_dir',
		'policy_only_experiment',
		'policy_only_source_dir',
		'policy_output_dir',
		'artifact_source_dir',
		'source_training_config',
		'policy_simplicity_tolerance',
		'module_min_positive_fold_fraction',
		'cluster_max_raw_rank',
		'cluster_cap_grid',
		'minimum_allocation_deployment_blend',
		'minimum_exposure_deployment_blend',
		'cluster_cap_enabled',
	}
	return {
		key: {
			'trained': saved_config.get(key),
			'live': live_config.get(key),
		}
		for key in sorted(set(saved_config) | set(live_config))
		if key not in ignored_keys
		and saved_config.get(key) != live_config.get(key)
	}


def main():
	# 锁箱回放通过环境变量固定 as-of 日期与临时输出，默认行为仍是当前目录最新日推理。
	output_dir = os.environ.get('MODEL_OUTPUT_DIR', config['output_dir'])
	prediction_output_dir = os.environ.get('PREDICTION_OUTPUT_DIR', './output')
	strategy_config_path = os.path.join(output_dir, 'config.json')
	policy_path = os.path.join(output_dir, 'ensemble_policy.json')
	output_path = os.path.join(prediction_output_dir, 'result.csv')
	diagnostics_path = os.path.join(prediction_output_dir, 'prediction_diagnostics.json')
	os.makedirs(prediction_output_dir, exist_ok=True)

	for path, description in [
		(strategy_config_path, '策略配置快照'),
		(policy_path, 'ensemble 策略'),
	]:
		if not os.path.exists(path):
			raise FileNotFoundError(f'未找到{description}: {path}')
	with open(strategy_config_path, 'r', encoding='utf-8') as f:
		strategy_config = json.load(f)
	with open(policy_path, 'r', encoding='utf-8') as f:
		policy = json.load(f)
	artifact_source = policy.get('artifact_source_dir', '.')
	artifact_source_dir = (
		artifact_source
		if os.path.isabs(artifact_source)
		else os.path.normpath(os.path.join(output_dir, artifact_source))
	)
	trained_config_path = os.path.join(
		artifact_source_dir,
		policy.get('config_path', 'config.json'),
	)
	stock_mapping_path = os.path.join(
		artifact_source_dir,
		'stockid2idx.json',
	)
	for path, description in [
		(trained_config_path, '训练配置快照'),
		(stock_mapping_path, '股票 ID 映射'),
	]:
		if not os.path.exists(path):
			raise FileNotFoundError(f'未找到{description}: {path}')
	with open(trained_config_path, 'r', encoding='utf-8') as f:
		trained_config = json.load(f)
	for key in ('min_exposure', 'max_exposure'):
		if not np.isclose(
			float(policy[key]),
			float(trained_config[key]),
			rtol=0.0,
			atol=1e-12,
		):
			raise ValueError(
				f'ensemble policy 的 {key} 与训练配置不一致'
			)
	config_drift = compare_runtime_configs(trained_config, config)
	if config_drift:
		print('检测到源码配置与训练快照差异，推理采用训练快照:')
		print(json.dumps(config_drift, indent=2, ensure_ascii=False))
	promotion = policy.get('promotion_criteria', {})
	if promotion.get('applicable', True) and not promotion.get('passed', True):
		print('警告: OOF ensemble 未满足全部 promotion criteria，请结合报告判断是否提交。')

	scaler_path = os.path.join(
		artifact_source_dir,
		policy.get('scaler_path', 'scaler.pkl'),
	)
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到 Scaler: {scaler_path}')
	model_paths = [
		os.path.join(artifact_source_dir, relative_path)
		for relative_path in policy.get('model_paths', [])
	]
	ensemble_enabled = bool(policy.get(
		'ensemble_enabled',
		len(model_paths) > 1,
	))
	if not model_paths:
		raise ValueError('ensemble_policy.json 未声明全量模型')
	if ensemble_enabled and len(model_paths) < 2:
		raise ValueError('ensemble 模式至少需要两个全量模型')
	if not ensemble_enabled and len(model_paths) != 1:
		raise ValueError('单模型模式必须且只能声明一个全量模型')
	model_seeds = policy.get('ensemble_seeds', [])
	if len(model_seeds) != len(model_paths):
		raise ValueError('训练 seed 与模型路径数量不一致')
	for model_path in model_paths:
		if not os.path.exists(model_path):
			raise FileNotFoundError(f'未找到 ensemble 模型: {model_path}')

	data_file = os.path.join(trained_config['data_path'], 'train.csv')
	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	requested_date = os.environ.get('PREDICTION_DATE')
	if requested_date:
		as_of_date = pd.Timestamp(requested_date)
		raw_df = raw_df.loc[raw_df['日期'] <= as_of_date].copy()
		if raw_df.empty or raw_df['日期'].max() != as_of_date:
			raise ValueError(
				f'PREDICTION_DATE 必须是可用交易日: {requested_date}'
			)
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	with open(stock_mapping_path, 'r', encoding='utf-8') as f:
		stockid2idx = json.load(f)
	if trained_config['feature_num'] not in feature_cloums_map:
		raise ValueError(
			'当前推理源码不支持训练工件的 feature_num: '
			f'{trained_config["feature_num"]}'
		)
	scaler = joblib.load(scaler_path)
	artifact_validation = validate_artifact_compatibility(
		trained_config,
		policy,
		artifact_source_dir,
		feature_cloums_map[trained_config['feature_num']],
		stockid2idx,
		scaler,
		model_paths,
	)
	print(
		'工件兼容性校验通过: '
		f'{artifact_validation["feature_num"]} / '
		f'{artifact_validation["feature_count"]}维 / '
		f'{artifact_validation["status"]}'
	)

	processed, features = preprocess_predict_data(
		raw_df,
		stockid2idx,
		runtime_config=trained_config,
	)
	if trained_config.get('regime_gate_enabled', False):
		expected_market_indices = [
			features.index(name)
			for name in [
				*RELATIVE_MARKET_FEATURES[-5:],
				*MARKET_PRESSURE_FEATURES,
			]
		]
		for config_key in (
			'market_state_feature_indices',
			'regime_market_feature_indices',
		):
			configured_indices = [
				int(index)
				for index in trained_config.get(config_key, [])
			]
			if configured_indices != expected_market_indices:
				raise ValueError(
					f'{config_key} 与训练特征位置不一致: '
					f'{configured_indices} != {expected_market_indices}'
				)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
	# LightGBM 使用未经 StandardScaler 变换的预测日表格输入；仅保留最后
	# 时点，避免复制整张五年特征表或把 Transformer 专用缩放误喂给树模型。
	lgbm_raw_latest_by_stock = processed.sort_values(
		['股票代码', '日期'],
	).groupby('股票代码', sort=False).tail(1).set_index('股票代码')[features]
	processed[features] = scaler.transform(processed[features])

	sequence_length = trained_config['sequence_length']
	sequences_np, sequence_stock_ids, sequence_stock_indices = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
		stockid2idx,
	)
	lgbm_raw_latest = lgbm_raw_latest_by_stock.reindex(
		sequence_stock_ids,
	)[features].to_numpy(dtype=np.float32)
	if not np.isfinite(lgbm_raw_latest).all():
		raise ValueError('LightGBM 预测日特征与可用序列股票集合不一致')
	selection_risk_context = None
	if (
		'selection_risk_gamma' in policy
		or policy.get('cluster_cap_enabled', False)
	):
		selection_risk_context = extract_selection_risk_context(
			sequences_np,
			features,
			scaler,
			lookback=int(policy.get(
				'selection_risk_lookback',
				trained_config.get('selection_risk_lookback', 20),
			)),
		)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
	stock_index_tensor = torch.LongTensor(
		sequence_stock_indices
	).unsqueeze(0).to(device)
	stock_mask = torch.ones_like(stock_index_tensor, dtype=torch.float32)
	model_scores = []
	model_allocations = []
	model_exposures = []
	model_diagnostics = []
	model_regime_gates = []
	model_risk_1d = []
	model_risk_3d = []
	model_risk_5d = []
	model_tail_5d = []
	base_identity_seed = int(trained_config.get(
		'identity_sensitivity_seed',
		20260728,
	))
	with torch.inference_mode():
		for model_idx, model_path in enumerate(model_paths):
			model = StockTransformer(
				input_dim=len(features),
				config=trained_config,
				num_stocks=len(stockid2idx),
			)
			model.load_state_dict(
				torch.load(model_path, map_location=device, weights_only=True)
			)
			model.to(device)
			model.eval()
			def run_model(indices, return_aux=False):
				with torch.autocast(
					device_type=device.type,
					dtype=torch.float16,
					enabled=(
						device.type == 'cuda'
						and trained_config.get('amp_enabled', True)
					),
				):
					return model(
						x,
						indices,
						stock_mask,
						return_aux=return_aux,
					)

			(
				score_output,
				_,
				allocation_output,
				exposure_output,
				auxiliary_output,
			) = run_model(stock_index_tensor, return_aux=True)
			scores = score_output.squeeze(0).float().cpu().numpy()
			allocation_logits = (
				allocation_output.squeeze(0).float().cpu().numpy()
			)
			exposure = float(exposure_output.squeeze(0).float().cpu().item())
			model_scores.append(scores)
			model_allocations.append(allocation_logits)
			model_exposures.append(exposure)
			model_top = np.argsort(scores, kind='stable')[::-1][:5]
			risk_1d_probabilities = (
				torch.sigmoid(auxiliary_output['risk_1d_logits'])
				.squeeze(0).float().cpu().numpy()
				if auxiliary_output['risk_1d_logits'] is not None
				else np.full(len(scores), 0.5)
			)
			risk_3d_probabilities = (
				torch.sigmoid(auxiliary_output['risk_3d_logits'])
				.squeeze(0).float().cpu().numpy()
				if auxiliary_output['risk_3d_logits'] is not None
				else np.full(len(scores), 0.5)
			)
			risk_5d_probabilities = (
				torch.sigmoid(auxiliary_output['risk_5d_logits'])
				.squeeze(0).float().cpu().numpy()
				if auxiliary_output['risk_5d_logits'] is not None
				else np.full(len(scores), 0.5)
			)
			tail_5d_probabilities = (
				torch.sigmoid(auxiliary_output['tail_5d_logits'])
				.squeeze(0).float().cpu().numpy()
				if auxiliary_output.get('tail_5d_logits') is not None
				else np.full(len(scores), 0.5)
			)
			model_regime_gates.append(float(
				auxiliary_output['regime_gate']
				.squeeze(0).float().cpu().item()
			))
			model_risk_1d.append(risk_1d_probabilities)
			model_risk_3d.append(risk_3d_probabilities)
			model_risk_5d.append(risk_5d_probabilities)
			model_tail_5d.append(tail_5d_probabilities)
			unk_scores_output, _, _, unk_exposure_output = run_model(
				torch.ones_like(stock_index_tensor)
			)
			permutation_rng = np.random.default_rng(
				base_identity_seed + int(model_seeds[model_idx])
			)
			identity_permutation = np.arange(len(stockid2idx) + 2)
			known_identity_indices = np.arange(2, len(stockid2idx) + 2)
			identity_permutation[known_identity_indices] = (
				permutation_rng.permutation(known_identity_indices)
			)
			permuted_indices = identity_permutation[
				np.asarray(sequence_stock_indices)
			]
			permuted_index_tensor = torch.as_tensor(
				permuted_indices,
				dtype=torch.long,
				device=device,
			).unsqueeze(0)
			permuted_scores_output, _, _, permuted_exposure_output = run_model(
				permuted_index_tensor
			)

			def identity_comparison(alternative_output, alternative_exposure):
				alternative_scores = (
					alternative_output.squeeze(0).float().cpu().numpy()
				)
				alternative_top = np.argsort(
					alternative_scores,
					kind='stable',
				)[::-1][:5]
				correlation = spearmanr(scores, alternative_scores).statistic
				return {
					'score_spearman': float(
						correlation if np.isfinite(correlation) else 0.0
					),
					'top5_overlap': float(
						len(set(model_top).intersection(alternative_top)) / 5
					),
					'top5': [
						sequence_stock_ids[index]
						for index in alternative_top
					],
					'exposure': float(
						alternative_exposure.squeeze(0).float().cpu().item()
					),
				}

			model_diagnostics.append({
				'seed': int(model_seeds[model_idx]),
				'model_path': policy['model_paths'][model_idx],
				'exposure': exposure,
				'regime_gate': float(
					auxiliary_output['regime_gate']
					.squeeze(0).float().cpu().item()
				),
				'identity_gate': float(
					model.identity_gate_value().detach().float().cpu().item()
				),
				'exposure_base_probability': float(
					auxiliary_output['exposure_base_probability']
					.squeeze(0).float().cpu().item()
				),
				'exposure_regime_penalty': (
					float(auxiliary_output['exposure_regime_penalty'].item())
					if auxiliary_output['exposure_regime_penalty'] is not None
					else None
				),
				'exposure_risk_penalty': (
					float(auxiliary_output['exposure_risk_penalty'].item())
					if auxiliary_output['exposure_risk_penalty'] is not None
					else None
				),
				'top5': [sequence_stock_ids[index] for index in model_top],
				'top5_risk_1d': [
					float(risk_1d_probabilities[index])
					for index in model_top
				],
				'top5_risk_3d': [
					float(risk_3d_probabilities[index])
					for index in model_top
				],
				'top5_risk_5d': [
					float(risk_5d_probabilities[index])
					for index in model_top
				],
				'top5_tail_5d': [
					float(tail_5d_probabilities[index])
					for index in model_top
				],
				'all_unk_vs_real': identity_comparison(
					unk_scores_output,
					unk_exposure_output,
				),
				'permuted_vs_real': identity_comparison(
					permuted_scores_output,
					permuted_exposure_output,
				),
			})
			del model

	lgbm_ranker, lgbm_weight, lgbm_model_path = resolve_lgbm_ranker(
		policy,
		artifact_source_dir,
	)
	lgbm_diagnostics = {
		'enabled': False,
		'weight': lgbm_weight,
		'model_path': lgbm_model_path,
	}
	if lgbm_ranker is not None:
		if len(features) != 205:
			raise ValueError(
				'LightGBM 排序融合只允许使用205维 v1.17 特征工件'
			)
		lgbm_feature_count = validate_lgbm_feature_count(
			lgbm_ranker,
			len(features),
		)
		# 保留列名，既校验205维顺序，也避免 sklearn 对 ndarray 的无名列警告。
		lgbm_input = pd.DataFrame(lgbm_raw_latest, columns=features)
		lgbm_scores = np.asarray(
		lgbm_ranker.predict(lgbm_input),
			dtype=np.float64,
		).reshape(-1)
		if lgbm_scores.shape != (len(sequence_stock_ids),):
			raise ValueError(
				'LightGBM 排序器输出长度与可预测股票数不一致: '
				f'{lgbm_scores.shape} != {(len(sequence_stock_ids),)}'
			)
		if not np.isfinite(lgbm_scores).all():
			raise ValueError('LightGBM 排序器输出包含 NaN 或无穷值')
		lgbm_percentiles = cross_sectional_percentiles(lgbm_scores)
		# 两侧先转为同日百分位，避免模型 logit 与树模型分数尺度互相主导。
		model_scores = [
			(1.0 - lgbm_weight) * cross_sectional_percentiles(scores)
			+ lgbm_weight * lgbm_percentiles
			for scores in model_scores
		]
		lgbm_top = np.argsort(lgbm_percentiles, kind='stable')[::-1][:5]
		lgbm_diagnostics.update({
			'enabled': True,
			'feature_count': lgbm_feature_count,
			'top5': [sequence_stock_ids[index] for index in lgbm_top],
		})

	risk_blends = np.asarray([
		float(policy.get(
			'risk_1d_blend',
			trained_config.get('risk_1d_blend', 0.40),
		)),
		float(policy.get(
			'risk_3d_blend',
			trained_config.get('risk_3d_blend', 0.60),
		)),
		float(policy.get(
			'risk_5d_blend',
			trained_config.get('risk_5d_blend', 0.0),
		)),
		float(policy.get(
			'tail_5d_blend',
			trained_config.get('tail_5d_blend', 0.0),
		)),
	], dtype=np.float64)
	if (risk_blends < 0).any() or risk_blends.sum() <= 0:
		raise ValueError('推理风险头融合权重必须非负且权重和大于0')
	risk_blends /= risk_blends.sum()
	portfolio = build_ensemble_portfolio(
		np.stack(model_scores),
		np.stack(model_allocations),
		np.asarray(model_exposures),
		min_exposure=float(trained_config['min_exposure']),
		max_exposure=float(trained_config['max_exposure']),
		allocation_temperature=float(policy['allocation_temperature']),
		allocation_blend=float(policy['allocation_blend']),
		disagreement_gamma=float(policy['disagreement_gamma']),
		selection_risk_context=selection_risk_context,
		selection_risk_gamma=float(policy.get(
			'selection_risk_gamma',
			0.0,
		)),
		selection_candidate_k=int(policy.get(
			'selection_candidate_k',
			20,
		)),
		correlation_lookbacks=policy.get(
			'correlation_lookbacks',
			trained_config.get('selection_correlation_lookbacks', [20]),
		),
		cluster_cap_enabled=bool(policy.get(
			'cluster_cap_enabled',
			trained_config.get('selection_cluster_cap_enabled', False),
		)),
		cluster_correlation_threshold=float(policy.get(
			'cluster_correlation_threshold',
			trained_config.get(
				'selection_cluster_correlation_threshold',
				0.60,
			),
		)),
		max_stocks_per_cluster=int(policy.get(
			'max_stocks_per_cluster',
			trained_config.get('selection_max_stocks_per_cluster', 2),
		)),
		cluster_max_raw_rank=policy.get('cluster_max_raw_rank'),
		risk_probability_matrix=(
			risk_blends[0] * np.stack(model_risk_1d)
			+ risk_blends[1] * np.stack(model_risk_3d)
			+ risk_blends[2] * np.stack(model_risk_5d)
			+ risk_blends[3] * np.stack(model_tail_5d)
		),
		regime_gates=np.asarray(model_regime_gates),
		risk_score_penalty=float(policy.get(
			'risk_score_penalty',
			0.0,
		)),
		correlation_exposure_gamma=float(policy.get(
			'correlation_exposure_gamma',
			0.0,
		)),
		exposure_head_blend=float(policy.get(
			'exposure_head_blend',
			1.0,
		)),
		fixed_exposure_baseline=float(policy.get(
			'fixed_exposure_baseline',
			0.6231689453125,
		)),
		top_k=int(policy.get('top_k', 5)),
	)
	top_indices = portfolio['top_indices']
	position_weights, exposure = make_serialization_safe_positions(
		portfolio,
		trained_config,
	)
	top5 = [sequence_stock_ids[i] for i in top_indices]
	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': position_weights,
	})
	validate_result(output_df, stock_ids, runtime_config=trained_config)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	output_df.to_csv(output_path, index=False, encoding='utf-8')

	# 重新读取实际提交文件，避免序列化精度或股票代码格式破坏约束。
	written_df = pd.read_csv(output_path, dtype={'stock_id': str})
	weight_sum = validate_result(
		written_df,
		stock_ids,
		runtime_config=trained_config,
	)
	combined_risk = (
		risk_blends[0] * np.mean(np.stack(model_risk_1d), axis=0)
		+ risk_blends[1] * np.mean(np.stack(model_risk_3d), axis=0)
		+ risk_blends[2] * np.mean(np.stack(model_risk_5d), axis=0)
		+ risk_blends[3] * np.mean(np.stack(model_tail_5d), axis=0)
	)
	prediction_day = build_prediction_day_diagnostics(
		top5,
		written_df['weight'].to_numpy(dtype=np.float64),
		weight_sum,
		portfolio,
		latest_date,
		trained_config,
		model_regime_gates,
		combined_risk[top_indices],
	)
	diagnostics = {
		'prediction_date': latest_date.strftime('%Y-%m-%d'),
		'num_ranked_stocks': len(sequence_stock_ids),
		'artifact_source_dir': artifact_source_dir,
		'artifact_validation': artifact_validation,
		'lgbm': lgbm_diagnostics,
		'config_drift': config_drift,
		'promotion_criteria': policy.get('promotion_criteria', {}),
		'daily': [prediction_day],
		'policy': {
			'allocation_blend': float(policy['allocation_blend']),
			'disagreement_gamma': float(policy['disagreement_gamma']),
			'selection_risk_gamma': float(policy.get(
				'selection_risk_gamma',
				0.0,
			)),
			'selection_candidate_k': int(policy.get(
				'selection_candidate_k',
				20,
			)),
			'correlation_lookbacks': [
				int(value) for value in policy.get(
					'correlation_lookbacks',
					trained_config.get(
						'selection_correlation_lookbacks',
						[20],
					),
				)
			],
			'cluster_cap_enabled': bool(policy.get(
				'cluster_cap_enabled',
				trained_config.get(
					'selection_cluster_cap_enabled',
					False,
				),
			)),
			'cluster_correlation_threshold': float(policy.get(
				'cluster_correlation_threshold',
				trained_config.get(
					'selection_cluster_correlation_threshold',
					0.60,
				),
			)),
			'max_stocks_per_cluster': int(policy.get(
				'max_stocks_per_cluster',
				trained_config.get(
					'selection_max_stocks_per_cluster',
					2,
				),
			)),
			'cluster_max_raw_rank': policy.get(
				'cluster_max_raw_rank'
			),
			'risk_blends': {
				'1d': float(risk_blends[0]),
				'3d': float(risk_blends[1]),
				'5d_soft': float(risk_blends[2]),
				'5d_tail': float(risk_blends[3]),
			},
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
			'fixed_exposure_baseline': float(policy.get(
				'fixed_exposure_baseline',
				0.6231689453125,
			)),
			'min_exposure': float(policy['min_exposure']),
			'max_exposure': float(policy['max_exposure']),
		},
		'models': model_diagnostics,
		'ensemble': {
			'top5': top5,
			'raw_top5': [
				sequence_stock_ids[index]
				for index in portfolio['raw_top_indices']
			],
			'unadjusted_top5': [
				sequence_stock_ids[index]
				for index in portfolio['unadjusted_top_indices']
			],
			'selection_details': [
				{
					'stock_id': sequence_stock_ids[index],
					'selected_rank': selected_rank,
					'raw_rank': int(
						portfolio['selected_raw_ranks'][selected_rank - 1]
					),
					'reversal_risk': float(
						portfolio['selected_reversal_risk'][selected_rank - 1]
					),
					'correlation_risk': float(
						portfolio['selected_correlation_risk'][
							selected_rank - 1
						]
					),
					'cluster_id': int(
						portfolio['selected_cluster_ids'][
							selected_rank - 1
						]
					),
				}
				for selected_rank, index in enumerate(top_indices, start=1)
			],
			'weights': [float(weight) for weight in position_weights],
			'ensemble_scores': [
				float(portfolio['ensemble_scores'][index])
				for index in top_indices
			],
			'selected_disagreement': [
				float(value)
				for value in portfolio['selected_disagreement']
			],
			'mean_disagreement': float(portfolio['mean_disagreement']),
			'regime_gate': float(np.median(model_regime_gates)),
			'selected_risk_1d': [
				float(value)
				for value in np.mean(
					np.stack(model_risk_1d),
					axis=0,
				)[top_indices]
			],
			'selected_risk_3d': [
				float(value)
				for value in np.mean(
					np.stack(model_risk_3d),
					axis=0,
				)[top_indices]
			],
			'selected_risk_5d': [
				float(value)
				for value in np.mean(
					np.stack(model_risk_5d),
					axis=0,
				)[top_indices]
			],
			'selected_tail_5d': [
				float(value)
				for value in np.mean(
					np.stack(model_tail_5d),
					axis=0,
				)[top_indices]
			],
			'selected_combined_risk': [
				float(value)
				for value in (
					risk_blends[0] * np.mean(
						np.stack(model_risk_1d), axis=0,
					)
					+ risk_blends[1] * np.mean(
						np.stack(model_risk_3d), axis=0,
					)
					+ risk_blends[2] * np.mean(
						np.stack(model_risk_5d), axis=0,
					)
					+ risk_blends[3] * np.mean(
						np.stack(model_tail_5d), axis=0,
					)
				)[top_indices]
			],
			'selected_reversal_risk': [
				float(value)
				for value in portfolio['selected_reversal_risk']
			],
			'selected_correlation_risk': [
				float(value)
				for value in portfolio['selected_correlation_risk']
			],
			'mean_positive_correlation': float(
				portfolio['mean_positive_correlation']
			),
			'raw_mean_positive_correlation': float(
				portfolio['raw_mean_positive_correlation']
			),
			'selected_cluster_ids': [
				int(value)
				for value in portfolio['selected_cluster_ids']
			],
			'raw_cluster_ids': [
				int(value)
				for value in portfolio['raw_cluster_ids']
			],
			'num_candidate_clusters': int(
				portfolio['num_candidate_clusters']
			),
			'requested_candidate_k': int(
				portfolio['requested_candidate_k']
			),
			'effective_candidate_k': int(
				portfolio['effective_candidate_k']
			),
			'candidate_pool_expanded': bool(
				portfolio['candidate_pool_expanded']
			),
			'cluster_constraint_applied': bool(
				portfolio['cluster_constraint_applied']
			),
			'cluster_constraint_skipped': bool(
				portfolio['cluster_constraint_skipped']
			),
			'max_selected_raw_rank': int(
				portfolio['max_selected_raw_rank']
			),
			'head_base_exposure': float(
				portfolio['head_base_exposure']
			),
			'base_exposure': float(portfolio['base_exposure']),
			'final_exposure': weight_sum,
			'cash_weight': 1.0 - weight_sum,
		},
	}
	with open(diagnostics_path, 'w', encoding='utf-8') as f:
		json.dump(diagnostics, f, indent=2, ensure_ascii=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(sequence_stock_ids)}')
	print(f'模型平均排名分歧: {portfolio["mean_disagreement"]:.6f}')
	print(
		f'市场压力门控: {float(np.median(model_regime_gates)):.6f}'
	)
	print(
		f'OOF 风险分数惩罚: '
		f'{float(policy.get("risk_score_penalty", 0.0)):.4f}'
	)
	print(
		f'风险选择 gamma/相关性: '
		f'{float(policy.get("selection_risk_gamma", 0.0)):.4f} / '
		f'{portfolio["mean_positive_correlation"]:.4f} '
		f'(原始 {portfolio["raw_mean_positive_correlation"]:.4f})'
	)
	if portfolio.get('cluster_cap_enabled', False):
		print(
			f'相关簇: 候选 {portfolio["num_candidate_clusters"]} 簇, '
			f'候选池 Top-{portfolio["effective_candidate_k"]}'
			f'{"（已扩展）" if portfolio["candidate_pool_expanded"] else ""}, '
			f'Top-5 簇编号 '
			f'{[int(value) for value in portfolio["selected_cluster_ids"]]}'
		)
	print(
		f'相关性降仓 gamma: '
		f'{float(policy.get("correlation_exposure_gamma", 0.0)):.4f}'
	)
	print(
		f'Exposure Head混合: '
		f'{float(policy.get("exposure_head_blend", 1.0)):.2f}; '
		f'Head仓位 {portfolio["head_base_exposure"]:.6f}'
	)
	print(f'分歧调整前仓位: {portfolio["base_exposure"]:.12f}')
	print(f'提交权重和: {weight_sum:.12f}')
	print(f'现金权重: {1.0 - weight_sum:.12f}')
	print(f'结果已写入: {output_path}')
	print(f'推理诊断已写入: {diagnostics_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()

import json
import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import add_relative_market_features
from utils import build_ensemble_portfolio
from utils import engineer_features_39, engineer_features_158plus39
from utils import RELATIVE_MARKET_FEATURES, RELATIVE_MARKET_FEATURE_SET


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
	if feature_num == RELATIVE_MARKET_FEATURE_SET:
		processed = add_relative_market_features(processed)
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


def compare_runtime_configs(saved_config, live_config):
	"""报告源码配置与训练快照漂移；推理始终以训练快照为准。"""
	ignored_keys = {'output_dir'}
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
	output_dir = config['output_dir']
	trained_config_path = os.path.join(output_dir, 'config.json')
	policy_path = os.path.join(output_dir, 'ensemble_policy.json')
	stock_mapping_path = os.path.join(output_dir, 'stockid2idx.json')
	output_path = os.path.join('./output/', 'result.csv')
	diagnostics_path = os.path.join('./output/', 'prediction_diagnostics.json')

	for path, description in [
		(trained_config_path, '训练配置快照'),
		(policy_path, 'ensemble 策略'),
		(stock_mapping_path, '股票 ID 映射'),
	]:
		if not os.path.exists(path):
			raise FileNotFoundError(f'未找到{description}: {path}')
	with open(trained_config_path, 'r', encoding='utf-8') as f:
		trained_config = json.load(f)
	with open(policy_path, 'r', encoding='utf-8') as f:
		policy = json.load(f)
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
		output_dir,
		policy.get('scaler_path', 'scaler.pkl'),
	)
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到 Scaler: {scaler_path}')
	model_paths = [
		os.path.join(output_dir, relative_path)
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
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	with open(stock_mapping_path, 'r', encoding='utf-8') as f:
		stockid2idx = json.load(f)

	processed, features = preprocess_predict_data(
		raw_df,
		stockid2idx,
		runtime_config=trained_config,
	)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
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
			with torch.autocast(
				device_type=device.type,
				dtype=torch.float16,
				enabled=(
					device.type == 'cuda'
					and trained_config.get('amp_enabled', True)
				),
			):
				score_output, _, allocation_output, exposure_output = model(
					x,
					stock_index_tensor,
					stock_mask,
				)
			scores = score_output.squeeze(0).float().cpu().numpy()
			allocation_logits = (
				allocation_output.squeeze(0).float().cpu().numpy()
			)
			exposure = float(exposure_output.squeeze(0).float().cpu().item())
			model_scores.append(scores)
			model_allocations.append(allocation_logits)
			model_exposures.append(exposure)
			model_top = np.argsort(scores, kind='stable')[::-1][:5]
			model_diagnostics.append({
				'seed': int(model_seeds[model_idx]),
				'model_path': policy['model_paths'][model_idx],
				'exposure': exposure,
				'top5': [sequence_stock_ids[index] for index in model_top],
			})
			del model

	portfolio = build_ensemble_portfolio(
		np.stack(model_scores),
		np.stack(model_allocations),
		np.asarray(model_exposures),
		min_exposure=float(trained_config['min_exposure']),
		max_exposure=float(trained_config['max_exposure']),
		allocation_temperature=float(policy['allocation_temperature']),
		allocation_blend=float(policy['allocation_blend']),
		disagreement_gamma=float(policy['disagreement_gamma']),
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
	diagnostics = {
		'prediction_date': latest_date.strftime('%Y-%m-%d'),
		'num_ranked_stocks': len(sequence_stock_ids),
		'config_drift': config_drift,
		'promotion_criteria': policy.get('promotion_criteria', {}),
		'policy': {
			'allocation_blend': float(policy['allocation_blend']),
			'disagreement_gamma': float(policy['disagreement_gamma']),
			'min_exposure': float(policy['min_exposure']),
			'max_exposure': float(policy['max_exposure']),
		},
		'models': model_diagnostics,
		'ensemble': {
			'top5': top5,
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
	print(f'分歧调整前仓位: {portfolio["base_exposure"]:.12f}')
	print(f'提交权重和: {weight_sum:.12f}')
	print(f'现金权重: {1.0 - weight_sum:.12f}')
	print(f'结果已写入: {output_path}')
	print(f'推理诊断已写入: {diagnostics_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()

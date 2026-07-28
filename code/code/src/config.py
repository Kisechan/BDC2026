# 配置参数
sequence_length = 60
feature_num = '158+39_reduced25'
patience_num = 12
experiment_name = 'single_seed_3fold'
config = {
    # 单个样本输入最近 60 个交易日；最早时点的 60 日特征可追溯到
    # t-119，因此显式窗口已覆盖约 120 个行情观测。
    'sequence_length': sequence_length,
    'd_model': 128,          # Transformer隐藏维度；原始特征维度由 feature_num 对应的特征表决定
    'nhead': 4,             # 注意力头数量
    'num_layers': 2,        # Transformer层数
    'dim_feedforward': 256, # 前馈网络维度
    'batch_size': 8,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'max_epochs': 50,
    'min_final_epochs': 8,
    'patience': patience_num,
    'learning_rate': 3e-5,
    'dropout': 0.1,
    'stock_embedding_dim': 4,
    'id_dropout': 0.1,
    'embedding_dropout': 0.1,
    'feature_num': feature_num,  # 166 个实际连续输入；名称保留“158+39”特征族血缘
    'max_grad_norm': 5.0,
    'grad_clip': True,
    'num_folds': 3,
    'validation_months': 2,
    'evaluation_stride': 5,
    'seed': 42,
    'ensemble_enabled': False,
    # 默认只运行 seed=42 的三折；仅在 ensemble_enabled=True 时使用下列种子。
    'ensemble_seeds': [42, 142, 242],
    'checkpoint_metric': 'weighted_portfolio_return_plus_rank_ic',
    'checkpoint_rank_ic_weight': 0.1,
    'purge_days': 5,

    'regression_weight': 0.05,
    'regression_beta': 0.02,
    'allocation_weight': 0.1,
    'exposure_weight': 1.0,
    'allocation_temperature': 1.0,
    'allocation_target_temperature': 0.10,
    'exposure_target_temperature': 0.02,
    'min_exposure': 0.20,
    'max_exposure': 0.999999,
    'allocation_blend_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
    'disagreement_gamma_grid': [0.0, 2.0, 4.0, 8.0],
    'ensemble_downside_weight': 0.5,
    'listwise_temperature': 0.2,
    'listwise_weight': 0.2,
    'ic_weight': 0.15,
    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    'weight_decay': 1e-4,

    # 保持损失为 FP32，仅对 Transformer 前向启用 AMP。
    'amp_enabled': True,
    'tf32_enabled': True,
    'fused_optimizer': True,
    'pin_memory': True,
    'non_blocking_transfer': True,
    'num_workers': 0,
    'deterministic_training': True,

    'output_dir': f'./model/{sequence_length}_{feature_num}_{experiment_name}',
    'data_path': './data',
}

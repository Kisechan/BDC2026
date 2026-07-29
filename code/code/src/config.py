# 配置参数
sequence_length = 60
feature_num = '158+39_reduced25_relmarket12_risk15'
patience_num = 12
experiment_name = 'nested_oof_forward_policy_v7_1'
artifact_experiment_name = 'nested_oof_diverse_tailregime_v6_decay5y'
config = {
    # 单个样本输入最近 60 个交易日；最早时点的 60 日特征可追溯到
    # t-119，因此显式窗口已覆盖约 120 个行情观测。
    'sequence_length': sequence_length,
    'd_model': 128,          # Transformer隐藏维度；原始特征维度由 feature_num 对应的特征表决定
    'nhead': 4,             # 注意力头数量
    'num_layers': 2,        # Transformer层数
    'dim_feedforward': 256, # 前馈网络维度
    'batch_size': 20,       # 2080 Ti 显存余量充足；仍需留意股票维度的二次复杂度
    'max_epochs': 50,
    'min_final_epochs': 1,
    'patience': patience_num,
    'learning_rate': 3e-5,
    'dropout': 0.1,
    'stock_embedding_dim': 4,
    'id_dropout': 0.2,
    'embedding_dropout': 0.1,
    'id_gate_enabled': True,
    'id_gate_init': 0.20,
    'id_gate_regularization': 0.01,
    'identity_sensitivity_seed': 20260728,
    # 166个reduced25 + 12个相对市场 + 15个短期风险/市场压力输入。
    'feature_num': feature_num,
    'max_grad_norm': 5.0,
    'grad_clip': True,
    'num_folds': 3,
    'validation_months': 2,
    'evaluation_stride': 5,
    'seed': 42,
    'ensemble_enabled': False,
    # 默认只运行 seed=42 的三折；仅在 ensemble_enabled=True 时使用下列种子。
    'ensemble_seeds': [42, 142, 242],
    'checkpoint_metric': 'top5_return_plus_rank_ic',
    'checkpoint_rank_ic_weight': 0.2,
    'purge_days': 5,

    'regression_weight': 0.05,
    'regression_beta': 0.02,
    'allocation_weight': 0.1,
    'exposure_weight': 1.0,
    'allocation_candidate_k': 20,
    'allocation_return_clip': 0.10,
    'allocation_temperature': 1.0,
    'allocation_target_temperature': 0.05,
    'exposure_target_temperature': 0.02,
    'exposure_selected_return_weight': 0.70,
    'exposure_market_return_weight': 0.30,
    'exposure_downside_weight': 0.25,
    'exposure_market_encoder_enabled': True,
    'exposure_market_hidden_size': 16,
    'exposure_portfolio_summary_enabled': True,
    # 原5个市场状态 + 新增8个市场压力特征。
    'market_state_feature_indices': [
        173, 174, 175, 176, 177,
        185, 186, 187, 188, 189, 190, 191, 192,
    ],
    'risk_heads_enabled': True,
    'risk_5d_head_enabled': True,
    'tail_5d_head_enabled': True,
    'risk_1d_blend': 0.15,
    'risk_3d_blend': 0.20,
    'risk_5d_blend': 0.30,
    'tail_5d_blend': 0.35,
    # 排序主干输出原始分数；风险惩罚强度只用 OOF 网格校准。
    'risk_penalty_scale': 0.0,
    'oof_risk_penalty_enabled': True,
    'risk_score_penalty_grid': [0.0, 0.05, 0.10, 0.15, 0.25],
    'risk_1d_target_temperature': 0.01,
    'risk_3d_target_temperature': 0.02,
    'risk_5d_target_temperature': 0.03,
    'risk_1d_weight': 0.10,
    'risk_3d_weight': 0.15,
    'risk_5d_weight': 0.20,
    'tail_5d_weight': 0.20,
    'tail_5d_threshold': -0.03,
    'regime_gate_enabled': True,
    'regime_market_hidden_size': 16,
    'regime_market_feature_indices': [
        173, 174, 175, 176, 177,
        185, 186, 187, 188, 189, 190, 191, 192,
    ],
    'regime_market_return_temperature': 0.01,
    'regime_tail_share_baseline': 0.20,
    'regime_tail_share_temperature': 0.10,
    'regime_weight': 0.10,
    # 市场压力或已选股票风险升高时，Exposure 只能单调下降。
    'monotonic_exposure_enabled': True,
    'exposure_regime_penalty_init': 0.25,
    'exposure_risk_penalty_init': 0.25,
    'min_exposure': 0.20,
    'max_exposure': 0.999999,
    'allocation_blend_grid': [0.0, 0.25, 0.5, 0.75, 1.0],
    'disagreement_gamma_grid': [0.0, 2.0, 4.0, 8.0],
    'selection_risk_gamma_grid': [0.0, 0.05, 0.10, 0.20],
    'correlation_exposure_gamma_grid': [0.0, 0.5, 1.0, 2.0],
    # Exposure Head 必须保留；OOF 只校准其相对固定基线的占比。
    'exposure_head_blend_grid': [0.25, 0.50, 0.75, 1.0],
    'selection_candidate_k': 30,
    'selection_risk_lookback': 60,
    'selection_correlation_lookbacks': [20, 60],
    'cluster_cap_enabled': False,
    'cluster_cap_grid': [False, True],
    'cluster_correlation_threshold': 0.60,
    'max_stocks_per_cluster': 2,
    'cluster_max_raw_rank': 10,
    'nested_oof_enabled': True,
    # 策略层纯收益优先，但保留较轻的下行波动惩罚。
    'ensemble_downside_weight': 0.25,
    'policy_only_experiment': True,
    'policy_simplicity_tolerance': 0.001,
    'module_min_positive_fold_fraction': 2 / 3,
    'forward_module_max_fold_loss': 0.0025,
    'forward_module_max_p10_loss': 0.005,
    'minimum_allocation_deployment_blend': 0.25,
    'minimum_exposure_deployment_blend': 0.25,
    'listwise_temperature': 0.2,
    'listwise_weight': 0.2,
    'ic_weight': 0.15,
    'pairwise_weight': 0.3, # 降低 LambdaRank@5 对排序主目标的支配
    'lambdarank_candidate_k': 20,
    'lambdarank_hard_negative_k': 20,
    'lambdarank_return_gap_scale': 0.02,
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    'weight_decay': 1e-4,

    # 五年排序样本按交易日以两年半衰期衰减，降低过旧市场状态的影响。
    'ranking_recency_half_life_days': 504,
    # Ranking → Risk/Regime → Allocation → Exposure 四阶段训练。
    'ranking_learning_rate': 3e-5,
    'ranking_max_epochs': 50,
    'ranking_patience': 12,
    'ranking_checkpoint_metric': 'top5_return_plus_rank_ic',
    'ranking_min_final_epochs': 1,
    'risk_learning_rate': 1e-4,
    'risk_max_epochs': 12,
    'risk_patience': 4,
    'risk_checkpoint_metric': 'negative_eval_loss',
    'risk_min_final_epochs': 1,
    'allocation_learning_rate': 1e-4,
    'allocation_max_epochs': 12,
    'allocation_patience': 4,
    'allocation_checkpoint_metric': 'allocation_contribution',
    'allocation_min_final_epochs': 1,
    'exposure_learning_rate': 1e-4,
    'exposure_max_epochs': 12,
    'exposure_patience': 4,
    'exposure_checkpoint_metric': 'weighted_portfolio_risk_adjusted',
    'exposure_min_final_epochs': 1,
    # Ranking 每轮验证；后三个冻结主干的辅助阶段每两轮验证一次。
    'auxiliary_eval_interval': 2,
    # Ranking 完成后缓存主干表示，Risk/Allocation/Exposure 不再重复 Transformer 前向。
    'cache_frozen_backbone': True,

    # 当前 relmarket12 基线的晋级门槛；测试周不参与这些阈值。
    'promotion_mean_weighted_return': 0.019902,
    'promotion_worst_fold_weighted_return': 0.012523,
    'promotion_mean_top5_return': 0.0300,
    'promotion_p10_weighted_return': -0.025672,
    'promotion_mean_rank_ic': 0.0514,
    'promotion_id_score_correlation': 0.90,
    'promotion_id_top5_overlap': 0.40,
    'promotion_min_exposure_std': 0.01,
    'promotion_min_regime_gate_std': 0.01,
    'fixed_exposure_baseline': 0.6231689453125,

    # 保持损失为 FP32，仅对 Transformer 前向启用 AMP。
    'amp_enabled': True,
    'tf32_enabled': True,
    'fused_optimizer': True,
    'pin_memory': True,
    'non_blocking_transfer': True,
    'num_workers': 0,
    'deterministic_training': True,

    'output_dir': f'./model/{sequence_length}_{feature_num}_{experiment_name}',
    'policy_output_dir': (
        f'./model/{sequence_length}_{feature_num}_{experiment_name}'
    ),
    'policy_only_source_dir': (
        f'./model/{sequence_length}_{feature_num}_{artifact_experiment_name}'
    ),
    # 五年历史数据与原三年数据隔离存放，避免覆盖已有实验的切分文件。
    'data_path': './data_5y',
}

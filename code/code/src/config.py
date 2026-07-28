# 配置参数
sequence_length = 60
feature_num = '158+39_reduced25'
patience_num = 12
experiment_name = 'linear_redundancy_removed'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 128,          # Transformer隐藏维度；原始特征维度由 feature_num 对应的特征表决定
    'nhead': 4,             # 注意力头数量
    'num_layers': 2,        # Transformer层数
    'dim_feedforward': 256, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
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
    'checkpoint_metric': 'top5_return_plus_rank_ic',
    'checkpoint_rank_ic_weight': 0.1,
    'purge_days': 5,

    'regression_weight': 0.05,
    'regression_beta': 0.02,
    'listwise_temperature': 0.2,
    'listwise_weight': 0.2,
    'ic_weight': 0.15,
    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    'weight_decay': 1e-4,

    'output_dir': f'./model/{sequence_length}_{feature_num}_{experiment_name}',
    'data_path': './data',
}

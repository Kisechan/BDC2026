# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去60个交易日的量价特征序列，以及独立的股票 ID；
- 模型：`StockTransformer`，使用股票 ID Embedding、时序编码和股票间注意力；
- 输出：每只股票的 ranking score 和辅助收益预测，最终选择 Top-5 组合。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（`data/stock_data.csv`）；
2. 做特征工程（39特征或`158+39`特征）；
3. 构建标签：未来收益率（代码中为 `open_t1` 到 `open_t5` 的相对收益）；
4. 按实际交易日构造三折 walk-forward 验证，每折训练/验证之间 purge 5 日；
5. 联合优化排序损失和原始收益 SmoothL1 回归损失；
6. 汇总平均/最差折 Top-5、Rank IC 和泛化差距；随后按各折最佳 epoch
   的中位数，用全部标签有效样本重训最终模型。

---

## 2. 代码结构说明

### [config.py](config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60）；
- 模型超参数（`d_model`、`nhead`、`num_layers` 等）；
- 训练超参数（`batch_size`、`max_epochs`、`patience`、`learning_rate`、`weight_decay`）；
- 排序损失权重参数（`pairwise_weight`、`top5_weight`、`base_weight`）；
- 多折切分、ID 正则化与辅助回归参数（`num_folds`、`validation_months`、`purge_days`、
  `id_dropout`、`embedding_dropout`、`regression_weight`）；
- 数据路径和输出路径（默认输出到 `output/`）。

### [model.py](model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- 股票 ID Embedding：训练时随机将部分 ID 替换成 UNK，并对 embedding 向量做 dropout；
- `CrossStockAttention`：使用 padding mask 建模同日股票间关系；
- `score_head` 与 `return_head`：分别输出排序分数和原始收益预测。

输入包含特征张量、股票索引和有效股票 mask；模型返回两个
`[batch, num_stocks]` 张量，分别对应排序分数和预测收益。

### [utils.py](utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：39个技术指标特征；
- `engineer_features()`：158个Alpha类特征；
- `engineer_features_158plus39()`：合并 `158 + 39` 特征；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。
原 `RSQR5/10/20/30/60` 的滚动索引实现会使绝大部分结果变成 NaN 后填 0，
现已从特征计算和训练/推理特征表中删除；`158+39` 名称为兼容旧配置保留。

### [train.py](train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：在完整历史上按股票计算严格因果特征和标签；
	- `build_walk_forward_folds()`：按实际交易日构造扩展窗口验证折和 5 日 purge。
	- 每折最多训练 `max_epochs`，验证指标连续 `patience` 轮无提升时提前停止；
	- 三折完成后，重新拟合全量 scaler 并训练最终推理模型。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`WeightedRankingLoss`
	- 组合 `listwise_loss`、`pairwise_loss` 与原始收益 SmoothL1 辅助损失；
	- 对真实Top-k样本施加更高权重。
- 评估指标：`calculate_ranking_metrics()`
	- 计算等权 Top-5 收益、Rank IC、回归 MAE 和原有归一化指标；
	- 汇总多折均值、最差折表现及训练—验证差距。

训练产物：
- `fold_N/best_model.pth`、`fold_N/scaler.pkl`、`fold_N/metrics.json`：逐折产物；
- `best_model.pth`、`scaler.pkl`：全量重训后供推理使用的最终产物；
- `final_training.json`、`full_train/log/`：全量重训轮数、样本范围和日志；
- `stockid2idx.json`：训练与推理共用的股票 ID 映射；
- `cross_validation_summary.json`：多折汇总；
- `config.json`：训练时配置快照；
- `fold_N/log/`：逐折 TensorBoard 日志。

### [predict.py](predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程；
3. 加载 `scaler.pkl` 进行特征标准化；
4. 用 `best_model.pth` 对全部可预测股票打分；
5. 按分数降序取前5只，输出到 `output.csv`：
	 - `stock_id`
	 - `weight`（固定等权 `0.2`）

写出前后都会检查列名、股票数量、股票唯一性、候选范围、权重有限性和权重和，
避免浮点序列化导致提交权重超过 1。

### [get_stock_data.py](get_stock_data.py)
数据抓取脚本（Baostock）：
- 获取沪深300成分股；
- 抓取历史日线数据并保存为训练所需格式。

---

## 3. 数据与输入输出约定

默认训练数据文件：
- `data/train.csv`

关键列：
- `股票代码`、`日期`、`开盘`、`收盘`、`最高`、`最低`、`成交量`、`成交额`、`换手率`、`涨跌幅` 等。

预测输出文件：
- output目录下 `result.csv`（由 `predict.py` 生成）。

---

## 4. 运行方法（推荐使用 uv）

1) 使用 `uv` 安装依赖

`uv sync`

2) 激活虚拟环境

`source .venv/bin/activate`

3) 训练模型

```
sh train.sh
```

4) 生成预测结果

```
sh test.sh
```

---

## 5. 常见问题

1) `TA-Lib` 安装失败  
本项目特征工程依赖 `TA-Lib`，需要先安装系统层面的 `ta-lib` 库，再安装Python包。
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

2) 多进程相关问题  
`train.py` 与 `predict.py` 均在入口使用了 `spawn` 模式，Linux/macOS下请保持通过脚本入口运行（不要在交互式环境里直接多进程调用主逻辑）。

3) GPU/CPU自动选择  
代码会按 `CUDA -> MPS -> CPU` 顺序自动选择设备；无GPU时可直接CPU运行。

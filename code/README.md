# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去60个交易日的量价特征序列，以及独立的股票 ID；
- 模型：`StockTransformer`，使用股票 ID Embedding、时序编码和股票间注意力；
- 输出：ranking score 选择 Top-5，Allocation Head 分配相对权重，Exposure Head 决定总仓位，回归头提供辅助收益预测。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取历史行情数据（`data/stock_data.csv`）；
2. 做特征工程（39特征或`158+39`特征）；
3. 构建标签：未来收益率（代码中为 `open_t1` 到 `open_t5` 的相对收益）；
4. 默认用固定随机种子 `42` 构造三折 walk-forward 验证，每折训练/验证之间 purge 5 日；
5. 联合优化平滑 Listwise、RankNet Pairwise、Rank IC、原始收益回归、相对仓位和总仓位损失；
6. 验证期从末端每隔 5 个交易日抽取非重叠锚点，保存三组 OOF 输出并标定
   Allocation 与等权的混合比例；
7. 取三个最佳 epoch 的中位数与最小全量轮数的较大值，用全部标签有效样本
   重训一个最终模型。

三随机种子集成仍作为可选实验保留。只有将 `ensemble_enabled=True` 后，程序才会
使用 `ensemble_seeds`，运行三种子 × 三折并训练三个最终模型；默认训练不会产生九折开销。

---

## 2. 代码结构说明

### [config.py](config.py)
统一管理训练与推理参数，包括：
- 序列长度 `sequence_length`（默认60；配合最长60日窗口指标，最早时点可追溯
  到 `t-119`，显式窗口已覆盖约120个行情观测）；
- 模型超参数（`d_model`、`nhead`、`num_layers` 等）；
- 训练超参数（`batch_size`、`max_epochs`、`patience`、`learning_rate`、`weight_decay`）；
- 排序与仓位损失参数（`pairwise_weight`、`top5_weight`、`base_weight`、
  `allocation_weight`、`exposure_weight` 及相应 temperature）；
- 多折切分、ID 正则化与辅助回归参数（`num_folds`、`validation_months`、`purge_days`、
  `id_dropout`、`embedding_dropout`、`regression_weight`）；
- 单种子/可选多种子模式、周频验证与 OOF 策略网格（`seed`、
  `ensemble_enabled`、`ensemble_seeds`、`evaluation_stride`、
  `allocation_blend_grid`、`disagreement_gamma_grid`）；
- CUDA AMP、TF32、fused AdamW、pinned memory 和 non-blocking 传输开关；
- 数据路径和输出路径（默认输出到 `output/`）。

### [model.py](model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- 股票 ID Embedding：训练时随机将部分 ID 替换成 UNK，并对 embedding 向量做 dropout；
- `CrossStockAttention`：使用 padding mask 建模同日股票间关系；
- `score_head` 与 `return_head`：分别输出排序分数和原始收益预测；
- `allocation_head`：输出 Top-5 内的相对仓位 logits；
- `exposure_head`：输出 `[0.20, 0.999999]` 内的总股票仓位，现金恒为 `1-exposure`。

输入包含特征张量、股票索引和有效股票 mask；模型返回排序分数、预测收益、
相对仓位 logits 三个 `[batch, num_stocks]` 张量，以及一个 `[batch]` 总仓位张量。

### [utils.py](utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：精简技术指标特征（兼容名称 `39`）；
- `engineer_features()`：Alpha 类特征；
- `engineer_features_158plus39()`：合并后的 171 列时序特征（兼容名称 `158+39`）；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。
- rank ensemble、OOF 对齐、收益分解和策略网格标定函数。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。
原 `RSQR5/10/20/30/60` 的滚动索引实现会使绝大部分结果变成 NaN 后填 0，
现已从特征计算和训练/推理特征表中删除。另删除 20 个可由保留列精确恢复的特征：
`IMXD=IMAX-IMIN`、`CNTD=CNTP-CNTN`、`SUMD=SUMP-SUMN`、
`VSUMD=VSUMP-VSUMN`（每组各 5 个窗口）。`158+39` 名称仅为兼容旧配置保留。

### [train.py](train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：在完整历史上按股票计算严格因果特征和标签；
	- `build_walk_forward_folds()`：按实际交易日构造扩展窗口验证折和 5 日 purge。
	- 每折最多训练 `max_epochs`，验证指标连续 `patience` 轮无提升时提前停止；
	- 默认固定随机种子42完成三折训练；checkpoint 只使用从验证期末向前每隔
	  5 个交易日抽取的非重叠日期；
	- 三折完成后统一选择最终 epoch，重新拟合一个全量 scaler，并训练一个最终模型；
	- 将 `ensemble_enabled=True` 可恢复三种子 × 三折及三个全量模型的稳健性实验；
	- 当前使用 `learning_rate=3e-5`、`patience=12` 和 `id_dropout=0.1`，让验证折有更充分的改善机会，同时减弱股票 ID 正则化。
	- CUDA 上使用 AMP 加速 Transformer 前向，排序和仓位损失保持 FP32；同时按配置启用 TF32、fused AdamW、pinned memory 与 non-blocking 传输。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`WeightedRankingLoss`
	- 将整数排名转成排名百分位，用 `listwise_temperature` 平滑目标分布，并通过 `listwise_weight` 控制 Listwise 尺度；
	- 组合归一化 `listwise_loss`、RankNet `pairwise_loss`、Rank IC 相关性损失与原始收益 SmoothL1 辅助损失；
	- 对真实Top-k样本施加更高权重。
	- TensorBoard 分别记录六个加权损失分量，便于检查目标是否失衡。
	- `allocation_loss` 监督 ranking head 当前 Top-5 内的相对仓位分布；
	- `exposure_loss` 根据该 Top-5 的真实平均收益监督总仓位。
- 评估指标：`calculate_ranking_metrics()`
	- 计算等权 Top-5 收益、动态权重组合收益、总仓位、现金、最大单股仓位、Rank IC、回归 MAE 和原有归一化指标；
	- 分离等权满仓、等权同仓位、Allocation 与 Exposure 的收益贡献；
	- 汇总均值、最差折、P10、标准差、正收益率、下行波动及模型排名分歧。
	- checkpoint 使用 `weighted_portfolio_return + checkpoint_rank_ic_weight * rank_ic` 组合指标，使选中股票后的权重头也参与模型选择。

训练产物：
- `seed_42/fold_N/`：默认三组 checkpoint、scaler、指标、日志与 OOF 输出；
- `seed_42/best_model.pth`：默认单个全量重训模型；
- `scaler.pkl`：全量标准化器；
- `ensemble_policy.json`：训练模式、模型路径及 OOF 选择的仓位策略；
- `stockid2idx.json`：训练与推理共用的股票 ID 映射；
- `cross_validation_summary.json`：多折汇总；
- `config.json`：训练时配置快照；
- `fold_N/log/`：逐折 TensorBoard 日志。

### [predict.py](predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程；
3. 从模型目录加载训练时 `config.json` 和全量 `scaler.pkl`，源码参数漂移只报告、不参与模型构造；
4. 默认加载一个全量模型；启用集成实验时加载多个模型，并将各自 ranking score
   转为横截面百分位后求均值；
5. 选择 Top-5。集成模式会平均各模型 Allocation 分布，并可按模型排名分歧将
   总仓位向 0.20 收缩，输出到 `result.csv`：
	 - `stock_id`
	 - `weight`（5只股票之和严格位于 `[0.20, 0.999999]`）

写出前后都会检查列名、股票数量、股票唯一性、候选范围、权重有限性和权重和，
避免浮点序列化导致提交权重超过 1。现金不写入股票行，隐含权重为
`1 - sum(weight)`。另写出 `output/prediction_diagnostics.json`，记录每个模型的
Top-5、排名分歧、策略参数、股票仓位和现金。

`sequence_length=60` 不代表模型只看到两个月的信息：单日输入已包含最长60日窗口
特征，再拼接60个时点后，最早的显式输入可追溯到`t-119`，覆盖约120个行情观测；
EMA 类特征还带有更早历史的衰减影响。直接改成90或120会增加显存、训练时间并减少
可用样本，且不能直接解决市场阶段反转。若后续比较序列长度，建议固定特征、损失与
随机种子，用单种子三折分别比较40/60/90，而不是仅看一次全量训练。

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
CUDA 默认启用 AMP、TF32、fused AdamW、pinned memory 和 non-blocking
传输；若 8 GB 显存仍出现 OOM，只需将 `batch_size` 从 8 降回 4，不要关闭
损失 FP32 或修改模型维度。

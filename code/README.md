# THU-BigDataCompetition-2026-baseline

本项目是一个面向沪深300成分股的**排序学习选股**方案：
- 输入：每只股票过去60个交易日的量价特征序列，以及独立的股票 ID；
- 模型：`StockTransformer`，使用受门控约束的股票 ID Embedding、时序编码和股票间注意力；
- 输出：ranking score 选择 Top-5，Allocation Head 分配相对权重，Exposure Head 决定总仓位，回归头提供辅助收益预测。

---

## 1. 项目目标与整体流程

核心目标是学习“当天应优先持有哪些股票”的排序函数，而不是单只股票二分类。

训练与推理主流程如下：
1. 读取五年历史行情数据（当前为 `data_5y/train.csv`）与行业历史快照；
2. 做单股量价特征工程，并增加短期反转、下行波动、市场压力和行业 as-of 残差特征；
3. 构建5日收益主标签、1/3日软下跌标签和市场状态软标签；
4. v1.17 用固定随机种子 `42` 构造三折 walk-forward 验证、2个月 lockbox，每折训练/验证之间 purge 5 日；v1.18 只严格重放这些候选工件的策略层；
5. Ranking阶段仅优化平滑Listwise、收益差加权LambdaRank@5、Rank IC和原始收益
   回归；随后冻结Ranking主干，独立训练1/3/5日软风险头、5日尾部事件头和市场状态门控；
6. 验证期从末端每隔 5 个交易日抽取非重叠锚点；策略晋级采用三折嵌套 OOF，
   每次只用另外两折依次标定 Ranking、Allocation、Exposure，并以配对收益、
   P10和最差折门控各模块，再在完全未参与决策的留出折评估；
7. 每折依次训练 Ranking、Risk/Regime、Allocation、Exposure；三折后把各阶段
   最佳optimizer更新步数中位数换算成全量epoch，进行四阶段全量重训。

三随机种子集成仍作为可选实验保留。只有将 `ensemble_enabled=True` 后，程序才会
使用 `ensemble_seeds`，运行三种子 × 三折并训练三个最终模型；默认训练不会产生九折开销。

### v1.18 严格 OOF 收益优先重放

v1.18 固定复用 v1.17 candidate 的 205 维 Transformer、scaler、三折 OOF 和已保存的
LightGBM 折模型，输出到独立的
`model/60_158+39_reduced25_relmarket12_risk15_indresid12_returnfirst_strict_oof_v18_policy/`。
运行 `POLICY_ONLY=1 ./train.sh` 不会出现训练 epoch：缺少时仅用已保存的 LightGBM 折模型
重建并原子缓存 OOF 分数。F1/F2 强制只用 Transformer；F3 只能用 F1/F2 完整标签选择融合
权重；部署权重取最后一个合法前向折。风险惩罚、反转、相关性降仓和相关簇替换均关闭；
Allocation Head 保留 25%，Exposure 为 25% Head 加 75% 近满仓 fallback。

`cross_validation_summary.json` 同时记录 v1.17 baseline、历史 v1.17 candidate 和校正后
v1.18 的逐折对比。晋级仅比较三折动态组合收益：均值至少 +10bp、至少两折正增益、P10 与
最差折不差于 -10bp、Rank IC 不低于 -0.005。通过后才可进行一次 lockbox 验收；`test.csv`
不参与该选择过程。

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
  `id_dropout`、`embedding_dropout`、`id_gate_init`、`id_gate_regularization`、
  `regression_weight`）；
- 单种子/可选多种子模式、周频验证与 OOF 策略网格（`seed`、
  `ensemble_enabled`、`ensemble_seeds`、`evaluation_stride`、
  `allocation_blend_grid`、`disagreement_gamma_grid`、
  `selection_risk_gamma_grid`、`risk_score_penalty_grid`、
  `correlation_exposure_gamma_grid`、`exposure_head_blend_grid`）；
- CUDA AMP、TF32、fused AdamW、pinned memory 和 non-blocking 传输开关；
- 数据路径和输出路径（默认输出到 `output/`）。

### [model.py](model.py)
定义核心模型 `StockTransformer`，主要由以下模块组成：
- `PositionalEncoding`：时序位置编码；
- 时序编码器 `TransformerEncoder`：提取单股票历史序列表示；
- `FeatureAttention`：对时间维特征做注意力聚合；
- 股票 ID Embedding：训练时随机将部分 ID 替换成 UNK，并对 embedding 向量做
  dropout；可学习门控初值为0.20并带平方正则，限制 ID 分支影响；
- `CrossStockAttention`：使用 padding mask 建模同日股票间关系；
- `score_head` 与 `return_head`：分别输出5日Alpha分数和原始收益预测；
- `risk_1d_head`、`risk_3d_head`、`risk_5d_head`：分别预测1、3、5日软下跌概率；
- `tail_5d_head`：直接预测 `future_return_5d <= -3%`，四个风险头按
  `0.15/0.20/0.30/0.35` 固定融合；
- 独立 `regime_market_encoder`：用未来市场平均收益和跌超3%的横截面扩散率监督
  市场压力门控；风险强度不再固定写入模型分数，而是仅通过OOF网格校准；
- `allocation_head`：输出相对仓位 logits，训练监督覆盖预测 Top-20，最终只在
  风险感知 Top-5 中重新 softmax；
- `exposure_head`：将股票池聚合表示、13维市场序列及Top-5分数离散度拼接，并通过
  正系数单调扣减市场压力和Top-5风险，输出`[0.20, 0.999999]`内的总股票仓位；
  OOF再校准其与固定仓位基线的混合比例，现金恒为`1-exposure`。

输入包含特征张量、股票索引和有效股票 mask；模型返回排序分数、预测收益、
相对仓位 logits 三个 `[batch, num_stocks]` 张量，以及一个 `[batch]` 总仓位张量。

### [utils.py](utils.py)
包含特征工程与数据集构建逻辑：
- `engineer_features_39()`：精简技术指标特征（兼容名称 `39`）；
- `engineer_features()`：Alpha 类特征；
- `engineer_features_158plus39()`：合并后的 171 列时序特征（兼容名称 `158+39`）；
- `add_relative_market_features()`：在单股特征合并后增加7个同日横截面百分位和
  5个市场状态特征；所有输入只依赖当前及过去行情；
- `create_ranking_dataset_vectorized()`：向量化构建按日排序样本（训练核心加速点）。
- rank ensemble、OOF 对齐、收益分解、分阶段策略标定及模块稳健门控函数；
- 风险感知 Top-5：反转软惩罚仍可在排名候选中校准；相关簇硬约束默认关闭，
  启用时仅能在原始 Top-10 内替换。若 Top-10 无法在每簇最多2只的前提下选满
  Top-5，当日完整保留原始 Top-5，不再向 Top-30 之外扩池。

说明：特征工程使用了 `TA-Lib`，若未正确安装会报错。
原 `RSQR5/10/20/30/60` 的滚动索引实现会使绝大部分结果变成 NaN 后填 0，
现已从特征计算和训练/推理特征表中删除。另删除 20 个可由保留列精确恢复的特征：
`IMXD=IMAX-IMIN`、`CNTD=CNTP-CNTN`、`SUMD=SUMP-SUMN`、
`VSUMD=VSUMP-VSUMN`（每组各 5 个窗口）。`158+39` 名称仅为兼容旧配置保留。
当前 `158+39_reduced25_relmarket12` 使用166个 `reduced25` 输入，加上：

- 5/20/60日收益、20日波动率、20日量比、MA20距离和MA60距离的同日横截面百分位；
- 市场等权5/20日收益、上涨家数占比、MA20以上家数占比和20日收益离散度。

共178个连续输入。横截面只使用同一交易日可观测股票，市场状态值广播给当天所有股票；
每折 StandardScaler 仍只使用该折训练期拟合。

`158+39_reduced25_relmarket12_risk15` 在上述178维上继续增加：

- 1/3日收益、5日相对20/60日动量差、5/20日下行波动和20日回撤的横截面百分位；
- 市场1/3日收益、5/20日下行波动、20日回撤、5日宽度变化、5日MA20宽度变化和
  20日市场拥挤度。

共193个连续输入，全部只依赖预测日及以前行情。

当前 v1.17 的 `158+39_reduced25_relmarket12_risk15_indresid12` 保留这193项，再增加12项
行业 as-of 输入，共205维：相对市场、相对所属行业的1/3/5日收益残差，以及它们各自的同日横截面百分位。
字段固定为 `indresid_market_return_{1,3,5}`、`indresid_industry_return_{1,3,5}`、
`indresid_market_return_{1,3,5}_pct` 和 `indresid_industry_return_{1,3,5}_pct`。行业代码、名称和
未来快照不进入 Transformer；每条记录只连接 `effective_date <= 日期` 的最后一份行业快照。

### v1.17 工件复现协议

205维 v1.17 工件根目录必须包含 `artifact_manifest.json`，并与 `config.json`、`scaler.pkl`、
`stockid2idx.json` 和每个 checkpoint 一起保留。manifest 至少包含：
`feature_num`、`feature_count`（205）、`sequence_length`、`stock_mapping_size`，以及
`architecture` 中的 `d_model`、`nhead`、`num_layers`、`dim_feedforward`。推理会交叉检查
manifest、Scaler 宽度、checkpoint 的 `input_proj.weight` 宽度和股票 embedding 行数；任意一项
不一致即拒绝运行。旧 v1.16 的193维工件没有 manifest 时仍可推理，但不能与205维策略、Scaler
或 checkpoint 混用。

### [train.py](train.py)
训练主脚本，关键内容：
- 数据预处理：
	- `_preprocess_common()`：在完整历史上按股票计算严格因果特征和标签；
	- `build_walk_forward_folds()`：按实际交易日构造扩展窗口验证折和 5 日 purge。
	- 每折最多训练 `max_epochs`，验证指标连续 `patience` 轮无提升时提前停止；
	- 默认固定随机种子42完成三折训练；checkpoint 只使用从验证期末向前每隔
	  5 个交易日抽取的非重叠日期；
	- 五年Ranking样本按504个交易日半衰期加权，降低过旧市场阶段的影响；
	- 三折完成后按最佳更新步数选择 Ranking、Risk、Allocation、Exposure 的最终
	  轮数，重新拟合全量 scaler 并按相同顺序训练最终模型；
	- 将 `ensemble_enabled=True` 可恢复三种子 × 三折及三个全量模型的稳健性实验；
	- 当前使用 `learning_rate=3e-5`、`patience=12`、`id_dropout=0.2`、
	  `embedding_dropout=0.1` 和 ID 门控，降低股票代码记忆风险。
	- CUDA 上使用 AMP 加速 Transformer 前向，排序和仓位损失保持 FP32；同时按配置启用 TF32、fused AdamW、pinned memory 与 non-blocking 传输。
- 数据集组织：
	- `RankingDataset` + `collate_fn`：处理每日股票数量不一致问题（padding + mask）。
- 损失函数：`WeightedRankingLoss`
	- 将整数排名转成排名百分位，用 `listwise_temperature` 平滑目标分布，并通过 `listwise_weight` 控制 Listwise 尺度；
	- LambdaRank只处理预测Top-20、真实Top-20和20只困难负样本，按
	  `ΔNDCG@5 × 真实收益差` 加权；
	- Ranking阶段只训练Rank IC、收益回归及排序目标；风险BCE和状态门控BCE在
	  冻结Ranking主干后独立训练，避免多任务负迁移；
	- 对真实Top-k样本施加更高权重。
	- TensorBoard分别记录各加权损失分量与ID门控正则，便于检查目标是否失衡；
	- `allocation_loss` 监督 ranking head 当前 Top-20 内的相对仓位分布，目标收益
	  先裁剪至 `[-10%, 10%]`；
	- `exposure_loss` 使用 Top-5 收益、全市场收益和 Top-5 下行波动构造软目标，
	  并以 BCE 训练总仓位。
	- Allocation阶段冻结Ranking与Risk；Exposure阶段冻结Ranking与Allocation，
	  避免两个弱辅助目标反向破坏选股表示。
- 评估指标：`calculate_ranking_metrics()`
	- 计算等权 Top-5 收益、动态权重组合收益、总仓位、现金、最大单股仓位、Rank IC、回归 MAE 和原有归一化指标；
	- 分离等权满仓、等权同仓位、Allocation 与 Exposure 的收益贡献；
	- 汇总均值、最差折、P10、标准差、正收益率、下行波动、组合相关性、反转风险
	  及模型排名分歧；
	- 每折最佳 checkpoint 额外执行真实 ID、全 UNK 和固定置换 ID 评估，报告分数
	  相关性、Top-5 重合率及收益变化。
	- Ranking、Risk、Allocation、Exposure分别使用`Top-5+0.2×Rank IC`、负验证
	  损失、同仓位Allocation增益和风险调整组合收益选择checkpoint。

训练产物：
- `seed_42/fold_N/`：默认三组 checkpoint、scaler、指标、日志与 OOF 输出；
- `seed_42/best_model.pth`：默认单个全量重训模型；
- `scaler.pkl`：全量标准化器；
- `ensemble_policy.json`：训练模式、模型路径及 OOF 选择的仓位策略；
- `stockid2idx.json`：训练与推理共用的股票 ID 映射；
- `cross_validation_summary.json`：多折汇总；
- `config.json`：训练时配置快照；
- `fold_N/log/`：逐折 TensorBoard 日志。

v1.16 的历史策略输出目录为
`model/60_158+39_reduced25_relmarket12_risk15_nested_oof_forward_policy_v7_1/`。
该目录不复制模型；`artifact_source_dir` 指向
`nested_oof_diverse_tailregime_v6_decay5y` 的 checkpoint、scaler、股票映射和
训练配置。`ensemble_policy.json` 同时保留严格前向策略、全 OOF 候选策略及经过
历史折模块资格过滤的实际部署策略。Fold 1 使用保守预热策略，Fold 2 只使用
Fold 1 已完成标签，Fold 3 只使用 Fold 1–2 已完成标签。

### [predict.py](predict.py)
推理主脚本，流程：
1. 加载历史数据，取最新交易日；
2. 执行与训练一致的特征工程；
3. 从策略目录加载部署参数，并通过 `artifact_source_dir` 从训练工件目录加载
   训练配置、全量 `scaler.pkl`、股票映射与 checkpoint；
4. 默认加载一个全量模型；启用集成实验时加载多个模型，并将各自 ranking score
   转为横截面百分位后求均值；
5. 先按稳健部署策略决定是否应用风险、反转和相关簇模块；相关簇启用时严格限制
   在原始 Top-10，无法满足时原样回退。交叉拟合 OOF 是唯一晋级依据，全量 OOF
   候选还必须通过跨折模块资格检查才能进入部署策略。集成模式会
   平均各模型 Allocation 分布，并可按模型排名分歧将总仓位向0.20收缩，输出到
   `result.csv`：
	 - `stock_id`
	 - `weight`（5只股票之和严格位于 `[0.20, 0.999999]`）

v1.17 在特征工程前还会验证 manifest、205维特征表、Scaler、股票映射和 checkpoint，防止
不同输入维度的工件混用。写出前后都会检查列名、股票数量、股票唯一性、候选范围、权重有限性和权重和，
避免浮点序列化导致提交权重超过 1。现金不写入股票行，隐含权重为
`1 - sum(weight)`。另写出 `output/prediction_diagnostics.json`，记录每个模型的
Top-5、原始 Top-5、选择前后名次、相关簇编号、ID 消融、反转风险、组合相关性、
策略参数、1/3/5日软风险、5日尾部事件风险、市场状态门控、股票仓位和现金。v1.17 另记录逐日
组合状态、行业集中度（as-of 行业 HHI/权重）、资金约束和融合风险诊断；缺少行业历史时只跳过该
诊断，不改变选股结果。

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
- `data_5y/train.csv`

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

3) 训练当前 v1.17 工件（会生成205维 manifest、Scaler 和 checkpoint）

```
./train.sh
```

开发比较先运行 `V17_PROFILE=baseline ./train.sh`，再运行默认 candidate；二者都冻结末两
个自然月。只有锁箱验收完成后，才允许以固定配置执行
`V17_INCLUDE_LOCKBOX=1 LOCKBOX_ACCEPTED=1 ./train.sh` 的部署重训，禁止据此再调参。

4) 生成预测结果

```
./test.sh
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
传输；若 8 GB 显存仍出现 OOM，可将 `batch_size` 从 12 降至 8 或 4，不要关闭
损失 FP32 或修改模型维度。

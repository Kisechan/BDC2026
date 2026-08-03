# BDC2026 — 沪深300 排序学习选股方案

> [!NOTE]
> 本 README 文档大部分内容由 AI 编写。

[第十一届「中国高校计算机大赛—大数据挑战赛」(BDC2026)](https://www.heywhale.com/org/r9xi8/competition/area/69c0dfa34f302f8f0122e1bb/content) 参赛项目，在 [THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026) 官方基线（`StockTransformer` 排序选股）上迭代改进，最终以 **LightGBM LambdaRank 严格 Top-5 选股** 方案参赛。

- **赛题**：基于沪深300成分股历史日线行情，预测未来一周收益最大的 ≤5 只股票组合（T+1 开盘买入、T+5 开盘卖出），权重累加和 ≤ 1，按组合收益率 `R_total = Σ wᵢ × (P_open,T+5 − P_open,T+1) / P_open,T+1` 排名。
- **提交格式**：`result.csv`，仅两列 `stock_id,weight`（UTF-8）。
- **比赛文档**：官方大赛介绍、赛制、提交评审等资料存放在本地 `docs/`（未纳入版本库），可到[比赛页面](https://www.heywhale.com/org/r9xi8/competition/area/69c0dfa34f302f8f0122e1bb/content)查看。

## 技术方案

当前部署方案为 **v1.22（严格前向 Allocation 权重）**，选股链路：

| 阶段 | 方案 |
| --- | --- |
| 特征 | 205 维量价特征 `158+39_reduced25_relmarket12_risk15_indresid12`（TA-Lib 技术指标 + 横截面百分位 + 市场状态 + 行业 as-of 残差，全部严格因果） |
| 选股 | LightGBM `LGBMRanker(lambdarank)`，每日按官方 `open_t5/open_t1 − 1` 恰好前五名的 relevance=1，优化 `ndcg@5` |
| 权重 | 冻结 v1.20.1 Transformer 的 Allocation Head 重放，候选仅等权 / 25% / 50% 混合，须通过 +10bp 门槛与护栏，否则回退等权 |
| 验证 | 三折 walk-forward、5 日 purge、近期 504 日窗口、内层 40 日早停、固定种子 42；嵌套 OOF 严格前向标定（Fold N 只使用更早折的已解析标签） |
| 部署 | 独立最终目录，只以截至 2026-07-24 的已完成标签拟合，2026-07-31 仅作推理 as-of |

方案演进（各版本实验均保留独立模型目录与 OOF 报告）：

- **v1.16–v1.20**：基线 Transformer 改进——股票 ID 门控 Embedding、CrossStockAttention、RankGLU 排序头、1/3/5 日软风险头、尾部事件头、市场状态门控、Allocation/Exposure 头，特征从 171 维逐步扩到 205 维，策略经嵌套 OOF 前向策略标定。
- **v1.21**：严格 Top-5 LambdaRank，纯 LightGBM 选股 + 等权满仓，冻结 Transformer 只提供特征/推理工件。
- **v1.22（当前）**：保留 v1.21 选股，前向校准 Allocation Head 混合权重。

## 目录结构

```
├── code/
│   ├── code/src/       # config.py / model.py / train.py / predict.py / utils.py / report_metrics.py
│   ├── asset/          # 使用说明截图
│   ├── test/           # 回归测试
│   ├── get_stock_data.py  # Baostock 数据抓取（沪深300 日线）
│   ├── train.sh / test.sh # 训练 / 推理入口
│   ├── Dockerfile / docker-compose.yml  # 比赛 docker 复现环境
│   └── pyproject.toml / uv.lock
└── AGENTS.md           # 项目工程约定
```

> 行情数据、训练工件与推理输出（`data*/`、`model/`、`output/`、`temp/` 等）由 `.gitignore` 排除，不入库：数据需自行用 `get_stock_data.py` 抓取（或放置五年训练集 `data_5y/train.csv`），训练与推理产物在本地按实验目录生成。

## 快速开始

依赖 `uv` 与 TA-Lib（需先安装系统库 `ta-lib-0.4.0`）。

```bash
cd code
uv sync
source .venv/bin/activate

./train.sh          # 训练当前 v1.22 工件（三折内层早停 → 外层重训 → OOF 评估）
./test.sh           # 生成 output/result.csv 并打印持仓（股票代码/名称/权重）
```

训练在双卡 RTX 2080Ti 服务器上约 50 分钟完成（当前 v1.22 为纯 LightGBM 重训，不含 Transformer 训练；如需完整四阶段 Transformer 训练，参考 `code/code/src/train.py` 的 `ensemble_enabled` 与历史版本说明）。

最终提交重训（写入隔离部署目录）：

```bash
FINAL_SUBMISSION_FIT=1 FINAL_SUBMISSION_DATE=2026-07-31 ./train.sh
```

## 关键约定

- `result.csv` 严格两列 `stock_id,weight`，现金隐含为 `1 − Σweight`，不写入股票行。
- 2026-07-31 持仓尚无完成标签，其真实收益不得用于任何调参、验证或权重选择；所有部署决策只使用更早折的已解析 OOF 标签。
- 模型工件按实验分目录存放，不同架构/输入维度的 checkpoint 不可混用（推理时校验 manifest、Scaler 宽度与输入维度）。
- 训练/推理自动选择 `CUDA → MPS → CPU`；CUDA 下默认启用 AMP、TF32 等加速开关。

## 致谢

- 官方基线：[Sherlock1956/THU-BDC2026](https://github.com/Sherlock1956/THU-BDC2026)
- 赛事平台：[和鲸 Heywhale](https://www.heywhale.com/) 与 清华大学 · 大数据系统软件国家工程研究中心

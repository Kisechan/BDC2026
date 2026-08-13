#!/usr/bin/env bash
# 赛事推理入口。依赖已在镜像构建阶段安装，运行时禁止联网和依赖同步。
set -euo pipefail

cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# 最终提交工件存在时，默认验证它；开发期 candidate 仍可通过
# MODEL_OUTPUT_DIR 显式指定。这样容器的 data/run.sh 直接执行 test.sh
# 不会意外回退到仅用于 OOF 的候选树模型。
if [ -z "${MODEL_OUTPUT_DIR:-}" ] && [ -d "./model/60_158+39_reduced25_relmarket12_risk15_indresid12_v1.22_submission_2026-07-31" ]; then
  export MODEL_OUTPUT_DIR="./model/60_158+39_reduced25_relmarket12_risk15_indresid12_v1.22_submission_2026-07-31"
  echo "使用 v1.22 最终提交工件: ${MODEL_OUTPUT_DIR}"
fi

if [ -n "${HISTORICAL_SCORE_DATE:-}" ]; then
  : "${PREDICTION_DATE:=${HISTORICAL_SCORE_DATE}}"
  : "${PREDICTION_OUTPUT_DIR:=./output/historical_${HISTORICAL_SCORE_DATE}}"
  export PREDICTION_DATE PREDICTION_OUTPUT_DIR
  echo "阶段 历史官方评分：${HISTORICAL_SCORE_DATE} as-of 推理，输出 ${PREDICTION_OUTPUT_DIR}"
fi

python code/src/predict.py
python code/src/report_metrics.py

set -eu

if [ -n "${HISTORICAL_SCORE_DATE:-}" ]; then
  : "${PREDICTION_DATE:=${HISTORICAL_SCORE_DATE}}"
  : "${PREDICTION_OUTPUT_DIR:=./output/historical_${HISTORICAL_SCORE_DATE}}"
  export PREDICTION_DATE PREDICTION_OUTPUT_DIR
  echo "阶段 历史官方评分：${HISTORICAL_SCORE_DATE} as-of 推理，输出 ${PREDICTION_OUTPUT_DIR}"
fi

uv run --locked python code/src/predict.py
uv run --locked python code/src/report_metrics.py

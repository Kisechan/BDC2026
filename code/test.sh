set -eu

uv run --locked python code/src/predict.py
uv run --locked python code/src/report_metrics.py

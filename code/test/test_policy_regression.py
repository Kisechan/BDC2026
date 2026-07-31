import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from report_metrics import _official_open_window  # noqa: E402
from train import lgbm_relevance_labels, lockbox_realized_return  # noqa: E402
from utils import attach_label_end_dates  # noqa: E402


class PolicyRegressionTests(unittest.TestCase):
    def test_label_end_dates_follow_trading_calendar(self):
        records = [{"prediction_date": "2026-01-05"}]
        trading_dates = pd.bdate_range("2026-01-05", periods=7)
        enriched = attach_label_end_dates(
            records,
            trading_dates,
            horizon=5,
        )
        self.assertEqual(enriched[0]["label_end_date"], "2026-01-12")
        self.assertNotIn("label_end_date", records[0])

    def test_lgbm_relevance_uses_configured_official_return_label(self):
        frame = pd.DataFrame({
            '日期': ['2026-01-05'] * 4,
            'label': [-0.02, 0.01, 0.03, 0.00],
            'industry_neutral_label': [0.9, 0.1, 0.2, 0.3],
        })
        labels = lgbm_relevance_labels(frame, 'label').to_numpy()
        self.assertEqual(int(labels[2]), int(labels.max()))
        self.assertEqual(int(labels[0]), int(labels.min()))

    def test_lockbox_uses_open_to_open_official_return(self):
        raw = pd.DataFrame({
            '股票代码': ['000001', '000001'],
            '日期': pd.to_datetime(['2026-01-06', '2026-01-12']),
            '开盘': [10.0, 11.0],
            '收盘': [100.0, 1.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / 'result.csv'
            pd.DataFrame({'stock_id': ['000001'], 'weight': [1.0]}).to_csv(
                result_path, index=False,
            )
            realized, _ = lockbox_realized_return(
                result_path, raw,
                pd.Timestamp('2026-01-06'), pd.Timestamp('2026-01-12'),
            )
        self.assertTrue(np.isclose(realized, 0.10))

    def test_official_window_uses_first_and_fifth_future_opens(self):
        dates = pd.bdate_range('2026-01-05', periods=6)
        prices = pd.DataFrame({
            '股票代码': ['000001'] * len(dates),
            '日期': dates,
            '开盘': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        realized, entry_date, exit_date = _official_open_window(
            prices, dates[0], strict_calendar=True,
        )
        self.assertEqual(entry_date, dates[1])
        self.assertEqual(exit_date, dates[5])
        self.assertTrue(np.isclose(float(realized.iloc[0]['return']), 15.0 / 11.0 - 1.0))

    def test_current_forward_replay_fails_promotion(self):
        summary_path = (
            PROJECT_ROOT
            / "model"
            / (
                "60_158+39_reduced25_relmarket12_risk15_"
                "nested_oof_forward_policy_v7_1"
            )
            / "cross_validation_summary.json"
        )
        if not summary_path.exists():
            self.skipTest("先运行 POLICY_ONLY=1 ./train.sh 生成回放产物")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(
            summary["cross_fitted_oof"]["method"],
            "strict_forward_historical_folds_only",
        )
        self.assertFalse(summary["promotion_criteria"]["passed"])
        self.assertFalse(
            summary["promotion_criteria"]["mean_weighted_return"]
        )
        self.assertFalse(
            summary["promotion_criteria"]["p10_weighted_return"]
        )
        self.assertIn("worst_daily_rank_ic", summary)
        self.assertIn("worst_fold_mean_rank_ic", summary)
        self.assertIn("mean_tail_5d_brier_skill", summary)
        folds = summary["cross_fitted_oof"]["fold_policies"]
        self.assertEqual(folds[0]["calibration_mode"], "warmup_fallback")
        self.assertEqual(folds[0]["calibration_folds"], [])
        self.assertEqual(folds[1]["calibration_folds"], [1])
        self.assertEqual(folds[2]["calibration_folds"], [1, 2])


if __name__ == "__main__":
    unittest.main()

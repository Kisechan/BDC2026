import json
import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

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

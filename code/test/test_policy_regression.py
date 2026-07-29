import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from model import StockTransformer  # noqa: E402
from predict import validate_result  # noqa: E402
from train import WeightedRankingLoss  # noqa: E402
from utils import (  # noqa: E402
    attach_industry_indices_asof,
    attach_label_end_dates,
    build_industry_mapping,
    select_risk_aware_top_indices,
)


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

    def test_industry_asof_join_never_reads_future_snapshot(self):
        history = pd.DataFrame({
            "effective_date": pd.to_datetime([
                "2026-01-01",
                "2026-01-10",
            ]),
            "stock_id": ["000001", "000001"],
            "industry": ["旧行业", "新行业"],
            "industry_classification": ["申万", "申万"],
        })
        mapping = build_industry_mapping(history)
        panel = pd.DataFrame({
            "股票代码": ["000001", "000001"],
            "日期": pd.to_datetime(["2026-01-09", "2026-01-10"]),
        })
        joined, coverage = attach_industry_indices_asof(
            panel,
            history,
            mapping,
            minimum_coverage=1.0,
        )
        self.assertEqual(coverage, 1.0)
        self.assertEqual(joined["industry_name"].tolist(), ["旧行业", "新行业"])

    def test_zero_soft_penalties_exactly_reproduce_raw_top5(self):
        scores = 1.0 - np.arange(20) * 0.005
        momentum = np.zeros((20, 3), dtype=np.float64)
        returns = np.zeros((20, 20), dtype=np.float64)
        industries = np.repeat(np.arange(1, 5), 5)
        baseline = select_risk_aware_top_indices(
            scores,
            momentum,
            returns,
            candidate_k=10,
            industry_indices=industries,
            industry_penalty=0.0,
            soft_correlation_penalty=0.0,
        )
        self.assertEqual(
            baseline["top_indices"].tolist(),
            baseline["raw_top_indices"].tolist(),
        )
        diversified = select_risk_aware_top_indices(
            scores,
            momentum,
            returns,
            candidate_k=10,
            industry_indices=industries,
            industry_penalty=0.10,
        )
        self.assertTrue(set(diversified["top_indices"]).issubset(set(range(10))))
        self.assertLess(diversified["industry_hhi"], baseline["industry_hhi"])

    def test_industry_residual_loss_is_finite_and_differentiable(self):
        criterion = WeightedRankingLoss(
            industry_residual_ranking_weight=0.15,
        )
        scores = torch.randn(2, 8, requires_grad=True)
        returns = torch.randn(2, 8) * 0.02
        relevance = torch.argsort(
            torch.argsort(returns, dim=1),
            dim=1,
        ).float()
        industries = torch.tensor([
            [1, 1, 1, 2, 2, 2, 0, 0],
            [1, 1, 2, 2, 3, 3, 3, 3],
        ])
        loss, components = criterion(
            scores,
            relevance,
            torch.zeros_like(scores),
            returns,
            torch.zeros_like(scores),
            torch.full((2,), 0.5),
            industry_indices=industries,
            stage="ranking",
            return_components=True,
        )
        self.assertIn("industry_residual_ranking_loss", components)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_old_structure_loads_and_heads_always_exist(self):
        legacy_config = deepcopy(config)
        legacy_config["exposure_industry_summary_enabled"] = False
        legacy = StockTransformer(
            input_dim=193,
            config=legacy_config,
            num_stocks=10,
        )
        reloaded = StockTransformer(
            input_dim=193,
            config=legacy_config,
            num_stocks=10,
        )
        reloaded.load_state_dict(legacy.state_dict(), strict=True)
        self.assertTrue(hasattr(reloaded, "allocation_head"))
        self.assertTrue(hasattr(reloaded, "exposure_head"))

    def test_result_csv_constraints(self):
        output = pd.DataFrame({
            "stock_id": [f"{index:06d}" for index in range(5)],
            "weight": [0.1] * 5,
        })
        runtime_config = {"min_exposure": 0.20, "max_exposure": 0.999999}
        self.assertAlmostEqual(
            validate_result(
                output,
                [f"{index:06d}" for index in range(10)],
                runtime_config=runtime_config,
            ),
            0.5,
        )
        output.loc[0, "weight"] = 1.1
        with self.assertRaises(ValueError):
            validate_result(
                output,
                [f"{index:06d}" for index in range(10)],
                runtime_config=runtime_config,
            )


if __name__ == "__main__":
    unittest.main()

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from model import StockTransformer  # noqa: E402
from predict import validate_result  # noqa: E402
from train import (  # noqa: E402
    FrozenBackboneDataset,
    RankingDataset,
    WeightedRankingLoss,
    _build_label_and_clean,
    build_walk_forward_folds,
    calculate_checkpoint_score,
    collate_fn,
    configure_model_for_stage,
    detached_deployment_policy,
)
from utils import (  # noqa: E402
    attach_industry_indices_asof,
    attach_label_end_dates,
    build_industry_mapping,
    create_ranking_dataset_vectorized,
    select_risk_aware_top_indices,
)


class PolicyRegressionTests(unittest.TestCase):
    def test_batched_ranking_loss_matches_per_day_reference(self):
        torch.manual_seed(7)
        counts = [8, 6, 5]
        batch_size, max_items = len(counts), max(counts)
        masks = torch.zeros(batch_size, max_items, dtype=torch.bool)
        for index, count in enumerate(counts):
            masks[index, :count] = True
        relevance = torch.randn(batch_size, max_items)
        raw_returns = torch.randn(batch_size, max_items) * 0.03
        recency = torch.tensor([0.3, 0.7, 1.0])
        allocation = torch.zeros_like(relevance)
        exposure = torch.full((batch_size,), 0.6)
        criterion = WeightedRankingLoss(
            listwise_weight=0.2,
            pairwise_weight=1.0,
            regression_weight=0.05,
            ic_weight=0.2,
            id_gate_regularization=0.01,
            industry_residual_ranking_weight=0.0,
        )

        batched_scores = torch.randn(
            batch_size,
            max_items,
            requires_grad=True,
        )
        batched_returns = torch.randn(
            batch_size,
            max_items,
            requires_grad=True,
        )
        batched_gate = torch.tensor(0.2, requires_grad=True)
        batched_loss, batched_components = criterion(
            batched_scores,
            relevance,
            batched_returns,
            raw_returns,
            allocation,
            exposure,
            identity_gate=batched_gate,
            item_mask=masks,
            sample_weights=recency,
            stage="ranking",
            return_components=True,
        )
        batched_loss.backward()

        reference_scores = batched_scores.detach().clone().requires_grad_()
        reference_returns = (
            batched_returns.detach().clone().requires_grad_()
        )
        reference_gate = batched_gate.detach().clone().requires_grad_()
        reference_loss = 0.0
        reference_components = {}
        for index, count in enumerate(counts):
            day_loss, day_components = criterion(
                reference_scores[index:index + 1, :count],
                relevance[index:index + 1, :count],
                reference_returns[index:index + 1, :count],
                raw_returns[index:index + 1, :count],
                allocation[index:index + 1, :count],
                exposure[index:index + 1],
                identity_gate=reference_gate,
                stage="ranking",
                return_components=True,
            )
            reference_loss = reference_loss + recency[index] * day_loss
            for name, value in day_components.items():
                reference_components[name] = (
                    reference_components.get(name, 0.0)
                    + recency[index] * value
                )
        reference_loss = reference_loss / recency.sum()
        reference_components = {
            name: value / recency.sum()
            for name, value in reference_components.items()
        }
        reference_loss.backward()

        self.assertLess(
            abs(batched_loss.item() - reference_loss.item()),
            1e-6,
        )
        self.assertEqual(
            set(batched_components),
            set(reference_components),
        )
        for name in batched_components:
            self.assertLess(
                abs(
                    batched_components[name].item()
                    - reference_components[name].item()
                ),
                1e-6,
            )
        self.assertLess(
            (
                batched_scores.grad - reference_scores.grad
            ).abs().max().item(),
            1e-5,
        )
        self.assertLess(
            (
                batched_returns.grad - reference_returns.grad
            ).abs().max().item(),
            1e-5,
        )
        self.assertLess(
            abs(batched_gate.grad.item() - reference_gate.grad.item()),
            1e-5,
        )
        for index, count in enumerate(counts):
            self.assertEqual(
                torch.argsort(
                    batched_scores.detach()[index, :count],
                    descending=True,
                    stable=True,
                )[:5].tolist(),
                torch.argsort(
                    reference_scores.detach()[index, :count],
                    descending=True,
                    stable=True,
                )[:5].tolist(),
            )
        self.assertTrue(torch.isfinite(batched_loss))

    def test_frozen_backbone_batch_omits_raw_sequences(self):
        sequences = [
            np.arange(24, dtype=np.float64).reshape(2, 3, 4),
            np.arange(12, dtype=np.float32).reshape(1, 3, 4),
        ]
        targets = [
            np.asarray([0.1, -0.1], dtype=np.float32),
            np.asarray([0.2], dtype=np.float32),
        ]
        dataset = RankingDataset(
            sequences,
            targets,
            [np.asarray([1, 0]), np.asarray([1])],
            [np.asarray([2, 3]), np.asarray([4])],
            [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
        )
        first = dataset[0]["sequences"]
        second = dataset[0]["sequences"]
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(dataset.sequences[0].flags.c_contiguous)
        self.assertTrue(dataset.sequences[0].flags.writeable)

        cached = FrozenBackboneDataset(dataset, [
            {
                "cached_ranking_features": torch.zeros(2, 5),
                "cached_regime_sequence": torch.zeros(3, 2),
                "cached_market_sequence": torch.zeros(3, 2),
            },
            {
                "cached_ranking_features": torch.zeros(1, 5),
                "cached_regime_sequence": torch.zeros(3, 2),
                "cached_market_sequence": torch.zeros(3, 2),
            },
        ])
        self.assertNotIn("sequences", cached[0])
        batch = collate_fn([cached[0], cached[1]])
        self.assertNotIn("sequences", batch)
        self.assertEqual(
            tuple(batch["cached_ranking_features"].shape),
            (2, 2, 5),
        )
        cached_subset = FrozenBackboneDataset(
            Subset(dataset, [1]),
            [cached.cached_samples[1]],
        )
        self.assertNotIn("sequences", cached_subset[0])
        self.assertEqual(cached_subset[0]["targets"].numel(), 1)

    def test_holding_path_tail_detects_recovered_drawdown(self):
        rows = []
        paths = {
            "000001": [99, 100, 95, 101, 102, 101],
            "000002": [99, 100, 99, 100, 101, 102],
        }
        for stock_id, opens in paths.items():
            for offset, value in enumerate(opens):
                rows.append({
                    "股票代码": stock_id,
                    "日期": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(days=offset),
                    "开盘": value,
                })
        processed = _build_label_and_clean(pd.DataFrame(rows))
        targets = processed.set_index("股票代码")
        self.assertGreater(targets.loc["000001", "label"], 0.0)
        self.assertEqual(targets.loc["000001", "tail_5d_target"], 1.0)
        self.assertEqual(targets.loc["000002", "tail_5d_target"], 0.0)
        self.assertEqual(
            targets.loc["000001", "ranking_target"],
            targets.loc["000001", "label"],
        )
        self.assertEqual(
            targets.loc["000002", "ranking_target"],
            targets.loc["000002", "label"],
        )

    def test_ranking_checkpoint_penalizes_top5_downside(self):
        base_metrics = {
            "top5_return": 0.02,
            "rank_ic": 0.05,
            "top5_downside_deviation": 0.0,
        }
        safe_score = calculate_checkpoint_score(
            base_metrics,
            "risk_adjusted_top5_plus_rank_ic",
        )
        risky_score = calculate_checkpoint_score(
            {**base_metrics, "top5_downside_deviation": 0.04},
            "risk_adjusted_top5_plus_rank_ic",
        )
        self.assertGreater(safe_score, risky_score)

    def test_dataset_relevance_uses_downside_adjusted_target(self):
        data = pd.DataFrame({
            "instrument": np.arange(10),
            "日期": pd.Timestamp("2026-01-05"),
            "feature": 0.0,
            "label": np.arange(10, dtype=np.float32),
            "ranking_target": -np.arange(10, dtype=np.float32),
            "risk_1d_target": 0.5,
            "risk_3d_target": 0.5,
            "tail_5d_target": 0.0,
            "regime_target": 0.5,
        })
        parts = create_ranking_dataset_vectorized(
            data,
            ["feature"],
            sequence_length=1,
            minimum_industry_coverage=0.0,
        )
        self.assertNotEqual(
            int(np.argmax(parts[1][0])),
            int(np.argmax(parts[2][0])),
        )

    def test_six_walk_forward_folds_keep_five_day_purge(self):
        dates = pd.bdate_range("2021-07-20", "2026-07-13")
        folds = build_walk_forward_folds(
            pd.DataFrame({"日期": dates}),
            num_folds=6,
            validation_months=3,
            purge_days=5,
        )
        self.assertEqual(len(folds), 6)
        for fold in folds:
            train_position = dates.get_loc(fold["train_end"])
            validation_position = dates.get_loc(fold["val_start"])
            self.assertEqual(validation_position - train_position - 1, 5)
        for previous, current in zip(folds, folds[1:]):
            self.assertLess(previous["val_end"], current["val_start"])

    def test_tail_loss_uses_explicit_path_target(self):
        criterion = WeightedRankingLoss(tail_5d_weight=1.0)
        scores = torch.zeros(1, 2)
        loss, components = criterion(
            scores,
            scores,
            scores,
            torch.full_like(scores, 0.02),
            scores,
            torch.full((1,), 0.5),
            tail_5d_logits=torch.full_like(scores, 5.0),
            tail_5d_targets=torch.ones_like(scores),
            stage="risk",
            return_components=True,
        )
        self.assertIn("tail_5d_loss", components)
        self.assertLess(loss.item(), 0.01)

    def test_deployment_policy_can_embed_cross_fitted_report(self):
        cross_fitted = {
            "robust_deployment_policy": {"allocation_blend": 0.25},
            "metrics": {"mean_top5_return": 0.01},
        }
        policy = detached_deployment_policy(cross_fitted)
        policy["cross_fitted_oof"] = cross_fitted
        serialized = json.dumps(policy)
        self.assertIn("cross_fitted_oof", serialized)
        self.assertIsNot(
            policy,
            cross_fitted["robust_deployment_policy"],
        )

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

    def test_disabled_industry_residual_loss_is_not_computed(self):
        criterion = WeightedRankingLoss(
            industry_residual_ranking_weight=0.0,
        )
        scores = torch.randn(1, 8)
        returns = torch.randn(1, 8) * 0.02
        loss, components = criterion(
            scores,
            torch.arange(8).float().unsqueeze(0),
            torch.zeros_like(scores),
            returns,
            torch.zeros_like(scores),
            torch.full((1,), 0.5),
            industry_indices=torch.ones_like(scores, dtype=torch.long),
            stage="ranking",
            return_components=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("industry_residual_ranking_loss", components)

    def test_risk_stage_keeps_ranking_and_position_heads_frozen(self):
        model = StockTransformer(input_dim=193, config=config, num_stocks=10)
        configure_model_for_stage(model, "risk")
        trainable = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("risk_") for name in trainable))
        self.assertFalse(any(name.startswith("transformer") for name in trainable))
        self.assertFalse(any(name.startswith("allocation_head") for name in trainable))
        self.assertFalse(any(name.startswith("exposure_head") for name in trainable))

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

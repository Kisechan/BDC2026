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
from train import (  # noqa: E402
    fixed_equal_top5_policy,
    lgbm_relevance_labels,
    lockbox_realized_return,
    remap_oof_records_to_official_labels,
    strict_lgbm_inner_split,
)
from utils import (  # noqa: E402
    _runtime_module_fallbacks,
    attach_label_end_dates,
    build_ensemble_portfolio,
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

    def test_lgbm_relevance_uses_configured_official_return_label(self):
        frame = pd.DataFrame({
            '日期': ['2026-01-05'] * 4,
            'label': [-0.02, 0.01, 0.03, 0.00],
            'industry_neutral_label': [0.9, 0.1, 0.2, 0.3],
        })
        labels = lgbm_relevance_labels(frame, 'label').to_numpy()
        self.assertEqual(int(labels[2]), int(labels.max()))
        self.assertEqual(int(labels[0]), int(labels.min()))

    def test_top5_binary_relevance_is_stable_for_ties(self):
        from config import config

        previous_mode = config.get('lgbm_label_mode')
        previous_top_k = config.get('lgbm_top_k')
        config['lgbm_label_mode'] = 'top5_binary'
        config['lgbm_top_k'] = 5
        try:
            frame = pd.DataFrame({
                '日期': ['2026-01-05'] * 8,
                'instrument': [8, 7, 6, 5, 4, 3, 2, 1],
                'label': [0.01, 0.01, 0.05, 0.04, 0.03, 0.02, 0.00, -0.01],
            })
            labels = lgbm_relevance_labels(frame, 'label').to_numpy()
        finally:
            config['lgbm_label_mode'] = previous_mode
            config['lgbm_top_k'] = previous_top_k
        self.assertEqual(int(labels.sum()), 5)
        self.assertEqual(int(labels[0]), 0)  # 0.01 并列时 instrument=6 优先。
        self.assertEqual(int(labels[1]), 1)

    def test_inner_lgbm_split_has_explicit_purge(self):
        dates = pd.bdate_range('2026-01-01', periods=12)
        frame = pd.DataFrame({
            '日期': np.repeat(dates, 2),
            'instrument': np.tile([2, 3], len(dates)),
            'label': 0.01,
        })
        inner_train, inner_valid, boundary = strict_lgbm_inner_split(
            frame, validation_days=3, purge_days=2,
        )
        self.assertEqual(boundary['inner_train_end'], str(dates[6].date()))
        self.assertEqual(boundary['inner_purge_start'], str(dates[7].date()))
        self.assertEqual(boundary['inner_purge_end'], str(dates[8].date()))
        self.assertEqual(boundary['inner_val_start'], str(dates[9].date()))
        self.assertLess(inner_train['日期'].max(), inner_valid['日期'].min())

    def test_fixed_equal_policy_and_runtime_fallback_are_zero_blend(self):
        policy = fixed_equal_top5_policy()
        self.assertEqual(policy['allocation_blend'], 0.0)
        self.assertEqual(policy['exposure_head_blend'], 0.0)
        self.assertEqual(policy['fixed_exposure_baseline'], 0.999999)
        fallbacks = _runtime_module_fallbacks({
            'min_exposure': 0.2,
            'max_exposure': 0.999999,
            'allocation_temperature': 1.0,
            'minimum_allocation_blend': 0.0,
            'minimum_exposure_blend': 0.0,
        })
        self.assertEqual(fallbacks['allocation'], 0.0)
        self.assertEqual(fallbacks['exposure_head'], 0.0)

    def test_fixed_equal_portfolio_weights_are_exact(self):
        policy = fixed_equal_top5_policy()
        portfolio = build_ensemble_portfolio(
            np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]),
            np.zeros((1, 6)),
            np.array([0.5]),
            min_exposure=policy['min_exposure'],
            max_exposure=policy['max_exposure'],
            allocation_temperature=policy['allocation_temperature'],
            allocation_blend=policy['allocation_blend'],
            disagreement_gamma=0.0,
            selection_risk_gamma=0.0,
            risk_score_penalty=0.0,
            correlation_exposure_gamma=0.0,
            exposure_head_blend=policy['exposure_head_blend'],
            fixed_exposure_baseline=policy['fixed_exposure_baseline'],
            top_k=5,
        )
        self.assertTrue(np.allclose(portfolio['positions'], 0.999999 / 5))
        self.assertTrue(np.isclose(portfolio['positions'].sum(), 0.999999))

    def test_frozen_baseline_scores_are_replayed_with_official_labels(self):
        date = pd.Timestamp('2026-01-05')
        instruments = np.arange(2, 7, dtype=np.int64)
        data = pd.DataFrame({
            '日期': [date] * len(instruments),
            'instrument': instruments,
            'label': np.linspace(-0.02, 0.02, len(instruments)),
            'risk_1d_target': np.full(len(instruments), 0.5),
            'risk_3d_target': np.full(len(instruments), 0.5),
            'tail_5d_target': np.zeros(len(instruments)),
            'regime_target': np.full(len(instruments), 0.5),
        })
        source = {
            1: [{
                'prediction_date': '2026-01-05',
                'label_end_date': '2026-01-12',
                'stock_indices': instruments,
                'scores': np.arange(len(instruments), dtype=np.float64),
                'targets': np.full(len(instruments), 99.0),
            }]
        }
        replayed = remap_oof_records_to_official_labels(source, data)
        record = replayed[1][0]
        self.assertTrue(np.allclose(record['scores'], source[1][0]['scores']))
        self.assertTrue(np.allclose(record['targets'], data['label']))

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

"""Print a compact, reproducible model report after inference.

Cross-validation metrics are always available after training.  Realized-return
metrics are printed only when a local ``data/test.csv`` is present; in a real
submission those future prices are unavailable and the report deliberately
does not fabricate a score.
"""

import json
import os

import numpy as np
import pandas as pd

from config import config


def _pct(value: float) -> str:
    return f"{value:+.4%}"


def print_cross_validation() -> None:
    summary_path = os.path.join(config["output_dir"], "cross_validation_summary.json")
    if not os.path.exists(summary_path):
        print(f"\n[模型验证] 未找到交叉验证报告: {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

    seeds = summary.get("ensemble_seeds", [])
    cross_fitted = summary.get("cross_fitted_oof", {})
    validation_label = (
        "嵌套交叉拟合 OOF"
        if cross_fitted.get("method") != "disabled"
        else "OOF"
    )
    if len(seeds) > 1:
        print(
            f"\n[模型验证：{validation_label}，{len(seeds)} 随机种子 × "
            f"{int(summary.get('num_folds', 3))} 折 walk-forward，"
            f"每 {summary.get('evaluation_stride', 1)} 个交易日评估]"
        )
    else:
        seed_label = seeds[0] if seeds else "-"
        print(
            f"\n[模型验证：{validation_label}，单随机种子 {seed_label} × "
            f"{int(summary.get('num_folds', 3))} 折 walk-forward，"
            f"每 {summary.get('evaluation_stride', 1)} 个交易日评估]"
        )
    print(f"平均 Top-5 收益: {_pct(float(summary['mean_top5_return']))}")
    print(f"最差折 Top-5 收益: {_pct(float(summary['worst_fold_top5_return']))}")
    if "mean_weighted_portfolio_return" in summary:
        print(
            "平均动态权重组合收益: "
            f"{_pct(float(summary['mean_weighted_portfolio_return']))}"
        )
        print(
            "最差折动态权重组合收益: "
            f"{_pct(float(summary['worst_fold_weighted_portfolio_return']))}"
        )
        if 'max_drawdown' in summary:
            print(
                f"最差单日/最大回撤: "
                f"{_pct(float(summary['worst_weighted_portfolio_return']))} / "
                f"{_pct(float(summary['max_drawdown']))}"
            )
        print(
            f"平均股票仓位/现金: {float(summary['mean_gross_exposure']):.2%} / "
            f"{float(summary['mean_cash_weight']):.2%}"
        )
        if "p10_weighted_portfolio_return" in summary:
            print(
                "动态组合收益 P10/标准差/正收益率: "
                f"{_pct(float(summary['p10_weighted_portfolio_return']))} / "
                f"{float(summary['std_weighted_portfolio_return']):.4%} / "
                f"{float(summary['weighted_portfolio_positive_rate']):.2%}"
            )
            print(
                "平均 Allocation贡献/Exposure相对固定仓位贡献: "
                f"{_pct(float(summary['mean_allocation_contribution']))} / "
                f"{_pct(float(summary.get(
                    'mean_exposure_policy_contribution',
                    summary['mean_exposure_contribution'],
                )))}"
            )
            print(
                f"平均模型排名分歧: "
                f"{float(summary['mean_model_disagreement']):.4f}"
            )
    print(f"平均 Rank IC: {float(summary['mean_rank_ic']):+.4f}")
    if "worst_fold_mean_rank_ic" in summary:
        print(
            "最差日/最差折平均 Rank IC: "
            f"{float(summary['worst_daily_rank_ic']):+.4f} / "
            f"{float(summary['worst_fold_mean_rank_ic']):+.4f}"
        )
    ranking_baseline = summary.get("original_ranking_baseline", {})
    if ranking_baseline:
        print(
            "原始 Ranking 基线—平均收益/下行波动/目标: "
            f"{_pct(float(ranking_baseline['mean_top5_return']))} / "
            f"{float(ranking_baseline['top5_downside_deviation']):.4%} / "
            f"{_pct(float(ranking_baseline['ranking_policy_objective']))}"
        )
    if "mean_tail_5d_brier" in summary:
        print(
            "5日尾部风险均值/Brier、融合风险与未来Top-5收益相关: "
            f"{float(summary['mean_selected_tail_5d']):.4f} / "
            f"{float(summary['mean_tail_5d_brier']):.4f} / "
            f"{float(summary['combined_risk_return_spearman']):+.4f}"
        )
        if "mean_tail_5d_brier_skill" in summary:
            print(
                "5日尾部风险事件率/Brier Skill/ROC-AUC/PR-AUC: "
                f"{float(summary['mean_tail_5d_event_rate']):.4f} / "
                f"{float(summary['mean_tail_5d_brier_skill']):+.4f} / "
                f"{float(summary['mean_tail_5d_roc_auc']):.4f} / "
                f"{float(summary['mean_tail_5d_pr_auc']):.4f}"
            )
            print(
                "1/3/5日软风险 Brier Skill: "
                f"{float(summary['mean_risk_1d_brier_skill']):+.4f} / "
                f"{float(summary['mean_risk_3d_brier_skill']):+.4f} / "
                f"{float(summary['mean_risk_5d_brier_skill']):+.4f}"
            )
        print(
            "Regime Gate 与 Top-5/市场收益/尾部扩散相关: "
            f"{float(summary['regime_return_spearman']):+.4f} / "
            f"{float(summary['regime_market_return_spearman']):+.4f} / "
            f"{float(summary['regime_tail_share_spearman']):+.4f}"
        )
    if "raw_mean_positive_correlation" in summary:
        print(
            "原始/相关簇约束后平均正相关及选股收益变化: "
            f"{float(summary['raw_mean_positive_correlation']):.4f} / "
            f"{float(summary['mean_positive_correlation']):.4f} / "
            f"{_pct(float(summary['mean_diversification_return_contribution']))}"
        )
        print(
            "候选池平均/最大规模及扩展日期占比: "
            f"{float(summary['mean_effective_candidate_k']):.1f} / "
            f"{int(summary['max_effective_candidate_k'])} / "
            f"{float(summary['candidate_pool_expansion_rate']):.2%}"
        )
        if "cluster_constraint_application_rate" in summary:
            print(
                "相关簇约束应用率/跳过率/最大原始排名: "
                f"{float(summary['cluster_constraint_application_rate']):.2%} / "
                f"{float(summary['cluster_constraint_skip_rate']):.2%} / "
                f"{int(summary['max_selected_raw_rank'])}"
            )
    if "risk_score_penalty" in summary:
        print(
            "OOF Allocation混合/Exposure混合/风险分数惩罚/"
            "选择gamma/相关性降仓: "
            f"{float(summary.get('allocation_blend', 0.0)):.2f} / "
            f"{float(summary.get('exposure_head_blend', 1.0)):.2f} / "
            f"{float(summary['risk_score_penalty']):.2f} / "
            f"{float(summary.get('selection_risk_gamma', 0.0)):.2f} / "
            f"{float(summary.get('correlation_exposure_gamma', 0.0)):.2f}"
        )
    if cross_fitted.get("fold_policies"):
        print("嵌套 OOF 留出折策略:")
        for row in cross_fitted["fold_policies"]:
            selected = row["policy"]
            print(
                f"  Fold {row['held_out_fold']} <- calibration "
                f"{row['calibration_folds']}: "
                f"Allocation={float(selected['allocation_blend']):.2f}, "
                f"Exposure={float(selected['exposure_head_blend']):.2f}, "
                f"Risk={float(selected['risk_score_penalty']):.2f}, "
                f"Reversal={float(selected['selection_risk_gamma']):.2f}, "
                f"CorrExposure="
                f"{float(selected['correlation_exposure_gamma']):.2f}"
            )
        module_reports = summary.get("module_alternative_reports", {})
        if module_reports:
            print("全量 OOF 模块最佳替代方案（仅诊断，不直接部署）:")
            for module, details in module_reports.items():
                if not details.get("available", False):
                    print(f"  {module}: 无非基线候选")
                    continue
                gate = details["gate"]
                fold_values = gate.get("fold_contributions", {})
                print(
                    f"  {module}: value={details['best_alternative_value']}, "
                    f"配对均值={_pct(float(gate['mean_paired_contribution']))}, "
                    f"各折={{{', '.join(f'{k}: {_pct(float(v))}' for k, v in fold_values.items())}}}, "
                    f"P10变化={_pct(float(gate['p10_change']))}, "
                    f"最差折变化={_pct(float(gate['worst_fold_change']))}, "
                    f"gate={'PASS' if gate['enabled'] else 'FAIL'}"
                )
        candidate = summary.get("all_oof_candidate_policy", {})
        robust = summary.get("robust_deployment_policy", {})
        if candidate and robust:
            compared_fields = (
                "risk_score_penalty",
                "selection_risk_gamma",
                "cluster_cap_enabled",
                "allocation_blend",
                "exposure_head_blend",
                "correlation_exposure_gamma",
            )
            print(
                "全量 OOF 候选 → 稳健部署: "
                + ", ".join(
                    f"{field}={candidate.get(field)}→{robust.get(field)}"
                    for field in compared_fields
                )
            )
        deployment = summary.get("deployment_policy", {})
        if deployment:
            print(
                "全量 OOF 部署策略（不用于晋级）: "
                f"Allocation={float(deployment['allocation_blend']):.2f}, "
                f"Exposure={float(deployment['exposure_head_blend']):.2f}, "
                f"Risk={float(deployment['risk_score_penalty']):.2f}, "
                f"Reversal={float(deployment['selection_risk_gamma']):.2f}, "
                f"CorrExposure="
                f"{float(deployment['correlation_exposure_gamma']):.2f}"
            )
        differences = cross_fitted.get(
            "deployment_policy_differences",
            {},
        )
        if differences:
            print(
                "部署策略相对三个留出策略的差值: "
                + ", ".join(
                    f"{name}={['%+.2f' % float(value) for value in values]}"
                    for name, values in differences.items()
                )
            )
    random_baselines = []
    policy_folds = {
        int(row["fold"]): row
        for row in summary.get("ensemble_oof", {}).get("folds", [])
    }
    # 策略重放的 ``folds`` 是 OOF 收益摘要；训练期 val_metrics 则保留在
    # source_training_folds。优先读取后者，兼容普通训练和 policy-only 工件。
    training_folds = summary.get("source_training_folds", summary.get("folds", []))
    for fold in training_folds:
        if "val_metrics" not in fold:
            continue
        metrics = fold["val_metrics"]
        # random_return_sum is the expected sum of five randomly selected
        # stocks' returns, so divide it by five to compare with Top-5 mean.
        random_top5 = float(metrics["random_return_sum"]) / 5.0
        policy_fold = policy_folds.get(int(fold["fold"]), {})
        top5_return = float(policy_fold.get(
            "mean_top5_return",
            metrics["top5_return"],
        ))
        rank_ic = float(policy_fold.get(
            "mean_rank_ic",
            metrics["rank_ic"],
        ))
        random_baselines.append(random_top5)
        print(
            f"  Seed {fold.get('base_seed', '-')} Fold {fold['fold']} "
            f"({fold['val_start']} ~ {fold['val_end']}): "
            f"Top-5 {_pct(top5_return)}, "
            f"随机/全池等权 {_pct(random_top5)}, "
            f"超额 {_pct(top5_return - random_top5)}, "
            f"Rank IC {rank_ic:+.4f}"
        )
    if random_baselines:
        mean_random = float(np.mean(random_baselines))
        print(f"平均随机/全池等权: {_pct(mean_random)}")
        print(f"平均相对随机超额: {_pct(float(summary['mean_top5_return']) - mean_random)}")
    promotion = summary.get("promotion_criteria")
    if promotion and promotion.get("applicable", True):
        print(
            "Ensemble promotion criteria: "
            f"{'PASS' if promotion.get('passed') else 'FAIL'}"
        )
    candidates = summary.get('pre_registered_candidates', {})
    if candidates:
        print('预注册策略候选（不做事后融合）:')
        for name, candidate in candidates.items():
            metrics = candidate.get('metrics', {})
            gate = candidate.get('promotion_criteria', {})
            print(
                f"  {name}: LGBM权重={float(candidate.get('lgbm_weight', 0.0)):.2f}, "
                f"动态收益={_pct(float(metrics.get('mean_weighted_portfolio_return', 0.0)))}, "
                f"P10={_pct(float(metrics.get('p10_weighted_portfolio_return', 0.0)))}, "
                f"Rank IC={float(metrics.get('mean_rank_ic', 0.0)):+.4f}, "
                f"晋级={'PASS' if gate.get('passed') else 'FAIL'}"
            )
    if 'industry_constraint_application_rate' in summary:
        print(
            '行业约束—应用/回退率、行业数/HHI/最大行业权重: '
            f"{float(summary['industry_constraint_application_rate']):.2%} / "
            f"{float(summary['industry_constraint_fallback_rate']):.2%}, "
            f"{float(summary['mean_industry_count']):.2f} / "
            f"{float(summary['mean_industry_hhi']):.4f} / "
            f"{_pct(float(summary['mean_max_industry_weight']))}"
        )
    if summary.get('market_state_diagnostics'):
        print('市场状态诊断已写入 cross_validation_summary.json（不参与选型）。')
    pair_rows = summary.get('lgbm_pair_report', [])
    if pair_rows:
        paired_return = np.mean([
            row['weighted_return_delta_lgbm_minus_transformer']
            for row in pair_rows
        ])
        paired_ic = np.mean([
            row['rank_ic_delta_lgbm_minus_transformer']
            for row in pair_rows
        ])
        overlap = np.mean([row['top5_overlap'] for row in pair_rows])
        print(
            '  LGBM−Transformer 配对：'
            f'收益 {_pct(float(paired_return))}，'
            f'Rank IC {float(paired_ic):+.4f}，'
            f'Top-5 重合率 {float(overlap):.2%}'
        )


def _number(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _legacy_prediction_day(diagnostics: dict) -> dict:
    """让 v1.16 diagnostics 在新增逐日报告下仍可展示。"""
    ensemble = diagnostics.get("ensemble", {})
    risks = ensemble.get("selected_combined_risk", [])
    return {
        "date": diagnostics.get("prediction_date", "-"),
        "stock_ids": ensemble.get("top5", []),
        "portfolio_state": {
            "raw_top5_changed": ensemble.get("top5") != ensemble.get("raw_top5"),
            "cluster_constraint_applied": ensemble.get("cluster_constraint_applied"),
            "head_base_exposure": ensemble.get("head_base_exposure"),
            "base_exposure": ensemble.get("base_exposure"),
        },
        "funding_constraints": {
            "gross_exposure": ensemble.get("final_exposure"),
            "cash_weight": ensemble.get("cash_weight"),
            "max_position": max(ensemble.get("weights", []), default=None),
        },
        "risk_state": {
            "regime_gate": ensemble.get("regime_gate"),
            "combined_risk_mean": float(np.mean(risks)) if risks else None,
            "combined_risk_max": float(np.max(risks)) if risks else None,
            "mean_positive_correlation": ensemble.get("mean_positive_correlation"),
        },
    }


def print_prediction_diagnostics() -> None:
    """打印预测日的工件、状态、行业、资金和风险诊断。"""
    path = os.path.join("output", "prediction_diagnostics.json")
    if not os.path.exists(path):
        print(f"\n[推理诊断] 未找到: {path}")
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            diagnostics = json.load(handle)
    except (OSError, ValueError) as error:
        print(f"\n[推理诊断] 无法读取: {error}")
        return

    artifact = diagnostics.get("artifact_validation")
    if artifact:
        print(
            "\n[推理工件校验] "
            f"{artifact.get('status', '-')}: "
            f"{artifact.get('feature_num', '-')} / "
            f"{artifact.get('feature_count', '-')}维，"
            f"Scaler {artifact.get('scaler_feature_count', '-')}维，"
            f"股票映射 {artifact.get('stock_mapping_size', '-')}"
        )
    lgbm = diagnostics.get("lgbm")
    if lgbm:
        print(
            "[LightGBM排序融合] "
            f"启用={lgbm.get('enabled', False)}, "
            f"权重={_number(lgbm.get('weight'))}, "
            f"特征数={lgbm.get('feature_count', '-')}, "
            f"工件={lgbm.get('model_path', '-') or '-'}"
        )
    daily = diagnostics.get("daily")
    if not isinstance(daily, list) or not daily:
        daily = [_legacy_prediction_day(diagnostics)]
    print("\n[推理逐日状态]")
    for day in daily:
        state = day.get("portfolio_state", {})
        funding = day.get("funding_constraints", {})
        risk = day.get("risk_state", {})
        print(f"{day.get('date', '-')}: {', '.join(day.get('stock_ids', [])) or '-'}")
        print(
            "  状态—"
            f"Top-5变化={state.get('raw_top5_changed', '-')}, "
            f"相关簇约束={state.get('cluster_constraint_applied', '-')}, "
            f"Head/基准仓位={_number(state.get('head_base_exposure'))} / "
            f"{_number(state.get('base_exposure'))}"
        )
        print(
            "  资金—"
            f"股票/现金={_number(funding.get('gross_exposure'))} / "
            f"{_number(funding.get('cash_weight'))}, "
            f"最大单股={_number(funding.get('max_position'))}, "
            f"边界/非负={funding.get('within_exposure_bounds', '-')} / "
            f"{funding.get('non_negative', '-')}"
        )
        print(
            "  风险—"
            f"Regime={_number(risk.get('regime_gate'))}, "
            f"融合风险均值/最大={_number(risk.get('combined_risk_mean'))} / "
            f"{_number(risk.get('combined_risk_max'))}, "
            f"正相关={_number(risk.get('mean_positive_correlation'))}"
        )
        industry = day.get("industry_concentration", {})
        if industry.get("available"):
            print(
                "  行业集中度—"
                f"行业数={industry.get('industry_count', '-')}, "
                f"HHI={_number(industry.get('hhi'))}, "
                f"最大行业={_number(industry.get('max_industry_weight'))}, "
                f"覆盖/未分类={industry.get('covered_stocks', '-')} / "
                f"{industry.get('unclassified_stocks', '-')}"
            )
            for item in industry.get("weights", []):
                print(
                    f"    {item['industry']}: {_pct(item['weight'])} "
                    f"({', '.join(item['stock_ids'])})"
                )
        elif industry:
            print(f"  行业集中度—未启用: {industry.get('reason', '-')}")


def _normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


def load_stock_names(data_path: str) -> dict[str, str]:
    """读取沪深300候选清单中的代码—名称映射。"""
    stock_list_path = os.path.join(data_path, "hs300_stock_list.csv")
    if not os.path.exists(stock_list_path):
        print(f"[本地后验评分] 未找到股票名称映射: {stock_list_path}")
        return {}
    stock_list = pd.read_csv(stock_list_path, dtype={"code": str})
    required_columns = {"code", "code_name"}
    if not required_columns.issubset(stock_list.columns):
        raise ValueError(
            f"股票名称映射缺少字段 {sorted(required_columns)}: {stock_list_path}"
        )
    stock_list["stock_id"] = stock_list["code"].str.extract(r"(\d{6})$")[0]
    stock_list = stock_list.dropna(subset=["stock_id", "code_name"])
    return stock_list.set_index("stock_id")["code_name"].to_dict()


def print_portfolio_return_breakdown(
    selected: pd.DataFrame,
    returns: pd.Series,
    gross_exposure: float,
    stock_names: dict[str, str],
) -> None:
    """打印选股的逐只收益和同仓位事后最优组合基准。"""
    selected = selected.copy()
    selected["weighted_return"] = selected["weight"] * selected["return"]
    selected["cash_adjusted_all_in_return"] = (
        gross_exposure * selected["return"]
    )
    selected["stock_name"] = selected["stock_id"].map(stock_names).fillna("未知")
    selected = selected.sort_values("weight", ascending=False)

    print("\n[模型组合逐股收益]")
    print(
        selected[[
            "stock_id", "stock_name", "weight", "return", "weighted_return",
            "cash_adjusted_all_in_return",
        ]].rename(columns={
            "stock_id": "股票代码",
            "stock_name": "股票名称",
            "weight": "模型权重",
            "return": "股票实际收益",
            "weighted_return": "加权收益贡献",
            "cash_adjusted_all_in_return": "同仓位全仓收益",
        }).to_string(
            index=False,
            formatters={
                "模型权重": _pct,
                "股票实际收益": _pct,
                "加权收益贡献": _pct,
                "同仓位全仓收益": _pct,
            },
        )
    )
    raw_return_sum = float(selected["return"].sum())
    raw_return_mean = float(selected["return"].mean())
    weighted_return_sum = float(selected["weighted_return"].sum())
    invested_weighted_return = (
        weighted_return_sum / gross_exposure if gross_exposure > 0 else 0.0
    )
    print(
        "所选股票未加权收益："
        f"求和 {_pct(raw_return_sum)}，等权平均 {_pct(raw_return_mean)}"
    )
    print(
        "按模型权重收益："
        f"求和 {_pct(weighted_return_sum)}，"
        f"仅股票仓位归一化后 {_pct(invested_weighted_return)}"
    )
    print(
        "若将当前总股票仓位全部投入任一已选股票（其余为现金）："
    )
    for stock_id, stock_name, stock_return, all_in_return in selected[[
        "stock_id", "stock_name", "return", "cash_adjusted_all_in_return",
    ]].itertuples(index=False, name=None):
        print(
            f"  {stock_id} {stock_name}: {_pct(float(all_in_return))} "
            f"（股票本身 {_pct(float(stock_return))}）"
        )

    oracle_top5 = returns.nlargest(min(5, len(returns)))
    oracle_top5_mean = float(oracle_top5.mean())
    oracle_all_in_id = str(oracle_top5.index[0])
    oracle_all_in_name = stock_names.get(oracle_all_in_id, "未知")
    oracle_all_in_return = float(oracle_top5.iloc[0])
    oracle_top5_return = oracle_top5_mean * gross_exposure
    oracle_all_in_portfolio_return = oracle_all_in_return * gross_exposure
    print("\n[同一总股票仓位下的理论事后最优]")
    print(
        f"单股全仓最优：{oracle_all_in_id} {oracle_all_in_name}，"
        f"股票收益 {_pct(oracle_all_in_return)}，"
        f"组合收益 {_pct(oracle_all_in_portfolio_return)}"
    )
    print(
        "全池 Top-5 等权最优组合："
        f"组合收益 {_pct(oracle_top5_return)}，成分如下"
    )
    for stock_id, stock_return in oracle_top5.items():
        stock_name = stock_names.get(str(stock_id), "未知")
        print(
            f"  {stock_id} {stock_name}: {_pct(float(stock_return))}，"
            f"权重 {_pct(gross_exposure / len(oracle_top5))}"
        )


def print_realized_return() -> None:
    result_path = os.path.join("output", "result.csv")
    test_path = os.path.join(config["data_path"], "test.csv")
    if not os.path.exists(test_path):
        print("\n[本地后验评分] 未找到 data/test.csv；未来行情未知，跳过真实收益与基线比较。")
        return
    if not os.path.exists(result_path):
        print(f"\n[本地后验评分] 未找到预测文件: {result_path}")
        return

    try:
        result = pd.read_csv(result_path)
        id_column = "stock_id" if "stock_id" in result.columns else "股票代码"
        weight_column = "weight" if "weight" in result.columns else "权重"
        result = result[[id_column, weight_column]].copy()
        result.columns = ["stock_id", "weight"]
        result["stock_id"] = _normalize_ids(result["stock_id"])
        result["weight"] = pd.to_numeric(result["weight"], errors="raise")

        test = pd.read_csv(test_path, dtype={"股票代码": str})
        test["股票代码"] = _normalize_ids(test["股票代码"])
        test["日期"] = pd.to_datetime(test["日期"])
        diagnostics_path = os.path.join("output", "prediction_diagnostics.json")
        if not os.path.exists(diagnostics_path):
            raise ValueError('缺少预测日期诊断，拒绝把 test.csv 当作未来收益')
        with open(diagnostics_path, "r", encoding="utf-8") as handle:
            prediction_date = pd.Timestamp(json.load(handle)['prediction_date'])
        if test['日期'].max() <= prediction_date:
            print(
                '\n[本地后验评分] 拒绝：test.csv 不含预测日之后的未来标签；'
                '仅完成工件与资金约束验证。'
            )
            return
        returns = (
            test.sort_values("日期")
            .groupby("股票代码", sort=False)["开盘"]
            .agg(lambda prices: prices.iloc[-1] / prices.iloc[0] - 1.0)
            .rename("return")
        )
        stock_names = load_stock_names(config["data_path"])

        selected = result.merge(returns, left_on="stock_id", right_index=True, how="left")
        if selected["return"].isna().any():
            missing = ", ".join(selected.loc[selected["return"].isna(), "stock_id"])
            raise ValueError(f"测试集缺少预测股票: {missing}")

        gross_exposure = float(selected["weight"].sum())
        portfolio_return = float((selected["weight"] * selected["return"]).sum())
        universe_return = float(returns.mean()) * gross_exposure
        oracle_top5_return = float(returns.nlargest(5).mean()) * gross_exposure
        start_date = test["日期"].min().date()
        end_date = test["日期"].max().date()

        print(f"\n[本地后验评分：{start_date} ~ {end_date}]")
        print(
            f"模型组合收益: {_pct(portfolio_return)} "
            f"(股票 {gross_exposure:.2%}, 现金 {1.0 - gross_exposure:.2%})"
        )
        print_portfolio_return_breakdown(
            selected,
            returns,
            gross_exposure,
            stock_names,
        )
        print(f"基线—全股票等权: {_pct(universe_return)}")
        print(f"基线—现金: {_pct(0.0)}")
        print(f"事后上界—全池 Top-5 等权: {_pct(oracle_top5_return)}")
        print(f"相对全股票等权超额: {_pct(portfolio_return - universe_return)}")
    except (KeyError, ValueError, TypeError, pd.errors.ParserError) as error:
        print(f"\n[本地后验评分] 无法计算: {error}")


if __name__ == "__main__":
    print_cross_validation()
    print_prediction_diagnostics()
    print_realized_return()

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
    if len(seeds) > 1:
        print(
            f"\n[模型验证：{len(seeds)} 随机种子 × "
            f"{int(summary.get('num_folds', 3))} 折 walk-forward，"
            f"每 {summary.get('evaluation_stride', 1)} 个交易日评估]"
        )
    else:
        seed_label = seeds[0] if seeds else "-"
        print(
            f"\n[模型验证：单随机种子 {seed_label} × "
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
                "平均 Allocation/Exposure 贡献: "
                f"{_pct(float(summary['mean_allocation_contribution']))} / "
                f"{_pct(float(summary['mean_exposure_contribution']))}"
            )
            print(
                f"平均模型排名分歧: "
                f"{float(summary['mean_model_disagreement']):.4f}"
            )
    print(f"平均 Rank IC: {float(summary['mean_rank_ic']):+.4f}")
    if "risk_score_penalty" in summary:
        print(
            "OOF Allocation混合/风险分数惩罚/选择gamma/相关性降仓: "
            f"{float(summary.get('allocation_blend', 0.0)):.2f} / "
            f"{float(summary['risk_score_penalty']):.2f} / "
            f"{float(summary.get('selection_risk_gamma', 0.0)):.2f} / "
            f"{float(summary.get('correlation_exposure_gamma', 0.0)):.2f}"
        )
    random_baselines = []
    policy_folds = {
        int(row["fold"]): row
        for row in summary.get("ensemble_oof", {}).get("folds", [])
    }
    for fold in summary.get("folds", []):
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
    print_realized_return()

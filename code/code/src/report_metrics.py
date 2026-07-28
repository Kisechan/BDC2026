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

    seeds = summary.get("ensemble_seeds")
    if seeds:
        print(
            "\n[模型验证：三随机种子 × 三折 walk-forward，"
            f"每 {summary.get('evaluation_stride', 1)} 个交易日评估]"
        )
    else:
        print("\n[模型验证：三折 walk-forward]")
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
    random_baselines = []
    for fold in summary.get("folds", []):
        metrics = fold["val_metrics"]
        # random_return_sum is the expected sum of five randomly selected
        # stocks' returns, so divide it by five to compare with Top-5 mean.
        random_top5 = float(metrics["random_return_sum"]) / 5.0
        top5_return = float(metrics["top5_return"])
        random_baselines.append(random_top5)
        print(
            f"  Seed {fold.get('base_seed', '-')} Fold {fold['fold']} "
            f"({fold['val_start']} ~ {fold['val_end']}): "
            f"Top-5 {_pct(top5_return)}, "
            f"随机/全池等权 {_pct(random_top5)}, "
            f"超额 {_pct(top5_return - random_top5)}, "
            f"Rank IC {float(metrics['rank_ic']):+.4f}"
        )
    if random_baselines:
        mean_random = float(np.mean(random_baselines))
        print(f"平均随机/全池等权: {_pct(mean_random)}")
        print(f"平均相对随机超额: {_pct(float(summary['mean_top5_return']) - mean_random)}")
    if "promotion_criteria" in summary:
        print(
            "Ensemble promotion criteria: "
            f"{'PASS' if summary['promotion_criteria'].get('passed') else 'FAIL'}"
        )


def _normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)


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
        print(f"基线—全股票等权: {_pct(universe_return)}")
        print(f"基线—现金: {_pct(0.0)}")
        print(f"事后上界—全池 Top-5 等权: {_pct(oracle_top5_return)}")
        print(f"相对全股票等权超额: {_pct(portfolio_return - universe_return)}")
    except (KeyError, ValueError, TypeError, pd.errors.ParserError) as error:
        print(f"\n[本地后验评分] 无法计算: {error}")


if __name__ == "__main__":
    print_cross_validation()
    print_realized_return()

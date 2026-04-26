from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "target_hit_rate",
    "expected_shortfall",
    "mean_terminal_wealth",
    "median_terminal_wealth",
    "terminal_wealth_p05",
    "mean_gross_leverage",
    "mean_concentration",
    "mean_turnover",
    "max_drawdown",
    "wealth_volatility",
    "train_time_sec",
    "solve_time_sec",
    "eval_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze revised fold-aware experiment outputs.")
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def load_manifest(results_dir: Path) -> dict:
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_summaries(results_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(results_dir.glob("*/validation/summary/main_results.csv")) + sorted(
        results_dir.glob("*/test/summary/main_results.csv")
    ):
        frame = pd.read_csv(summary_path)
        parts = summary_path.relative_to(results_dir).parts
        frame["fold"] = parts[0]
        frame["evaluation_set"] = parts[1]
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"No per-fold summary CSV files found under {results_dir}")
    return pd.concat(rows, ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["evaluation_set", "method_family", "method_name", "n_assets", "initial_wealth"]
    available_metrics = [metric for metric in METRICS if metric in frame.columns]
    pieces = []
    grouped = frame.groupby(group_cols, dropna=False)
    for metric in available_metrics:
        stats = grouped[metric].agg(["count", "mean", "std"]).reset_index()
        stats["metric"] = metric
        stats = stats.rename(columns={"count": "n", "mean": "mean", "std": "std"})
        stats["std"] = stats["std"].fillna(0.0)
        stats["sem"] = stats["std"] / np.sqrt(stats["n"].clip(lower=1))
        stats["ci95_half_width"] = 1.96 * stats["sem"]
        pieces.append(stats[group_cols + ["metric", "n", "mean", "std", "sem", "ci95_half_width"]])
    return pd.concat(pieces, ignore_index=True)


def make_validation_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    validation = frame[frame["evaluation_set"] == "validation"].copy()
    ranking_cols = ["method_name", "n_assets", "initial_wealth"]
    needed = ["target_hit_rate", "expected_shortfall", "mean_turnover"]
    available = [col for col in needed if col in validation.columns]
    ranking = validation.groupby(ranking_cols, dropna=False)[available].mean().reset_index()
    sort_cols = [col for col in ["n_assets", "initial_wealth", "target_hit_rate", "expected_shortfall", "mean_turnover"] if col in ranking.columns]
    ascending = [True, True, False, True, True][: len(sort_cols)]
    return ranking.sort_values(sort_cols, ascending=ascending)


def select_best_by_scenario(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return ranking
    sort_cols = [col for col in ["n_assets", "initial_wealth", "target_hit_rate", "expected_shortfall", "mean_turnover"] if col in ranking.columns]
    ascending = [True, True, False, True, True][: len(sort_cols)]
    ordered = ranking.sort_values(sort_cols, ascending=ascending)
    return ordered.groupby(["n_assets", "initial_wealth"], as_index=False).first()


def attach_test_results(frame: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    test = frame[frame["evaluation_set"] == "test"].copy()
    key_cols = ["method_name", "n_assets", "initial_wealth"]
    metrics = [metric for metric in METRICS if metric in test.columns]
    test_summary = test.groupby(key_cols, dropna=False)[metrics].mean().reset_index()
    return selected[key_cols].merge(test_summary, on=key_cols, how="left")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(results_dir)
    combined = collect_summaries(results_dir)
    combined.to_csv(analysis_dir / "combined_results.csv", index=False)

    metric_summary = summarize(combined)
    metric_summary.to_csv(analysis_dir / "metric_summary.csv", index=False)

    ranking = make_validation_ranking(combined)
    ranking.to_csv(analysis_dir / "validation_ranking.csv", index=False)

    selected = select_best_by_scenario(ranking)
    selected.to_csv(analysis_dir / "selected_methods_by_validation.csv", index=False)

    selected_test = attach_test_results(combined, selected)
    selected_test.to_csv(analysis_dir / "selected_methods_test_results.csv", index=False)

    print(f"[analysis] wrote {analysis_dir}")
    if manifest:
        print(f"[analysis] experiment={manifest.get('experiment_name')} config_hash={manifest.get('config_hash')}")


if __name__ == "__main__":
    main()


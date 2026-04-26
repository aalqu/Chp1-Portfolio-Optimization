from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_optim.core.types import EvaluationResult, SolverResult


def save_run(result_dir: Path, solver_result: SolverResult, evaluation_result: EvaluationResult) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    metadata = solver_result.metadata or {}
    initial_wealth = float(metadata.get("initial_wealth", np.nan))
    target_wealth = float(metadata.get("target_wealth", np.nan))
    initial_tag = "na" if np.isnan(initial_wealth) else f"{int(round(initial_wealth * 1000)):04d}"
    path = result_dir / f"{solver_result.method_name}_n{solver_result.n_assets}_w{initial_tag}_seed{solver_result.seed}.npz"
    np.savez_compressed(
        path,
        method_family=solver_result.method_family,
        method_name=solver_result.method_name,
        n_assets=solver_result.n_assets,
        seed=solver_result.seed,
        initial_wealth=initial_wealth,
        target_wealth=target_wealth,
        train_time_sec=-1.0 if solver_result.train_time_sec is None else solver_result.train_time_sec,
        solve_time_sec=-1.0 if solver_result.solve_time_sec is None else solver_result.solve_time_sec,
        wealth_grid=np.array([]) if solver_result.wealth_grid is None else solver_result.wealth_grid,
        value_grid=np.array([]) if solver_result.value_grid is None else solver_result.value_grid,
        policy_grid=np.array([]) if solver_result.policy_grid is None else solver_result.policy_grid,
        terminal_wealth=evaluation_result.terminal_wealth,
        wealth_paths=evaluation_result.wealth_paths,
        weight_paths=evaluation_result.weight_paths,
        metadata_json=json.dumps(metadata),
        metrics_json=json.dumps(evaluation_result.metrics),
    )
    return path


def aggregate_npz_results(raw_dir: Path, summary_path: Path) -> pd.DataFrame:
    rows = []
    for npz_path in sorted(raw_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        row = {
            "method_family": str(data["method_family"]),
            "method_name": str(data["method_name"]),
            "n_assets": int(data["n_assets"]),
            "seed": int(data["seed"]),
            "initial_wealth": float(data["initial_wealth"]) if "initial_wealth" in data else np.nan,
            "target_wealth": float(data["target_wealth"]) if "target_wealth" in data else np.nan,
            "train_time_sec": float(data["train_time_sec"]),
            "solve_time_sec": float(data["solve_time_sec"]),
        }
        if not np.isnan(row["initial_wealth"]) and not np.isnan(row["target_wealth"]) and row["initial_wealth"] > 0:
            row["required_return_pct"] = 100.0 * (row["target_wealth"] / row["initial_wealth"] - 1.0)
        row.update(json.loads(str(data["metrics_json"])))
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_path, index=False)
    return frame

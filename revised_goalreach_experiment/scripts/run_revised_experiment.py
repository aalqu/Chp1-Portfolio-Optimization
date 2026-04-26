from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_optim.core.config import ExperimentConfig, periods_per_year_for_frequency
from portfolio_optim.core.types import MarketDataSplit, MarketParameters, SolverResult
from portfolio_optim.data.loaders import compute_log_returns, load_price_frame
from portfolio_optim.evaluation.io import aggregate_npz_results, save_run
from portfolio_optim.evaluation.rollout import evaluate_solver_forward_test
from portfolio_optim.fd.solver import FiniteDifferencePortfolioSolver
from portfolio_optim.market.constraints import equal_weight, project_weights
from portfolio_optim.market.estimators import estimate_market_parameters
from portfolio_optim.nn.solver import NeuralPortfolioSolver


BASELINE_METHODS = {"baseline_cash", "baseline_equal_weight", "baseline_mean_variance"}
FD_METHODS = {"fd_hjb_viscosity"}
NN_ARCHITECTURES = {"mlp_shared", "mlp_deep", "bsde", "pinn", "recurrent", "transformer"}
KNOWN_METHODS = BASELINE_METHODS | FD_METHODS | {f"nn_{name}" for name in NN_ARCHITECTURES}


class CashOnlySolver:
    method_family = "baseline"
    method_name = "baseline_cash"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.n_assets = 0

    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        self.n_assets = n_assets
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=n_assets,
            seed=seed,
            solve_time_sec=0.0,
            metadata={"tickers": market.tickers, "baseline": "cash_only"},
        )

    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        del t_index, wealth, market_features
        return np.zeros(self.n_assets, dtype=float)


class EqualWeightSolver:
    method_family = "baseline"
    method_name = "baseline_equal_weight"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.weights: np.ndarray | None = None

    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        self.weights = equal_weight(n_assets, self.config.problem.constraints)
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=n_assets,
            seed=seed,
            solve_time_sec=0.0,
            policy_grid=self.weights.reshape(1, 1, -1),
            metadata={"tickers": market.tickers, "baseline": "equal_weight"},
        )

    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        del t_index, wealth, market_features
        if self.weights is None:
            raise RuntimeError("Call fit before policy.")
        return self.weights


class MeanVarianceSolver:
    method_family = "baseline"
    method_name = "baseline_mean_variance"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.weights: np.ndarray | None = None

    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        per_period_rf = self.config.problem.risk_free_rate / market.periods_per_year
        excess = market.mean_returns - per_period_rf
        try:
            raw = np.linalg.solve(market.covariance, excess)
        except np.linalg.LinAlgError:
            raw = np.linalg.pinv(market.covariance) @ excess
        raw = np.maximum(raw, 0.0)
        if float(raw.sum()) <= 1e-12:
            raw = np.ones(n_assets, dtype=float)
        raw = raw / raw.sum()
        self.weights = project_weights(raw, self.config.problem.constraints)
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=n_assets,
            seed=seed,
            solve_time_sec=0.0,
            policy_grid=self.weights.reshape(1, 1, -1),
            metadata={"tickers": market.tickers, "baseline": "long_only_mean_variance"},
        )

    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        del t_index, wealth, market_features
        if self.weights is None:
            raise RuntimeError("Call fit before policy.")
        return self.weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised fold-aware goal-reaching experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a revised experiment JSON config.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Optional override for the config results_dir.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method subset.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip per-fold plot generation.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def config_hash(raw_config: dict[str, Any]) -> str:
    payload = json.dumps(raw_config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def update_dataclass(target: Any, values: dict[str, Any], path_fields: set[str] | None = None) -> None:
    path_fields = path_fields or set()
    valid_fields = {field.name for field in fields(target)}
    for key, value in values.items():
        if key not in valid_fields:
            raise KeyError(f"Unsupported config field {target.__class__.__name__}.{key}")
        if key in path_fields:
            value = resolve_repo_path(value)
        setattr(target, key, value)


def build_experiment_config(raw: dict[str, Any], results_dir: Path) -> ExperimentConfig:
    config = ExperimentConfig(results_dir=results_dir, summary_dir=results_dir / "summary")
    data_values = dict(raw.get("data", {}))
    ticker_groups = data_values.pop("ticker_groups", None)
    update_dataclass(config.data, data_values, path_fields={"data_root"})
    if ticker_groups is not None:
        config.data.ticker_groups.update({int(k): v for k, v in ticker_groups.items()})
    update_dataclass(config.problem, raw.get("problem", {}))
    update_dataclass(config.problem.constraints, raw.get("constraints", {}))
    update_dataclass(config.fd, raw.get("fd", {}))
    update_dataclass(config.nn, raw.get("nn", {}))
    if config.problem.initial_wealths:
        config.problem.initial_wealth = float(config.problem.initial_wealths[0])
    return config


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def write_run_manifest(results_dir: Path, config_path: Path, raw_config: dict[str, Any], selected_methods: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    copied_config = results_dir / "config.json"
    shutil.copyfile(config_path, copied_config)
    manifest = {
        "experiment_name": raw_config.get("experiment_name"),
        "created_unix_time": time.time(),
        "config_path": str(config_path.resolve()),
        "config_hash": config_hash(raw_config),
        "selected_methods": selected_methods,
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "git_revision": git_revision(),
    }
    with (results_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def build_fold_split(config: ExperimentConfig, tickers: list[str], fold: dict[str, str]) -> tuple[MarketDataSplit, dict[str, Any]]:
    prices = load_price_frame(config.data, tickers)
    returns = compute_log_returns(prices, config.problem.rebalance_frequency)

    def slice_period(start_key: str, end_key: str, label: str) -> np.ndarray:
        sliced = returns.loc[fold[start_key] : fold[end_key]]
        if sliced.empty:
            raise ValueError(f"{fold['name']} {label} period has no returns for tickers={tickers}")
        return sliced.to_numpy()

    train = slice_period("train_start", "train_end", "train")
    validation = slice_period("validation_start", "validation_end", "validation")
    test = slice_period("test_start", "test_end", "test")
    horizon = config.problem.horizon_steps
    for label, values in {"validation": validation, "test": test}.items():
        if len(values) < horizon:
            raise ValueError(f"{fold['name']} {label} has {len(values)} observations, fewer than horizon_steps={horizon}")

    metadata = {
        "fold_name": fold["name"],
        "train_start": fold["train_start"],
        "train_end": fold["train_end"],
        "validation_start": fold["validation_start"],
        "validation_end": fold["validation_end"],
        "test_start": fold["test_start"],
        "test_end": fold["test_end"],
        "train_observations": int(len(train)),
        "validation_observations": int(len(validation)),
        "test_observations": int(len(test)),
    }
    return (
        MarketDataSplit(
            tickers=tickers,
            train_returns=train,
            validation_returns=validation,
            test_returns=test,
            rebalance_frequency=config.problem.rebalance_frequency,
            periods_per_year=periods_per_year_for_frequency(config.problem.rebalance_frequency),
        ),
        metadata,
    )


def make_solver(method: str, config: ExperimentConfig):
    if method == "baseline_cash":
        return CashOnlySolver(config)
    if method == "baseline_equal_weight":
        return EqualWeightSolver(config)
    if method == "baseline_mean_variance":
        return MeanVarianceSolver(config)
    if method == "fd_hjb_viscosity":
        return FiniteDifferencePortfolioSolver(config)
    if method.startswith("nn_"):
        return NeuralPortfolioSolver(config, method.removeprefix("nn_"))
    raise ValueError(f"Unknown method {method!r}")


def add_metadata(
    solver_result: SolverResult,
    config: ExperimentConfig,
    fold_metadata: dict[str, Any],
    evaluation_set: str,
    market: MarketParameters,
    config_id: str,
) -> None:
    solver_result.metadata.update(fold_metadata)
    solver_result.metadata.update(
        {
            "initial_wealth": config.problem.initial_wealth,
            "target_wealth": config.problem.target_wealth,
            "evaluation_mode": "calendar_forward_window",
            "evaluation_set": evaluation_set,
            "rebalance_frequency": config.problem.rebalance_frequency,
            "periods_per_year": market.periods_per_year,
            "config_hash": config_id,
        }
    )


def aggregate_and_plot(eval_dirs: set[Path], make_plots: bool) -> None:
    for eval_dir in sorted(eval_dirs):
        raw_dir = eval_dir / "raw"
        summary_dir = eval_dir / "summary"
        if not raw_dir.exists():
            continue
        aggregate_npz_results(raw_dir, summary_dir / "main_results.csv")
        if make_plots:
            from portfolio_optim.plots.comparison import generate_experiment_plots

            generate_experiment_plots(eval_dir)


def run(raw_config: dict[str, Any], config_path: Path, results_dir: Path, methods_override: list[str] | None, skip_plots: bool) -> None:
    selected_methods = methods_override or list(raw_config.get("methods", []))
    if not selected_methods:
        raise ValueError("No methods configured.")
    unknown = sorted(set(selected_methods) - KNOWN_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods in config: {unknown}")

    evaluation_sets = raw_config.get("evaluation_sets", ["validation", "test"])
    invalid_eval_sets = sorted(set(evaluation_sets) - {"validation", "test"})
    if invalid_eval_sets:
        raise ValueError(f"Unsupported evaluation_sets: {invalid_eval_sets}")

    base_config = build_experiment_config(raw_config, results_dir)
    config_id = config_hash(raw_config)
    make_plots = bool(raw_config.get("make_plots", False)) and not skip_plots
    write_run_manifest(results_dir, config_path, raw_config, selected_methods)

    eval_dirs: set[Path] = set()
    folds = raw_config.get("folds", [])
    if not folds:
        raise ValueError("Config must include at least one fold.")

    for fold in folds:
        print(f"[fold] {fold['name']}")
        for initial_wealth in base_config.problem.initial_wealths:
            run_config = copy.deepcopy(base_config)
            run_config.problem.initial_wealth = float(initial_wealth)
            for n_assets in run_config.problem.asset_counts:
                tickers = run_config.data.ticker_groups[n_assets]
                split, fold_metadata = build_fold_split(run_config, tickers, fold)
                market = estimate_market_parameters(split)
                for seed in run_config.problem.seed_list:
                    for method in selected_methods:
                        print(
                            f"  [run] w0={initial_wealth:.2f} n={n_assets} seed={seed} method={method}",
                            flush=True,
                        )
                        solver = make_solver(method, run_config)
                        solver_result = solver.fit(market, n_assets, seed)
                        for evaluation_set in evaluation_sets:
                            returns = split.validation_returns if evaluation_set == "validation" else split.test_returns
                            result_for_save = copy.deepcopy(solver_result)
                            add_metadata(result_for_save, run_config, fold_metadata, evaluation_set, market, config_id)
                            evaluation_result = evaluate_solver_forward_test(solver, returns, run_config)
                            result_for_save.eval_time_sec = evaluation_result.metrics["eval_time_sec"]
                            eval_dir = results_dir / fold["name"] / evaluation_set
                            eval_dirs.add(eval_dir)
                            save_run(eval_dir / "raw", result_for_save, evaluation_result)

    aggregate_and_plot(eval_dirs, make_plots)


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    raw_config = load_json(config_path)
    configured_results = args.results_dir or raw_config.get("results_dir")
    if configured_results is None:
        raise ValueError("Provide --results-dir or set results_dir in the config.")
    results_dir = resolve_repo_path(configured_results)
    run(raw_config, config_path, results_dir, args.methods, args.skip_plots)


if __name__ == "__main__":
    main()

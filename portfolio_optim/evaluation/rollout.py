from __future__ import annotations

import time

import numpy as np

from portfolio_optim.core.config import ExperimentConfig, periods_per_year_for_frequency
from portfolio_optim.core.types import EvaluationResult
from portfolio_optim.evaluation.metrics import compute_path_risk_metrics, compute_terminal_metrics, compute_weight_metrics
from portfolio_optim.market.simulators import bootstrap_historical_paths, wealth_step


def evaluate_solver(solver, market, config: ExperimentConfig, seed: int) -> EvaluationResult:
    start = time.perf_counter()
    paths = bootstrap_historical_paths(market, config.problem, config.problem.n_eval_paths, seed)
    n_paths, horizon_steps, n_assets = paths.shape
    wealth = np.full((n_paths, horizon_steps + 1), config.problem.initial_wealth, dtype=float)
    weights = np.zeros((n_paths, horizon_steps, n_assets), dtype=float)

    for t in range(horizon_steps):
        current_wealth = wealth[:, t]
        step_weights = np.vstack(
            [
                solver.policy(t, w, (paths[i, :t], wealth[i, : t + 1]))
                for i, w in enumerate(current_wealth)
            ]
        )
        weights[:, t, :] = step_weights
        wealth[:, t + 1] = wealth_step(
            current_wealth,
            step_weights,
            paths[:, t, :],
            config.problem.risk_free_rate,
            market.periods_per_year,
        )

    metrics = {}
    metrics.update(compute_terminal_metrics(wealth[:, -1], config.problem.target_wealth))
    metrics.update(compute_weight_metrics(weights))
    metrics.update(compute_path_risk_metrics(wealth))
    metrics["eval_time_sec"] = time.perf_counter() - start
    return EvaluationResult(
        terminal_wealth=wealth[:, -1],
        wealth_paths=wealth,
        weight_paths=weights,
        metrics=metrics,
    )


def rolling_forward_paths(test_returns: np.ndarray, horizon_steps: int) -> np.ndarray:
    if len(test_returns) < horizon_steps:
        raise ValueError(f"Need at least {horizon_steps} unseen observations, found {len(test_returns)}")
    windows = [test_returns[start : start + horizon_steps] for start in range(len(test_returns) - horizon_steps + 1)]
    return np.stack(windows, axis=0)


def evaluate_solver_forward_test(
    solver,
    test_returns: np.ndarray,
    config: ExperimentConfig,
) -> EvaluationResult:
    start = time.perf_counter()
    paths = rolling_forward_paths(test_returns, config.problem.horizon_steps)
    n_paths, horizon_steps, n_assets = paths.shape
    wealth = np.full((n_paths, horizon_steps + 1), config.problem.initial_wealth, dtype=float)
    weights = np.zeros((n_paths, horizon_steps, n_assets), dtype=float)

    for t in range(horizon_steps):
        current_wealth = wealth[:, t]
        step_weights = np.vstack(
            [
                solver.policy(t, w, (paths[i, :t], wealth[i, : t + 1]))
                for i, w in enumerate(current_wealth)
            ]
        )
        weights[:, t, :] = step_weights
        wealth[:, t + 1] = wealth_step(
            current_wealth,
            step_weights,
            paths[:, t, :],
            config.problem.risk_free_rate,
            periods_per_year_for_frequency(config.problem.rebalance_frequency),
        )

    metrics = {}
    metrics.update(compute_terminal_metrics(wealth[:, -1], config.problem.target_wealth))
    metrics.update(compute_weight_metrics(weights))
    metrics.update(compute_path_risk_metrics(wealth))
    metrics["eval_time_sec"] = time.perf_counter() - start
    metrics["n_forward_windows"] = int(n_paths)
    return EvaluationResult(
        terminal_wealth=wealth[:, -1],
        wealth_paths=wealth,
        weight_paths=weights,
        metrics=metrics,
    )

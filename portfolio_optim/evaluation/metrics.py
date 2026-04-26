from __future__ import annotations

import numpy as np


def max_drawdown(wealth_paths: np.ndarray) -> float:
    running_max = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = 1.0 - wealth_paths / np.maximum(running_max, 1e-12)
    return float(drawdowns.max())


def compute_terminal_metrics(terminal_wealth: np.ndarray, target_wealth: float) -> dict[str, float]:
    shortfall = np.maximum(target_wealth - terminal_wealth, 0.0)
    return {
        "target_hit_rate": float(np.mean(terminal_wealth >= target_wealth)),
        "mean_terminal_wealth": float(np.mean(terminal_wealth)),
        "median_terminal_wealth": float(np.median(terminal_wealth)),
        "terminal_wealth_p05": float(np.quantile(terminal_wealth, 0.05)),
        "expected_shortfall": float(np.mean(shortfall)),
    }


def compute_weight_metrics(weight_paths: np.ndarray) -> dict[str, float]:
    gross = np.sum(np.abs(weight_paths), axis=-1)
    concentration = np.sum(weight_paths ** 2, axis=-1)
    turnover = np.abs(np.diff(weight_paths, axis=1)).sum(axis=-1) if weight_paths.shape[1] > 1 else np.zeros(weight_paths.shape[0])
    return {
        "mean_gross_leverage": float(np.mean(gross)),
        "max_gross_leverage": float(np.max(gross)),
        "mean_concentration": float(np.mean(concentration)),
        "mean_turnover": float(np.mean(turnover)),
        "max_single_weight": float(np.max(np.abs(weight_paths))),
    }


def compute_path_risk_metrics(wealth_paths: np.ndarray) -> dict[str, float]:
    path_returns = wealth_paths[:, 1:] / np.maximum(wealth_paths[:, :-1], 1e-12) - 1.0
    return {
        "wealth_volatility": float(np.std(path_returns)),
        "max_drawdown": max_drawdown(wealth_paths),
    }


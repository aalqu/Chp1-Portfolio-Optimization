from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MarketDataSplit:
    tickers: list[str]
    train_returns: np.ndarray
    validation_returns: np.ndarray
    test_returns: np.ndarray
    rebalance_frequency: str
    periods_per_year: int


@dataclass
class MarketParameters:
    tickers: list[str]
    mean_returns: np.ndarray
    covariance: np.ndarray
    chol: np.ndarray
    historical_returns: np.ndarray
    rebalance_frequency: str
    periods_per_year: int


@dataclass
class SolverResult:
    method_family: str
    method_name: str
    n_assets: int
    seed: int
    train_time_sec: float | None = None
    solve_time_sec: float | None = None
    eval_time_sec: float | None = None
    wealth_grid: np.ndarray | None = None
    value_grid: np.ndarray | None = None
    policy_grid: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    terminal_wealth: np.ndarray
    wealth_paths: np.ndarray
    weight_paths: np.ndarray
    metrics: dict[str, float]

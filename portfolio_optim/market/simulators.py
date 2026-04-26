from __future__ import annotations

import numpy as np

from portfolio_optim.core.config import ProblemConfig
from portfolio_optim.core.types import MarketParameters


def simulate_gaussian_paths(
    market: MarketParameters,
    problem: ProblemConfig,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(size=(n_paths, problem.horizon_steps, len(market.tickers)))
    correlated = shocks @ market.chol.T
    drift = market.mean_returns.reshape(1, 1, -1)
    return drift + correlated


def bootstrap_historical_paths(
    market: MarketParameters,
    problem: ProblemConfig,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    history = market.historical_returns
    idx = rng.integers(0, len(history), size=(n_paths, problem.horizon_steps))
    return history[idx]


def wealth_step(
    wealth: np.ndarray,
    weights: np.ndarray,
    asset_returns: np.ndarray,
    risk_free_rate: float,
    periods_per_year: int,
) -> np.ndarray:
    risk_weight = weights.sum(axis=-1, keepdims=True)
    cash_weight = 1.0 - risk_weight
    gross_return = np.sum(weights * np.exp(asset_returns) - weights, axis=-1)
    cash_return = cash_weight.squeeze(-1) * (np.exp(risk_free_rate / periods_per_year) - 1.0)
    return wealth * (1.0 + gross_return + cash_return)

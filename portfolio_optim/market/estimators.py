from __future__ import annotations

import numpy as np

from portfolio_optim.core.types import MarketDataSplit, MarketParameters


def estimate_market_parameters(split: MarketDataSplit, ridge: float = 1e-6) -> MarketParameters:
    returns = split.train_returns
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance = covariance + ridge * np.eye(covariance.shape[0])
    chol = np.linalg.cholesky(covariance)
    return MarketParameters(
        tickers=split.tickers,
        mean_returns=mean_returns,
        covariance=covariance,
        chol=chol,
        historical_returns=returns,
        rebalance_frequency=split.rebalance_frequency,
        periods_per_year=split.periods_per_year,
    )

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from portfolio_optim.core.types import MarketParameters, SolverResult


class BaseSolver(ABC):
    """Common interface for finite-difference and neural portfolio solvers."""

    method_family: str
    method_name: str

    @abstractmethod
    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        raise NotImplementedError

    @abstractmethod
    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


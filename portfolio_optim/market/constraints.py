from __future__ import annotations

import numpy as np

from portfolio_optim.core.config import ConstraintConfig


def project_weights(weights: np.ndarray, constraints: ConstraintConfig) -> np.ndarray:
    projected = weights.copy()
    projected = np.clip(projected, constraints.min_weight, constraints.max_weight)
    gross = float(np.sum(np.abs(projected)))
    if gross > constraints.leverage_limit and gross > 0:
        projected = projected / gross * constraints.leverage_limit
    total = float(projected.sum())
    if constraints.long_only:
        total = max(total, 1e-12)
        if total > 1.0 or not constraints.allow_cash:
            projected = projected / total
    return projected


def equal_weight(n_assets: int, constraints: ConstraintConfig) -> np.ndarray:
    weights = np.full(n_assets, 1.0 / n_assets)
    return project_weights(weights, constraints)


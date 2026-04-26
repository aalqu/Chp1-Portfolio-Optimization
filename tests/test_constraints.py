import numpy as np

from portfolio_optim.core.config import ConstraintConfig
from portfolio_optim.market.constraints import project_weights


def test_project_weights_respects_long_only_and_budget():
    weights = np.array([1.5, -0.2, 0.7])
    projected = project_weights(weights, ConstraintConfig())
    assert np.all(projected >= 0.0)
    assert projected.sum() <= 1.0 + 1e-9


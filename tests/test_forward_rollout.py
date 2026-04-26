import numpy as np

from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.evaluation.rollout import evaluate_solver_forward_test, rolling_forward_paths


class _ConstantSolver:
    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def policy(self, t_index: int, wealth: float, market_features=None):
        del t_index, wealth, market_features
        return np.full(self.n_assets, 1.0 / self.n_assets)


def test_rolling_forward_paths_shape():
    test_returns = np.zeros((8, 3))
    paths = rolling_forward_paths(test_returns, 4)
    assert paths.shape == (5, 4, 3)


def test_evaluate_solver_forward_test_runs():
    config = ExperimentConfig()
    config.problem.horizon_steps = 4
    config.problem.initial_wealth = 0.9
    config.problem.target_wealth = 1.0
    test_returns = np.zeros((8, 2))
    result = evaluate_solver_forward_test(_ConstantSolver(2), test_returns, config)
    assert result.wealth_paths.shape == (5, 5)
    assert result.weight_paths.shape == (5, 4, 2)
    assert result.metrics["n_forward_windows"] == 5

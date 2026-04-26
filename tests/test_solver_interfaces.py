import numpy as np

from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.core.types import MarketParameters
from portfolio_optim.fd.solver import FiniteDifferencePortfolioSolver, fd_solve_viscosity_multiasset
from portfolio_optim.nn.solver import NeuralPortfolioSolver
from portfolio_optim.nn.trainer import resolve_torch_device


def _market(n_assets: int) -> MarketParameters:
    covariance = 0.01 * np.eye(n_assets)
    return MarketParameters(
        tickers=[f"a{i}" for i in range(n_assets)],
        mean_returns=np.full(n_assets, 0.001),
        covariance=covariance,
        chol=np.linalg.cholesky(covariance),
        historical_returns=np.full((100, n_assets), 0.001),
    )


def test_fd_solver_multiasset_interface():
    config = ExperimentConfig()
    solver = FiniteDifferencePortfolioSolver(config)
    result = solver.fit(_market(5), 5, 1)
    weights = solver.policy(0, 1.0)
    assert result.n_assets == 5
    assert result.policy_grid is not None
    assert result.policy_grid.shape == (config.problem.horizon_steps, config.fd.n_wealth_points + 1, 5)
    assert weights.shape == (5,)


def test_nn_solver_transformer_interface():
    config = ExperimentConfig()
    config.nn.epochs = 1
    config.nn.train_paths = 64
    config.problem.horizon_steps = 4
    solver = NeuralPortfolioSolver(config, "transformer")
    result = solver.fit(_market(3), 3, 1)
    weights = solver.policy(0, 1.0, np.zeros(3))
    assert result.n_assets == 3
    assert weights.shape == (3,)


def test_nn_solver_bsde_interface():
    config = ExperimentConfig()
    config.nn.epochs = 1
    config.nn.train_paths = 16
    config.problem.horizon_steps = 3
    solver = NeuralPortfolioSolver(config, "bsde")
    result = solver.fit(_market(2), 2, 1)
    weights = solver.policy(0, 0.9, np.zeros(2))
    assert result.n_assets == 2
    assert weights.shape == (2,)
    assert "training_loss" in result.metadata


def test_nn_solver_pinn_interface():
    config = ExperimentConfig()
    config.nn.epochs = 1
    config.nn.train_paths = 16
    config.problem.horizon_steps = 3
    solver = NeuralPortfolioSolver(config, "pinn")
    result = solver.fit(_market(2), 2, 1)
    weights = solver.policy(0, 0.9, np.zeros(2))
    assert result.n_assets == 2
    assert weights.shape == (2,)
    assert "training_loss" in result.metadata


def test_fd_solve_viscosity_multiasset_shapes():
    mu = np.array([0.12, 0.09])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    w, V, Pi, snapshots, policy_snaps, controls, value_hist, policy_hist = fd_solve_viscosity_multiasset(
        mu=mu,
        r=0.03,
        covariance=covariance,
        T=1.0,
        A=1.5,
        Nw=40,
        Nt=20,
        d=0.0,
        u=1.0,
        utility_fn=lambda wealth: (np.asarray(wealth, float) >= 1.0).astype(float),
        goal=1.0,
        record_taus=[0.1],
        n_control_samples=32,
        long_only=True,
        seed=7,
        store_history=True,
    )
    assert w.shape == (41,)
    assert V.shape == (41,)
    assert Pi.shape == (41, 2)
    assert controls.shape[1] == 2
    assert value_hist.shape == (21, 41)
    assert policy_hist.shape == (20, 41, 2)
    assert np.all(V >= -1e-9)
    assert np.all(V <= 1.0 + 1e-9)
    assert 0.1 in snapshots
    assert 0.1 in policy_snaps


def test_resolve_torch_device_cpu_explicit():
    assert resolve_torch_device("cpu").type == "cpu"

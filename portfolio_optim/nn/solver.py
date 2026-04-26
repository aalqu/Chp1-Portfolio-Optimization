from __future__ import annotations

import time

import numpy as np
import torch

from portfolio_optim.core.base import BaseSolver
from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.core.types import MarketParameters, SolverResult
from portfolio_optim.market.constraints import project_weights
from portfolio_optim.nn.architectures import build_architecture
from portfolio_optim.nn.features import build_flat_features_numpy, build_sequence_features_numpy, flat_feature_dim, token_dim
from portfolio_optim.nn.trainer import train_policy_network


class NeuralPortfolioSolver(BaseSolver):
    method_family = "nn"

    def __init__(self, config: ExperimentConfig, architecture_name: str) -> None:
        self.config = config
        self.architecture_name = architecture_name
        self.method_name = f"nn_{architecture_name}"
        self.model = None
        self.n_assets: int | None = None
        self._is_sequence_model = architecture_name in {"recurrent", "transformer"}

    def fit(self, market: MarketParameters, n_assets: int, seed: int) -> SolverResult:
        feature_dim = token_dim(n_assets, self.config.nn.fourier_modes) if self._is_sequence_model else flat_feature_dim(n_assets, self.config.nn.history_window, self.config.nn.fourier_modes)
        model = build_architecture(self.architecture_name, feature_dim, n_assets, self.config.nn.hidden_dim, self.config.nn.depth, self.config.nn.dropout)
        self.model, train_time, history = train_policy_network(model, self.architecture_name, market, self.config, seed)
        self.n_assets = n_assets

        wealth_grid = np.linspace(self.config.fd.wealth_min, self.config.fd.wealth_max, self.config.fd.n_wealth_points)
        policy_grid = np.vstack([self.policy(0, wealth) for wealth in wealth_grid])[None, :, :]
        return SolverResult(
            method_family=self.method_family,
            method_name=self.method_name,
            n_assets=n_assets,
            seed=seed,
            train_time_sec=train_time,
            wealth_grid=wealth_grid,
            policy_grid=policy_grid,
            metadata={
                "training_loss": history["loss"],
                "validation_loss": history.get("val_loss", []),
                "training_shortfall": history.get("shortfall", []),
                "validation_shortfall": history.get("val_shortfall", []),
                "tickers": market.tickers,
                "hidden_dim": self.config.nn.hidden_dim,
                "depth": self.config.nn.depth,
                "epochs": self.config.nn.epochs,
                "train_paths": self.config.nn.train_paths,
                "validation_paths": self.config.nn.validation_paths,
                "history_window": self.config.nn.history_window,
                "fourier_modes": self.config.nn.fourier_modes,
                "path_sampler": self.config.nn.path_sampler,
                "goal_loss_weight": self.config.nn.goal_loss_weight,
                "shortfall_loss_weight": self.config.nn.shortfall_loss_weight,
                "weight_regularization_weight": self.config.nn.weight_regularization_weight,
                "rebalance_frequency": self.config.problem.rebalance_frequency,
            },
        )

    def policy(self, t_index: int, wealth: float, market_features: np.ndarray | None = None) -> np.ndarray:
        if self.model is None or self.n_assets is None:
            raise RuntimeError("Call fit before policy.")
        if isinstance(market_features, tuple):
            return_history, wealth_history = market_features
            return_history = np.asarray(return_history, dtype=np.float32).reshape(-1, self.n_assets)
            wealth_history = np.asarray(wealth_history, dtype=np.float32).reshape(-1)
        else:
            return_history = np.zeros((0, self.n_assets), dtype=np.float32)
            wealth_history = np.array([wealth], dtype=np.float32)
        if self._is_sequence_model:
            x = build_sequence_features_numpy(
                t_index,
                self.config.problem.horizon_steps,
                wealth_history,
                return_history,
                self.config.nn.history_window,
                self.config.nn.fourier_modes,
                self.n_assets,
            )
            tensor = torch.tensor(x[None, :, :], dtype=torch.float32)
        else:
            x = build_flat_features_numpy(
                t_index,
                self.config.problem.horizon_steps,
                wealth_history,
                return_history,
                self.config.nn.history_window,
                self.config.nn.fourier_modes,
                self.n_assets,
            )
            tensor = torch.tensor(x[None, :], dtype=torch.float32)
        with torch.no_grad():
            raw = self.model(tensor).cpu().numpy()[0]
        weights = np.exp(raw - raw.max())
        weights = weights / np.sum(weights)
        return project_weights(weights, self.config.problem.constraints)

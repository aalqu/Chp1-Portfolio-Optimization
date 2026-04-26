from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from portfolio_optim.core.config import periods_per_year_for_frequency
from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.core.types import MarketParameters
from portfolio_optim.market.simulators import bootstrap_historical_paths, simulate_gaussian_paths
from portfolio_optim.nn.features import (
    build_flat_features_torch,
    build_sequence_features_torch,
    current_flat_time_slice,
    current_flat_wealth_index,
    time_chain_rule_coeffs,
)
from portfolio_optim.nn.losses import pinn_residual_loss, smooth_goal_loss, target_shortfall_loss, weight_regularization


def resolve_torch_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _policy_logits(model: nn.Module, architecture_name: str, state: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "value_and_policy"):
        return model.value_and_policy(state)["policy_logits"]
    return model(state)


def _apply_policy_weights(raw_weights: torch.Tensor) -> torch.Tensor:
    return torch.softmax(raw_weights, dim=-1)


def _wealth_update(
    wealth: torch.Tensor,
    weights: torch.Tensor,
    asset_returns: torch.Tensor,
    risk_free_rate: float,
    periods_per_year: int,
) -> torch.Tensor:
    risk_weight = weights.sum(dim=-1)
    cash_weight = 1.0 - risk_weight
    risky_growth = torch.sum(weights * (torch.exp(asset_returns) - 1.0), dim=-1)
    cash_growth = cash_weight * (float(np.exp(risk_free_rate / periods_per_year)) - 1.0)
    return wealth * (1.0 + risky_growth + cash_growth)


def _simulate_training_paths(
    market: MarketParameters,
    config: ExperimentConfig,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    if config.nn.path_sampler == "bootstrap":
        return bootstrap_historical_paths(market, config.problem, n_paths, seed)
    if config.nn.path_sampler == "gaussian":
        return simulate_gaussian_paths(market, config.problem, n_paths, seed)
    raise ValueError(f"Unsupported nn.path_sampler={config.nn.path_sampler!r}")


def _value_derivatives(
    value: torch.Tensor,
    state: torch.Tensor,
    time_index: int,
    horizon_steps: int,
    history_window: int,
    fourier_modes: int,
    n_assets: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad = torch.autograd.grad(value.sum(), state, create_graph=True)[0]
    time_grad = grad[:, current_flat_time_slice(history_window, fourier_modes, n_assets)]
    coeffs = torch.tensor(time_chain_rule_coeffs(time_index, horizon_steps, fourier_modes), dtype=torch.float32, device=state.device)
    v_t = torch.sum(time_grad * coeffs.unsqueeze(0), dim=1)
    wealth_index = current_flat_wealth_index(history_window, fourier_modes, n_assets)
    v_w = grad[:, wealth_index]
    v_ww = torch.autograd.grad(v_w.sum(), state, create_graph=True)[0][:, wealth_index]
    return v_t, v_w, v_ww


def _rollout_policy_network(
    model: nn.Module,
    architecture_name: str,
    paths_tensor: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    wealth = torch.full((paths_tensor.shape[0],), config.problem.initial_wealth, dtype=torch.float32, device=device)
    wealth_hist: list[torch.Tensor] = [wealth]
    weights_hist: list[torch.Tensor] = []

    for t in range(config.problem.horizon_steps):
        wealth_tensor = torch.stack(wealth_hist, dim=1)
        return_history = paths_tensor[:, :t, :]
        if architecture_name in {"recurrent", "transformer"}:
            model_input = build_sequence_features_torch(
                t,
                config.problem.horizon_steps,
                wealth_tensor,
                return_history,
                config.nn.history_window,
                config.nn.fourier_modes,
                paths_tensor.shape[-1],
            )
        else:
            model_input = build_flat_features_torch(
                t,
                config.problem.horizon_steps,
                wealth_tensor,
                return_history,
                config.nn.history_window,
                config.nn.fourier_modes,
                paths_tensor.shape[-1],
            )
        raw_weights = _policy_logits(model, architecture_name, model_input)
        weights = _apply_policy_weights(raw_weights)
        wealth = _wealth_update(
            wealth,
            weights,
            paths_tensor[:, t, :],
            config.problem.risk_free_rate,
            periods_per_year_for_frequency(config.problem.rebalance_frequency),
        )
        weights_hist.append(weights)
        wealth_hist.append(wealth)

    return {
        "weights": weights_hist,
        "wealth_paths": wealth_hist,
        "terminal_wealth": wealth_hist[-1],
    }


def _policy_rollout_objective(
    model: nn.Module,
    architecture_name: str,
    paths_tensor: torch.Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout = _rollout_policy_network(model, architecture_name, paths_tensor, config, device)
    terminal_wealth = rollout["terminal_wealth"]
    goal_loss = smooth_goal_loss(terminal_wealth, config.problem.target_wealth)
    shortfall_loss = target_shortfall_loss(terminal_wealth, config.problem.target_wealth)
    regularization = config.nn.weight_regularization_weight * torch.stack([weight_regularization(w) for w in rollout["weights"]]).mean()
    loss = config.nn.goal_loss_weight * goal_loss + config.nn.shortfall_loss_weight * shortfall_loss + regularization
    stats = {
        "loss": float(loss.detach().cpu()),
        "goal": float(goal_loss.detach().cpu()),
        "shortfall": float(shortfall_loss.detach().cpu()),
    }
    return loss, stats


def _simulate_hjb_rollout(
    model: nn.Module,
    architecture_name: str,
    paths_tensor: torch.Tensor,
    market: MarketParameters,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, list[torch.Tensor] | torch.Tensor]:
    del architecture_name
    horizon_steps = config.problem.horizon_steps
    n_assets = paths_tensor.shape[-1]
    covariance = torch.tensor(market.covariance, dtype=torch.float32, device=device)
    chol = torch.tensor(market.chol, dtype=torch.float32, device=device)
    mean_returns = torch.tensor(market.mean_returns, dtype=torch.float32, device=device)
    wealth = torch.full((paths_tensor.shape[0],), config.problem.initial_wealth, dtype=torch.float32, device=device)

    values: list[torch.Tensor] = []
    weights_hist: list[torch.Tensor] = []
    wealth_hist: list[torch.Tensor] = [wealth]
    z_hist: list[torch.Tensor] = []
    noise_hist: list[torch.Tensor] = []
    hamiltonian_hist: list[torch.Tensor] = []

    for t in range(horizon_steps):
        wealth_tensor = torch.stack(wealth_hist, dim=1)
        return_history = paths_tensor[:, :t, :]
        state = build_flat_features_torch(
            t,
            horizon_steps,
            wealth_tensor,
            return_history,
            config.nn.history_window,
            config.nn.fourier_modes,
            n_assets,
        )
        state.requires_grad_(True)
        output = model.value_and_policy(state)
        raw_weights = output["policy_logits"]
        weights = _apply_policy_weights(raw_weights)
        value = output["value"]
        v_t, v_w, v_ww = _value_derivatives(
            value,
            state,
            t,
            horizon_steps,
            config.nn.history_window,
            config.nn.fourier_modes,
            n_assets,
        )
        mu_eff = torch.sum(
            weights * (mean_returns - config.problem.risk_free_rate / market.periods_per_year),
            dim=-1,
        )
        sigma_eff2 = torch.sum((weights @ covariance) * weights, dim=-1)
        hamiltonian = v_t + wealth * mu_eff * v_w + 0.5 * wealth.pow(2) * sigma_eff2 * v_ww
        next_wealth = _wealth_update(
            wealth,
            weights,
            paths_tensor[:, t, :],
            config.problem.risk_free_rate,
            market.periods_per_year,
        )

        values.append(value)
        weights_hist.append(weights)
        wealth_hist.append(next_wealth)
        hamiltonian_hist.append(hamiltonian)
        if "z" in output:
            centered = paths_tensor[:, t, :] - mean_returns
            eps = torch.linalg.solve_triangular(chol, centered.T, upper=False).T
            z_hist.append(output["z"])
            noise_hist.append(eps)
        wealth = next_wealth

    terminal_state = build_flat_features_torch(
        horizon_steps,
        horizon_steps,
        torch.stack(wealth_hist, dim=1),
        paths_tensor,
        config.nn.history_window,
        config.nn.fourier_modes,
        n_assets,
    )
    terminal_output = model.value_and_policy(terminal_state)
    terminal_value = terminal_output["value"]
    terminal_indicator = (wealth >= config.problem.target_wealth).float()
    return {
        "values": values,
        "weights": weights_hist,
        "wealth_paths": wealth_hist,
        "terminal_value": terminal_value,
        "terminal_indicator": terminal_indicator,
        "z": z_hist,
        "noise": noise_hist,
        "hamiltonian": hamiltonian_hist,
    }


def _hjb_objective(
    model: nn.Module,
    architecture_name: str,
    paths_tensor: torch.Tensor,
    market: MarketParameters,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    rollout = _simulate_hjb_rollout(model, architecture_name, paths_tensor, market, config, device)
    terminal_wealth = rollout["wealth_paths"][-1]
    terminal_value = rollout["terminal_value"]
    terminal_indicator = rollout["terminal_indicator"]
    terminal_loss = torch.mean((terminal_value - terminal_indicator) ** 2)
    goal_loss = smooth_goal_loss(terminal_wealth, config.problem.target_wealth)
    shortfall_loss = target_shortfall_loss(terminal_wealth, config.problem.target_wealth)
    regularization = config.nn.weight_regularization_weight * torch.stack([weight_regularization(w) for w in rollout["weights"]]).mean()

    if architecture_name == "bsde":
        residual_terms = []
        dt = 1.0 / max(config.problem.horizon_steps, 1)
        for t, value in enumerate(rollout["values"]):
            next_value = rollout["terminal_value"] if t == config.problem.horizon_steps - 1 else rollout["values"][t + 1]
            martingale = torch.sum(rollout["z"][t] * rollout["noise"][t], dim=-1)
            residual_terms.append(next_value - value + dt * rollout["hamiltonian"][t] - martingale)
        residual_loss = torch.mean(torch.stack([term.pow(2) for term in residual_terms]))
    else:
        residual_loss = torch.mean(torch.stack([term.pow(2) for term in rollout["hamiltonian"]]))
        residual_loss = residual_loss + pinn_residual_loss(terminal_value, terminal_indicator)

    loss = (
        terminal_loss
        + config.nn.goal_loss_weight * goal_loss
        + config.nn.shortfall_loss_weight * shortfall_loss
        + regularization
        + 0.1 * residual_loss
    )
    stats = {
        "loss": float(loss.detach().cpu()),
        "goal": float(goal_loss.detach().cpu()),
        "shortfall": float(shortfall_loss.detach().cpu()),
        "terminal": float(terminal_loss.detach().cpu()),
        "residual": float(residual_loss.detach().cpu()),
    }
    return loss, stats


def _train_rollout_policy_model(
    model: nn.Module,
    architecture_name: str,
    market: MarketParameters,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)

    train_paths = _simulate_training_paths(market, config, config.nn.train_paths, seed)
    train_tensor = torch.tensor(train_paths, dtype=torch.float32, device=device)
    val_paths = _simulate_training_paths(market, config, config.nn.validation_paths, seed + 10_000)
    val_tensor = torch.tensor(val_paths, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.nn.learning_rate)
    history = {"loss": [], "val_loss": [], "shortfall": [], "val_shortfall": [], "goal": [], "val_goal": []}
    start = time.perf_counter()

    for _ in range(config.nn.epochs):
        model.train()
        optimizer.zero_grad()
        loss, diagnostics = _policy_rollout_objective(model, architecture_name, train_tensor, config, device)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, val_diag = _policy_rollout_objective(model, architecture_name, val_tensor, config, device)
        history["loss"].append(diagnostics["loss"])
        history["shortfall"].append(diagnostics["shortfall"])
        history["goal"].append(diagnostics["goal"])
        history["val_loss"].append(float(val_loss.detach().cpu()))
        history["val_shortfall"].append(float(val_diag["shortfall"]))
        history["val_goal"].append(float(val_diag["goal"]))

    elapsed = time.perf_counter() - start
    model.cpu()
    return model, elapsed, history


def _train_hjb_model(
    model: nn.Module,
    architecture_name: str,
    market: MarketParameters,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)

    train_paths = _simulate_training_paths(market, config, config.nn.train_paths, seed)
    train_tensor = torch.tensor(train_paths, dtype=torch.float32, device=device)
    val_paths = _simulate_training_paths(market, config, config.nn.validation_paths, seed + 10_000)
    val_tensor = torch.tensor(val_paths, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.nn.learning_rate)
    history = {
        "loss": [],
        "val_loss": [],
        "terminal": [],
        "val_terminal": [],
        "residual": [],
        "val_residual": [],
        "shortfall": [],
        "val_shortfall": [],
        "goal": [],
        "val_goal": [],
    }
    start = time.perf_counter()

    for _ in range(config.nn.epochs):
        model.train()
        optimizer.zero_grad()
        loss, train_stats = _hjb_objective(model, architecture_name, train_tensor, market, config, device)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.enable_grad():
            val_loss, val_stats = _hjb_objective(model, architecture_name, val_tensor, market, config, device)
        history["loss"].append(train_stats["loss"])
        history["terminal"].append(train_stats["terminal"])
        history["residual"].append(train_stats["residual"])
        history["shortfall"].append(train_stats["shortfall"])
        history["goal"].append(train_stats["goal"])
        history["val_loss"].append(float(val_loss.detach().cpu()))
        history["val_terminal"].append(val_stats["terminal"])
        history["val_residual"].append(val_stats["residual"])
        history["val_shortfall"].append(val_stats["shortfall"])
        history["val_goal"].append(val_stats["goal"])

    elapsed = time.perf_counter() - start
    model.cpu()
    return model, elapsed, history


def train_policy_network(
    model: nn.Module,
    architecture_name: str,
    market: MarketParameters,
    config: ExperimentConfig,
    seed: int,
) -> tuple[nn.Module, float, dict[str, list[float]]]:
    device = resolve_torch_device(config.nn.device)
    if architecture_name in {"bsde", "pinn"}:
        return _train_hjb_model(model, architecture_name, market, config, seed, device)
    return _train_rollout_policy_model(model, architecture_name, market, config, seed, device)

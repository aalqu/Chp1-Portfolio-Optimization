from __future__ import annotations

import torch


def target_shortfall_loss(terminal_wealth: torch.Tensor, target_wealth: float) -> torch.Tensor:
    return torch.relu(target_wealth - terminal_wealth).mean()


def smooth_goal_loss(terminal_wealth: torch.Tensor, target_wealth: float, temperature: float = 40.0) -> torch.Tensor:
    hit_proxy = torch.sigmoid(temperature * (terminal_wealth - target_wealth))
    return 1.0 - hit_proxy.mean()


def weight_regularization(weights: torch.Tensor) -> torch.Tensor:
    return (weights.pow(2).mean(dim=-1)).mean()


def pinn_residual_loss(wealth_now: torch.Tensor, wealth_next: torch.Tensor) -> torch.Tensor:
    return torch.mean((wealth_next - wealth_now) ** 2)

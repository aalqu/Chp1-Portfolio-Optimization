from __future__ import annotations

import math

import numpy as np
import torch


def token_dim(n_assets: int, fourier_modes: int) -> int:
    return 2 + 2 * fourier_modes + 1 + n_assets


def flat_feature_dim(n_assets: int, history_window: int, fourier_modes: int) -> int:
    return history_window * token_dim(n_assets, fourier_modes)


def current_flat_wealth_index(history_window: int, fourier_modes: int, n_assets: int) -> int:
    return (history_window - 1) * token_dim(n_assets, fourier_modes) + (2 + 2 * fourier_modes)


def current_flat_time_slice(history_window: int, fourier_modes: int, n_assets: int) -> slice:
    start = (history_window - 1) * token_dim(n_assets, fourier_modes)
    stop = start + 2 + 2 * fourier_modes
    return slice(start, stop)


def time_chain_rule_coeffs(time_index: int, horizon_steps: int, fourier_modes: int) -> np.ndarray:
    t_norm = float(time_index) / max(horizon_steps - 1, 1)
    coeffs = [1.0, -1.0]
    for k in range(1, fourier_modes + 1):
        coeffs.append(2.0 * math.pi * k * math.cos(2.0 * math.pi * k * t_norm))
        coeffs.append(-2.0 * math.pi * k * math.sin(2.0 * math.pi * k * t_norm))
    return np.asarray(coeffs, dtype=np.float32)


def _time_features_np(time_index: int, horizon_steps: int, fourier_modes: int) -> np.ndarray:
    t_norm = float(time_index) / max(horizon_steps - 1, 1)
    tau_norm = 1.0 - t_norm
    feats = [t_norm, tau_norm]
    for k in range(1, fourier_modes + 1):
        feats.append(math.sin(2.0 * math.pi * k * t_norm))
        feats.append(math.cos(2.0 * math.pi * k * t_norm))
    return np.asarray(feats, dtype=np.float32)


def _time_features_torch(time_index: int, horizon_steps: int, fourier_modes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    t_norm = float(time_index) / max(horizon_steps - 1, 1)
    tau_norm = 1.0 - t_norm
    feats = [torch.full((batch_size, 1), t_norm, dtype=torch.float32, device=device), torch.full((batch_size, 1), tau_norm, dtype=torch.float32, device=device)]
    for k in range(1, fourier_modes + 1):
        feats.append(torch.full((batch_size, 1), math.sin(2.0 * math.pi * k * t_norm), dtype=torch.float32, device=device))
        feats.append(torch.full((batch_size, 1), math.cos(2.0 * math.pi * k * t_norm), dtype=torch.float32, device=device))
    return torch.cat(feats, dim=1)


def build_sequence_features_numpy(
    time_index: int,
    horizon_steps: int,
    wealth_history: np.ndarray,
    return_history: np.ndarray,
    history_window: int,
    fourier_modes: int,
    n_assets: int,
) -> np.ndarray:
    dim = token_dim(n_assets, fourier_modes)
    tokens = np.zeros((history_window, dim), dtype=np.float32)
    observed = min(len(return_history), history_window - 1)
    start_slot = history_window - (observed + 1)
    for i, s in enumerate(range(len(return_history) - observed, len(return_history))):
        tf = _time_features_np(s + 1, horizon_steps, fourier_modes)
        tokens[start_slot + i] = np.concatenate([tf, np.array([wealth_history[s + 1]], dtype=np.float32), return_history[s].astype(np.float32)])
    current_tf = _time_features_np(time_index, horizon_steps, fourier_modes)
    current_returns = np.zeros(n_assets, dtype=np.float32)
    tokens[-1] = np.concatenate([current_tf, np.array([wealth_history[-1]], dtype=np.float32), current_returns])
    return tokens


def build_flat_features_numpy(
    time_index: int,
    horizon_steps: int,
    wealth_history: np.ndarray,
    return_history: np.ndarray,
    history_window: int,
    fourier_modes: int,
    n_assets: int,
) -> np.ndarray:
    return build_sequence_features_numpy(
        time_index,
        horizon_steps,
        wealth_history,
        return_history,
        history_window,
        fourier_modes,
        n_assets,
    ).reshape(-1)


def build_sequence_features_torch(
    time_index: int,
    horizon_steps: int,
    wealth_history: torch.Tensor,
    return_history: torch.Tensor,
    history_window: int,
    fourier_modes: int,
    n_assets: int,
) -> torch.Tensor:
    batch_size = wealth_history.shape[0]
    device = wealth_history.device
    dim = token_dim(n_assets, fourier_modes)
    tokens = torch.zeros((batch_size, history_window, dim), dtype=torch.float32, device=device)
    observed = min(return_history.shape[1], history_window - 1)
    start_slot = history_window - (observed + 1)
    for offset, s in enumerate(range(return_history.shape[1] - observed, return_history.shape[1])):
        tf = _time_features_torch(s + 1, horizon_steps, fourier_modes, batch_size, device)
        wealth_col = wealth_history[:, s + 1].unsqueeze(1)
        tokens[:, start_slot + offset, :] = torch.cat([tf, wealth_col, return_history[:, s, :]], dim=1)
    current_tf = _time_features_torch(time_index, horizon_steps, fourier_modes, batch_size, device)
    current_returns = torch.zeros((batch_size, n_assets), dtype=torch.float32, device=device)
    tokens[:, -1, :] = torch.cat([current_tf, wealth_history[:, -1].unsqueeze(1), current_returns], dim=1)
    return tokens


def build_flat_features_torch(
    time_index: int,
    horizon_steps: int,
    wealth_history: torch.Tensor,
    return_history: torch.Tensor,
    history_window: int,
    fourier_modes: int,
    n_assets: int,
) -> torch.Tensor:
    return build_sequence_features_torch(
        time_index,
        horizon_steps,
        wealth_history,
        return_history,
        history_window,
        fourier_modes,
        n_assets,
    ).reshape(wealth_history.shape[0], -1)

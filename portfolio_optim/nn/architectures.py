from __future__ import annotations

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * depth
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HJBValuePolicyBase(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_and_policy(x)["policy_logits"]

    def value_and_policy(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class BSDEPolicy(HJBValuePolicyBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim, output_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.z_head = nn.Linear(hidden_dim, output_dim)

    def value_and_policy(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encoder(x)
        return {
            "policy_logits": self.policy_head(latent),
            "value": self.value_head(latent).squeeze(-1),
            "z": self.z_head(latent),
        }


class RecurrentPolicy(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :])


class PINNPolicy(HJBValuePolicyBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim, output_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def value_and_policy(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.network(x)
        return {
            "policy_logits": self.policy_head(latent),
            "value": self.value_head(latent).squeeze(-1),
        }


class TransformerPolicy(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        encoded = self.encoder(embedded)
        return self.head(encoded[:, -1, :])


def build_architecture(name: str, feature_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float) -> nn.Module:
    if name == "mlp_shared":
        return MLPPolicy(feature_dim, output_dim, hidden_dim, depth, dropout)
    if name == "mlp_deep":
        return MLPPolicy(feature_dim, output_dim, hidden_dim * 2, depth + 2, dropout)
    if name == "bsde":
        return BSDEPolicy(feature_dim, output_dim, hidden_dim)
    if name == "recurrent":
        return RecurrentPolicy(feature_dim, output_dim, hidden_dim)
    if name == "pinn":
        return PINNPolicy(feature_dim, output_dim, hidden_dim)
    if name == "transformer":
        return TransformerPolicy(feature_dim, output_dim, hidden_dim)
    raise KeyError(f"Unknown architecture '{name}'")

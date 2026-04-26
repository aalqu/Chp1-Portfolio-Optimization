from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
}


def periods_per_year_for_frequency(rebalance_frequency: str) -> int:
    key = rebalance_frequency.lower()
    if key not in PERIODS_PER_YEAR:
        supported = ", ".join(sorted(PERIODS_PER_YEAR))
        raise ValueError(f"Unsupported rebalance_frequency={rebalance_frequency!r}. Expected one of: {supported}")
    return PERIODS_PER_YEAR[key]


DEFAULT_TICKER_GROUPS = {
    1: ["spy.us.txt"],
    5: ["spy.us.txt", "qqq.us.txt", "xlf.us.txt", "xle.us.txt", "xlp.us.txt"],
    10: ["spy.us.txt", "qqq.us.txt", "iwm.us.txt", "xlf.us.txt", "xle.us.txt", "xlp.us.txt", "xlv.us.txt", "xli.us.txt", "xly.us.txt", "xlu.us.txt"],
    20: [
        "spy.us.txt",
        "qqq.us.txt",
        "iwm.us.txt",
        "dia.us.txt",
        "xlf.us.txt",
        "xle.us.txt",
        "xlp.us.txt",
        "xlv.us.txt",
        "xli.us.txt",
        "xly.us.txt",
        "xlu.us.txt",
        "xlk.us.txt",
        "xlb.us.txt",
        "xlre.us.txt",
        "xop.us.txt",
        "vwo.us.txt",
        "vea.us.txt",
        "vnq.us.txt",
        "agg.us.txt",
        "gld.us.txt",
    ],
}


@dataclass
class DataConfig:
    data_root: Path = Path("real world data")
    ticker_groups: dict[int, list[str]] = field(default_factory=lambda: DEFAULT_TICKER_GROUPS.copy())
    min_history: int = 756
    train_fraction: float = 0.7
    validation_fraction: float = 0.15


@dataclass
class ConstraintConfig:
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    leverage_limit: float = 1.0
    allow_cash: bool = True


@dataclass
class ProblemConfig:
    horizon_steps: int = 24
    risk_free_rate: float = 0.01
    initial_wealth: float = 0.8
    initial_wealths: list[float] = field(default_factory=lambda: [0.8, 0.85, 0.9, 0.95])
    target_wealth: float = 1.0
    asset_counts: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    rebalance_frequency: str = "monthly"
    n_eval_paths: int = 10000
    seed_list: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class FDConfig:
    wealth_min: float = 0.3
    wealth_max: float = 2.0
    n_wealth_points: int = 200
    n_control_samples: int = 256
    risk_aversion: float = 6.0
    use_quantile_stress: bool = True


@dataclass
class NNConfig:
    hidden_dim: int = 64
    depth: int = 3
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 25
    train_paths: int = 4096
    validation_paths: int = 1024
    device: str = "auto"
    history_window: int = 6
    fourier_modes: int = 3
    path_sampler: str = "bootstrap"
    goal_loss_weight: float = 2.0
    shortfall_loss_weight: float = 0.1
    weight_regularization_weight: float = 0.001


@dataclass
class ExperimentConfig:
    project_root: Path = Path(".")
    data: DataConfig = field(default_factory=DataConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    fd: FDConfig = field(default_factory=FDConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    results_dir: Path = Path("results")
    summary_dir: Path = Path("results/summary")

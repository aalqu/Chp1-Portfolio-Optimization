from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.data.loaders import compute_log_returns, load_price_frame


BASELINE_METHODS = {"baseline_cash", "baseline_equal_weight", "baseline_mean_variance"}
FD_METHODS = {"fd_hjb_viscosity"}
NN_ARCHITECTURES = {"mlp_shared", "mlp_deep", "bsde", "pinn", "recurrent", "transformer"}
KNOWN_METHODS = BASELINE_METHODS | FD_METHODS | {f"nn_{name}" for name in NN_ARCHITECTURES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate revised experiment config and data availability.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def update_dataclass(target: Any, values: dict[str, Any], path_fields: set[str] | None = None) -> None:
    path_fields = path_fields or set()
    valid_fields = {field.name for field in fields(target)}
    for key, value in values.items():
        if key not in valid_fields:
            raise KeyError(f"Unsupported config field {target.__class__.__name__}.{key}")
        if key in path_fields:
            value = resolve_repo_path(value)
        setattr(target, key, value)


def build_config(raw: dict[str, Any]) -> ExperimentConfig:
    results_dir = resolve_repo_path(raw.get("results_dir", "revised_goalreach_experiment/outputs/validation_check"))
    config = ExperimentConfig(results_dir=results_dir, summary_dir=results_dir / "summary")
    data_values = dict(raw.get("data", {}))
    ticker_groups = data_values.pop("ticker_groups", None)
    update_dataclass(config.data, data_values, path_fields={"data_root"})
    if ticker_groups is not None:
        config.data.ticker_groups.update({int(k): v for k, v in ticker_groups.items()})
    update_dataclass(config.problem, raw.get("problem", {}))
    update_dataclass(config.problem.constraints, raw.get("constraints", {}))
    update_dataclass(config.fd, raw.get("fd", {}))
    update_dataclass(config.nn, raw.get("nn", {}))
    return config


def validate_methods(raw: dict[str, Any]) -> None:
    methods = raw.get("methods", [])
    unknown = sorted(set(methods) - KNOWN_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if any(method.startswith("nn_") for method in methods):
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise RuntimeError("Config includes NN methods, but torch could not be imported.") from exc


def validate_folds(raw: dict[str, Any], config: ExperimentConfig) -> None:
    folds = raw.get("folds", [])
    if not folds:
        raise ValueError("Config must define at least one fold.")
    for n_assets in config.problem.asset_counts:
        if n_assets not in config.data.ticker_groups:
            raise KeyError(f"No ticker group configured for n_assets={n_assets}")
        tickers = config.data.ticker_groups[n_assets]
        prices = load_price_frame(config.data, tickers)
        returns = compute_log_returns(prices, config.problem.rebalance_frequency)
        print(f"[data] n={n_assets} tickers={len(tickers)} returns={len(returns)}")
        for fold in folds:
            counts = {}
            for label, start_key, end_key in [
                ("train", "train_start", "train_end"),
                ("validation", "validation_start", "validation_end"),
                ("test", "test_start", "test_end"),
            ]:
                sliced = returns.loc[fold[start_key] : fold[end_key]]
                counts[label] = len(sliced)
                if label in {"validation", "test"} and len(sliced) < config.problem.horizon_steps:
                    raise ValueError(
                        f"{fold['name']} n={n_assets} {label} has {len(sliced)} observations, "
                        f"fewer than horizon_steps={config.problem.horizon_steps}"
                    )
                min_train_observations = max(3 * config.problem.horizon_steps, 36)
                if label == "train" and len(sliced) < min_train_observations:
                    raise ValueError(
                        f"{fold['name']} n={n_assets} train period looks too short: "
                        f"{len(sliced)} observations; expected at least {min_train_observations}"
                    )
            print(
                f"  [fold] {fold['name']} train={counts['train']} "
                f"validation={counts['validation']} test={counts['test']}"
            )


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    raw = load_json(config_path)
    validate_methods(raw)
    config = build_config(raw)
    validate_folds(raw, config)
    print("[ok] revised experiment setup validated")


if __name__ == "__main__":
    main()

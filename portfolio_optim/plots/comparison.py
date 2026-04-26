from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _method_time(row: pd.Series) -> float:
    train = float(row.get("train_time_sec", -1.0))
    solve = float(row.get("solve_time_sec", -1.0))
    if train > 0:
        return train
    return max(solve, 0.0)


def _plot_metric_by_assets(summary: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    initial_wealths = sorted(summary["initial_wealth"].dropna().unique())
    methods = sorted(summary["method_name"].unique())
    fig, axes = plt.subplots(1, len(initial_wealths), figsize=(5.2 * max(len(initial_wealths), 1), 4), squeeze=False)
    for ax, initial_wealth in zip(axes.ravel(), initial_wealths):
        subset = summary[summary["initial_wealth"] == initial_wealth]
        grouped = subset.groupby(["method_name", "n_assets"], as_index=False)[metric].mean()
        for method in methods:
            method_rows = grouped[grouped["method_name"] == method].sort_values("n_assets")
            if method_rows.empty:
                continue
            ax.plot(method_rows["n_assets"], method_rows[metric], marker="o", label=method)
        ax.set_title(f"start={initial_wealth:.2f}")
        ax.set_xlabel("number of assets")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(summary: pd.DataFrame, output_path: Path) -> None:
    data = summary.copy()
    data["effective_time_sec"] = data.apply(_method_time, axis=1)
    _plot_metric_by_assets(data, "effective_time_sec", "runtime (sec)", output_path)


def _plot_nn_loss_curves(raw_dir: Path, output_dir: Path) -> None:
    nn_files = sorted(raw_dir.glob("nn_*.npz"))
    series: dict[tuple[int, float], list[tuple[str, list[float], list[float]]]] = {}
    for npz_path in nn_files:
        data = np.load(npz_path, allow_pickle=True)
        if int(data["n_assets"]) <= 1:
            continue
        metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
        train_loss = metadata.get("training_loss", [])
        val_loss = metadata.get("validation_loss", [])
        if not train_loss:
            continue
        key = (int(data["n_assets"]), float(data["initial_wealth"]))
        series.setdefault(key, []).append((str(data["method_name"]), train_loss, val_loss))

    for (n_assets, initial_wealth), items in series.items():
        items = sorted(items, key=lambda item: item[0])
        fig, axes = plt.subplots(len(items), 1, figsize=(7, 2.6 * len(items)), squeeze=False)
        for ax, (method, train_loss, val_loss) in zip(axes.ravel(), items):
            epochs = np.arange(1, len(train_loss) + 1)
            ax.plot(epochs, train_loss, label="train")
            if val_loss:
                ax.plot(epochs, val_loss, label="validation")
            ax.set_title(method)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        fig.suptitle(f"NN loss curves | assets={n_assets} | start={initial_wealth:.2f}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"nn_loss_curves_n{n_assets}_w{int(round(initial_wealth * 1000)):04d}.png", bbox_inches="tight")
        plt.close(fig)


def generate_experiment_plots(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    summary_path = results_dir / "summary" / "main_results.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return

    plots_dir = results_dir / "plots"
    _plot_metric_by_assets(summary, "target_hit_rate", "goal-hit probability", plots_dir / "goal_hit_rate_by_assets.png")
    _plot_metric_by_assets(summary, "expected_shortfall", "test expected shortfall", plots_dir / "test_error_expected_shortfall.png")
    _plot_runtime(summary, plots_dir / "runtime_by_assets.png")
    _plot_nn_loss_curves(results_dir / "raw", plots_dir)

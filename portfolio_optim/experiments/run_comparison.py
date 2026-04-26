from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from portfolio_optim.core.config import ExperimentConfig
from portfolio_optim.data.loaders import make_market_split
from portfolio_optim.evaluation.io import aggregate_npz_results, save_run
from portfolio_optim.evaluation.rollout import evaluate_solver
from portfolio_optim.fd.solver import FiniteDifferencePortfolioSolver
from portfolio_optim.market.estimators import estimate_market_parameters
from portfolio_optim.nn.solver import NeuralPortfolioSolver
from portfolio_optim.nn.trainer import resolve_torch_device
from portfolio_optim.plots.comparison import generate_experiment_plots


NN_ARCHITECTURES = ["mlp_shared", "mlp_deep", "bsde", "recurrent", "pinn", "transformer"]


def run_experiment(config: ExperimentConfig, methods: list[str] | None = None) -> None:
    selected_methods = methods or ["fd_hjb_sampled"] + [f"nn_{name}" for name in NN_ARCHITECTURES]
    if any(method.startswith("nn_") for method in selected_methods):
        print(f"[nn] training on device: {resolve_torch_device(config.nn.device).type}")
    print(
        "[experiment] target wealth:",
        config.problem.target_wealth,
        "| initial wealths:",
        ", ".join(f"{w:.2f}" for w in config.problem.initial_wealths),
    )
    for initial_wealth in config.problem.initial_wealths:
        run_config = deepcopy(config)
        run_config.problem.initial_wealth = float(initial_wealth)
        for n_assets in run_config.problem.asset_counts:
            tickers = run_config.data.ticker_groups[n_assets]
            split = make_market_split(run_config.data, tickers, run_config.problem.rebalance_frequency)
            market = estimate_market_parameters(split)
            for seed in run_config.problem.seed_list:
                for method in selected_methods:
                    if method in {"fd_hjb_sampled", "fd_hjb_viscosity"}:
                        solver = FiniteDifferencePortfolioSolver(run_config)
                    else:
                        solver = NeuralPortfolioSolver(run_config, method.removeprefix("nn_"))
                    solver_result = solver.fit(market, n_assets, seed)
                    solver_result.metadata["initial_wealth"] = run_config.problem.initial_wealth
                    solver_result.metadata["target_wealth"] = run_config.problem.target_wealth
                    solver_result.metadata["rebalance_frequency"] = run_config.problem.rebalance_frequency
                    solver_result.metadata["periods_per_year"] = market.periods_per_year
                    evaluation_result = evaluate_solver(solver, market, run_config, seed)
                    solver_result.eval_time_sec = evaluation_result.metrics["eval_time_sec"]
                    save_run(run_config.results_dir / "raw", solver_result, evaluation_result)
    aggregate_npz_results(config.results_dir / "raw", config.summary_dir / "main_results.csv")
    generate_experiment_plots(config.results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FD vs NN portfolio optimization comparison.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional subset of methods to run.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--nn-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device for neural-network methods.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        default=None,
        choices=["daily", "weekly", "monthly", "quarterly"],
        help="Sampling frequency used to build returns and apply the cash leg.",
    )
    parser.add_argument("--target-wealth", type=float, default=1.0, help="Terminal goal wealth.")
    parser.add_argument(
        "--initial-wealths",
        type=float,
        nargs="*",
        default=None,
        help="Starting wealth levels to evaluate, e.g. --initial-wealths 0.8 0.9 0.95",
    )
    parser.add_argument("--asset-counts", type=int, nargs="*", default=None, help="Asset counts to run, e.g. --asset-counts 20")
    parser.add_argument("--seed-list", type=int, nargs="*", default=None, help="Seeds to run, e.g. --seed-list 1 2 3")
    parser.add_argument("--n-wealth-points", type=int, default=None, help="Override FD wealth grid size.")
    parser.add_argument("--horizon-steps", type=int, default=None, help="Override number of time steps.")
    parser.add_argument("--n-control-samples", type=int, default=None, help="Override sampled control count for FD.")
    parser.add_argument("--nn-hidden-dim", type=int, default=None, help="Override NN hidden width.")
    parser.add_argument("--nn-depth", type=int, default=None, help="Override NN depth.")
    parser.add_argument("--nn-epochs", type=int, default=None, help="Override NN training epochs.")
    parser.add_argument("--nn-train-paths", type=int, default=None, help="Override NN training path count.")
    parser.add_argument("--nn-validation-paths", type=int, default=None, help="Override NN validation path count.")
    parser.add_argument("--nn-batch-size", type=int, default=None, help="Override NN batch size.")
    parser.add_argument("--nn-learning-rate", type=float, default=None, help="Override NN learning rate.")
    parser.add_argument(
        "--nn-path-sampler",
        default=None,
        choices=["bootstrap", "gaussian"],
        help="Path sampler used to train and validate the neural-network policies.",
    )
    parser.add_argument("--n-eval-paths", type=int, default=None, help="Override evaluation path count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(results_dir=args.results_dir, summary_dir=args.results_dir / "summary")
    config.nn.device = args.nn_device
    if args.rebalance_frequency is not None:
        config.problem.rebalance_frequency = args.rebalance_frequency
    config.problem.target_wealth = args.target_wealth
    if args.initial_wealths:
        config.problem.initial_wealths = list(args.initial_wealths)
        config.problem.initial_wealth = float(args.initial_wealths[0])
    if args.asset_counts:
        config.problem.asset_counts = list(args.asset_counts)
    if args.seed_list:
        config.problem.seed_list = list(args.seed_list)
    if args.n_wealth_points is not None:
        config.fd.n_wealth_points = args.n_wealth_points
    if args.horizon_steps is not None:
        config.problem.horizon_steps = args.horizon_steps
    if args.n_control_samples is not None:
        config.fd.n_control_samples = args.n_control_samples
    if args.nn_hidden_dim is not None:
        config.nn.hidden_dim = args.nn_hidden_dim
    if args.nn_depth is not None:
        config.nn.depth = args.nn_depth
    if args.nn_epochs is not None:
        config.nn.epochs = args.nn_epochs
    if args.nn_train_paths is not None:
        config.nn.train_paths = args.nn_train_paths
    if args.nn_validation_paths is not None:
        config.nn.validation_paths = args.nn_validation_paths
    if args.nn_batch_size is not None:
        config.nn.batch_size = args.nn_batch_size
    if args.nn_learning_rate is not None:
        config.nn.learning_rate = args.nn_learning_rate
    if args.nn_path_sampler is not None:
        config.nn.path_sampler = args.nn_path_sampler
    if args.n_eval_paths is not None:
        config.problem.n_eval_paths = args.n_eval_paths
    run_experiment(config, args.methods)


if __name__ == "__main__":
    main()

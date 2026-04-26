import json

import numpy as np

from portfolio_optim.core.types import EvaluationResult, SolverResult
from portfolio_optim.evaluation.io import aggregate_npz_results, save_run


def test_save_run_includes_initial_wealth_in_filename_and_payload(tmp_path):
    solver_result = SolverResult(
        method_family="fd",
        method_name="fd_hjb_viscosity",
        n_assets=5,
        seed=2,
        solve_time_sec=0.5,
        wealth_grid=np.array([0.8, 1.0]),
        value_grid=np.array([[0.0, 1.0]]),
        policy_grid=np.array([[[0.5, 0.5, 0.0, 0.0, 0.0]]]),
        metadata={"initial_wealth": 0.8, "target_wealth": 1.0},
    )
    evaluation_result = EvaluationResult(
        terminal_wealth=np.array([0.9, 1.1]),
        wealth_paths=np.array([[0.8, 0.9], [0.8, 1.1]]),
        weight_paths=np.array([[[0.5, 0.5, 0.0, 0.0, 0.0]]]),
        metrics={"target_hit_rate": 0.5},
    )

    path = save_run(tmp_path, solver_result, evaluation_result)
    data = np.load(path, allow_pickle=True)

    assert "w0800" in path.name
    assert float(data["initial_wealth"]) == 0.8
    assert float(data["target_wealth"]) == 1.0
    assert json.loads(str(data["metadata_json"]))["initial_wealth"] == 0.8


def test_aggregate_npz_results_computes_required_return(tmp_path):
    solver_result = SolverResult(
        method_family="fd",
        method_name="fd_hjb_viscosity",
        n_assets=1,
        seed=1,
        solve_time_sec=0.1,
        metadata={"initial_wealth": 0.9, "target_wealth": 1.0},
    )
    evaluation_result = EvaluationResult(
        terminal_wealth=np.array([1.0]),
        wealth_paths=np.array([[0.9, 1.0]]),
        weight_paths=np.array([[[1.0]]]),
        metrics={"target_hit_rate": 1.0},
    )
    save_run(tmp_path, solver_result, evaluation_result)

    frame = aggregate_npz_results(tmp_path, tmp_path / "summary.csv")

    assert frame.loc[0, "initial_wealth"] == 0.9
    assert frame.loc[0, "target_wealth"] == 1.0
    assert round(float(frame.loc[0, "required_return_pct"]), 4) == round(100.0 * (1.0 / 0.9 - 1.0), 4)

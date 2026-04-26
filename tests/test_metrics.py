import numpy as np

from portfolio_optim.evaluation.metrics import compute_terminal_metrics


def test_terminal_metrics_contains_expected_keys():
    terminal = np.array([0.9, 1.0, 1.2, 1.3])
    metrics = compute_terminal_metrics(terminal, 1.1)
    assert metrics["target_hit_rate"] == 0.5
    assert "expected_shortfall" in metrics


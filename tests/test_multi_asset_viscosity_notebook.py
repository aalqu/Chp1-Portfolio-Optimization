import json
import os
from pathlib import Path


def test_multi_asset_viscosity_notebook_executes():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    notebook_path = Path(__file__).resolve().parents[1] / "07_multi_asset_viscosity_goalreach.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__name__": "__main__"}

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        exec(compile(code, f"{notebook_path.name}", "exec"), namespace)

    summary = namespace["NOTEBOOK_SMOKE_SUMMARY"]
    errors = namespace["ASYMPTOTIC_ERROR_TABLE"]

    assert sorted(summary) == [1, 5, 10, 20]
    for n, item in summary.items():
        assert 0.0 <= item["V_0p80"] <= 1.0
        assert 0.0 <= item["V_0p90"] <= 1.0
        assert 0.0 <= item["V_1p00"] <= 1.0 + 1e-9
        assert 0.0 <= item["policy_sum_0p80"] <= 1.0 + 1e-9

    assert 0.005 in errors
    assert 0.01 in errors
    assert all(value >= 0.0 for value in errors.values())

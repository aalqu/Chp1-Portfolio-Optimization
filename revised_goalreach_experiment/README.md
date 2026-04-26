# Revised Goal-Reaching Experiment Package

This folder is a clean experiment kit built from the current Chapter 1 portfolio-optimization codebase. It keeps the original notebooks, modules, and result folders untouched, while adding a stronger methodology layer around them.

## What This Adds

- A written critique of the current methodology and implementation assumptions.
- A revised experiment protocol with pre-specified train, validation, and test folds.
- Baseline strategies so FD and neural methods are not compared in isolation.
- Fold-aware validation and test outputs, with each fold saved separately.
- Config files for a quick pilot and a heavier confirmatory run.
- Analysis scripts that combine fold summaries and produce uncertainty-aware ranking tables.

## Files

- `methodology_critique.md` - concise critique of the current experiment design.
- `experiment_protocol.md` - revised methodology to use for the next experiment.
- `configs/pilot.json` - small smoke/pilot run to verify the workflow.
- `configs/confirmatory.json` - fuller design for the main experiment.
- `scripts/validate_setup.py` - checks data availability, fold sizes, and method names.
- `scripts/run_revised_experiment.py` - runs the revised fold-aware experiment.
- `scripts/analyze_revised_results.py` - combines fold results and creates ranking tables.
- `outputs/` - intended location for generated outputs.

## Quick Start

Run these commands from the repository root:

```bash
python revised_goalreach_experiment/scripts/validate_setup.py \
  --config revised_goalreach_experiment/configs/pilot.json
```

```bash
python revised_goalreach_experiment/scripts/run_revised_experiment.py \
  --config revised_goalreach_experiment/configs/pilot.json
```

```bash
python revised_goalreach_experiment/scripts/analyze_revised_results.py \
  --results-dir revised_goalreach_experiment/outputs/pilot_run
```

For the larger run, replace `pilot.json` with `confirmatory.json`.

## Methodological Intent

The revised package treats the existing FD and NN solvers as research instruments, then wraps them in a more defensible experimental design:

- train on past data only,
- use validation folds for model comparison and tuning,
- reserve test folds for final reporting,
- include simple baselines,
- report variability across seeds and market regimes,
- keep FD multi-asset results labelled as approximate sampled-control benchmarks,
- save enough metadata for the results to be reproducible.

The pilot config is intentionally small. Use it first to make sure the data paths, folds, and dependencies are healthy before spending time on the confirmatory grid.


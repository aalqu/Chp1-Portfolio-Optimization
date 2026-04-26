# Revised Experiment Protocol

## Objective

Compare finite-difference HJB and neural-network portfolio policies for the goal-reaching problem under a methodology that supports reproducible, out-of-sample claims.

The primary claim should be about whether neural policies can match or improve the target-achievement and shortfall profile of the FD benchmark under the same market data, horizon, wealth target, and portfolio constraints.

## Primary Outcomes

- Target-hit rate: `P(W_T >= target)`.
- Expected shortfall: `E[max(target - W_T, 0)]`.

The primary ranking rule is:

1. higher validation target-hit rate,
2. lower validation expected shortfall,
3. lower turnover or concentration if the first two metrics are materially tied.

Final test summaries should be interpreted after this validation ranking is fixed.

## Secondary Outcomes

- Mean and median terminal wealth.
- Terminal wealth 5th percentile.
- Wealth volatility.
- Maximum drawdown.
- Gross leverage.
- Concentration.
- Maximum single-name weight.
- Turnover.
- Runtime and training/solve time.

## Methods

Required baselines:

- cash only,
- equal weight,
- mean-variance/tangency-style long-only baseline.

Core research methods:

- `fd_hjb_viscosity`,
- selected neural architectures, for example `nn_mlp_shared`, `nn_mlp_deep`, `nn_bsde`, `nn_pinn`, `nn_recurrent`, and `nn_transformer`.

For `n = 1`, FD can be treated as the strongest reference benchmark. For `n > 1`, FD must be reported as a sampled-control wealth-grid benchmark, not as an exact full-state HJB solution.

## Data Design

Use calendar folds. Each fold has:

- training period for parameter estimation and NN path simulation,
- validation period for model ranking and tuning,
- test period for final reporting.

The confirmatory config uses expanding training windows and multiple market regimes. The pilot config is intentionally small and should only be used to verify workflow health.

## Training And Evaluation Rules

- Estimate market parameters from the training period only.
- Do not use validation or test returns to fit market parameters.
- Neural training seeds must be repeated and reported.
- Evaluate each fitted solver on validation and test return windows separately.
- Keep validation and test outputs in separate directories.
- Do not choose architectures based on test results.

## Fairness Rules

- Use the same target wealth, initial wealth, horizon, frequency, risk-free rate, and constraints for all methods.
- Report the effective risk budget actually used: gross leverage, concentration, and max single-name weight.
- Include the baselines in every asset-count and initial-wealth scenario.
- For multi-asset FD, run or cite grid/control-sample sensitivity checks before using it as a reference curve.

## Statistical Reporting

Report summary statistics across folds and seeds:

- mean,
- standard deviation,
- standard error,
- 95 percent interval using `1.96 * SE` as a first-pass summary.

Because forward windows overlap, do not treat individual windows as independent observations for headline uncertainty. Use fold-level and seed-level aggregation for the main claims.

## Reproducibility Artifacts

Each run should save:

- copied config JSON,
- run manifest with Python/platform metadata,
- fold definitions,
- ticker groups,
- raw NPZ files,
- per-fold summary CSV files,
- combined analysis CSV files.


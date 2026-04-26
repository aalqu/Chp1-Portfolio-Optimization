# Methodology Critique

## Strengths

The current project has a solid experimental spine. The codebase separates configuration, data loading, solvers, evaluation, metrics, plotting, and tests. It also already includes both simulated-path evaluation and a forward-test runner on unseen historical returns. The raw NPZ plus summary CSV design is a good base for reproducibility.

The main issue is not that the experiment is weak. It is that the methodology is not yet strict enough for a chapter-level empirical claim about FD versus neural portfolio control.

## Key Concerns

1. The default Monte Carlo evaluation is in-sample unless the forward-test runner is used.

   `portfolio_optim/evaluation/rollout.py:13-16` samples evaluation paths from `market.historical_returns`, and `portfolio_optim/market/estimators.py` builds those market parameters from the training split. This is useful for controlled simulation, but it should not be treated as out-of-sample evidence.

2. The data split is a single chronological fraction split.

   `portfolio_optim/data/loaders.py:65-73` uses fixed train, validation, and test fractions. That is simple and clean, but one split cannot distinguish method quality from one lucky or unlucky market regime. A chapter experiment needs multiple calendar folds or expanding-window folds.

3. The forward test is helpful but still one split.

   `portfolio_optim/experiments/run_forward_test.py:21-55` evaluates on the test segment, but it does not produce repeated folds, validation-based model selection, or a confirmatory test-only report.

4. Rolling forward windows overlap.

   `portfolio_optim/evaluation/rollout.py:50-54` turns a test return sequence into overlapping windows. This increases path count but not independent evidence. Report the number of windows, but compute uncertainty at the fold or block level rather than treating all windows as independent observations.

5. The multi-asset FD solver should be described as an approximate benchmark.

   `portfolio_optim/fd/solver.py:116-122` states that the state remains one-dimensional in wealth and the control is sampled. This is a practical extension, not an exact high-dimensional HJB reference. It is credible as a sampled-control wealth-reduction benchmark, especially with convergence checks, but the text should avoid calling it the definitive PDE truth for `n > 1`.

6. The FD and NN policy classes do not have identical effective policy parameterizations.

   FD multi-asset controls are sampled from an admissible set in `portfolio_optim/fd/solver.py:55-71`. Neural policies use a softmax in `portfolio_optim/nn/solver.py:97-101`, which makes them naturally long-only and nearly fully invested before projection. This is fine for long-only experiments, but it should be named as a design choice and stress-tested under cash/risk-budget controls.

7. Baselines are missing from the core comparison.

   FD and NN methods should be compared against cash, equal-weight, and a simple mean-variance/tangency-style baseline. Without simple baselines, it is hard to tell whether a sophisticated method is actually adding value.

8. Reported metrics need uncertainty and selection discipline.

   `portfolio_optim/evaluation/metrics.py:12-40` includes useful terminal, weight, and path-risk metrics. The methodology should add confidence intervals, paired comparisons by fold/seed/scenario, and a rule that final test results are reported after model selection on validation only.

9. Transaction costs and turnover are not yet tied together.

   Turnover is computed in `portfolio_optim/evaluation/metrics.py:23-32`, but it is not translated into wealth drag. This can overstate the value of frequently changing policies.

10. Reproducibility metadata is incomplete.

   The current run outputs contain raw arrays and some metadata, but each run should also save the exact config file, fold dates, data root, ticker list, code revision when available, Python version, and dependency context.

## Additions Incorporated In This Package

- Calendar folds instead of one fraction split.
- Separate validation and test outputs for each fold.
- Simple baseline solvers.
- A pilot config and a confirmatory config.
- Artifact manifests copied into each run directory.
- Combined summary tables with mean, standard deviation, standard error, and 95 percent normal-approximation intervals.
- A written protocol that distinguishes exploratory tuning from final reporting.


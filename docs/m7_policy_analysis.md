# M7 — Cost-sensitive treatment policies and paired uncertainty

**Status:** exploratory; validation only. This milestone uses frozen predictions from six already-trained models. It does **not** retrain models, tune hyperparameters or access test outcomes.

## Research questions

1. If treating everybody costs resources, when might selective targeting be preferable?
2. Which model gives the highest estimated net response under fixed treatment budgets?
3. How uncertain are the comparisons to treating everyone and no one?
4. Are differences in Qini large relative to their paired bootstrap uncertainty?

## Data and policy definition

All four validation prediction files are joined using `client_id`. The tool rejects duplicate/missing IDs, nonfinite uplift values, mismatched treatment/outcome labels, inconsistent population sizes, and inconsistent training treatment fractions.

For model score `tau_hat(x)`, hypothetical cost `c` (measured in *conversion-probability equivalents per treated customer*), and budget `b` (fraction of **all** validation customers):

- Eligible customers have predicted uplift `> c`.
- Treat the highest-scoring eligible customers up to `floor(b * N)`.
- Other eligible customers are withheld if the budget is exhausted.
- Comparator policies are always treat-all and treat-none, **not** constrained by `b`; this is an explicit reference point, not a feasible budget policy when `b < 1`.

Expected net value is estimated by **IPW** minus `c * fraction_treated`. These are **illustrative unit conversion costs**, not observed business expenses. Never describe the resulting values as monetary profit.

## Uncertainty

Paired arm-stratified bootstrap resamples the same validation rows across models and reference policies. It gives percentile 95% intervals conditional on frozen predictions, fixed scenario parameters and the assumed treatment probability. Paired Qini intervals use the same bootstrap index across learners and report differences relative to X-Learner. These intervals do not account for refitting or choosing a winner based on validation results.

## Run

```bash
python -m upliftlab.evaluation.m7_policy \
  --m3-predictions models/m3_baselines/validation_predictions.csv \
  --m3-summary models/m3_baselines/validation_summary.json \
  --m4-predictions models/m4_x_learner/validation_predictions.csv \
  --m4-summary models/m4_x_learner/validation_summary.json \
  --m5-predictions models/m5_orthogonal_learners/validation_predictions.csv \
  --m5-summary models/m5_orthogonal_learners/validation_summary.json \
  --m6-predictions models/m6_causal_forest/validation_predictions.csv \
  --m6-summary models/m6_causal_forest/validation_summary.json \
  --output models/m7_policy_analysis \
  --costs 0 0.01 0.03 \
  --budgets 0.1 0.3 0.5 1.0 \
  --bootstrap 400 --qini-bootstrap 200 --seed 42
```

Outputs (ignored by Git under root `/models/`):

- `policy_analysis.json`: paired interval summaries for every scenario and Qini differences.
- `policy_scenarios.csv`: compact, plot-ready rows for every policy and scenario.

## Critical causal limitations

- Randomized assignment and the true assignment propensity have **not been independently verified**. The framework reuses the empirical training treatment frequency as an assumed propensity; thus policy values are exploratory, **not identified causal effects**.
- Verify that all engineered purchase-history features are strictly pre-treatment. Dataset-relative recency cannot prove this on its own.
- This comparison holds the validation population and the existing fitted models fixed; it does not establish model superiority in a new population.
- The treatment-cost and budget grids were predeclared for sensitivity analysis; inspecting this table and choosing a winner would be model selection, requiring a separate confirmatory evaluation.
- No claim of improvement should be made from a point estimate alone. A strong final study will also audit treatment balance, assignment provenance, propensity calibration and feature timing.
- The held-out 30,062-person test set stays untouched until a final protocol has been frozen.

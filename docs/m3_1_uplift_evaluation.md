# M3.1 — Shared exploratory uplift evaluation

## Research question

Do S-Learner and T-Learner rank likely incremental responders, and do their
zero-threshold policies improve *estimated* response over treating everyone?

This milestone does **not** establish causal effectiveness: the treatment
assignment mechanism and exact timing of engineered features require validation.

## Inputs and guarantees

The evaluator reads only M3's saved `validation_predictions.csv` and
`validation_summary.json`. It does not retrain or open the test dataset.
Predictions are fixed before evaluation; no threshold tuning is performed here.
Anonymized identifiers are checked for duplicates. M3's empirical treatment
fraction is reused only as an **unverified stand-in** for assignment propensity.

## Run

```bash
python -m upliftlab.evaluation.report \
  --predictions models/m3_baselines/validation_predictions.csv \
  --summary models/m3_baselines/validation_summary.json \
  --output models/m3_evaluation \
  --bootstrap 400 \
  --seed 42

pytest
```

Outputs (under the ignored root `models/` directory):

- `evaluation_summary.json`: per-model AUUC, Qini, uplift@10/20/30%,
  policy value, paired 95% percentile bootstrap confidence intervals
  for policy value and differences versus treat-all.
- `uplift_curves.csv`: full cumulative gains and random-targeting reference.

## Definitions

For binary outcome `Y`, treatment `T`, and assumed propensity `e`, the
transformed outcome is `Z = Y * (T/e - (1-T)/(1-e))`.

- **Cumulative gain** at top `k`: sum of `Z` for top-ranked customers divided
  by *all* validation customers, not just selected customers.
- **AUUC**: trapezoidal area under cumulative gain versus fraction selected.
- **Qini**: area between that gain curve and the straight random-targeting
  reference joining the origin and full-sample gain.
- **Uplift@K**: mean `Z` within the top K% (conditional incremental response)
  plus sum of `Z` divided by all validation customers (population contribution).
- **Policy value**: unnormalized inverse-propensity estimate of the binary
  zero-threshold policy's response; compared with treat-all and treat-none.
- **Confidence intervals**: arm-stratified paired row bootstrap, retracing
  identical rows for model policy and treat-all, with fixed trained predictions.

## Interpretation limits

1. X5 treatment assignment is not yet independently verified. Propensity is
   estimated from training data, not retrieved from experimental design.
2. Temporal leakage checks still need to establish pre-treatment status of all
   engineered features; dataset-relative dates are not actual intervention dates.
3. Bootstrap intervals condition on fixed predictions and policies and do not
   account for training variability, hyperparameter selection, or propensity
   uncertainty. They are **not** calibrated causal confidence intervals unless
   the requisite design assumptions hold.
4. AUUC/Qini use IPW transformed outcomes and depend on the propensity assumption.
   With tied scores, stable input ordering affects the finite-sample gain path.
5. A higher validation metric is exploratory; the held-out test should be used
   only once with a pre-registered final policy and comparison procedure.
6. These curves are not outcome-model calibration or ground-truth CATE accuracy.

## Next research checkpoints

Document the assignment mechanism, treatment timing, outcome window,
and covariate timing before making causal claims. Consider diagnostics of
pre-treatment balance, sensitivity to assumed propensity, and separate
policy-selection versus final-evaluation splits.

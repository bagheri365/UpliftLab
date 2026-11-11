# M8.3 — Conservative-feature sensitivity analysis

This is a **post-hoc validation-only sensitivity study**, not a confirmatory causal analysis.
It does not change the draft M8.2 protocol or open the 30,062-row test split.

## Rationale

The source labels `purchases.csv` as pre-communication history, but the exact
campaign cutoff has not been verified. The following three features depend on
the dataset-relative reference date and are therefore excluded:

- `historical_recency_days_dataset_relative`
- `historical_lines_last_30d_dataset_relative`
- `historical_lines_last_90d_dataset_relative`

**Important:** This exclusion reduces one timing concern but does **not** prove
that the remaining features are pre-treatment. In particular, the meaning and
timing of other historical aggregates still need independent verification.

## Apply and test

```bash
pytest
python -m upliftlab.evaluation.m83_sensitivity \
  --data data/interim/x5_features_enriched.csv \
  --audit models/m8_1_audit/audit_summary.json \
  --output models/m8_3_preflight \
  --prepare-only
```

The preflight validates the full input's SHA-256 against the M8.1 audit,
checks split counts, and writes a reduced **train+validation-only** CSV. Test
rows are skipped without looking at their outcome values. The reduced CSV is
stored under ignored `models/` outputs and should **never be committed**.

## Full six-model sensitivity run

```bash
python -m upliftlab.evaluation.m83_sensitivity \
  --data data/interim/x5_features_enriched.csv \
  --audit models/m8_1_audit/audit_summary.json \
  --original models/m7_policy_analysis/policy_analysis.json \
  --output models/m8_3_sensitivity \
  --folds 3 --trees 200 --min-leaf 20 \
  --bootstrap 400 --qini-bootstrap 200 --seed 42
```

The runner reuses the **existing** M3 S/T, M4 X, M5 R/DR and M6
`CausalForestDML` implementations, then the M7 evaluator. It fits only on the
train split and evaluates only on validation. EconML must be installed as in
`requirements-m6.txt`. The full run may be computationally expensive.

Outputs in `models/m8_3_sensitivity/`:

- `conservative_train_validation.csv` (no test rows, no excluded features)
- `m3_baselines/`, `m4_x_learner/`, `m5_orthogonal_learners/`, `m6_causal_forest/`
- `policy_comparison/policy_analysis.json` and `policy_scenarios.csv`
- `sensitivity_comparison.csv`: matched cost/budget/model changes in validation
  net policy value and targeting fractions
- `reproducibility_manifest.json`: input/audit hashes and environment versions

Always use a **fresh output directory** per run. The runner refuses to
silently overwrite a reduced dataset. Preserve prior artifacts to compare.

## Interpretation and research gates

The original and reduced-feature experiments share validation rows, so the
scenario table indicates robustness of **validation point estimates**, not a
new independent holdout or a p-value for the change. All policies are
conditional on estimated empirical propensity, not verified design propensity.
The M7 `treat_all` row remains an unconstrained reference when budgets are
below 100%; compare budget-feasible policies separately.

No treatment-assignment design, per-customer feature timing, dataset license,
or prospective pre-registration has been independently established. The
M8.2 protocol stays `draft`; the pre-test gate remains **closed**. Do not use
test outcomes or advertise confirmatory causal conclusions.

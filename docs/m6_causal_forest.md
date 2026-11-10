# M6 — Honest Causal Forest

## Research objective

Evaluate whether a **Causal Forest DML** with honest tree construction and
cross-fitted nuisance predictions improves uplift ranking or decision value over
S-, T-, X-, R-, and DR-Learners on the *same validation split*.

## Method

This milestone uses `econml.dml.CausalForestDML` (EconML 0.17.x), not a
standard random forest trained on pseudo-effects. Nuisance models predict the
binary outcome and treatment using LightGBM classifiers; EconML internally
cross-fits nuisance models (`cv=3`) and estimates heterogeneous effects with
an honest causal forest (`honest=True`, 200 trees, leaf size 20).

Model preprocessing is fitted on training rows only, using the existing M3
`FeatureEncoder` and approved feature allowlist. The forest is trained on the
training split; only validation records are scored and evaluated.

## Install

```bash
python -m pip install -r requirements-modeling.txt
python -m pip install -r requirements-m6.txt
pytest
```

## Train

```bash
python -m upliftlab.models.causal_forest \
  --data data/interim/x5_features_enriched.csv \
  --output models/m6_causal_forest \
  --folds 3 --trees 200 --min-leaf 20 --seed 42
```

Artifacts: `causal_forest.joblib`, `feature_encoder.joblib`,
`validation_predictions.csv`, `validation_summary.json`.

## Compare all six

```bash
python -m upliftlab.evaluation.m6_compare \
  --m3-predictions models/m3_baselines/validation_predictions.csv \
  --m3-summary models/m3_baselines/validation_summary.json \
  --m4-predictions models/m4_x_learner/validation_predictions.csv \
  --m4-summary models/m4_x_learner/validation_summary.json \
  --m5-predictions models/m5_orthogonal_learners/validation_predictions.csv \
  --m5-summary models/m5_orthogonal_learners/validation_summary.json \
  --m6-predictions models/m6_causal_forest/validation_predictions.csv \
  --m6-summary models/m6_causal_forest/validation_summary.json \
  --output models/m6_comparison --bootstrap 400 --seed 42
```

The comparison enforces matching customer IDs, treatment, outcome, validation
counts, and assumed propensities. It computes Qini, AUUC, top-K uplift,
zero-threshold policy value and fixed-policy bootstrap intervals. It leaves
`split == test` untouched.

## Limitations / responsible interpretation

- Treatment allocation probability is approximated using the training sample's
  treatment frequency. Randomization and design propensity are **not verified**.
- The campaign start time and pre-treatment timestamp validity of derived
  features have **not been independently verified**.
- Estimates and confidence intervals are **exploratory**, not established causal
  impacts. Bootstrap intervals condition on already-trained models.
- The forest is a different estimator family with different inductive biases;
  performance differences should not be attributed solely to honesty or cross-fitting.
- EconML is an optional dependency; the lightweight unit tests mock its backend.
  **Run training locally with EconML installed before claiming M6 verified.**
- M7 should compare model pairs using paired bootstrap confidence intervals,
  consider treatment costs, and reserve the held-out test split until the final
  frozen comparison.

## EconML API

https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html

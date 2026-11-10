# M4 — Cross-Fitted X-Learner

## Research question

Does a cross-fitted X-Learner improve uplift ranking (Qini/AUUC) or the estimated value of a positive-uplift targeting policy compared with the existing S- and T-Learners?

## Method

1. Use the same encoded pre-treatment candidate features and fixed customer-level splits as M3.
2. On the **training** split, form stratified cross-validation folds by treatment and binary outcome. For each fold, estimate conditional outcome models for both arms on the other folds; predict outcomes out of fold.
3. For treated training customers, impute effects as `Y - mu0_oof(X)`; for control customers, impute effects as `mu1_oof(X) - Y`.
4. Fit one LightGBM regression model to each arm's pseudo-outcomes.
5. Weight estimated control-group pseudo-effects by the training treatment fraction `e`, and treated-group pseudo-effects by `1-e`, then clip predicted differences to [-1, 1]. This is an X-Learner with a constant empirical blending propensity, **not** a verified randomization propensity.
6. Use the original **validation** split for evaluation; do not fit on validation or access the test set for model selection.

## Commands

```bash
python -m pip install -r requirements-modeling.txt
pytest

python -m upliftlab.models.x_learner \
  --data data/interim/x5_features_enriched.csv \
  --output models/m4_x_learner \
  --folds 3 --seed 42

python -m upliftlab.evaluation.m4_compare \
  --m3-predictions models/m3_baselines/validation_predictions.csv \
  --m3-summary models/m3_baselines/validation_summary.json \
  --m4-predictions models/m4_x_learner/validation_predictions.csv \
  --m4-summary models/m4_x_learner/validation_summary.json \
  --output models/m4_comparison \
  --bootstrap 400 --seed 42
```

## Generated outputs (ignored by Git)

- `models/m4_x_learner/validation_predictions.csv`: IDs, treatment, observed outcome, X-Learner effect prediction.
- `models/m4_x_learner/validation_summary.json`: training configuration, sample sizes, diagnostics and causal cautions.
- `models/m4_x_learner/x_learner.joblib`, `feature_encoder.joblib`: trained objects.
- `models/m4_comparison/evaluation_summary.json`: comparable Qini, AUUC, top-k, fixed-policy value, and bootstrap CI for all three learners.
- `models/m4_comparison/uplift_curves.csv`: per-model cumulative gain curves.

## Limitations

- Treatment assignment mechanism/design propensity is unverified; all IPW comparisons are **exploratory**, not confirmed causal estimates.
- Existing feature timing still requires audit. Dataset-relative recency is not necessarily intervention-relative recency.
- Bootstrap CIs hold trained models and zero-threshold policies fixed; they do not include training or model-selection uncertainty.
- Three-fold cross-fitting is internal to **training**, not a replacement for held-out validation.
- M4 model choice and hyperparameters are initially fixed for comparability; no test-set access until the final pre-specified comparison.

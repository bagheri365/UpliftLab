# M5 — Cross-fitted R-Learner and DR-Learner

## Goal

Add two orthogonalized treatment-effect estimators to the existing S/T/X comparison without changing the held-out test set or retraining earlier models.

## Methods

- **R-Learner:** Cross-fit the pooled outcome nuisance `m(x) = E[Y | X]`, then fit a weighted regression with pseudo-response `(Y - m(x)) / (T - e)` and squared residual-treatment weight `(T - e)^2`. This minimizes the empirical residualized R-loss with constant propensity `e`.
- **DR-Learner:** Cross-fit arm-specific outcome models `mu0(x)` and `mu1(x)`; fit a regression on the augmented inverse propensity pseudo-outcome `mu1-mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e)`.
- Nuisance prediction is out-of-fold with stratification across treatment/outcome cells. LightGBM is used throughout, matching earlier baselines.
- The propensity is the **empirical training treatment fraction**, not a confirmed experimental probability. Despite its name, the DR-Learner here does *not* justify a causal interpretation in the absence of design verification.

## Run

```bash
python -m upliftlab.models.orthogonal \
  --data data/interim/x5_features_enriched.csv \
  --output models/m5_orthogonal_learners \
  --folds 3 --seed 42

python -m upliftlab.evaluation.m5_compare \
  --m3-predictions models/m3_baselines/validation_predictions.csv \
  --m3-summary models/m3_baselines/validation_summary.json \
  --m4-predictions models/m4_x_learner/validation_predictions.csv \
  --m4-summary models/m4_x_learner/validation_summary.json \
  --m5-predictions models/m5_orthogonal_learners/validation_predictions.csv \
  --m5-summary models/m5_orthogonal_learners/validation_summary.json \
  --output models/m5_comparison --bootstrap 400 --seed 42
```

Training creates two learner joblib artifacts, an encoder, `validation_predictions.csv`, and `validation_summary.json`. Comparison writes `evaluation_summary.json` and `uplift_curves.csv` for all five learners. Generated results stay under the git-ignored root `models/`.

## Interpretation constraints

- All conclusions are **exploratory** until treatment assignment, propensity, and feature timing are verified.
- Validation model comparison reuses the same individuals and treatment/outcome labels.
- Bootstraps condition on fitted models and fixed policies; they do not account for training, hyperparameter search, or selecting the best model.
- The `test` split is never used in M5.
- Causal rankings and effect estimates are not established by these outputs alone.

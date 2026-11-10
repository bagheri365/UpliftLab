# M3 — S-Learner and T-Learner baselines

## Research objective

Can a personalized show/hide policy estimated from historical treatment data outperform treating everyone or no one, **on held-out validation data**?

**Causal caveat:** X5 treatment assignment and the timing of engineered features still need independent verification. All IPW policy-value estimates in this milestone are **provisional diagnostics**, not established causal effects. Do not optimize the business policy on these numbers yet.

## Setup

From the project root (Python 3.12 virtual environment already active):

```bash
python -m pip install -r requirements-modeling.txt
pytest
```

The project retains its existing `pyproject.toml`; modeling dependencies are isolated in `requirements-modeling.txt` to keep this patch additive and avoid changing the existing M0 build configuration. Pin exact versions in a future lockfile for stronger cross-machine reproducibility.

## Training

First ensure the M2.1 feature table exists:

```bash
python -m upliftlab.data.x5_enrich \
  --features data/interim/x5_features.csv \
  --purchases data/raw/x5/purchases.csv.gz \
  --output data/interim/x5_features_enriched.csv
```

Then:

```bash
python -m upliftlab.models.baselines \
  --data data/interim/x5_features_enriched.csv \
  --output models/m3_baselines
```

## Models and protocol

- **S-Learner:** LightGBM classifier using treatment as an additional input. Predict each customer's outcome under `T=0` and `T=1`, then subtract.
- **T-Learner:** Two LightGBM classifiers trained separately by treatment arm. Predict outcomes under each arm, then subtract.
- Both use the same base hyperparameters and seed.
- Inputs are only `age`, `gender`, `has_profile`, and `historical_*` features. Customer ID, split, treatment and outcome are never included in the underlying covariate matrix.
- Numeric imputation and categorical mapping are fitted from **train only**.
- The provided `train` split is used for fitting; `validation` is used for diagnostics; `test` is never used for prediction or tuning in M3.
- Predicted uplift is a difference in two model-estimated probabilities, not an observed individual treatment effect.

## Outputs

In `models/m3_baselines/` (Git ignored by the existing `.gitignore`):

- `s_learner.joblib`, `t_learner.joblib`: fitted treatment-outcome models
- `feature_encoder.joblib`: train-fitted encoder
- `validation_predictions.csv`: per-customer validation `p0`, `p1`, and uplift predictions
- `validation_summary.json`: train/validation/test counts, features, mean predicted uplift, share targeted at threshold 0, and **provisional** IPW policy-value diagnostics

## Limitations and next checks

1. **Randomization not yet proven.** The IPW diagnostics use the training treatment rate as an *approximation* of the experimental propensity; this is not a substitute for the actual assignment probabilities.
2. **Feature leakage risk.** X5 documentation describes historical purchases as before communication, but the exact campaign cutoff and the reference date for recency features must be independently checked.
3. **Model selection.** Comparing policies on the same validation split used for threshold/hyperparameter tuning introduces selection bias. Select and freeze the policy before the final untouched test evaluation.
4. **Policy value.** IPW is high variance and can have estimates outside [0,1] in small samples. Later milestones should add confidence intervals, doubly robust policy evaluation, uplift/Qini curves, and formally documented identification assumptions.
5. **Artifact safety.** Only load `.joblib` files you generated or trust; pickle-based formats can execute code during loading.

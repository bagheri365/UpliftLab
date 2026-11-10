"""M5 cross-fitted R-Learner and DR-Learner for binary uplift.

Exploratory only: X5 assignment propensities and treatment timing unverified.
No test-set outcomes are used in training or evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold

from upliftlab.models.baselines import (
    FeatureEncoder, base_model, ipw_policy_value, read_table, select_features,
)


def _effect_model(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=120, learning_rate=0.05, num_leaves=15,
        min_child_samples=25, random_state=seed, n_jobs=1, verbosity=-1,
    )


def _check_inputs(X: pd.DataFrame, t: np.ndarray, y: np.ndarray, folds: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(X, pd.DataFrame) or len(X) == 0 or len(X) != len(t) or len(t) != len(y):
        raise ValueError("X, treatment and outcome must have matching nonempty lengths")
    t = np.asarray(t)
    y = np.asarray(y)
    if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Treatment and outcome must be binary")
    if folds < 2 or np.min(np.bincount((2 * t + y).astype(int), minlength=4)) < folds:
        raise ValueError("Each treatment/outcome cell must contain at least folds observations")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Encoded features must all be finite")
    return t.astype(int), y.astype(int)


class OrthogonalLearner:
    """Cross-fitted nuisance regressions with R or doubly robust second stage.

    Assumes constant propensity estimated from training treatment frequency.
    This is a provisional approximation, not verified randomization.
    Cross-fitting avoids in-sample outcome pseudo-labels; the second stage
    still fits on the full training sample and validation is held out.
    """

    def __init__(self, method: str, seed: int = 42, folds: int = 3):
        if method not in {"r_learner", "dr_learner"}:
            raise ValueError("method must be r_learner or dr_learner")
        self.method = method
        self.seed = seed
        self.folds = folds

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "OrthogonalLearner":
        t, y = _check_inputs(X, t, y, self.folds)
        self.propensity_ = float(t.mean())
        p = self.propensity_
        # Pooled outcome regression estimates E[Y|X]. Arm models estimate E[Y|T=a,X].
        oof_m = np.full(len(y), np.nan)
        oof0 = np.full(len(y), np.nan)
        oof1 = np.full(len(y), np.nan)
        folds = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        strata = 2 * t + y
        for fit_idx, hold_idx in folds.split(X, strata):
            pooled = base_model(self.seed).fit(X.iloc[fit_idx], y[fit_idx])
            oof_m[hold_idx] = pooled.predict_proba(X.iloc[hold_idx])[:, 1]
            for arm, pred in ((0, oof0), (1, oof1)):
                arm_fit = fit_idx[t[fit_idx] == arm]
                model = base_model(self.seed).fit(X.iloc[arm_fit], y[arm_fit])
                pred[hold_idx] = model.predict_proba(X.iloc[hold_idx])[:, 1]
        if not (np.isfinite(oof_m).all() and np.isfinite(oof0).all() and np.isfinite(oof1).all()):
            raise RuntimeError("Incomplete out-of-fold nuisance predictions")

        if self.method == "r_learner":
            # Robinson / R-loss: (Y-m(X) - (T-e)tau(X))^2.
            residual_t = t - p
            pseudo = (y - oof_m) / residual_t
            weights = residual_t ** 2
        else:
            # AIPW/DR pseudo-outcome for E[Y(1)-Y(0)|X].
            pseudo = oof1 - oof0 + t * (y - oof1) / p - (1 - t) * (y - oof0) / (1 - p)
            weights = None
        self.effect_model_ = _effect_model(self.seed)
        self.effect_model_.fit(X, pseudo, sample_weight=weights)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "effect_model_"):
            raise RuntimeError("Fit learner before predicting")
        result = np.asarray(self.effect_model_.predict(X), dtype=float)
        return np.clip(result, -1.0, 1.0)


def run(data: str | Path, output_dir: str | Path, *, seed: int = 42, folds: int = 3) -> dict:
    df = read_table(data)
    train, valid = df.loc[df.split == "train"], df.loc[df.split == "validation"]
    features = select_features(train)
    encoder = FeatureEncoder().fit(train, features)
    X_train, X_valid = encoder.transform(train), encoder.transform(valid)
    t_train = train.treatment_flg.to_numpy(int)
    y_train = train.target.to_numpy(int)
    p = float(t_train.mean())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    preds = valid[["client_id", "treatment_flg", "target"]].copy()
    summary = {
        "status": "exploratory_only_assignment_unverified", "split": "validation_only",
        "seed": seed, "folds": folds, "train_rows": len(train),
        "validation_rows": len(valid), "test_rows_not_used": int((df.split == "test").sum()),
        "approved_features": features, "encoded_features": list(X_train.columns),
        "train_treatment_fraction": p,
        "propensity_source": "empirical_train_fraction_not_verified_design_probability",
        "models": {},
        "notes": ["Assignment probability is assumed constant and estimated from training frequency.",
                  "Pre-treatment feature timing and randomization remain unverified.",
                  "All nuisance models are cross-fitted; validation only; test remains unused."],
    }
    for name in ("r_learner", "dr_learner"):
        learner = OrthogonalLearner(name, seed=seed, folds=folds).fit(X_train, t_train, y_train)
        scores = learner.predict_uplift(X_valid)
        preds[f"{name}_uplift"] = scores
        summary["models"][name] = {
            "mean_predicted_uplift": float(scores.mean()),
            "fraction_targeted_at_zero_threshold": float(np.mean(scores > 0)),
            "provisional_ipw_policy_value": ipw_policy_value(
                valid.treatment_flg.to_numpy(int), valid.target.to_numpy(int),
                (scores > 0).astype(int), p,
            ),
        }
        joblib.dump(learner, output / f"{name}.joblib")
    joblib.dump(encoder, output / "feature_encoder.joblib")
    preds.to_csv(output / "validation_predictions.csv", index=False)
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train cross-fitted R- and DR-Learners (validation only)")
    parser.add_argument("--data", default="data/interim/x5_features_enriched.csv")
    parser.add_argument("--output", default="models/m5_orthogonal_learners")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.output, seed=args.seed, folds=args.folds), indent=2))


if __name__ == "__main__":
    main()

"""M4 cross-fitted X-Learner for binary uplift, evaluated on validation only.

Exploratory until treatment assignment and feature timing are established.
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
    FeatureEncoder, base_model, read_table, select_features, ipw_policy_value,
)


class XLearner:
    """Arm-specific outcome models + cross-fitted imputed-effect regressors.

    Training-arm outcomes are never predicted by outcome models fitted on the
    same row. Each fold is stratified by (treatment, outcome), so both arms
    have both outcome classes in the fitting sample.
    """

    def __init__(self, seed: int = 42, folds: int = 3):
        if folds < 2:
            raise ValueError("folds must be >= 2")
        self.seed = seed
        self.folds = folds

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "XLearner":
        t = np.asarray(t, dtype=int)
        y = np.asarray(y, dtype=int)
        if not isinstance(X, pd.DataFrame) or len(X) != len(t) or len(t) != len(y):
            raise ValueError("X, treatment and outcome must have matching lengths")
        if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Treatment and outcome must be binary")
        strata = 2 * t + y
        counts = np.bincount(strata, minlength=4)
        if np.min(counts) < self.folds:
            raise ValueError("Each treatment/outcome cell needs at least folds observations")
        self.propensity_ = float(t.mean())
        oof0 = np.full(len(t), np.nan)
        oof1 = np.full(len(t), np.nan)
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        for train_idx, held_idx in skf.split(X, strata):
            for arm, pred in ((0, oof0), (1, oof1)):
                arm_idx = train_idx[t[train_idx] == arm]
                nuisance = base_model(self.seed)
                nuisance.fit(X.iloc[arm_idx], y[arm_idx])
                pred[held_idx] = nuisance.predict_proba(X.iloc[held_idx])[:, 1]
        if not np.isfinite(oof0).all() or not np.isfinite(oof1).all():
            raise RuntimeError("Incomplete out-of-fold predictions")
        effect_treated = y[t == 1] - oof0[t == 1]
        effect_control = oof1[t == 0] - y[t == 0]
        def regressor() -> LGBMRegressor:
            return LGBMRegressor(n_estimators=120, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=25, random_state=self.seed,
                                 n_jobs=1, verbosity=-1)
        self.effect_treated_ = regressor().fit(X.iloc[np.flatnonzero(t == 1)], effect_treated)
        self.effect_control_ = regressor().fit(X.iloc[np.flatnonzero(t == 0)], effect_control)
        self.outcome_control_ = base_model(self.seed).fit(X.iloc[np.flatnonzero(t == 0)], y[t == 0])
        self.outcome_treated_ = base_model(self.seed).fit(X.iloc[np.flatnonzero(t == 1)], y[t == 1])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "effect_treated_"):
            raise RuntimeError("Fit learner before prediction")
        # With constant propensity e, weight control effect model by e.
        tau = (self.propensity_ * self.effect_control_.predict(X)
               + (1 - self.propensity_) * self.effect_treated_.predict(X))
        return np.clip(np.asarray(tau, dtype=float), -1.0, 1.0)


def run(data: str | Path, output_dir: str | Path, *, seed: int = 42, folds: int = 3) -> dict:
    df = read_table(data)
    train, valid = df.loc[df.split == "train"], df.loc[df.split == "validation"]
    features = select_features(train)
    encoder = FeatureEncoder().fit(train, features)
    x_train, x_valid = encoder.transform(train), encoder.transform(valid)
    t_train = train.treatment_flg.to_numpy(int)
    y_train = train.target.to_numpy(int)
    learner = XLearner(seed=seed, folds=folds).fit(x_train, t_train, y_train)
    uplift = learner.predict_uplift(x_valid)
    preds = valid[["client_id", "treatment_flg", "target"]].copy()
    preds["x_learner_uplift"] = uplift
    propensity = float(t_train.mean())
    summary = {
        "status": "exploratory_only_assignment_unverified", "split": "validation_only",
        "seed": seed, "folds": folds, "train_rows": len(train),
        "validation_rows": len(valid), "test_rows_not_used": int((df.split == "test").sum()),
        "approved_features": features, "encoded_features": list(x_train.columns),
        "train_treatment_fraction": propensity,
        "mean_predicted_uplift": float(np.mean(uplift)),
        "fraction_targeted_at_zero_threshold": float(np.mean(uplift > 0)),
        "provisional_ipw_policy_value": ipw_policy_value(
            valid.treatment_flg.to_numpy(int), valid.target.to_numpy(int),
            (uplift > 0).astype(int), propensity),
        "notes": ["Propensity is empirical, not confirmed design probability.",
                  "Pre-treatment feature timing and randomization remain unverified.",
                  "Only training data were used to fit models, validation for evaluation; test untouched."],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "validation_predictions.csv", index=False)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    joblib.dump(learner, out / "x_learner.joblib")
    joblib.dump(encoder, out / "feature_encoder.joblib")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted X-Learner on existing X5 train/validation splits")
    parser.add_argument("--data", default="data/interim/x5_features_enriched.csv")
    parser.add_argument("--output", default="models/m4_x_learner")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.output, seed=args.seed, folds=args.folds), indent=2))


if __name__ == "__main__":
    main()

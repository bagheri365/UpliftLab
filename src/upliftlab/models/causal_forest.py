"""M6 honest causal forest on X5 train data; validation-only effect scoring.

Requires optional EconML 0.17.x. Causal interpretation requires verified assignment
and pre-intervention covariates; neither is established for X5 here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from upliftlab.models.baselines import (
    FeatureEncoder, base_model, ipw_policy_value, read_table, select_features,
)


def _forest_class():
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError(
            "M6 requires EconML. Run: python -m pip install -r requirements-m6.txt"
        ) from exc
    return CausalForestDML


class HonestCausalForest:
    """Binary-treatment CATE estimation with cross-fit nuisance models and honest trees."""

    def __init__(self, *, seed: int = 42, folds: int = 3, trees: int = 200,
                 min_leaf: int = 20):
        if folds < 2 or trees < 4 or trees % 4 or min_leaf < 2:
            raise ValueError("Require folds >=2, trees >=4 divisible by 4, min_leaf >=2")
        self.seed, self.folds, self.trees, self.min_leaf = seed, folds, trees, min_leaf

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "HonestCausalForest":
        t, y = np.asarray(t), np.asarray(y)
        if not isinstance(X, pd.DataFrame) or len(X) == 0 or len(X) != len(t) or len(t) != len(y):
            raise ValueError("X, treatment, outcome must have matching, nonempty lengths")
        if not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Treatment and outcome must be binary")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Encoded features must be finite")
        if min(np.bincount(2 * t.astype(int) + y.astype(int), minlength=4)) < self.folds:
            raise ValueError("Need >= folds samples in each treatment/outcome cell")
        forest_type = _forest_class()
        self.model_ = forest_type(
            model_y=base_model(self.seed),
            model_t=base_model(self.seed + 1),
            discrete_outcome=True,
            discrete_treatment=True,
            cv=self.folds,
            honest=True,
            n_estimators=self.trees,
            min_samples_leaf=self.min_leaf,
            random_state=self.seed,
            n_jobs=1,
            inference=False,
        )
        self.model_.fit(y.astype(int), t.astype(int), X=X.to_numpy(dtype=float))
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Fit the forest before predicting")
        effects = np.asarray(self.model_.effect(X.to_numpy(dtype=float)), dtype=float).reshape(-1)
        if len(effects) != len(X) or not np.isfinite(effects).all():
            raise ValueError("Invalid causal forest effect predictions")
        return np.clip(effects, -1.0, 1.0)


def run(data: str | Path, output_dir: str | Path, *, seed: int = 42, folds: int = 3,
        trees: int = 200, min_leaf: int = 20) -> dict:
    df = read_table(data)
    train = df.loc[df.split == "train"]
    valid = df.loc[df.split == "validation"]
    features = select_features(train)
    encoder = FeatureEncoder().fit(train, features)
    X_train, X_valid = encoder.transform(train), encoder.transform(valid)
    t_train, y_train = train.treatment_flg.to_numpy(int), train.target.to_numpy(int)
    p = float(t_train.mean())
    learner = HonestCausalForest(seed=seed, folds=folds, trees=trees, min_leaf=min_leaf)
    learner.fit(X_train, t_train, y_train)
    uplift = learner.predict_uplift(X_valid)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = valid[["client_id", "treatment_flg", "target"]].copy()
    preds["causal_forest_uplift"] = uplift
    preds.to_csv(out / "validation_predictions.csv", index=False)
    summary = {
        "status": "exploratory_only_assignment_unverified",
        "split": "validation_only", "seed": seed, "folds": folds,
        "trees": trees, "min_leaf": min_leaf,
        "train_rows": len(train), "validation_rows": len(valid),
        "test_rows_not_used": int((df.split == "test").sum()),
        "approved_features": features, "encoded_features": list(X_train.columns),
        "train_treatment_fraction": p,
        "propensity_source": "empirical_train_fraction_not_verified_design_probability",
        "models": {"causal_forest": {
            "mean_predicted_uplift": float(uplift.mean()),
            "fraction_targeted_at_zero_threshold": float(np.mean(uplift > 0)),
            "provisional_ipw_policy_value": ipw_policy_value(
                valid.treatment_flg.to_numpy(int), valid.target.to_numpy(int),
                (uplift > 0).astype(int), p),
        }},
        "notes": [
            "EconML cross-fits outcome and propensity nuisance models; forest uses honest splitting.",
            "Treatment assignment and pre-treatment feature timing have not been verified.",
            "Validation only; no held-out test set outcomes used.",
            "No EconML inference intervals calculated; policy intervals are conditional on fitted models.",
        ],
    }
    joblib.dump(learner, out / "causal_forest.joblib")
    joblib.dump(encoder, out / "feature_encoder.joblib")
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M6 honest causal forest, validation only")
    parser.add_argument("--data", default="data/interim/x5_features_enriched.csv")
    parser.add_argument("--output", default="models/m6_causal_forest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--min-leaf", type=int, default=20)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.output, seed=args.seed, folds=args.folds,
                         trees=args.trees, min_leaf=args.min_leaf), indent=2))


if __name__ == "__main__":
    main()

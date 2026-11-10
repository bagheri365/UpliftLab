"""M3 S- and T-Learners for binary treatment and binary outcomes.

Features are selected using an allowlist to prevent treatment/outcome/ID leakage.
All preprocessing is fitted on the training split only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROTECTED = frozenset({"client_id", "split", "treatment_flg", "target"})


def select_features(df: pd.DataFrame) -> list[str]:
    """Select documented pre-treatment customer features (never treatment/outcome)."""
    cols = [c for c in df if c in {"age", "gender", "has_profile"} or c.startswith("historical_")]
    if not cols:
        raise ValueError("No approved pre-treatment features found")
    return cols


def read_table(path: str | Path) -> pd.DataFrame:
    """Validate the existing customer-level split table."""
    df = pd.read_csv(path, low_memory=False)
    required = PROTECTED
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty or df.client_id.isna().any() or df.client_id.duplicated().any():
        raise ValueError("Customer IDs must be unique and nonmissing")
    if not df.split.isin({"train", "validation", "test"}).all():
        raise ValueError("Unexpected split values")
    if any(df.loc[:, c].isna().any() or not df[c].isin({0, 1}).all() for c in ("treatment_flg", "target")):
        raise ValueError("Treatment and outcome must be binary without missing values")
    for split in ("train", "validation"):
        subset = df[df.split == split]
        if subset.empty or set(subset.treatment_flg) != {0, 1}:
            raise ValueError(f"{split} must contain both treatment arms")
        # A constant outcome in either training arm makes a classifier invalid.
        if split == "train":
            for arm in (0, 1):
                if subset.loc[subset.treatment_flg == arm, "target"].nunique() != 2:
                    raise ValueError(f"Training arm {arm} must include both outcome classes")
    return df


class FeatureEncoder:
    """Training-fitted numeric imputation and deterministic gender encoding."""

    def fit(self, df: pd.DataFrame, features: list[str]) -> "FeatureEncoder":
        self.features_ = list(features)
        self.numeric_ = [c for c in features if c != "gender"]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        if self.numeric_:
            self.imputer_.fit(self._numeric(df))
        return self

    def _numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(columns=self.numeric_).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "features_"):
            raise RuntimeError("Encoder must be fitted before transform")
        if any(c not in df.columns for c in self.features_):
            raise ValueError("Missing approved feature in evaluation table")
        result = pd.DataFrame(index=df.index)
        if self.numeric_:
            values = self.imputer_.transform(self._numeric(df))
            for i, name in enumerate(self.numeric_):
                result[name] = values[:, i]
        if "gender" in self.features_:
            gender = df.gender.fillna("U").astype(str).str.upper()
            result["gender_F"] = (gender == "F").astype(int)
            result["gender_M"] = (gender == "M").astype(int)
        return result


def base_model(seed: int) -> LGBMClassifier:
    """Use the same classifier specification for both meta-learners."""
    return LGBMClassifier(
        n_estimators=120, learning_rate=0.05, num_leaves=15,
        min_child_samples=25, random_state=seed, n_jobs=1, verbosity=-1,
    )


class SLearner:
    """One model, conditioning on a binary treatment feature."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "SLearner":
        self.model_ = base_model(self.seed)
        self.model_.fit(X.assign(__treatment=np.asarray(t)), y)
        return self

    def potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p0 = self.model_.predict_proba(X.assign(__treatment=0))[:, 1]
        p1 = self.model_.predict_proba(X.assign(__treatment=1))[:, 1]
        return p0, p1

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.potential_outcomes(X)
        return p1 - p0


class TLearner:
    """Two independent outcome models, one per randomized arm."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> "TLearner":
        t = np.asarray(t)
        self.control_ = base_model(self.seed)
        self.treated_ = base_model(self.seed)
        self.control_.fit(X.loc[t == 0], np.asarray(y)[t == 0])
        self.treated_.fit(X.loc[t == 1], np.asarray(y)[t == 1])
        return self

    def potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.control_.predict_proba(X)[:, 1], self.treated_.predict_proba(X)[:, 1]

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.potential_outcomes(X)
        return p1 - p0


def ipw_policy_value(t: np.ndarray, y: np.ndarray, actions: np.ndarray, propensity: float) -> float:
    """Unnormalized inverse-propensity policy value; assumes valid known assignment probability.

    Use only for provisional diagnostics until X5 assignment is verified.
    """
    if not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between 0 and 1")
    t, y, actions = map(np.asarray, (t, y, actions))
    if not (len(t) == len(y) == len(actions)) or len(t) == 0:
        raise ValueError("Treatment, outcome and actions must have equal nonzero length")
    if not np.isin(actions, [0, 1]).all():
        raise ValueError("Policy actions must be binary")
    w = np.where(t == 1, propensity, 1 - propensity)
    return float(np.mean((t == actions) * y / w))


def run(path: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    df = read_table(path)
    train, valid = df[df.split == "train"], df[df.split == "validation"]
    features = select_features(train)
    encoder = FeatureEncoder().fit(train, features)
    X_train, X_valid = encoder.transform(train), encoder.transform(valid)
    t_train = train.treatment_flg.to_numpy(dtype=int)
    y_train = train.target.to_numpy(dtype=int)
    t_valid = valid.treatment_flg.to_numpy(dtype=int)
    y_valid = valid.target.to_numpy(dtype=int)
    propensity = float(t_train.mean())  # Diagnostic approximation, NOT verified design propensity.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = valid[["client_id", "treatment_flg", "target"]].copy()
    summary = {
        "status": "exploratory_only_assignment_unverified",
        "data": str(path), "seed": seed, "train_rows": len(train), "validation_rows": len(valid),
        "test_rows_not_used": int((df.split == "test").sum()), "approved_features": features,
        "encoded_features": list(X_train.columns), "train_treatment_fraction": propensity,
        "note": "IPW diagnostics use empirical training treatment fraction as if it were the design propensity; do not claim causal validity until assignment and feature timing are verified.",
        "models": {},
    }
    for name, learner in (("s_learner", SLearner(seed)), ("t_learner", TLearner(seed))):
        learner.fit(X_train, t_train, y_train)
        p0, p1 = learner.potential_outcomes(X_valid)
        uplift = p1 - p0
        predictions[f"{name}_p0"] = p0
        predictions[f"{name}_p1"] = p1
        predictions[f"{name}_uplift"] = uplift
        summary["models"][name] = {
            "mean_predicted_uplift": float(np.mean(uplift)),
            "fraction_targeted_at_zero_threshold": float(np.mean(uplift > 0)),
            "provisional_ipw_policy_value": ipw_policy_value(t_valid, y_valid, uplift > 0, propensity),
        }
        joblib.dump(learner, out / f"{name}.joblib")
    summary["provisional_comparators"] = {
        "treat_all": ipw_policy_value(t_valid, y_valid, np.ones(len(valid), dtype=int), propensity),
        "treat_none": ipw_policy_value(t_valid, y_valid, np.zeros(len(valid), dtype=int), propensity),
    }
    joblib.dump(encoder, out / "feature_encoder.joblib")
    predictions.to_csv(out / "validation_predictions.csv", index=False)
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit S-/T-Learner baselines on X5 validation split")
    parser.add_argument("--data", default="data/interim/x5_features_enriched.csv")
    parser.add_argument("--output", default="models/m3_baselines")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.data, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

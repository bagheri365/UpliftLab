"""M3.1: exploratory offline uplift evaluation of saved M3 predictions.

IMPORTANT: IPW estimands require known or otherwise justified assignment
propensities. The X5 propensity supplied by M3 is empirical, not confirmed by
experiment documentation. Do not interpret these diagnostics as proven causality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAMES = ("s_learner", "t_learner")
TOP_FRACTIONS = (0.10, 0.20, 0.30)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read the untouched M3 validation predictions and enforce their schema."""
    df = pd.read_csv(path, low_memory=False)
    required = {"client_id", "treatment_flg", "target"}
    for name in MODEL_NAMES:
        required.update((f"{name}_uplift", f"{name}_p0", f"{name}_p1"))
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df.client_id.isna().any() or df.client_id.duplicated().any():
        raise ValueError("Validation client_id must be unique and nonmissing")
    for c in ("treatment_flg", "target"):
        if df[c].isna().any() or not df[c].isin([0, 1]).all():
            raise ValueError(f"{c} must be binary without missing values")
    if set(df.treatment_flg) != {0, 1}:
        raise ValueError("Both treatment arms are required")
    for name in MODEL_NAMES:
        cols = [f"{name}_uplift", f"{name}_p0", f"{name}_p1"]
        vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f"Missing or nonfinite predictions for {name}")
        if ((vals[:, 1:] < 0) | (vals[:, 1:] > 1)).any():
            raise ValueError(f"Predicted probabilities out of range for {name}")
    return df


def transformed_outcome(t: np.ndarray, y: np.ndarray, propensity: float) -> np.ndarray:
    """IPW transformed outcome for incremental purchase probability."""
    if not np.isfinite(propensity) or not 0 < propensity < 1:
        raise ValueError("propensity must be strictly between zero and one")
    t, y = np.asarray(t), np.asarray(y)
    if len(t) == 0 or len(t) != len(y) or not np.isin(t, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("treatment/outcome require nonempty equally sized binary arrays")
    return y * (t / propensity - (1 - t) / (1 - propensity))


def policy_value(t: np.ndarray, y: np.ndarray, actions: np.ndarray, propensity: float) -> float:
    """Unnormalized IPW policy value (outcome probability in a target population)."""
    t, y, actions = map(np.asarray, (t, y, actions))
    if len(t) != len(actions) or not np.isin(actions, [0, 1]).all():
        raise ValueError("Policy actions must be binary and match outcome length")
    transformed_outcome(t, y, propensity)  # Validate inputs.
    p_action = np.where(actions == 1, propensity, 1 - propensity)
    return float(np.mean((actions == t) * y / p_action))


def uplift_curve(scores: np.ndarray, t: np.ndarray, y: np.ndarray, propensity: float) -> dict:
    """Sorted cumulative incremental gain per *total* evaluated customers.

    Qini is area between the model's gain curve and random targeting; AUUC here
    means the unadjusted area under the cumulative gain curve. Both are normalized
    by the number of customers, and use stable sorting for tied scores.
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(t) or not np.isfinite(scores).all():
        raise ValueError("Uplift scores must be finite and match outcomes")
    z = transformed_outcome(t, y, propensity)
    idx = np.argsort(-scores, kind="stable")
    n = len(scores)
    x = np.arange(n + 1, dtype=float) / n
    gain = np.r_[0.0, np.cumsum(z[idx]) / n]
    random_gain = x * gain[-1]
    return {
        "auuc": float(np.trapezoid(gain, x)),
        "qini": float(np.trapezoid(gain - random_gain, x)),
        "total_incremental_gain": float(gain[-1]),
        "fractions": x,
        "gain": gain,
        "random_gain": random_gain,
    }


def bootstrap_policy_comparison(t: np.ndarray, y: np.ndarray, actions: np.ndarray,
                                propensity: float, *, reps: int = 400, seed: int = 42) -> dict:
    """Paired, arm-stratified percentile CIs for fixed policies vs treat-all.

    Reuses the same sampled row indices for both policies. This estimates
    uncertainty conditional on *already-trained* models and fixed decisions;
    it does not account for learning, selection, or propensity uncertainty.
    """
    if reps < 2:
        raise ValueError("Bootstrap repetitions must be >= 2")
    t, y, actions = map(np.asarray, (t, y, actions))
    baseline = np.ones(len(t), dtype=int)
    point = policy_value(t, y, actions, propensity)
    base = policy_value(t, y, baseline, propensity)
    rng = np.random.default_rng(seed)
    arm0, arm1 = np.flatnonzero(t == 0), np.flatnonzero(t == 1)
    deltas = np.empty(reps)
    values = np.empty(reps)
    for i in range(reps):
        idx = np.r_[rng.choice(arm0, len(arm0), replace=True),
                    rng.choice(arm1, len(arm1), replace=True)]
        values[i] = policy_value(t[idx], y[idx], actions[idx], propensity)
        deltas[i] = values[i] - policy_value(t[idx], y[idx], baseline[idx], propensity)
    def ci(a: np.ndarray) -> list[float]:
        return [float(v) for v in np.quantile(a, [0.025, 0.975])]
    return {
        "policy_value": point,
        "policy_value_ci95": ci(values),
        "difference_vs_treat_all": point - base,
        "difference_vs_treat_all_ci95": ci(deltas),
    }


def evaluate(predictions: pd.DataFrame, summary: dict, *, reps: int = 400, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Compute fixed-threshold policies and uplift rankings on validation only."""
    if summary.get("status") != "exploratory_only_assignment_unverified":
        raise ValueError("Unexpected M3 summary status; review evaluator assumptions")
    if int(summary.get("validation_rows", -1)) != len(predictions):
        raise ValueError("Validation row count does not match M3 summary")
    p = float(summary["train_treatment_fraction"])
    t = predictions.treatment_flg.to_numpy(dtype=int)
    y = predictions.target.to_numpy(dtype=int)
    base_all = policy_value(t, y, np.ones(len(t), dtype=int), p)
    base_none = policy_value(t, y, np.zeros(len(t), dtype=int), p)
    result = {
        "status": "exploratory_only_assignment_unverified",
        "split": "validation_only", "validation_rows": len(t),
        "propensity_source": "empirical_train_fraction_not_verified_design_probability",
        "propensity": p, "bootstrap_repetitions": reps, "bootstrap_seed": seed,
        "notes": [
            "Assignment mechanism and pre-treatment feature timing remain unverified.",
            "Bootstrap intervals condition on trained models, fixed policies, and assumed propensity.",
            "Do not use the held-out test set until the final pre-specified comparison.",
            "Do not select a winner solely from exploratory validation metrics.",
        ],
        "comparators": {"treat_all": base_all, "treat_none": base_none}, "models": {},
    }
    rows = []
    for name in MODEL_NAMES:
        scores = predictions[f"{name}_uplift"].to_numpy(float)
        decisions = (scores > 0).astype(int)
        curve = uplift_curve(scores, t, y, p)
        boot = bootstrap_policy_comparison(t, y, decisions, p, reps=reps, seed=seed)
        top = {}
        for fraction in TOP_FRACTIONS:
            n_top = max(1, int(np.ceil(fraction * len(scores))))
            selected = np.argsort(-scores, kind="stable")[:n_top]
            z = transformed_outcome(t[selected], y[selected], p)
            top[str(fraction)] = {
                "selected_customers": int(n_top),
                "incremental_response_among_selected": float(z.mean()),
                "incremental_conversions_per_evaluated_customer": float(z.sum() / len(scores)),
            }
        result["models"][name] = {
            "mean_predicted_uplift": float(np.mean(scores)),
            "fraction_targeted_at_zero_threshold": float(np.mean(decisions)),
            "auuc": curve["auuc"], "qini": curve["qini"],
            "top_k": top, "zero_threshold_policy": boot,
        }
        for i in range(len(curve["fractions"])):
            rows.append({"model": name, "fraction_targeted": float(curve["fractions"][i]),
                         "gain": float(curve["gain"][i]),
                         "random_targeting_gain": float(curve["random_gain"][i])})
    return result, pd.DataFrame(rows)


def run(predictions_path: str | Path, summary_path: str | Path, output_dir: str | Path,
        *, reps: int = 400, seed: int = 42) -> dict:
    predictions = load_predictions(predictions_path)
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    report, curve_rows = evaluate(predictions, summary, reps=reps, seed=seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    curve_rows.to_csv(output / "uplift_curves.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate M3 models without retraining or using the test set")
    parser.add_argument("--predictions", default="models/m3_baselines/validation_predictions.csv")
    parser.add_argument("--summary", default="models/m3_baselines/validation_summary.json")
    parser.add_argument("--output", default="models/m3_evaluation")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.predictions, args.summary, args.output,
                         reps=args.bootstrap, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

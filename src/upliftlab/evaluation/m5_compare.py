"""M5: five-learner validation comparison on identical customer rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upliftlab.evaluation.report import (
    TOP_FRACTIONS, bootstrap_policy_comparison, load_predictions,
    policy_value, transformed_outcome, uplift_curve,
)

NAMES = ("s_learner", "t_learner", "x_learner", "r_learner", "dr_learner")
STATUS = "exploratory_only_assignment_unverified"


def compare(m3_predictions: str | Path, m3_summary: str | Path,
            m4_predictions: str | Path, m4_summary: str | Path,
            m5_predictions: str | Path, m5_summary: str | Path,
            output_dir: str | Path, *, reps: int = 400, seed: int = 42) -> dict:
    base = load_predictions(m3_predictions)
    s3 = json.loads(Path(m3_summary).read_text(encoding="utf-8"))
    s4 = json.loads(Path(m4_summary).read_text(encoding="utf-8"))
    s5 = json.loads(Path(m5_summary).read_text(encoding="utf-8"))
    p = float(s3["train_treatment_fraction"])
    if not 0 < p < 1:
        raise ValueError("Invalid propensity")
    for s in (s3, s4, s5):
        if s.get("status") != STATUS or int(s.get("validation_rows", -1)) != len(base):
            raise ValueError("Unexpected status or validation count")
        if not np.isclose(float(s["train_treatment_fraction"]), p, rtol=0, atol=1e-12):
            raise ValueError("Inconsistent treatment propensity across models")

    merged = base
    for path, names in ((m4_predictions, ("x_learner",)),
                        (m5_predictions, ("r_learner", "dr_learner"))):
        new = pd.read_csv(path, low_memory=False)
        required = ["client_id", "treatment_flg", "target", *(n + "_uplift" for n in names)]
        if not set(required).issubset(new.columns) or new.client_id.isna().any() or new.client_id.duplicated().any():
            raise ValueError("Missing fields or duplicate/invalid IDs")
        if len(new) != len(base):
            raise ValueError("Validation row count mismatch")
        joined = merged.merge(new[required], on="client_id", how="inner",
                              validate="one_to_one", suffixes=("", "_new"))
        if (len(joined) != len(base) or
            not np.array_equal(joined.treatment_flg, joined.treatment_flg_new) or
            not np.array_equal(joined.target, joined.target_new)):
            raise ValueError("Validation IDs, treatment or outcome mismatch")
        merged = joined.drop(columns=["treatment_flg_new", "target_new"])

    t, y = merged.treatment_flg.to_numpy(int), merged.target.to_numpy(int)
    report = {
        "status": STATUS, "split": "validation_only", "validation_rows": len(merged),
        "propensity_source": "empirical_train_fraction_not_verified_design_probability",
        "propensity": p, "bootstrap_repetitions": reps, "bootstrap_seed": seed,
        "comparators": {
            "treat_all": policy_value(t, y, np.ones(len(t), int), p),
            "treat_none": policy_value(t, y, np.zeros(len(t), int), p),
        },
        "models": {},
        "note": "Fixed-policy exploratory IPW, not established causality. Test set unused.",
    }
    curves = []
    for name in NAMES:
        scores = pd.to_numeric(merged[name + "_uplift"], errors="coerce").to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite uplift scores: {name}")
        curve = uplift_curve(scores, t, y, p)
        boot = bootstrap_policy_comparison(t, y, (scores > 0).astype(int), p, reps=reps, seed=seed)
        order = np.argsort(-scores, kind="stable")
        top = {}
        for frac in TOP_FRACTIONS:
            n = max(1, int(np.ceil(frac * len(scores))))
            z = transformed_outcome(t[order[:n]], y[order[:n]], p)
            top[str(frac)] = {
                "selected_customers": n,
                "incremental_response_among_selected": float(z.mean()),
                "incremental_conversions_per_evaluated_customer": float(z.sum() / len(scores)),
            }
        report["models"][name] = {
            "mean_predicted_uplift": float(scores.mean()),
            "fraction_targeted_at_zero_threshold": float(np.mean(scores > 0)),
            "auuc": curve["auuc"], "qini": curve["qini"],
            "top_k": top, "zero_threshold_policy": boot,
        }
        curves.extend({"model": name, "fraction_targeted": float(f), "gain": float(g),
                       "random_targeting_gain": float(r)}
                      for f, g, r in zip(curve["fractions"], curve["gain"], curve["random_gain"]))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(curves).to_csv(out / "uplift_curves.csv", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare S/T/X/R/DR learners on validation only")
    parser.add_argument("--m3-predictions", default="models/m3_baselines/validation_predictions.csv")
    parser.add_argument("--m3-summary", default="models/m3_baselines/validation_summary.json")
    parser.add_argument("--m4-predictions", default="models/m4_x_learner/validation_predictions.csv")
    parser.add_argument("--m4-summary", default="models/m4_x_learner/validation_summary.json")
    parser.add_argument("--m5-predictions", default="models/m5_orthogonal_learners/validation_predictions.csv")
    parser.add_argument("--m5-summary", default="models/m5_orthogonal_learners/validation_summary.json")
    parser.add_argument("--output", default="models/m5_comparison")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.m3_predictions, args.m3_summary, args.m4_predictions,
                             args.m4_summary, args.m5_predictions, args.m5_summary,
                             args.output, reps=args.bootstrap, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

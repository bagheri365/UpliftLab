"""Evaluate M4 alongside existing M3 predictions with identical M3.1 metrics."""
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


def compare(m3_predictions: str | Path, m3_summary: str | Path,
            m4_predictions: str | Path, m4_summary: str | Path,
            output_dir: str | Path, *, reps: int = 400, seed: int = 42) -> dict:
    base = load_predictions(m3_predictions)
    new = pd.read_csv(m4_predictions)
    s3 = json.loads(Path(m3_summary).read_text(encoding="utf-8"))
    s4 = json.loads(Path(m4_summary).read_text(encoding="utf-8"))
    for summary in (s3, s4):
        if summary.get("status") != "exploratory_only_assignment_unverified":
            raise ValueError("Unexpected causal status")
    if s4.get("split") != "validation_only" or len(new) != len(base):
        raise ValueError("M4 rows/split differ from M3 validation")
    if new.client_id.isna().any() or new.client_id.duplicated().any():
        raise ValueError("M4 customer IDs must be unique and nonmissing")
    merged = base.merge(new[["client_id", "treatment_flg", "target", "x_learner_uplift"]],
                        on="client_id", how="inner", validate="one_to_one", suffixes=("", "_m4"))
    if len(merged) != len(base) or not np.array_equal(merged.treatment_flg, merged.treatment_flg_m4) or not np.array_equal(merged.target, merged.target_m4):
        raise ValueError("M3/M4 validation identifiers, treatment or outcomes do not match")
    p = float(s3["train_treatment_fraction"])
    if not np.isclose(p, float(s4["train_treatment_fraction"]), rtol=0, atol=1e-12):
        raise ValueError("M3/M4 propensities differ")
    t, y = merged.treatment_flg.to_numpy(int), merged.target.to_numpy(int)
    result = {"status": "exploratory_only_assignment_unverified", "split": "validation_only",
              "propensity_source": "empirical_train_fraction_not_verified_design_probability",
              "propensity": p, "validation_rows": len(merged), "bootstrap_repetitions": reps,
              "comparators": {"treat_all": policy_value(t, y, np.ones(len(t), int), p),
                              "treat_none": policy_value(t, y, np.zeros(len(t), int), p)},
              "models": {}, "note": "Fixed-policy exploratory IPW; assignment and feature timing unverified; test untouched."}
    rows = []
    for name in ("s_learner", "t_learner", "x_learner"):
        scores = pd.to_numeric(merged[f"{name}_uplift"], errors="coerce").to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite uplift scores for {name}")
        curve = uplift_curve(scores, t, y, p)
        boot = bootstrap_policy_comparison(t, y, (scores > 0).astype(int), p, reps=reps, seed=seed)
        ranked = np.argsort(-scores, kind="stable")
        top = {}
        for fraction in TOP_FRACTIONS:
            n = max(1, int(np.ceil(fraction * len(scores))))
            z = transformed_outcome(t[ranked[:n]], y[ranked[:n]], p)
            top[str(fraction)] = {
                "selected_customers": n, "incremental_response_among_selected": float(z.mean()),
                "incremental_conversions_per_evaluated_customer": float(z.sum() / len(scores))}
        result["models"][name] = {"mean_predicted_uplift": float(scores.mean()),
                                  "fraction_targeted_at_zero_threshold": float(np.mean(scores > 0)),
                                  "auuc": curve["auuc"], "qini": curve["qini"],
                                  "top_k": top, "zero_threshold_policy": boot}
        rows.extend({"model": name, "fraction_targeted": float(f), "gain": float(g),
                     "random_targeting_gain": float(r)}
                    for f, g, r in zip(curve["fractions"], curve["gain"], curve["random_gain"]))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "uplift_curves.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare M3 and M4 uplift learners on identical validation rows")
    parser.add_argument("--m3-predictions", default="models/m3_baselines/validation_predictions.csv")
    parser.add_argument("--m3-summary", default="models/m3_baselines/validation_summary.json")
    parser.add_argument("--m4-predictions", default="models/m4_x_learner/validation_predictions.csv")
    parser.add_argument("--m4-summary", default="models/m4_x_learner/validation_summary.json")
    parser.add_argument("--output", default="models/m4_comparison")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.m3_predictions, args.m3_summary,
                             args.m4_predictions, args.m4_summary, args.output,
                             reps=args.bootstrap, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

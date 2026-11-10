"""Compare six estimators on exactly the same validation customers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upliftlab.evaluation.m5_compare import compare as compare_five
from upliftlab.evaluation.report import (
    TOP_FRACTIONS, bootstrap_policy_comparison, transformed_outcome, uplift_curve,
)

STATUS = "exploratory_only_assignment_unverified"


def compare(m3_predictions: str | Path, m3_summary: str | Path,
            m4_predictions: str | Path, m4_summary: str | Path,
            m5_predictions: str | Path, m5_summary: str | Path,
            m6_predictions: str | Path, m6_summary: str | Path,
            output_dir: str | Path, *, reps: int = 400, seed: int = 42) -> dict:
    # Reuse the same exact evaluation implementation for the original five models.
    out = Path(output_dir)
    five = compare_five(m3_predictions, m3_summary, m4_predictions, m4_summary,
                        m5_predictions, m5_summary, out, reps=reps, seed=seed)
    six_summary = json.loads(Path(m6_summary).read_text(encoding="utf-8"))
    p = float(five["propensity"])
    if (six_summary.get("status") != STATUS or
        int(six_summary.get("validation_rows", -1)) != five["validation_rows"] or
        not np.isclose(float(six_summary["train_treatment_fraction"]), p, atol=1e-12, rtol=0)):
        raise ValueError("M6 summary does not match experiment settings")
    base = pd.read_csv(m3_predictions, usecols=["client_id", "treatment_flg", "target"])
    forest = pd.read_csv(m6_predictions)
    required = {"client_id", "treatment_flg", "target", "causal_forest_uplift"}
    if not required.issubset(forest.columns) or forest.client_id.isna().any() or forest.client_id.duplicated().any():
        raise ValueError("M6 predictions missing fields or have invalid IDs")
    joined = base.merge(forest[list(required)], on="client_id", how="inner",
                        validate="one_to_one", suffixes=("", "_m6"))
    if (len(joined) != len(base) or
        not np.array_equal(joined.treatment_flg, joined.treatment_flg_m6) or
        not np.array_equal(joined.target, joined.target_m6)):
        raise ValueError("M6 customer IDs, treatment or outcomes differ from M3")
    scores = pd.to_numeric(joined.causal_forest_uplift, errors="coerce").to_numpy(float)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite forest uplift predictions")
    t, y = joined.treatment_flg.to_numpy(int), joined.target.to_numpy(int)
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
    five["models"]["causal_forest"] = {
        "mean_predicted_uplift": float(scores.mean()),
        "fraction_targeted_at_zero_threshold": float(np.mean(scores > 0)),
        "auuc": curve["auuc"], "qini": curve["qini"],
        "top_k": top, "zero_threshold_policy": boot,
    }
    curves = pd.read_csv(out / "uplift_curves.csv")
    extra = pd.DataFrame({
        "model": "causal_forest", "fraction_targeted": curve["fractions"],
        "gain": curve["gain"], "random_targeting_gain": curve["random_gain"],
    })
    pd.concat([curves, extra], ignore_index=True).to_csv(out / "uplift_curves.csv", index=False)
    five["models_compared"] = 6
    five["note"] = "Six-learner fixed-policy validation IPW; causal assumptions unverified; test untouched."
    (out / "evaluation_summary.json").write_text(json.dumps(five, indent=2) + "\n", encoding="utf-8")
    return five


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare six uplift estimators on validation")
    for step in ("m3", "m4", "m5", "m6"):
        default = {"m3": "m3_baselines", "m4": "m4_x_learner",
                   "m5": "m5_orthogonal_learners", "m6": "m6_causal_forest"}[step]
        parser.add_argument(f"--{step}-predictions", default=f"models/{default}/validation_predictions.csv")
        parser.add_argument(f"--{step}-summary", default=f"models/{default}/validation_summary.json")
    parser.add_argument("--output", default="models/m6_comparison")
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.m3_predictions, args.m3_summary, args.m4_predictions,
                             args.m4_summary, args.m5_predictions, args.m5_summary,
                             args.m6_predictions, args.m6_summary, args.output,
                             reps=args.bootstrap, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()

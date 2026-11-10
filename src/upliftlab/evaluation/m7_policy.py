"""M7: validation-only policy value, budget and cost sensitivity across six learners.

Results are exploratory: true assignment propensities and feature timing remain unverified.
Treatment costs are expressed in outcome-probability equivalent units per treated person.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upliftlab.evaluation.report import policy_value, uplift_curve

MODELS = ("s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "causal_forest")
PREDICTION_FIELDS = {
    "m3": ("s_learner", "t_learner"),
    "m4": ("x_learner",),
    "m5": ("r_learner", "dr_learner"),
    "m6": ("causal_forest",),
}
STATUS = "exploratory_only_assignment_unverified"


def load_six(paths: dict[str, str | Path], summaries: dict[str, str | Path]) -> tuple[pd.DataFrame, float]:
    """Require the same IDs, treatment labels, outcomes and experiment settings."""
    if set(paths) != set(PREDICTION_FIELDS) or set(summaries) != set(PREDICTION_FIELDS):
        raise ValueError("Expected prediction and summary paths for m3, m4, m5, and m6")
    base: pd.DataFrame | None = None
    propensity: float | None = None
    n: int | None = None
    for stage, models in PREDICTION_FIELDS.items():
        summary = json.loads(Path(summaries[stage]).read_text(encoding="utf-8"))
        p = float(summary["train_treatment_fraction"])
        if (summary.get("status") != STATUS or summary.get("split", "validation_only") != "validation_only"
                or not np.isfinite(p) or not 0 < p < 1):
            raise ValueError(f"Invalid {stage} experiment metadata")
        if propensity is None:
            propensity, n = p, int(summary["validation_rows"])
        elif (not np.isclose(p, propensity, rtol=0, atol=1e-12)
              or int(summary["validation_rows"]) != n):
            raise ValueError("Mismatch in validation rows or assumed propensity")
        data = pd.read_csv(paths[stage], low_memory=False)
        needed = ["client_id", "treatment_flg", "target", *(f"{m}_uplift" for m in models)]
        if not set(needed).issubset(data.columns) or len(data) != n:
            raise ValueError(f"Invalid columns or number of {stage} prediction rows")
        data = data[needed]
        if data.client_id.isna().any() or data.client_id.duplicated().any():
            raise ValueError(f"Invalid IDs in {stage}")
        if stage == "m3":
            base = data.copy()
            for field in ("treatment_flg", "target"):
                if data[field].isna().any() or not data[field].isin([0, 1]).all():
                    raise ValueError(f"{field} must be binary")
            if set(data.treatment_flg) != {0, 1}:
                raise ValueError("Both treatment arms must be represented")
        else:
            assert base is not None
            joined = base[["client_id", "treatment_flg", "target"]].merge(
                data, on="client_id", how="inner", validate="one_to_one", suffixes=("", "_new")
            )
            if (len(joined) != n or not np.array_equal(joined.treatment_flg, joined.treatment_flg_new)
                    or not np.array_equal(joined.target, joined.target_new)):
                raise ValueError(f"Mismatch in validation IDs or observed data: {stage}")
            base = base.merge(data[["client_id", *(f"{m}_uplift" for m in models)]],
                              on="client_id", how="left", validate="one_to_one")
    assert base is not None and propensity is not None
    for model in MODELS:
        series = pd.to_numeric(base[f"{model}_uplift"], errors="coerce")
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite predictions for {model}")
        base[f"{model}_uplift"] = series
    return base, propensity


def decide(scores: np.ndarray, *, cost: float, budget: float) -> np.ndarray:
    """Treat positive net-uplift customers, up to the stated population budget.

    Thresholds are exploratory fixed choices; no threshold optimization on validation.
    """
    score = np.asarray(scores, dtype=float)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("Scores must be finite and one-dimensional")
    if not np.isfinite(cost) or cost < 0 or not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Invalid treatment cost or treatment budget")
    actions = np.zeros(len(score), dtype=int)
    eligible = np.flatnonzero(score > cost)
    k = min(len(eligible), int(np.floor(budget * len(score))))
    if k:
        order = eligible[np.argsort(-score[eligible], kind="stable")[:k]]
        actions[order] = 1
    return actions


def net_policy_value(t: np.ndarray, y: np.ndarray, actions: np.ndarray,
                     propensity: float, cost: float) -> float:
    """IPW conversion value minus cost per treatment assigned by the policy."""
    return policy_value(t, y, actions, propensity) - cost * float(np.mean(actions))


def fixed_policy_bootstrap(t: np.ndarray, y: np.ndarray,
                           actions: dict[str, np.ndarray], propensity: float,
                           cost: float, *, reps: int, seed: int) -> dict[str, dict]:
    """Paired arm-stratified bootstrap for fixed decisions and pairwise differences."""
    if reps < 2:
        raise ValueError("Need at least two bootstrap repetitions")
    t, y = np.asarray(t), np.asarray(y)
    keys = list(actions)
    values = {key: net_policy_value(t, y, actions[key], propensity, cost) for key in keys}
    arm0, arm1 = np.flatnonzero(t == 0), np.flatnonzero(t == 1)
    rng = np.random.default_rng(seed)
    samples = {key: np.empty(reps) for key in keys}
    for i in range(reps):
        idx = np.r_[rng.choice(arm0, len(arm0), replace=True),
                    rng.choice(arm1, len(arm1), replace=True)]
        for key in keys:
            samples[key][i] = net_policy_value(t[idx], y[idx], actions[key][idx], propensity, cost)
    out: dict[str, dict] = {}
    for key in keys:
        ci = np.quantile(samples[key], [0.025, 0.975]).tolist()
        differences = {}
        for reference in ("treat_all", "treat_none"):
            if reference == key:
                continue
            diff = samples[key] - samples[reference]
            differences[reference] = {
                "point": values[key] - values[reference],
                "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
            }
        out[key] = {"net_value": values[key], "ci95": ci, "differences": differences,
                    "treatment_fraction": float(np.mean(actions[key]))}
    return out


def paired_qini_bootstrap(scores: dict[str, np.ndarray], t: np.ndarray, y: np.ndarray,
                          propensity: float, *, reps: int, seed: int) -> dict:
    """Paired bootstrap of Qini differences, conditional on frozen scores."""
    if reps < 2:
        raise ValueError("Need at least two bootstrap repetitions")
    names = list(scores)
    arm0, arm1 = np.flatnonzero(t == 0), np.flatnonzero(t == 1)
    rng = np.random.default_rng(seed)
    values = {n: uplift_curve(scores[n], t, y, propensity)["qini"] for n in names}
    sample = {n: np.empty(reps) for n in names}
    for i in range(reps):
        idx = np.r_[rng.choice(arm0, len(arm0), replace=True),
                    rng.choice(arm1, len(arm1), replace=True)]
        for n in names:
            sample[n][i] = uplift_curve(scores[n][idx], t[idx], y[idx], propensity)["qini"]
    return {
        "qini": {n: {"point": values[n], "ci95": np.quantile(sample[n], [0.025, 0.975]).tolist()}
                 for n in names},
        "paired_difference_vs_x_learner": {
            n: {"point": values[n] - values["x_learner"],
                "ci95": np.quantile(sample[n] - sample["x_learner"], [0.025, 0.975]).tolist()}
            for n in names if n != "x_learner"
        },
    }


def evaluate(data: pd.DataFrame, propensity: float, *, costs: tuple[float, ...],
             budgets: tuple[float, ...], reps: int, seed: int,
             qini_bootstrap: int = 0) -> tuple[dict, pd.DataFrame]:
    """Apply predeclared scenarios to validation, without fitting or selecting models."""
    if not 0 < propensity < 1 or not np.isfinite(propensity):
        raise ValueError("Invalid propensity")
    t = data.treatment_flg.to_numpy(dtype=int)
    y = data.target.to_numpy(dtype=int)
    scores = {m: data[f"{m}_uplift"].to_numpy(dtype=float) for m in MODELS}
    result = {
        "status": STATUS, "split": "validation_only", "validation_rows": len(t),
        "propensity": propensity, "propensity_source": "empirical_train_fraction_not_verified_design_probability",
        "units": "Cost is purchase-probability equivalent per treated customer; no monetary cost data supplied.",
        "bootstrapping": {"repetitions": reps, "seed": seed,
                          "scope": "fixed scores, fixed policies; no model-fitting or selection uncertainty"},
        "warnings": [
            "No causal claims until assignment and pre-treatment feature timing are verified.",
            "Model/threshold selection using validation estimates is exploratory; test remains untouched.",
            "Conversion-equivalent cost scenarios are hypothetical, not observed promotion costs.",
        ],
        "scenarios": [],
    }
    records = []
    for cost in costs:
        for budget in budgets:
            actions = {"treat_all": np.ones(len(t), dtype=int),
                       "treat_none": np.zeros(len(t), dtype=int)}
            for m in MODELS:
                actions[m] = decide(scores[m], cost=cost, budget=budget)
            intervals = fixed_policy_bootstrap(t, y, actions, propensity, cost, reps=reps, seed=seed)
            scenario = {"cost": cost, "budget": budget, "policies": intervals}
            result["scenarios"].append(scenario)
            for name, stats in intervals.items():
                records.append({"cost": cost, "budget": budget, "policy": name,
                                "net_value": stats["net_value"],
                                "lower_ci95": stats["ci95"][0], "upper_ci95": stats["ci95"][1],
                                "treatment_fraction": stats["treatment_fraction"],
                                "gain_vs_treat_all": (0.0 if name == "treat_all" else
                                    stats["differences"]["treat_all"]["point"])})
    if qini_bootstrap:
        result["qini_uncertainty"] = paired_qini_bootstrap(
            scores, t, y, propensity, reps=qini_bootstrap, seed=seed)
    return result, pd.DataFrame.from_records(records)


def run(*, m3_predictions: str | Path, m3_summary: str | Path,
        m4_predictions: str | Path, m4_summary: str | Path,
        m5_predictions: str | Path, m5_summary: str | Path,
        m6_predictions: str | Path, m6_summary: str | Path,
        output: str | Path, costs: tuple[float, ...] = (0.0, 0.01, 0.03),
        budgets: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0),
        bootstrap: int = 400, qini_bootstrap: int = 200, seed: int = 42) -> dict:
    paths = dict(m3=m3_predictions, m4=m4_predictions, m5=m5_predictions, m6=m6_predictions)
    summaries = dict(m3=m3_summary, m4=m4_summary, m5=m5_summary, m6=m6_summary)
    data, p = load_six(paths, summaries)
    result, rows = evaluate(data, p, costs=costs, budgets=budgets,
                            reps=bootstrap, qini_bootstrap=qini_bootstrap, seed=seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    rows.to_csv(out / "policy_scenarios.csv", index=False)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M7 exploratory six-learner policy analysis (validation only)")
    for step, name in (("m3", "m3_baselines"), ("m4", "m4_x_learner"),
                       ("m5", "m5_orthogonal_learners"), ("m6", "m6_causal_forest")):
        parser.add_argument(f"--{step}-predictions", default=f"models/{name}/validation_predictions.csv")
        parser.add_argument(f"--{step}-summary", default=f"models/{name}/validation_summary.json")
    parser.add_argument("--output", default="models/m7_policy_analysis")
    parser.add_argument("--costs", nargs="+", type=float, default=[0.0, 0.01, 0.03])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--qini-bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = run(m3_predictions=args.m3_predictions, m3_summary=args.m3_summary,
                 m4_predictions=args.m4_predictions, m4_summary=args.m4_summary,
                 m5_predictions=args.m5_predictions, m5_summary=args.m5_summary,
                 m6_predictions=args.m6_predictions, m6_summary=args.m6_summary,
                 output=args.output, costs=tuple(args.costs), budgets=tuple(args.budgets),
                 bootstrap=args.bootstrap, qini_bootstrap=args.qini_bootstrap, seed=args.seed)
    print(json.dumps({"status": report["status"], "validation_rows": report["validation_rows"],
                      "scenarios": len(report["scenarios"]),
                      "outputs": [str(Path(args.output) / "policy_analysis.json"),
                                  str(Path(args.output) / "policy_scenarios.csv")],
                      "qini_bootstrap": args.qini_bootstrap}, indent=2))


if __name__ == "__main__":
    main()

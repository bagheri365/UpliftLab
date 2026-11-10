"""M8: validation-only, assumption-aware review of M7 policy scenarios.

This module intentionally does not unlock or read the held-out test set.
The equal-budget random policy is an *expected* randomized baseline derived
from treat-all and treat-none IPW values, not an observed online experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MODELS = ("s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "causal_forest")
EXPECTED_STATUS = "exploratory_only_assignment_unverified"


def assess(report: dict) -> tuple[dict, list[dict]]:
    """Produce budget-feasible exploratory comparisons from an M7 JSON report.

    Treat-all remains a descriptive reference when budget < 1, not an eligible
    policy. Random-budget values are calculated from group-level IPW policy
    values; uncertainty for their *paired differences* is not supplied here.
    """
    if report.get("status") != EXPECTED_STATUS or report.get("split") != "validation_only":
        raise ValueError("Expected M7 exploratory validation-only report")
    p = float(report.get("propensity", float("nan")))
    if not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("Invalid assumed propensity")
    scenarios = report.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Missing M7 scenarios")
    rows: list[dict] = []
    seen: set[tuple[float, float]] = set()
    for scenario in scenarios:
        cost, budget = float(scenario["cost"]), float(scenario["budget"])
        if not math.isfinite(cost) or cost < 0 or not math.isfinite(budget) or not 0 <= budget <= 1:
            raise ValueError("Invalid scenario budget or cost")
        if (cost, budget) in seen:
            raise ValueError("Duplicate scenario")
        seen.add((cost, budget))
        policies = scenario["policies"]
        if not set((*MODELS, "treat_all", "treat_none")).issubset(policies):
            raise ValueError("Incomplete model or reference policy results")
        all_policy, none_policy = policies["treat_all"], policies["treat_none"]
        if not math.isclose(float(all_policy["treatment_fraction"]), 1.0, abs_tol=1e-10):
            raise ValueError("Treat-all reference must treat everyone")
        if not math.isclose(float(none_policy["treatment_fraction"]), 0.0, abs_tol=1e-10):
            raise ValueError("Treat-none reference must treat no one")
        v_all = float(all_policy["net_value"])
        v_none = float(none_policy["net_value"])
        if not all(math.isfinite(v) for v in (v_all, v_none)):
            raise ValueError("Nonfinite baseline policy values")
        # The expected policy value for independent random targeting at rate b:
        # b V(all) + (1-b) V(none). Both reference values already include
        # treatment cost at the correct fraction, so do not subtract it twice.
        random_budget = budget * v_all + (1 - budget) * v_none
        winner = max(MODELS, key=lambda name: float(policies[name]["net_value"]))
        for name in MODELS:
            stats = policies[name]
            fraction = float(stats["treatment_fraction"])
            value = float(stats["net_value"])
            if not math.isfinite(value) or not math.isfinite(fraction) or fraction < 0 or fraction > budget + 1e-6:
                raise ValueError(f"Invalid value or budget violation for {name}")
            rows.append({
                "cost": cost, "budget": budget, "model": name,
                "treatment_fraction": fraction,
                "estimated_net_value": value,
                "random_at_budget_net_value": random_budget,
                "difference_vs_random_at_budget": value - random_budget,
                "difference_vs_treat_none": value - v_none,
                "difference_vs_unconstrained_treat_all": value - v_all,
                "treat_all_feasible": budget == 1.0,
                "scenario_leader_among_learners": name == winner,
            })
    best = [r for r in rows if r["scenario_leader_among_learners"]]
    summary = {
        "status": EXPECTED_STATUS,
        "split": "validation_only",
        "test_set_used": False,
        "validation_rows": report.get("validation_rows"),
        "scenario_count": len(scenarios),
        "learner_count": len(MODELS),
        "propensity_source": report.get("propensity_source", "unknown"),
        "causal_identification_verified": False,
        "feature_timing_verified": False,
        "test_unlocked": False,
        "paired_uncertainty_vs_random_baseline_available": False,
        "baseline_note": "Equal-budget random reference uses convex combination of estimated treat-all and treat-none values. It is feasible in expectation; no paired confidence interval is calculated.",
        "treat_all_note": "Treat-all is infeasible under budgets below 1.0 and must not be used as the budget-feasible comparator.",
        "scenario_leaders": best,
    }
    return summary, rows


def markdown_report(summary: dict, rows: list[dict], source: str) -> str:
    """Render an evidence-limited research note; no statistical winner claims."""
    lines = [
        "# UpliftLab — M8 Validation Research Review", "",
        "**Status:** Exploratory; causal assumptions not verified. **Held-out test:** not used.", "",
        f"**Input:** `{source}`", "",
        "## Research question", "",
        "Can a learned treatment policy improve expected net response relative to feasible simple policies under treatment budgets and hypothetical costs?", "",
        "## Evaluation design", "",
        "- Six pre-trained uplift models were assessed on the same validation population.",
        "- Costs are purchase-probability equivalents, not observed monetary costs.",
        "- A budget is a maximum treatment fraction, not a requirement to treat that fraction.",
        "- Equal-budget random targeting is an expected-value benchmark, calculated as `b * V(treat_all) + (1-b) * V(treat_none)`.",
        "- The treat-all policy is **infeasible** when `budget < 1.0`; its value is shown only as an unconstrained reference.",
        "- Differences against random targeting lack paired bootstrap intervals and **must not be called significant**.",
        "- Model, cost, and budget selection used validation data. Model-selection uncertainty is not quantified.", "",
        "## Point-estimate leaders among learned policies", "",
        "| Cost | Max treatment fraction | Best learned policy | Net value | Gain vs equal-budget random |",
        "| ---: | ---: | --- | ---: | ---: |",
    ]
    for r in rows:
        if r["scenario_leader_among_learners"]:
            lines.append(f'| {r["cost"]:.3f} | {r["budget"]:.0%} | {r["model"]} | {r["estimated_net_value"]:.4f} | {r["difference_vs_random_at_budget"]:+.4f} |')
    lines += [
        "", "## Interpretation constraints", "",
        "1. Treatment assignment probabilities are assumed from observed training frequencies; the original randomization protocol has not been independently verified.",
        "2. The intervention-time ordering of all historical features, including dataset-relative 30-/90-day features, has not been independently verified.",
        "3. Previous bootstrap intervals condition on trained models and fixed scores; they do not account for model or policy selection.",
        "4. The cost grid is hypothetical, and these estimates do not establish real-world profit improvements.",
        "5. The held-out test split remains untouched. It must not be used until the design and selection protocol are frozen.", "",
        "## Pre-test release gate", "",
        "- [ ] Verify treatment randomization/assignment from primary documentation or mark causal findings unsupported.",
        "- [ ] Verify feature timestamps precede intervention or remove unverified temporal features and retrain.",
        "- [ ] Pre-register a single primary metric and feasible reference policy for each budget scenario.",
        "- [ ] Select model, cost scenario, and threshold using validation only; freeze code and settings.",
        "- [ ] Add paired uncertainty for selected-vs-baseline policy and address selection/multiple-comparison uncertainty.",
        "- [ ] Document dataset licensing and data-processing provenance.",
        "- [ ] Only then perform one final evaluation on the held-out test sample.", "",
        "## Reproduce", "",
        "```bash", "python -m upliftlab.evaluation.m8_review \\",
        "  --analysis models/m7_policy_analysis/policy_analysis.json \\",
        "  --output models/m8_validation_review", "```", "",
    ]
    return "\n".join(lines)


def run(*, analysis: str | Path, output: str | Path) -> dict:
    source = Path(analysis)
    report = json.loads(source.read_text(encoding="utf-8"))
    summary, rows = assess(report)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (out / "feasible_policy_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "validation_research_review.md").write_text(markdown_report(summary, rows, str(source)), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M8 validation-only research review; test set remains sealed")
    parser.add_argument("--analysis", default="models/m7_policy_analysis/policy_analysis.json")
    parser.add_argument("--output", default="models/m8_validation_review")
    args = parser.parse_args(argv)
    result = run(analysis=args.analysis, output=args.output)
    print(json.dumps({"status": result["status"], "test_set_used": False,
                      "scenarios": result["scenario_count"],
                      "output": str(args.output), "pre_test_gate_open": False}, indent=2))


if __name__ == "__main__":
    main()

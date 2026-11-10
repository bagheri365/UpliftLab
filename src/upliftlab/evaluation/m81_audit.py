"""M8.1 reproducibility and causal-assumption audit, without opening test data.

The CLI reviews *metadata*, pre-selected validation artifacts, and source evidence.
It never trains models, scores held-out data, or unlocks the test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

AUDIT_STATUS = "blocked_pending_independent_evidence"
REQUIRED_PREDICTOR_GROUPS = {
    "demographics": ("age", "gender"),
    "behavior": ("historical_purchase_lines", "historical_active_days"),
    "relative_time": (
        "historical_recency_days_dataset_relative",
        "historical_lines_last_30d_dataset_relative",
        "historical_lines_last_90d_dataset_relative",
    ),
}


def sha256_file(path: Path) -> str:
    """Hash an artifact in bounded memory for reproducibility tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_feature_table(path: Path) -> dict:
    """Inspect columns and split counts without accessing outcome values.

    Test **rows** are counted for split integrity, but their features/outcomes
    are deliberately not inspected or used. This is an audit of an already
    produced split label, not a re-evaluation on the test set.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if "split" not in columns:
            raise ValueError("Feature table must contain the existing split column")
        counts: Counter[str] = Counter()
        for row in reader:
            name = str(row["split"]).lower()
            if name not in {"train", "validation", "test"}:
                raise ValueError(f"Unrecognized split: {name!r}")
            counts[name] += 1
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "total_rows": sum(counts.values()),
        "split_counts": dict(counts),
        "feature_groups_present": {
            group: [name for name in members if name in columns]
            for group, members in REQUIRED_PREDICTOR_GROUPS.items()
        },
        "missing_expected_columns": [
            name for members in REQUIRED_PREDICTOR_GROUPS.values()
            for name in members if name not in columns
        ],
        "split_labels_only_test_rows_not_examined": True,
        "timing_verification_performed": False,
    }


def assess(*, feature_summary: dict | None = None, repository_commit: str = "unknown") -> dict:
    """Build a conservative audit that cannot incorrectly certify causality."""
    return {
        "audit": "m8_1_causal_assumptions_and_reproducibility",
        "status": AUDIT_STATUS,
        "repository_commit_recorded": repository_commit,
        "treatment_assignment": {
            "verified_randomized": False,
            "verified_design_propensity": False,
            "evidence": "Official competition defines treatment_flg as communication, but no randomization protocol independently established.",
            "missing": ["Assignment procedure", "Eligibility and allocation details", "Design treatment propensity"],
        },
        "feature_timing": {
            "verified_pre_treatment": False,
            "evidence": "Official data description calls purchases.csv history prior to communication.",
            "missing": ["Exact treatment timestamp/cutoff", "Per-customer feature availability", "Validation of dataset-relative recency windows"],
            "dataset_relative_reference_date_observed": "2019-03-18",
            "reference_date_is_campaign_date": False,
        },
        "data_provenance": {
            "source": "https://ods.ai/competitions/x5-retailhero-uplift-modeling/data",
            "mirror": "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift",
            "dataset_license_verified": False,
            "license_review_required": True,
        },
        "feature_table": feature_summary,
        "evaluation_protocol": {
            "pre_registered": False,
            "primary_metric_frozen": False,
            "policy_and_cost_frozen": False,
            "feasible_baseline_frozen": False,
            "uncertainty_method_frozen": False,
            "model_selection_frozen": False,
        },
        "test_set_used": False,
        "pre_test_gate_open": False,
        "ready_for_confirmatory_causal_claim": False,
        "next_steps": [
            "Find and cite original treatment allocation protocol and campaign cutoff.",
            "If timing cannot be established, remove unverified temporal features and retrain all models.",
            "Decide and pre-register one primary comparison and uncertainty procedure.",
            "Record source hashes and dependency versions in a reproducibility manifest.",
            "Do not access the held-out test outcomes until the audit is resolved.",
        ],
    }


def render_markdown(report: dict) -> str:
    """Create a human-readable evidence ledger with explicit unresolved gates."""
    lines = [
        "# M8.1 — Causal Assumptions & Reproducibility Audit", "",
        "**Status:** Blocked pending independent evidence. **Test set:** sealed.", "",
        "## Evidence ledger", "",
        "| Question | Evidence | Decision |", "| --- | --- | --- |",
        "| Was treatment randomized? | Source defines communication flag; no allocation protocol verified | UNVERIFIED |",
        "| Is the assignment probability known? | Observed treatment fraction is not the design probability | UNVERIFIED |",
        "| Was purchase history before treatment? | Official X5 description says yes at file level | PARTIALLY SUPPORTED |",
        "| Is every engineered feature pre-treatment? | Exact campaign cutoff and per-customer timing unavailable | UNVERIFIED |",
        "| Is the intervention cost observed? | Synthetic cost scenarios only | NO |",
        "| Is the final model/policy frozen? | Exploratory validation selection | NO |",
        "| Is the held-out test set used? | Audit deliberately avoids scoring it | NO |", "",
        "## Dataset evidence", "",
        "- Original competition: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data",
        "- X5 documentation: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html",
        "- Mirror: https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift", "",
        "## Timing caution", "",
        "The observed dataset-relative reference date `2019-03-18` is **not verified as the treatment date**.",
        "The official description calls `purchases.csv` pre-communication history, but that is not",
        "equivalent to independently checking every engineered time-window feature.", "",
        "## Audit checks", "",
    ]
    features = report.get("feature_table")
    if features:
        lines += [
            f"- Feature table SHA-256: `{features['sha256']}`",
            f"- Split counts (labels only): `{features['split_counts']}`",
            f"- Missing selected expected columns: `{features['missing_expected_columns']}`",
            "- Held-out rows were counted by split label, not scored or analyzed.",
        ]
    else:
        lines += ["- No feature table supplied. Run the optional schema/hash audit when ready."]
    lines += [
        "", "## Required steps before a final held-out evaluation", "",
        "- [ ] Obtain original randomization/assignment evidence (or explicitly limit causal claims).",
        "- [ ] Verify the campaign cutoff and temporal feature derivations; run a timing sensitivity study if needed.",
        "- [ ] Establish dataset redistribution/usage permissions and capture input hashes.",
        "- [ ] Freeze one primary metric, model, cost/budget, comparator, and interval method.",
        "- [ ] Specify how model and scenario selection uncertainty will be handled.",
        "- [ ] Review and sign off the pre-test gate before any final test evaluation.", "",
        "## Conclusion", "",
        "This report records a **blocked gate**. It does not certify assignment, open the test",
        "set, or justify confirmatory treatment-effect claims.", "",
    ]
    return "\n".join(lines)


def run(*, output: str | Path, features: str | Path | None = None, commit: str = "unknown") -> dict:
    feature_summary = inspect_feature_table(Path(features)) if features else None
    report = assess(feature_summary=feature_summary, repository_commit=commit)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (target / "audit_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M8.1 conservative pre-test causal and reproducibility audit")
    parser.add_argument("--output", default="models/m8_1_audit")
    parser.add_argument("--features", help="Optional existing feature table for schema and hash only")
    parser.add_argument("--commit", default="fc4e96a", help="Git commit to document, not fetched or verified")
    args = parser.parse_args(argv)
    report = run(output=args.output, features=args.features, commit=args.commit)
    print(json.dumps({"status": report["status"], "test_set_used": False,
                      "pre_test_gate_open": False, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()

"""M8.2 research protocol review. Never reads customer-level records or test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_FIELDS = (
    "study_status", "primary_analysis", "causal_assumptions", "feature_policy",
    "uncertainty", "test_policy",
)
REQUIRED_PRIMARY = ("learner", "comparator", "budget", "cost", "metric")
VALID_STATES = {"draft", "frozen"}


def sha256(path: Path) -> str:
    """Compute a deterministic hash for a small research protocol artifact."""
    hash_obj = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def validate_protocol(protocol: dict) -> None:
    """Validate schema, but do not certify underlying assumptions."""
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    missing = set(REQUIRED_FIELDS) - protocol.keys()
    if missing:
        raise ValueError(f"Missing protocol fields: {sorted(missing)}")
    if protocol["study_status"] not in VALID_STATES:
        raise ValueError("study_status must be draft or frozen")
    primary = protocol["primary_analysis"]
    if not isinstance(primary, dict) or set(REQUIRED_PRIMARY) - primary.keys():
        raise ValueError("primary_analysis must specify learner, comparator, budget, cost, metric")
    budget = primary["budget"]
    cost = primary["cost"]
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
        raise ValueError("cost must be nonnegative")
    if budget < 1 and primary["comparator"] == "treat_all":
        raise ValueError("treat_all is infeasible at budget < 100%; use a feasible comparator")
    if protocol["test_policy"].get("allow_test_access") is not False:
        raise ValueError("M8.2 requires test access to remain disabled")


def review(audit: dict, protocol: dict, *, audit_digest: str, protocol_digest: str) -> dict:
    """Assess documented readiness without granting permission to open test data."""
    validate_protocol(protocol)
    assignment = audit.get("treatment_assignment") or {}
    timing = audit.get("feature_timing") or {}
    provenance = audit.get("data_provenance") or {}
    gaps = []
    if not assignment.get("verified_randomized", False):
        gaps.append("Randomized assignment protocol not independently verified")
    if not assignment.get("verified_design_propensity", False):
        gaps.append("Design treatment probability not independently verified")
    if not timing.get("verified_pre_treatment", False):
        gaps.append("Per-feature pre-treatment timing not established")
    if not provenance.get("dataset_license_verified", False):
        gaps.append("Dataset license and redistribution terms not verified")
    if protocol["study_status"] != "frozen":
        gaps.append("Primary comparison and inference protocol not frozen")
    if protocol["feature_policy"].get("exclude_unverified_relative_time") is not True:
        gaps.append("Unverified dataset-relative features have not been excluded by protocol")
    if protocol["causal_assumptions"].get("evidence_reviewed") is not True:
        gaps.append("Independent evidence review remains open")
    return {
        "audit": "m8_2_evidence_resolution_and_protocol",
        "status": "blocked_pending_independent_evidence",
        "test_set_used": False,
        "pre_test_gate_open": False,
        "ready_for_confirmatory_causal_claim": False,
        "protocol_status": protocol["study_status"],
        "primary_analysis_candidate": protocol["primary_analysis"],
        "unresolved_gaps": gaps,
        "evidence": {
            "audit_sha256": audit_digest,
            "protocol_sha256": protocol_digest,
            "feature_table_sha256_recorded": (audit.get("feature_table") or {}).get("sha256"),
            "source_commit_recorded": audit.get("repository_commit_recorded"),
        },
        "notes": [
            "This audit does not access customer rows, train models, or unlock held-out outcomes.",
            "Validation-driven candidate selection is exploratory, not a prospective pre-registration.",
            "An external review and a separate explicit final-evaluation gate are required.",
        ],
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=Path("models/m8_1_audit/audit_summary.json"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/m82_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("models/m8_2_protocol"))
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    result = review(audit, protocol, audit_digest=sha256(args.audit), protocol_digest=sha256(args.protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "readiness.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "test_set_used": False, "pre_test_gate_open": False, "output": str(args.output)}, indent=2))
    return result


if __name__ == "__main__":
    main()

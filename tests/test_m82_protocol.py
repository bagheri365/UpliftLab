"""M8.2 tests: protocol integrity and conservative gate behavior."""
import json
from pathlib import Path

import pytest

from upliftlab.evaluation.m82_protocol import main, review, validate_protocol


@pytest.fixture
def protocol():
    return {
        "study_status": "draft",
        "primary_analysis": {"learner": "s_learner", "comparator": "random_at_equal_budget", "budget": 0.3, "cost": 0.0, "metric": "ipw_net_policy_value_difference"},
        "causal_assumptions": {"evidence_reviewed": False},
        "feature_policy": {"exclude_unverified_relative_time": True},
        "uncertainty": {"method": "paired_bootstrap"},
        "test_policy": {"allow_test_access": False},
    }


def test_valid_draft(protocol):
    validate_protocol(protocol)


def test_cannot_treat_all_under_hard_budget(protocol):
    protocol["primary_analysis"]["comparator"] = "treat_all"
    with pytest.raises(ValueError, match="infeasible"):
        validate_protocol(protocol)


def test_reject_invalid_budget(protocol):
    protocol["primary_analysis"]["budget"] = 1.2
    with pytest.raises(ValueError, match="budget"):
        validate_protocol(protocol)


def test_reject_test_access(protocol):
    protocol["test_policy"]["allow_test_access"] = True
    with pytest.raises(ValueError, match="test access"):
        validate_protocol(protocol)


def test_unverified_assumptions_block(protocol):
    result = review({"treatment_assignment": {}, "feature_timing": {}, "data_provenance": {}}, protocol, audit_digest="a", protocol_digest="b")
    assert not result["pre_test_gate_open"]
    assert not result["ready_for_confirmatory_causal_claim"]
    assert len(result["unresolved_gaps"]) >= 4


def test_frozen_status_alone_cannot_unlock(protocol):
    protocol["study_status"] = "frozen"
    protocol["causal_assumptions"]["evidence_reviewed"] = True
    audit = {"treatment_assignment": {"verified_randomized": True, "verified_design_propensity": True}, "feature_timing": {"verified_pre_treatment": True}, "data_provenance": {"dataset_license_verified": True}}
    result = review(audit, protocol, audit_digest="a", protocol_digest="b")
    assert result["unresolved_gaps"] == []
    assert result["pre_test_gate_open"] is False


def test_cli_uses_only_audit_metadata(tmp_path, protocol):
    audit = tmp_path / "audit.json"
    policy = tmp_path / "protocol.json"
    audit.write_text(json.dumps({"treatment_assignment": {}, "feature_timing": {}, "data_provenance": {}, "feature_table": {"sha256": "abc"}}))
    policy.write_text(json.dumps(protocol))
    output = tmp_path / "out"
    result = main(["--audit", str(audit), "--protocol", str(policy), "--output", str(output)])
    assert (output / "readiness.json").is_file()
    assert result["test_set_used"] is False
    assert result["evidence"]["feature_table_sha256_recorded"] == "abc"

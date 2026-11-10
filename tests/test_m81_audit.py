"""M8.1 gate must remain closed absent independently verified evidence."""
import csv
import json

import pytest

from upliftlab.evaluation.m81_audit import assess, inspect_feature_table, render_markdown, run


def test_gate_fails_closed():
    report = assess()
    assert report["pre_test_gate_open"] is False
    assert report["ready_for_confirmatory_causal_claim"] is False
    assert report["treatment_assignment"]["verified_randomized"] is False
    assert report["feature_timing"]["verified_pre_treatment"] is False


def test_reference_date_not_treatment_date():
    report = assess()
    assert report["feature_timing"]["dataset_relative_reference_date_observed"] == "2019-03-18"
    assert report["feature_timing"]["reference_date_is_campaign_date"] is False


def test_feature_audit_split_only(tmp_path):
    path = tmp_path / "features.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "age", "gender", "target"])
        writer.writeheader()
        writer.writerows([{"split": "train", "age": "30", "gender": "F", "target": "1"},
                          {"split": "validation", "age": "25", "gender": "M", "target": "0"},
                          {"split": "test", "age": "99", "gender": "M", "target": "1"}])
    result = inspect_feature_table(path)
    assert result["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert len(result["sha256"]) == 64
    assert result["timing_verification_performed"] is False
    assert "historical_purchase_lines" in result["missing_expected_columns"]


def test_invalid_split_rejected(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("split,age\nproduction,25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized split"):
        inspect_feature_table(path)


def test_missing_split_rejected(tmp_path):
    path = tmp_path / "features.csv"
    path.write_text("age\n25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split column"):
        inspect_feature_table(path)


def test_report_does_not_claim_randomization():
    markdown = render_markdown(assess())
    assert "UNVERIFIED" in markdown
    assert "blocked gate" in markdown
    assert "not verified as the treatment date" in markdown


def test_cli_outputs_are_blocked(tmp_path):
    out = tmp_path / "audit"
    report = run(output=out, commit="fc4e96a")
    assert report["test_set_used"] is False
    saved = json.loads((out / "audit_summary.json").read_text(encoding="utf-8"))
    assert saved["pre_test_gate_open"] is False
    assert (out / "audit_report.md").exists()

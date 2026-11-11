"""M8.3 conservative-feature validation without opening test outcomes."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from upliftlab.evaluation.m83_sensitivity import EXCLUDED, compare_scenarios, prepare, run


def fixture_data(tmp_path: Path):
    source = tmp_path / "features.csv"
    columns = ["client_id", "split", "treatment_flg", "target", "age", "historical_spend", *EXCLUDED]
    rows = [
        ["1", "train", "0", "1", "30", "10", "4", "2", "3"],
        ["2", "validation", "1", "0", "40", "11", "5", "3", "4"],
        ["3", "test", "REDACTED", "REDACTED", "99", "99", "99", "99", "99"],
    ]
    with source.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "pre_test_gate_open": False, "test_set_used": False,
        "feature_table": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "split_counts": {"train": 1, "validation": 1, "test": 1}},
    }))
    return source, audit


def test_prepare_removes_three_columns_and_test_rows(tmp_path):
    src, audit = fixture_data(tmp_path)
    output = tmp_path / "safe.csv"
    report = prepare(src, output, audit=audit)
    text = output.read_text()
    assert "REDACTED" not in text and "99" not in text
    assert all(feature not in text for feature in EXCLUDED)
    assert report["derived_rows"] == 2
    assert report["source_split_counts"]["test"] == 1
    assert report["test_set_used"] is False


def test_prepare_rejects_source_hash_drift(tmp_path):
    src, audit = fixture_data(tmp_path)
    src.write_text(src.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare(src, tmp_path / "safe.csv", audit=audit)


def test_prepare_rejects_open_gate(tmp_path):
    src, audit = fixture_data(tmp_path)
    data = json.loads(audit.read_text())
    data["pre_test_gate_open"] = True
    audit.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="closed test gate"):
        prepare(src, tmp_path / "safe.csv", audit=audit)


def test_prepare_does_not_overwrite_original_or_output(tmp_path):
    src, audit = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        prepare(src, src, audit=audit)
    output = tmp_path / "safe.csv"
    output.write_text("important")
    with pytest.raises(FileExistsError):
        prepare(src, output, audit=audit)
    assert output.read_text() == "important"


def test_prepare_rejects_missing_temporal_column(tmp_path):
    src, audit = fixture_data(tmp_path)
    src.write_text("split,client_id,treatment_flg,target\ntrain,1,0,1\n")
    record = json.loads(audit.read_text())
    record["feature_table"]["sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    audit.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Missing expected columns"):
        prepare(src, tmp_path / "safe.csv", audit=audit)
    assert not (tmp_path / "safe.csv").exists()


def sample_policy(n: float):
    return {"net_value": n, "treatment_fraction": 0.3}


def sample_analysis(n: float):
    policies = {m: sample_policy(n) for m in ("s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "causal_forest")}
    policies["treat_none"] = sample_policy(0.6)
    return {"scenarios": [{"cost": 0.0, "budget": 0.3, "policies": policies}]}


def test_compare_scenario_delta():
    result = compare_scenarios(sample_analysis(0.62), sample_analysis(0.63))
    assert len(result) == 6
    assert result[0]["net_value_change"] == pytest.approx(0.01)
    assert result[0]["conservative_gain_vs_treat_none"] == pytest.approx(0.03)


def test_compare_rejects_unmatched_scenarios():
    original, new = sample_analysis(0.62), sample_analysis(0.63)
    new["scenarios"][0]["cost"] = 0.03
    with pytest.raises(ValueError, match="match"):
        compare_scenarios(original, new)


def test_prepare_only_does_not_train(tmp_path, monkeypatch):
    src, audit = fixture_data(tmp_path)
    def fail(*args, **kwargs):
        raise AssertionError("training called")
    monkeypatch.setattr("upliftlab.evaluation.m83_sensitivity.train_and_compare", fail)
    report = run(source=src, audit=audit, original=tmp_path / "unused.json",
                 output=tmp_path / "run", prepare_only=True)
    assert report["training_completed"] is False
    assert (tmp_path / "run" / "reproducibility_manifest.json").exists()

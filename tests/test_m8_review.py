"""M8 research review tests; these tests never load held-out records."""
import json

import pytest

from upliftlab.evaluation.m8_review import MODELS, assess, markdown_report, run


def fake_report(budget=0.3, cost=0.01):
    policies = {
        "treat_all": {"net_value": 0.63, "treatment_fraction": 1.0},
        "treat_none": {"net_value": 0.60, "treatment_fraction": 0.0},
    }
    for i, name in enumerate(MODELS):
        policies[name] = {"net_value": 0.605 + i * 0.002, "treatment_fraction": min(0.2, budget)}
    return {"status": "exploratory_only_assignment_unverified", "split": "validation_only",
            "validation_rows": 100, "propensity": 0.5,
            "scenarios": [{"cost": cost, "budget": budget, "policies": policies}]}


def test_random_reference_is_budget_feasible_in_expectation():
    summary, rows = assess(fake_report())
    assert summary["scenario_count"] == 1
    assert all(r["random_at_budget_net_value"] == pytest.approx(0.609) for r in rows)
    assert all(not r["treat_all_feasible"] for r in rows)
    assert sum(r["scenario_leader_among_learners"] for r in rows) == 1


def test_full_budget_marks_treat_all_feasible():
    _, rows = assess(fake_report(budget=1.0))
    assert all(r["treat_all_feasible"] for r in rows)
    assert rows[0]["random_at_budget_net_value"] == pytest.approx(0.63)


def test_budget_violation_rejected():
    d = fake_report()
    d["scenarios"][0]["policies"]["s_learner"]["treatment_fraction"] = 0.5
    with pytest.raises(ValueError, match="budget violation"):
        assess(d)


def test_non_validation_report_rejected():
    d = fake_report()
    d["split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        assess(d)


def test_duplicate_scenarios_rejected():
    d = fake_report()
    d["scenarios"].append(d["scenarios"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        assess(d)


def test_generated_markdown_has_explicit_caveats():
    summary, rows = assess(fake_report())
    text = markdown_report(summary, rows, "policy_analysis.json")
    assert "infeasible" in text
    assert "held-out test" in text
    assert "must not be called significant" in text


def test_run_generates_three_files_without_test_data(tmp_path):
    src = tmp_path / "policy_analysis.json"
    src.write_text(json.dumps(fake_report()))
    result = run(analysis=src, output=tmp_path / "out")
    assert result["test_set_used"] is False
    assert result["test_unlocked"] is False
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "feasible_policy_comparison.csv", "review_summary.json", "validation_research_review.md"
    ]

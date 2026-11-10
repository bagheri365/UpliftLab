"""Deterministic M3.1 evaluation tests; no real dataset required."""
import json

import numpy as np
import pandas as pd
import pytest

from upliftlab.evaluation.report import (
    bootstrap_policy_comparison, evaluate, load_predictions, policy_value,
    transformed_outcome, uplift_curve, run,
)


def fixture(n=100):
    t = np.tile([0, 1], n // 2)
    y = np.tile([0, 1, 1, 1], n // 4)
    rng = np.random.default_rng(11)
    score = np.linspace(-0.3, 0.3, n)
    df = pd.DataFrame({"client_id": [f"x{i}" for i in range(n)],
                       "treatment_flg": t, "target": y})
    for name in ("s_learner", "t_learner"):
        df[f"{name}_uplift"] = score if name == "s_learner" else -score
        df[f"{name}_p0"] = 0.4
        df[f"{name}_p1"] = 0.5
    summary = {"status": "exploratory_only_assignment_unverified",
               "validation_rows": n, "train_treatment_fraction": 0.5}
    return df, summary


def test_transformed_outcome_and_policy_value_known_example():
    t = np.array([0, 1, 0, 1])
    y = np.array([1, 1, 0, 0])
    np.testing.assert_allclose(transformed_outcome(t, y, 0.5), [-2, 2, 0, 0])
    assert policy_value(t, y, np.ones(4, int), 0.5) == pytest.approx(0.5)
    assert policy_value(t, y, np.zeros(4, int), 0.5) == pytest.approx(0.5)


def test_gain_curve_area_and_random_targeting():
    t = np.array([1, 0, 1, 0])
    y = np.array([1, 1, 1, 1])
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    curve = uplift_curve(scores, t, y, 0.5)
    assert curve["gain"][0] == 0
    assert curve["gain"][-1] == pytest.approx(0.0)
    assert curve["qini"] > 0
    assert curve["auuc"] == pytest.approx(curve["qini"])


def test_perfect_constant_effect_has_zero_qini_for_tied_scores():
    # With identical scores and balanced alternating outcome contributions,
    # cumulative curve is close to, but not exactly, random due to finite samples.
    n = 100
    t = np.tile([0, 1], n // 2)
    y = np.ones(n, dtype=int)
    curve = uplift_curve(np.ones(n), t, y, 0.5)
    assert abs(curve["qini"]) < 0.02


def test_bootstrap_is_reproducible_and_paired():
    df, _ = fixture()
    t, y = df.treatment_flg.to_numpy(), df.target.to_numpy()
    actions = np.ones(len(t), dtype=int)
    a = bootstrap_policy_comparison(t, y, actions, 0.5, reps=40, seed=3)
    b = bootstrap_policy_comparison(t, y, actions, 0.5, reps=40, seed=3)
    assert a == b
    assert a["difference_vs_treat_all"] == 0
    assert a["difference_vs_treat_all_ci95"] == [0, 0]


def test_validation_schema_rejects_duplicate_ids_and_nan(tmp_path):
    df, _ = fixture()
    df.loc[1, "client_id"] = df.loc[0, "client_id"]
    path = tmp_path / "validation.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_predictions(path)
    df.loc[1, "client_id"] = "new"
    df.loc[3, "s_learner_uplift"] = np.nan
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="nonfinite"):
        load_predictions(path)


def test_run_writes_report_and_full_curves(tmp_path):
    df, summary = fixture()
    df.to_csv(tmp_path / "pred.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    output = run(tmp_path / "pred.csv", tmp_path / "summary.json", tmp_path / "out", reps=20)
    assert set(output["models"]) == {"s_learner", "t_learner"}
    assert (tmp_path / "out/evaluation_summary.json").is_file()
    curves = pd.read_csv(tmp_path / "out/uplift_curves.csv")
    assert len(curves) == 202
    assert output["split"] == "validation_only"


def test_mismatched_validation_count_is_rejected():
    df, summary = fixture()
    summary["validation_rows"] = 99
    with pytest.raises(ValueError, match="row count"):
        evaluate(df, summary, reps=10)

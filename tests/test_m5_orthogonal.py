"""M5 estimator and comparison regression tests."""
import json

import numpy as np
import pandas as pd
import pytest

from upliftlab.models.orthogonal import OrthogonalLearner, run
from upliftlab.evaluation.m5_compare import compare


def fixture_arrays(n=120, seed=9):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"age": rng.normal(40, 10, n), "historical_spend": rng.uniform(0, 100, n)})
    t = np.tile(np.array([0, 0, 1, 1]), n // 4)
    y = np.tile(np.array([0, 1, 0, 1]), n // 4)
    return X, t, y


@pytest.mark.parametrize("method", ["r_learner", "dr_learner"])
def test_fit_predict_finite(method):
    X, t, y = fixture_arrays()
    learner = OrthogonalLearner(method, folds=3, seed=42).fit(X, t, y)
    result = learner.predict_uplift(X.iloc[:5])
    assert result.shape == (5,)
    assert np.isfinite(result).all()
    assert ((result >= -1) & (result <= 1)).all()


def test_deterministic():
    X, t, y = fixture_arrays()
    a = OrthogonalLearner("dr_learner", folds=3).fit(X, t, y).predict_uplift(X)
    b = OrthogonalLearner("dr_learner", folds=3).fit(X, t, y).predict_uplift(X)
    np.testing.assert_allclose(a, b)


def test_invalid_inputs():
    X, t, y = fixture_arrays()
    with pytest.raises(ValueError):
        OrthogonalLearner("other")
    with pytest.raises(ValueError):
        OrthogonalLearner("r_learner", folds=1).fit(X, t, y)
    with pytest.raises(ValueError):
        OrthogonalLearner("dr_learner", folds=3).fit(X, t, y[:-1])
    with pytest.raises(ValueError):
        OrthogonalLearner("r_learner", folds=3).fit(X, np.full(len(t), 1), y)
    with pytest.raises(RuntimeError):
        OrthogonalLearner("r_learner").predict_uplift(X)


def test_end_to_end_run_never_evaluates_test(tmp_path):
    rng = np.random.default_rng(4)
    n = 160
    df = pd.DataFrame({
        "client_id": np.arange(n),
        "split": ["train"] * 120 + ["validation"] * 20 + ["test"] * 20,
        "treatment_flg": np.tile([0, 0, 1, 1], n // 4),
        "target": np.tile([0, 1, 0, 1], n // 4),
        "age": rng.integers(20, 75, size=n),
        "historical_spend": rng.uniform(1, 200, n),
        "gender": rng.choice(["F", "M"], size=n),
    })
    df.loc[df.split == "test", "target"] = 1
    inp = tmp_path / "features.csv"
    df.to_csv(inp, index=False)
    result = run(inp, tmp_path / "m5", folds=3)
    assert result["test_rows_not_used"] == 20
    assert result["validation_rows"] == 20
    pred = pd.read_csv(tmp_path / "m5" / "validation_predictions.csv")
    assert len(pred) == 20
    assert set(pred.columns) == {"client_id", "treatment_flg", "target", "r_learner_uplift", "dr_learner_uplift"}
    assert (tmp_path / "m5" / "r_learner.joblib").exists()


def test_compare_alignment_and_outputs(tmp_path):
    n = 24
    ids = np.arange(n)
    t = np.tile([0, 0, 1, 1], n // 4)
    y = np.tile([0, 1, 0, 1], n // 4)
    m3 = pd.DataFrame({"client_id": ids, "treatment_flg": t, "target": y})
    for name in ("s_learner", "t_learner"):
        m3[name + "_p0"] = 0.45
        m3[name + "_p1"] = 0.55
        m3[name + "_uplift"] = np.linspace(-0.1, 0.2, n)
    m4 = m3[["client_id", "treatment_flg", "target"]].copy()
    m4["x_learner_uplift"] = np.linspace(-0.15, 0.2, n)
    m5 = m3[["client_id", "treatment_flg", "target"]].copy()
    m5["r_learner_uplift"] = np.linspace(-0.05, 0.2, n)
    m5["dr_learner_uplift"] = np.linspace(-0.2, 0.1, n)
    paths = []
    for i, frame in enumerate((m3, m4, m5)):
        p = tmp_path / f"m{i}.csv"
        frame.to_csv(p, index=False)
        paths.append(p)
    summaries = []
    for i in range(3):
        p = tmp_path / f"s{i}.json"
        p.write_text(json.dumps({"status": "exploratory_only_assignment_unverified",
                                 "validation_rows": n, "train_treatment_fraction": 0.5}), encoding="utf-8")
        summaries.append(p)
    result = compare(paths[0], summaries[0], paths[1], summaries[1], paths[2], summaries[2],
                     tmp_path / "comparison", reps=10)
    assert set(result["models"]) == {"s_learner", "t_learner", "x_learner", "r_learner", "dr_learner"}
    assert (tmp_path / "comparison" / "evaluation_summary.json").exists()
    assert len(pd.read_csv(tmp_path / "comparison" / "uplift_curves.csv")) == (n + 1) * 5
    m5.loc[0, "target"] = 1 - m5.loc[0, "target"]
    m5.to_csv(paths[2], index=False)
    with pytest.raises(ValueError, match="mismatch"):
        compare(paths[0], summaries[0], paths[1], summaries[1], paths[2], summaries[2], tmp_path / "bad", reps=10)

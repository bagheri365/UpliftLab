import json

import numpy as np
import pandas as pd
import pytest

from upliftlab.models.x_learner import XLearner, run
from upliftlab.evaluation.m4_compare import compare


def fixture_table(n=120):
    rng = np.random.default_rng(12)
    rows = []
    for split, count in (("train", n), ("validation", 48), ("test", 24)):
        for j in range(count):
            t = j % 2
            age = 18 + j % 60
            y = (j % 4 == 0 or j % 7 == 0 or (t == 1 and j % 5 == 0)) * 1
            rows.append({"client_id": f"{split}_{j}", "split": split,
                         "treatment_flg": t, "target": y, "age": age,
                         "gender": "M" if j % 3 else "F", "historical_spend": float(j % 13)})
    return pd.DataFrame(rows)


def test_x_learner_predictions_finite_and_reproducible():
    frame = fixture_table().query('split == "train"').reset_index(drop=True)
    X = frame[["age", "historical_spend"]]
    t, y = frame.treatment_flg.to_numpy(), frame.target.to_numpy()
    a = XLearner(seed=17, folds=3).fit(X, t, y).predict_uplift(X)
    b = XLearner(seed=17, folds=3).fit(X, t, y).predict_uplift(X)
    assert np.isfinite(a).all() and ((a >= -1) & (a <= 1)).all()
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_x_learner_rejects_insufficient_cell_counts():
    X = pd.DataFrame({"age": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="at least folds"):
        XLearner(folds=3).fit(X, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))


def test_x_learner_unfitted():
    with pytest.raises(RuntimeError):
        XLearner().predict_uplift(pd.DataFrame({"age": [25]}))


def test_train_uses_only_train_validation(tmp_path):
    table = fixture_table()
    data = tmp_path / "features.csv"
    table.to_csv(data, index=False)
    output = tmp_path / "m4"
    result = run(data, output, folds=3)
    assert result["validation_rows"] == 48
    assert result["test_rows_not_used"] == 24
    assert result["status"] == "exploratory_only_assignment_unverified"
    preds = pd.read_csv(output / "validation_predictions.csv")
    assert len(preds) == 48
    assert preds.client_id.str.startswith("validation_").all()
    assert (output / "x_learner.joblib").is_file()
    assert (output / "feature_encoder.joblib").is_file()


def test_comparison_aligns_predictions_and_rejects_label_mismatch(tmp_path):
    table = fixture_table()
    data = tmp_path / "features.csv"
    table.to_csv(data, index=False)
    m4dir = tmp_path / "m4"
    summary = run(data, m4dir, folds=3)
    preds = pd.read_csv(m4dir / "validation_predictions.csv")
    m3 = preds[["client_id", "treatment_flg", "target"]].copy()
    for name in ("s_learner", "t_learner"):
        m3[f"{name}_p0"] = 0.45
        m3[f"{name}_p1"] = 0.55
        m3[f"{name}_uplift"] = np.linspace(-0.1, 0.1, len(m3))
    m3path = tmp_path / "m3.csv"
    m3.to_csv(m3path, index=False)
    m3summary = tmp_path / "m3.json"
    m3summary.write_text(json.dumps({"status": summary["status"],
                                     "train_treatment_fraction": summary["train_treatment_fraction"]}))
    comparison = compare(m3path, m3summary, m4dir / "validation_predictions.csv",
                         m4dir / "validation_summary.json", tmp_path / "compare", reps=10)
    assert set(comparison["models"]) == {"s_learner", "t_learner", "x_learner"}
    assert (tmp_path / "compare" / "uplift_curves.csv").exists()
    damaged = preds.copy()
    damaged.loc[0, "target"] = 1 - damaged.loc[0, "target"]
    damaged.to_csv(m4dir / "bad.csv", index=False)
    with pytest.raises(ValueError, match="do not match"):
        compare(m3path, m3summary, m4dir / "bad.csv",
                m4dir / "validation_summary.json", tmp_path / "bad_result", reps=10)

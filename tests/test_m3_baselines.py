"""Small, deterministic integration tests for M3 baselines."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from upliftlab.models.baselines import (
    FeatureEncoder, SLearner, TLearner, ipw_policy_value, read_table, run, select_features,
)


def fake_table(n=240):
    rng = np.random.default_rng(7)
    t = np.tile([0, 1], n // 2)
    y = (rng.random(n) < np.clip(0.30 + 0.15*t + 0.07*rng.normal(size=n), 0.05, 0.9)).astype(int)
    return pd.DataFrame({
        "client_id": [f"c{i}" for i in range(n)],
        "split": ["train"]*160 + ["validation"]*60 + ["test"]*20,
        "age": [np.nan if i % 31 == 0 else 20 + i % 45 for i in range(n)],
        "gender": ["M", "F", "U"] * (n//3),
        "historical_spend": np.arange(n).astype(float),
        "treatment_flg": t,
        "target": y,
        "future_purchase_value": y * 100,
    })


def test_allowlist_excludes_outcomes_and_future_columns():
    features = select_features(fake_table())
    assert features == ["age", "gender", "historical_spend"]
    assert "target" not in features and "future_purchase_value" not in features


def test_encoder_uses_only_training_median():
    df = fake_table()
    encoder = FeatureEncoder().fit(df.iloc[:160], ["age", "gender"])
    encoded = encoder.transform(df.iloc[160:])
    assert encoded.isna().sum().sum() == 0
    assert list(encoded.columns) == ["age", "gender_F", "gender_M"]


def test_models_predict_valid_probabilities():
    df = fake_table()
    tr = df[df.split == "train"]
    X = FeatureEncoder().fit(tr, ["age", "gender"]).transform(tr)
    for learner in (SLearner(), TLearner()):
        learner.fit(X, tr.treatment_flg.to_numpy(), tr.target.to_numpy())
        p0, p1 = learner.potential_outcomes(X)
        assert len(p0) == len(tr)
        assert np.isfinite(learner.predict_uplift(X)).all()
        assert np.logical_and(p0 >= 0, p0 <= 1).all()
        assert np.logical_and(p1 >= 0, p1 <= 1).all()


def test_validation_only_and_reproducible_outputs(tmp_path):
    path = tmp_path / "features.csv"
    fake_table().to_csv(path, index=False)
    result1 = run(path, tmp_path / "one")
    result2 = run(path, tmp_path / "two")
    assert result1["test_rows_not_used"] == 20
    assert result1["models"] == result2["models"]
    assert len(pd.read_csv(tmp_path / "one/validation_predictions.csv")) == 60
    assert (tmp_path / "one/s_learner.joblib").exists()
    assert (tmp_path / "one/t_learner.joblib").exists()


def test_invalid_schema_rejected(tmp_path):
    df = fake_table()
    df.loc[0, "target"] = 2
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="binary"):
        read_table(path)


def test_ipw_policy_value_simple_case():
    t, y = np.array([0, 1]), np.array([1, 1])
    assert ipw_policy_value(t, y, np.array([0, 1]), 0.5) == 2.0
    assert ipw_policy_value(t, y, np.array([1, 1]), 0.5) == 1.0

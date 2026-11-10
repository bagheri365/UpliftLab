"""M6 tests: independent of optional EconML installation."""
import json

import numpy as np
import pandas as pd
import pytest

from upliftlab.models import causal_forest as cf


def sample(n=40):
    x = pd.DataFrame({"age": np.arange(n, dtype=float) + 20})
    t = np.tile([0, 1], n // 2)
    y = np.tile([0, 0, 1, 1], n // 4)
    return x, t, y


class FakeEconML:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, y, t, *, X):
        assert len(X) == len(y) == len(t)
        self.is_fitted = True
        return self

    def effect(self, X):
        return np.full(len(X), 0.04)


def test_forest_uses_honesty_crossfitting_and_discrete_outcome(monkeypatch):
    monkeypatch.setattr(cf, "_forest_class", lambda: FakeEconML)
    x, t, y = sample()
    forest = cf.HonestCausalForest(folds=3, trees=8).fit(x, t, y)
    assert forest.model_.kwargs["honest"] is True
    assert forest.model_.kwargs["discrete_outcome"] is True
    assert forest.model_.kwargs["discrete_treatment"] is True
    assert forest.model_.kwargs["cv"] == 3
    assert np.allclose(forest.predict_uplift(x), 0.04)


def test_invalid_model_settings():
    with pytest.raises(ValueError):
        cf.HonestCausalForest(trees=5)
    with pytest.raises(ValueError):
        cf.HonestCausalForest(folds=1)


def test_invalid_binary_inputs(monkeypatch):
    monkeypatch.setattr(cf, "_forest_class", lambda: FakeEconML)
    x, t, y = sample()
    t[0] = 2
    with pytest.raises(ValueError, match="binary"):
        cf.HonestCausalForest(trees=8).fit(x, t, y)


def test_predictions_before_fit_rejected():
    x, _, _ = sample()
    with pytest.raises(RuntimeError):
        cf.HonestCausalForest().predict_uplift(x)


def test_run_validation_only(monkeypatch, tmp_path):
    monkeypatch.setattr(cf, "_forest_class", lambda: FakeEconML)
    x, t, y = sample(80)
    df = pd.DataFrame({"client_id": np.arange(80), "age": x.age,
                       "treatment_flg": t, "target": y,
                       "split": ["train"] * 40 + ["validation"] * 20 + ["test"] * 20})
    data = tmp_path / "input.csv"
    df.to_csv(data, index=False)
    output = tmp_path / "artifacts"
    summary = cf.run(data, output, folds=3, trees=8)
    preds = pd.read_csv(output / "validation_predictions.csv")
    assert len(preds) == 20
    assert summary["test_rows_not_used"] == 20
    assert summary["validation_rows"] == 20
    assert np.allclose(preds.causal_forest_uplift, 0.04)
    assert json.loads((output / "validation_summary.json").read_text())["models"]["causal_forest"]


def test_compare_rejects_changed_labels(monkeypatch, tmp_path):
    from upliftlab.evaluation import m6_compare as ev
    baseline = pd.DataFrame({"client_id": [1, 2, 3, 4], "treatment_flg": [0, 1, 0, 1],
                             "target": [0, 1, 1, 0]})
    new = baseline.copy()
    new["causal_forest_uplift"] = [0.2, -0.1, 0.1, 0.4]
    new.loc[1, "target"] = 0
    p3, p6, s6 = (tmp_path / s for s in ("m3.csv", "m6.csv", "m6.json"))
    baseline.to_csv(p3, index=False)
    new.to_csv(p6, index=False)
    s6.write_text(json.dumps({"status": ev.STATUS, "validation_rows": 4,
                              "train_treatment_fraction": 0.5}))
    def stub(*args, **kwargs):
        return {"propensity": 0.5, "validation_rows": 4}
    monkeypatch.setattr(ev, "compare_five", stub)
    with pytest.raises(ValueError, match="differ"):
        ev.compare(p3, s6, p3, s6, p3, s6, p6, s6, tmp_path / "out")

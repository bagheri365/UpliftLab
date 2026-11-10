"""M7 policy-evaluation invariants and integration smoke test."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from upliftlab.evaluation.m7_policy import (
    MODELS, decide, evaluate, fixed_policy_bootstrap, load_six, net_policy_value,
)


def test_decide_respects_positive_net_uplift_and_budget():
    scores = np.array([0.1, -0.2, 0.04, 0.08, 0.02])
    assert decide(scores, cost=0.03, budget=0.4).tolist() == [1, 0, 0, 1, 0]
    assert decide(scores, cost=0.11, budget=1).sum() == 0
    assert decide(scores, cost=0, budget=0).sum() == 0


@pytest.mark.parametrize("cost,budget", [(-0.01, 1), (0.0, -0.1), (0.0, 1.1), (np.nan, 1)])
def test_decide_rejects_invalid_scenarios(cost, budget):
    with pytest.raises(ValueError):
        decide(np.array([0.1]), cost=cost, budget=budget)


def test_cost_deducted_only_from_treated_policy_actions():
    t = np.array([0, 1, 0, 1])
    y = np.array([0, 1, 0, 1])
    actions = np.array([1, 1, 0, 0])
    assert np.isclose(net_policy_value(t, y, actions, 0.5, 0.2),
                      net_policy_value(t, y, actions, 0.5, 0) - 0.1)


def test_paired_bootstrap_reproduces_zero_difference_for_same_policy():
    t = np.array([0, 1] * 10)
    y = np.array([0, 1, 1, 0] * 5)
    policy = np.ones(len(t), dtype=int)
    res = fixed_policy_bootstrap(t, y, {"treat_all": policy, "treat_none": 1 - policy,
                                          "clone": policy}, 0.5, 0.01, reps=20, seed=42)
    assert res["clone"]["differences"]["treat_all"]["point"] == 0
    assert res["clone"]["differences"]["treat_all"]["ci95"] == [0, 0]


def test_evaluate_runs_six_models_without_test_data():
    n = 40
    d = pd.DataFrame({"client_id": range(n), "treatment_flg": [0, 1] * 20,
                      "target": [0, 1, 1, 0] * 10})
    for idx, model in enumerate(MODELS):
        d[f"{model}_uplift"] = np.linspace(-0.1, 0.2, n) + idx * 0.002
    report, rows = evaluate(d, 0.5, costs=(0.0, 0.03), budgets=(0.5,),
                            reps=10, seed=42, qini_bootstrap=5)
    assert len(report["scenarios"]) == 2
    assert len(rows) == 16  # six learners + two comparators per scenario
    assert set(report["qini_uncertainty"]["qini"]) == set(MODELS)
    assert report["split"] == "validation_only"


def test_load_six_rejects_mismatched_outcomes(tmp_path):
    paths, summaries = {}, {}
    for step, models in {"m3": ("s_learner", "t_learner"), "m4": ("x_learner",),
                         "m5": ("r_learner", "dr_learner"), "m6": ("causal_forest",)}.items():
        d = pd.DataFrame({"client_id": [1, 2, 3, 4], "treatment_flg": [0, 1, 0, 1],
                          "target": [0, 1, 0, 1]})
        for model in models:
            d[model + "_uplift"] = [0.1, 0.2, -0.1, 0.1]
        if step == "m6":
            d.loc[0, "target"] = 1
        paths[step] = tmp_path / (step + ".csv")
        summaries[step] = tmp_path / (step + ".json")
        d.to_csv(paths[step], index=False)
        summaries[step].write_text(json.dumps({"status": "exploratory_only_assignment_unverified",
                                               "validation_rows": 4, "train_treatment_fraction": 0.5}))
    with pytest.raises(ValueError, match="Mismatch"):
        load_six(paths, summaries)

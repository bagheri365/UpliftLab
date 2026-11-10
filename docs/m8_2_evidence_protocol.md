# M8.2 — Evidence resolution and draft evaluation protocol

**Status: draft / blocked.** This milestone does not use the held-out test set, certify randomization, or authorize confirmatory causal claims.

## What primary sources actually support

| Claim | Finding | Evidence |
| --- | --- | --- |
| `treatment_flg` flags communication, `target` a later purchase | Supported | [X5 competition data description](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data) |
| `purchases.csv` is purchase history before communication | Supported *at dataset-description level* | [X5 data description](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data), [scikit-uplift](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html) |
| Original contest evaluates the top 30% of ranked customers | Supported | [Competition description](https://ods.ai/competitions/x5-retailhero-uplift-modeling) |
| Treatment was randomly assigned at propensity 0.5 | **Not established by available original description** | A third-party [tutorial](https://tom-beer.github.io/blog/uplift-part1/) calls X5 an RCT; this is not an original allocation protocol |
| Exact campaign cutoff and timing of all engineered features | **Not established** | Dataset description alone does not provide a per-customer cutoff |
| License permits redistribution | **Not established** | Review the original release/mirror license before redistributing raw data |

The original competition's 30%-targeting metric is a **ranking benchmark**, not proof of randomization. A balanced treatment fraction cannot, by itself, establish causal identification.

## Conservative feature policy

The following features use the latest observed transaction date (`2019-03-18`) as an **internal dataset-relative reference**, not a verified campaign date:

- `historical_recency_days_dataset_relative`
- `historical_lines_last_30d_dataset_relative`
- `historical_lines_last_90d_dataset_relative`

A draft sensitivity experiment should remove these relative-time features, then **retrain all six models from scratch** with the same fixed train/validation split. Do not compare newly retrained models to old predictions as though the feature representation were identical. Other purchase-history features still require evidence about their pre-treatment availability.

## Candidate primary analysis — not preregistered

`configs/m82_protocol.json` records an *exploratory candidate* chosen after examining validation results:

- S-Learner, top-30% budget, zero hypothetical treatment cost
- Equal-budget **random targeting** as feasible comparator (not unconstrained treat-all)
- Primary contrast: difference in expected policy value under an assumed propensity
- Paired arm-stratified bootstrap (proposed 2,000 repetitions), conditional on fixed scores

The 30% budget matches the original competition's uplift-ranking protocol; this does **not** make the selected S-Learner or metric prospectively registered. A secondary *illustrative* 50%-budget, 0.03-cost DR-Learner comparison is also listed. Do not treat its cost as monetary data.

Freezing a protocol is **not** proof of a valid causal experiment. Before any test-set analysis, obtain independent allocation evidence, justify propensity and overlap, investigate temporal leakage, record source/dependency hashes and licensing, and document a separate explicit gate decision. If evidence cannot be obtained, publish validation-only exploratory results with appropriate limitations.

## Run the metadata-only review

```bash
python -m upliftlab.evaluation.m82_protocol \
  --audit models/m8_1_audit/audit_summary.json \
  --protocol configs/m82_protocol.json \
  --output models/m8_2_protocol

pytest
```

It writes `models/m8_2_protocol/readiness.json`, including SHA-256 hashes of the audit and draft protocol, unresolved gaps, and `pre_test_gate_open: false`. It does **not** open customer records, run models, or read held-out outcomes. Altering a status field in a JSON file cannot unlock the test gate.

# M8.1 — Causal Assumptions and Reproducibility Audit

## Objective

Before making confirmatory causal claims or using held-out outcomes, record what is supported by documentation and what remains unknown. This milestone is an audit, **not** a certification of causal identification and **not** the final test evaluation.

## Evidence status

| Question | Current evidence | Status |
| --- | --- | --- |
| Is `treatment_flg` the promotional communication flag? | Yes, original X5 competition dataset description | Supported |
| Is `target` purchase after communication? | Yes, original dataset description | Supported |
| Is `purchases.csv` pre-communication history? | Yes, original competition and scikit-uplift descriptions | Supported at file level |
| Was communication randomized? | Treatment ratio ~0.5, but allocation protocol not verified | **Unknown** |
| Is propensity exactly 0.5 for all customers? | Previously estimated from training fractions, not specified by design | **Unknown** |
| Is the latest purchase date the intervention date? | Reference date `2019-03-18` derived from data | **Not established** |
| Do all temporal features precede assignment? | No per-customer timestamp audit against campaign cutoff | **Unknown** |
| Are treatment costs observed? | Only hypothetical probability-equivalent scenarios used | **No** |

Sources: [Original competition](https://ods.ai/competitions/x5-retailhero-uplift-modeling/data), [scikit-uplift X5 documentation](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html), [dataset mirror](https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift).

## Run

```bash
python -m upliftlab.evaluation.m81_audit \
  --output models/m8_1_audit \
  --features data/interim/x5_features_enriched.csv \
  --commit fc4e96a

pytest
git status
```

The optional feature table check counts split labels, inspects column names, and captures a SHA-256 hash; it **does not evaluate test outcomes or model predictions**. Omit `--features` to generate a documentation-only audit. Outputs: `audit_summary.json` and `audit_report.md` under the ignored `models/` artifact directory.

## Requirements before final test evaluation

1. Obtain campaign allocation/randomization evidence and probability, or frame all findings as assumption-dependent exploratory estimates.
2. Establish actual treatment/campaign cutoff and audit the timing of each derived feature; if this cannot be verified, remove suspect features, retrain models, and report sensitivity.
3. Record data licensing, hashes, dependencies, and training command versions.
4. Pre-register **one primary** policy comparison: metric, model-selection rule, intervention cost, budget, feasible comparator, and inference plan. Consider equal-budget random targeting, not unconstrained treat-all, for constrained budgets.
5. Define how validation-driven model/threshold selection uncertainty and multiple comparisons will be handled.
6. Explicitly approve the pre-test gate only after these checks. The current audit has **no automatic unlock path**.

This work does not change existing pipeline or model files and intentionally does not read any held-out test prediction or outcome file.

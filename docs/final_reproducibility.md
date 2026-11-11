# Final reproducibility guide (exploratory validation only)

This guide documents the pipeline executed by the project maintainer through M8.3. **Do not run a held-out test evaluation**: randomization, treatment propensity and feature timing were not independently verified and the evaluation gate is closed.

## System and setup

The maintainer ran Python **3.12.13** on macOS, with **85 unit tests passing** at Git commit `74d7b74`. Exact package versions may vary if reinstalled from unconstrained requirements; for a future reproducibility archive, capture `python -m pip freeze` locally and retain the environment metadata with the model outputs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install -r requirements-m6.txt
pytest
python scripts/check_docs.py
python -m pip freeze > models/local_environment_freeze.txt
```

The `models/` directory is locally ignored by Git. **Do not commit raw customer data or environment files containing confidential paths/tokens.** The repository source and documentation are version controlled, but generated training artifacts may need regeneration.

## Inputs and model training

Acquire X5 using the [dataset documentation](datasets.md) and follow [data preparation](reproducibility.md) to create `data/interim/x5_features_enriched.csv`. This project does not redistribute data, and dataset usage rights require independent review.

The following commands were executed on the original feature specification (all on the training split, with validation-only reporting):

```bash
python -m upliftlab.models.baselines --data data/interim/x5_features_enriched.csv --output models/m3_baselines --seed 42
python -m upliftlab.models.x_learner --data data/interim/x5_features_enriched.csv --output models/m4_x_learner --folds 3 --seed 42
python -m upliftlab.models.orthogonal --data data/interim/x5_features_enriched.csv --output models/m5_orthogonal_learners --folds 3 --seed 42
python -m upliftlab.models.causal_forest --data data/interim/x5_features_enriched.csv --output models/m6_causal_forest --folds 3 --trees 200 --min-leaf 20 --seed 42
```

Note: the M3 command shown is a convenience synopsis; use `python -m upliftlab.models.baselines --help` to verify its installed CLI options before rerunning. Exact details are recorded in milestone documentation. The downstream M7 tool expects the existing M3–M6 prediction artifacts.

## Cost-sensitive validation and audits

```bash
python -m upliftlab.evaluation.m7_policy --output models/m7_policy_analysis --costs 0 0.01 0.03 --budgets 0.1 0.3 0.5 1.0 --bootstrap 400 --qini-bootstrap 200 --seed 42
python -m upliftlab.evaluation.m8_review --analysis models/m7_policy_analysis/policy_analysis.json --output models/m8_validation_review
python -m upliftlab.evaluation.m81_audit --output models/m8_1_audit --features data/interim/x5_features_enriched.csv --commit fc4e96a
python -m upliftlab.evaluation.m82_protocol --audit models/m8_1_audit/audit_summary.json --protocol configs/m82_protocol.json --output models/m8_2_protocol
```

The M8.1 audit recorded source feature SHA-256:

```text
84218869a26677d701a2682c84bef5e89d45b1e6f473bcebf628a93bd8625295
```

A different hash requires documenting dataset/version differences; don't silently compare mismatched runs. The audit's `--commit fc4e96a` records its historical source reference, **not** proof that the current repository is still at that commit.

## Conservative-feature sensitivity (M8.3)

```bash
python -m upliftlab.evaluation.m83_sensitivity \
  --data data/interim/x5_features_enriched.csv \
  --audit models/m8_1_audit/audit_summary.json \
  --original models/m7_policy_analysis/policy_analysis.json \
  --output models/m8_3_sensitivity \
  --folds 3 --trees 200 --min-leaf 20 \
  --bootstrap 400 --qini-bootstrap 200 --seed 42
```

Check the locally generated `models/m8_3_sensitivity/sensitivity_comparison.csv` and `reproducibility_manifest.json` against the [final research report](final_research_report.md). The M8.3 runner prepared only **169,977 train/validation rows**, excluded three dataset-relative time predictors and did **not** open the test split. The randomization/timing audit remains unresolved.

## Scope of reproduction

The project implements six methods and 12 policy scenarios, not the planned ACIC ground-truth benchmark. No claim of precise bitwise repeatability across systems is made without pinned dependencies and an archived environment. The tested publication state is a validation-only exploratory study; no final test result or causal identification guarantee is available.

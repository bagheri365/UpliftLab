# Evaluation protocol (planned; not yet executed)

## Models

Compare S-Learner, T-Learner, X-Learner, R-Learner, DR-Learner, and Causal Forest against treat-all and treat-none baselines. Train on the same pre-treatment features and customer-level split. Use comparable tuning budgets; document deviations.

## Precommitment

- Fix train/validation/test assignments before training models.
- Fit preprocessing only on training data, and apply the learned transforms to validation/test.
- Cross-fit nuisance functions for orthogonalized and doubly robust approaches; use the known treatment probability if assignment is verified randomized, or a justified estimated propensity otherwise.
- Use validation data for algorithm selection, targeting thresholds, and hyperparameters.
- Leave the test set untouched until the final evaluation.

## Metrics

- Uplift/Qini curves and AUUC (record the chosen normalization and baseline).
- Estimated incremental policy value versus treat-all and treat-none with uncertainty intervals.
- Uplift at selected treatment budgets (e.g., top 10%, 20%, 30%).
- Runtime, peak memory, calibration diagnostics, and subgroup stability.
- On ACIC, treatment-effect error (e.g., PEHE) **only when ground-truth effects are available**.

For unbalanced or nonrandom assignment, raw treated-control outcome differences within predicted segments can be biased. Use appropriate propensity weighting, doubly robust estimation, and overlap checks, with assumptions clearly stated.

## Reporting

Report multiple seeds and confidence intervals where possible. Distinguish observed response differences, model estimates, offline policy-value estimates, and experimentally verified improvements. No claim of online incremental lift can be made without a new randomized policy experiment.

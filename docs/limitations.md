# Assumptions, limitations, and open checks

## Not yet verified

1. **Treatment assignment:** Near 50/50 balance does not establish randomization or ignorability.
2. **Temporal ordering:** The purchase dataset is described as historical, but we have not independently confirmed the intervention date or that every feature precedes it.
3. **Outcome horizon:** Confirm which time window `target` represents.
4. **Representation:** The observed customers and logged campaign may not generalize to other campaigns, products, or users.
5. **Feature semantics:** Check sign/units/refunds of spending and approximation behavior of product variety.
6. **Confidence:** The +3.3231 pp treated-control difference is descriptive until the design assumptions are verified.

## Modeling cautions

- Individual treatment effects are not observed in real-world intervention data; AUUC/Qini are policy-ranking proxies, not ground-truth CATE errors.
- Avoid conditioning on post-treatment variables or realized exposure that introduces selection bias.
- Strong outcomes or large sample sizes alone do not imply learnable treatment heterogeneity.
- Monitor overlap, calibration, subgroup robustness, and sensitivity to design assumptions.

## Release checklist

Before publishing causal conclusions, record an explicit evidence trail for assignment, timing, selected estimand, nuisance estimation, evaluation estimator, uncertainty, and exclusions.

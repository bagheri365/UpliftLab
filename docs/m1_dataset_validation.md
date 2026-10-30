# M1 — X5 RetailHero data validation

## Primary research question

Can a personalized treatment policy improve *expected* customer purchase probability compared with treating everyone or no one? Later milestones compare S-, T-, X-, R-, and DR-Learners and Causal Forest on the same held-out data.

## Dataset provenance

- Official competition: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
- `uplift_train.csv`: labeled subset; `treatment_flg` indicates communication, `target` indicates later purchase.
- `clients.csv`: customer metadata; `purchases.csv`: history preceding communication.
- Download manually under `data/raw/x5/`; do not commit customer-level data.
- Inspect and comply with dataset access and reuse conditions before downloading.
- The public `uplift_test.csv` does **not** contain outcomes; it cannot serve as an offline evaluation test set. A held-out sample of `uplift_train.csv` is required.

## Validate locally

```bash
python -m upliftlab.data.x5_validate data/raw/x5/uplift_train.csv \
  --output data/interim/x5_validation.json
pytest
```

The report includes group counts, purchase rates by treatment, observed response difference, and the four treatment-outcome cell counts. It **does not establish** causal identification or uplift heterogeneity.

## Mandatory manual causal audit before M2

- [ ] Confirm treatment randomization/assignment mechanism in the original study or organizer documentation; 50/50 group balance is not proof of randomization.
- [ ] Establish target population, eligibility, and sampling/selection criteria.
- [ ] Confirm when treatment was assigned, delivered, and recorded.
- [ ] Verify outcome definition and follow-up window, including exclusions and missing outcomes.
- [ ] Verify which columns and derived features were available **before** assignment; exclude post-treatment signals.
- [ ] Check joins against `clients.csv` and `purchases.csv` and ensure client-level uniqueness.
- [ ] Document treatment propensity assumptions and any sensitivity analysis needed.
- [ ] Choose client-disjoint train/validation/test splits from labeled rows, with a fixed seed and stratification.

**M1 completion condition:** validation report generated on actual downloaded labeled data *and* provenance/causal audit documented. Code and synthetic fixture tests alone do not complete M1.

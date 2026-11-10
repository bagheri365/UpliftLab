# Datasets and provenance

## Primary: X5 RetailHero uplift

Mirror: [Hugging Face pytorch-lifestream/retailhero-uplift](https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift).

Files used locally:

- `data/raw/x5/uplift_train.csv`: labeled intervention and outcome, 200,039 customers.
- `data/raw/x5/clients.csv.gz`: customer profile features.
- `data/raw/x5/purchases.csv.gz`: historical purchase events.

The raw competition data are excluded from Git. Check license and redistribution requirements from the original host and the mirror before publishing any derived data.

## Observed M1 statistics (not a verified ATE)

| Group | Customers | Purchasers | Response rate |
| --- | ---: | ---: | ---: |
| Control (0) | 100,058 | 60,363 | 60.3280% |
| Treated (1) | 99,981 | 63,639 | 63.6511% |

Observed difference: +3.3231 percentage points. This is **not proof of causal uplift** without verifying assignment and data timing. Response-rate figures come from the locally run M1 validator.

## Secondary: ACIC 2016

Semi-synthetic causal benchmark with simulated outcomes and known generating effects, useful for treatment-effect error analysis. Download and scenario-specific setup are future work; it is not yet integrated in the repository.

## Open provenance questions

- Verify and cite the primary source's method of assigning promotional communication.
- Confirm campaign/treatment date and the observation window of `target`.
- Confirm that every purchase record used for features precedes treatment.
- Record raw file hashes, retrieval dates, and any upstream version changes in future runs.

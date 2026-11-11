# Publication / release checklist

Target: an **exploratory v1.0.0 research-code release**, not a confirmatory experiment or a deployed targeting system. A Git tag does not validate causal assumptions.

## Review before tagging

- [ ] Confirm `git status --short` is empty after committing M9 and rerun `pytest` and `python scripts/check_docs.py`.
- [ ] Confirm the README, final report and reproduction guide agree about the **85 tests as of M8.3**, 30,062 untouched test rows and unverified assignment/timing.
- [ ] Verify GitHub does not contain raw X5 records, local model output, customer identifiers, credentials or accidental data exports.
- [ ] Distinguish repository's Apache-2.0 code license from unverified rights to redistribute X5 data.
- [ ] Verify all commands and paths in the reproduction guide on the intended OS/environment; save the environment freeze privately or in an appropriate sanitized archive.
- [ ] Confirm `configs/m82_protocol.json` is marked **draft**, not prospective pre-registration, and the M8.2 `pre_test_gate_open` flag remains `false`.
- [ ] Review final report values against local M6/M7/M8.3 result files. Do not infer statistical significance from point-estimate winner switches.
- [ ] Mark ACIC simulations, model refitting uncertainty, independent assignment verification and held-out test evaluation **out of scope** for this release.

## Commands after reviewing and committing M9

```bash
pytest
python scripts/check_docs.py
git status --short
git log -1 --oneline
# Tag ONLY after passing the checklist and inspecting the commit:
git tag -a v1.0.0 -m 'UpliftLab exploratory research release'
git push origin main
git push origin v1.0.0
```

To publish release notes, open **GitHub → Releases → Draft a new release**, choose `v1.0.0`, describe the six-learner benchmark, note validation-only exploratory evidence, and link the report. Do not attach datasets or generated customer-level predictions without a separate licensing/privacy review.

**Important:** Tagging and creating the GitHub Release are manual actions. This repository patch will not create or push a tag automatically.

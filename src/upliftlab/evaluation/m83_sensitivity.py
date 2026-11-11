"""M8.3 validation-only sensitivity to three unverified dataset-relative features.

Never reads test outcomes: the source table is streamed, test rows are ignored
before any treatment, outcome, or predictor value is accessed. Results remain
exploratory because treatment randomization and timing are unverified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path

EXCLUDED = (
    "historical_recency_days_dataset_relative",
    "historical_lines_last_30d_dataset_relative",
    "historical_lines_last_90d_dataset_relative",
)
STAGES = {"m3": "m3_baselines", "m4": "m4_x_learner",
          "m5": "m5_orthogonal_learners", "m6": "m6_causal_forest"}
MODELS = ("s_learner", "t_learner", "x_learner", "r_learner", "dr_learner", "causal_forest")
STATUS = "exploratory_only_assignment_unverified"


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: str | Path, destination: str | Path, *, audit: str | Path) -> dict:
    """Export train+validation only; reject missing exclusions and audit hash drift.

    Skip test rows by the split field before reading other row values.
    """
    source, destination = Path(source), Path(destination)
    report = json.loads(Path(audit).read_text(encoding="utf-8"))
    if report.get("pre_test_gate_open") is not False or report.get("test_set_used") is not False:
        raise ValueError("Input audit must certify a closed test gate")
    expected = report.get("feature_table", {}).get("sha256")
    if not expected or file_hash(source) != expected:
        raise ValueError("Feature table SHA-256 does not match M8.1 audit")
    if source.resolve() == destination.resolve():
        raise ValueError("Refusing to overwrite the original feature table")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    try:
        with source.open(newline="", encoding="utf-8") as src, destination.open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            fields = reader.fieldnames or []
            needed = {"split", "client_id", "treatment_flg", "target", *EXCLUDED}
            if not needed.issubset(fields):
                raise ValueError(f"Missing expected columns: {sorted(needed - set(fields))}")
            keep = [field for field in fields if field not in EXCLUDED]
            writer = csv.DictWriter(dst, fieldnames=keep, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                split = row["split"]
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"Invalid split label: {split!r}")
                counts[split] += 1
                if split != "test":
                    writer.writerow({field: row[field] for field in keep})
        expected_counts = report.get("feature_table", {}).get("split_counts", {})
        if any(counts[name] != expected_counts.get(name) for name in ("train", "validation", "test")):
            raise ValueError("Split counts differ from the original audit")
        if not counts["train"] or not counts["validation"] or not counts["test"]:
            raise ValueError("Source split counts must be nonzero")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"original_sha256": expected, "derived_sha256": file_hash(destination),
            "excluded_features": list(EXCLUDED), "source_split_counts": dict(counts),
            "derived_rows": counts["train"] + counts["validation"],
            "test_set_used": False, "pre_test_gate_open": False}


def compare_scenarios(original: dict, conservative: dict) -> list[dict]:
    """Compare matched six-model policy value differences vs treat-none.

    Comparisons condition on trained predictions and do not establish causal
    validity. Cost/budget scenarios must match exactly.
    """
    old = {(s["cost"], s["budget"]): s for s in original["scenarios"]}
    new = {(s["cost"], s["budget"]): s for s in conservative["scenarios"]}
    if not old or old.keys() != new.keys() or len(old) != len(original["scenarios"]) or len(new) != len(conservative["scenarios"]):
        raise ValueError("Original and conservative scenarios must match one-to-one")
    records = []
    for cost, budget in sorted(old):
        for model in MODELS:
            a, b = old[(cost, budget)]["policies"][model], new[(cost, budget)]["policies"][model]
            old_none = old[(cost, budget)]["policies"]["treat_none"]["net_value"]
            new_none = new[(cost, budget)]["policies"]["treat_none"]["net_value"]
            records.append({"cost": cost, "budget": budget, "model": model,
                            "original_net_value": a["net_value"],
                            "conservative_net_value": b["net_value"],
                            "net_value_change": b["net_value"] - a["net_value"],
                            "original_gain_vs_treat_none": a["net_value"] - old_none,
                            "conservative_gain_vs_treat_none": b["net_value"] - new_none,
                            "original_fraction_targeted": a["treatment_fraction"],
                            "conservative_fraction_targeted": b["treatment_fraction"]})
    return records


def manifest(*, source: str | Path, audit: str | Path, run_config: dict, preparation: dict) -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "econml", "joblib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {"status": STATUS, "python": sys.version, "platform": platform.platform(),
            "versions": versions, "source_file": str(source), "audit_file": str(audit),
            "configuration": run_config, "preparation": preparation,
            "test_set_used": False, "pre_test_gate_open": False,
            "note": "Validation-only, post-hoc sensitivity. No causal assumptions certified."}


def train_and_compare(data: Path, output: Path, *, seed: int, folds: int,
                      trees: int, min_leaf: int, bootstrap: int,
                      qini_bootstrap: int, original: Path) -> dict:
    # Import lazily so preparation and unit tests require no EconML installation.
    from upliftlab.models.baselines import run as m3_run
    from upliftlab.models.x_learner import run as m4_run
    from upliftlab.models.orthogonal import run as m5_run
    from upliftlab.models.causal_forest import run as m6_run
    from upliftlab.evaluation.m7_policy import run as m7_run

    stage_out = {key: output / name for key, name in STAGES.items()}
    m3_run(data, stage_out["m3"], seed)
    m4_run(data, stage_out["m4"], seed=seed, folds=folds)
    m5_run(data, stage_out["m5"], seed=seed, folds=folds)
    m6_run(data, stage_out["m6"], seed=seed, folds=folds, trees=trees, min_leaf=min_leaf)
    args = {}
    for stage, folder in stage_out.items():
        args[f"{stage}_predictions"] = folder / "validation_predictions.csv"
        args[f"{stage}_summary"] = folder / "validation_summary.json"
    current = m7_run(**args, output=output / "policy_comparison",
                     costs=(0.0, 0.01, 0.03), budgets=(0.1, 0.3, 0.5, 1.0),
                     bootstrap=bootstrap, qini_bootstrap=qini_bootstrap, seed=seed)
    baseline = json.loads(original.read_text(encoding="utf-8"))
    if baseline.get("validation_rows") != current.get("validation_rows") or baseline.get("status") != STATUS:
        raise ValueError("Original validation cohort/status mismatch")
    rows = compare_scenarios(baseline, current)
    path = output / "sensitivity_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"scenario_count": len(current["scenarios"]), "comparison_rows": len(rows),
            "sensitivity_comparison": str(path), "validation_rows": current["validation_rows"],
            "policy_analysis": str(output / "policy_comparison" / "policy_analysis.json")}


def run(*, source: str | Path, audit: str | Path, original: str | Path,
        output: str | Path, prepare_only: bool, seed: int = 42, folds: int = 3,
        trees: int = 200, min_leaf: int = 20, bootstrap: int = 400,
        qini_bootstrap: int = 200) -> dict:
    if folds < 2 or bootstrap < 2 or qini_bootstrap < 0 or trees % 4 or trees < 4 or min_leaf < 2:
        raise ValueError("Invalid training or bootstrap parameters")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    safe_table = output / "conservative_train_validation.csv"
    if safe_table.exists():
        raise FileExistsError(f"Output already contains derived table: {safe_table}. Use a new output directory")
    prep = prepare(source, safe_table, audit=audit)
    cfg = {"seed": seed, "folds": folds, "trees": trees, "min_leaf": min_leaf,
           "bootstrap": bootstrap, "qini_bootstrap": qini_bootstrap,
           "prepare_only": prepare_only, "original_policy_analysis": str(original)}
    info = manifest(source=source, audit=audit, run_config=cfg, preparation=prep)
    (output / "reproducibility_manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if not prepare_only:
        details = train_and_compare(safe_table, output, seed=seed, folds=folds, trees=trees,
                                    min_leaf=min_leaf, bootstrap=bootstrap,
                                    qini_bootstrap=qini_bootstrap, original=Path(original))
        info["results"] = details
        (output / "reproducibility_manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return {"status": STATUS, "output": str(output), "prepared_rows": prep["derived_rows"],
            "excluded_features": list(EXCLUDED), "training_completed": not prepare_only,
            "test_set_used": False, "pre_test_gate_open": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M8.3 conservative-feature sensitivity; no held-out outcomes")
    parser.add_argument("--data", default="data/interim/x5_features_enriched.csv")
    parser.add_argument("--audit", default="models/m8_1_audit/audit_summary.json")
    parser.add_argument("--original", default="models/m7_policy_analysis/policy_analysis.json")
    parser.add_argument("--output", default="models/m8_3_sensitivity")
    parser.add_argument("--prepare-only", action="store_true", help="Export safe train/validation data without fitting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--min-leaf", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--qini-bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    print(json.dumps(run(source=args.data, audit=args.audit, original=args.original,
                         output=args.output, prepare_only=args.prepare_only,
                         seed=args.seed, folds=args.folds, trees=args.trees,
                         min_leaf=args.min_leaf, bootstrap=args.bootstrap,
                         qini_bootstrap=args.qini_bootstrap), indent=2))


if __name__ == "__main__":
    main()

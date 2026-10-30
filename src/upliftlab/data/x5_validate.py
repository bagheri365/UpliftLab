"""Lightweight schema and quality checks for the X5 uplift training file.

The checks are descriptive, not evidence of randomized assignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REQUIRED_COLUMNS = ("client_id", "treatment_flg", "target")


def validate_x5(path: str | Path) -> dict:
    """Summarize a labeled CSV and fail on invalid treatment/outcome/IDs.

    Does not read or upload any customer-level values beyond counting them.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    counts: Counter[tuple[str, str]] = Counter()
    ids: set[str] = set()
    rows = 0
    with file_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is empty or missing a header")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV contains duplicated column names")
        for line_number, record in enumerate(reader, start=2):
            rows += 1
            client_id = (record["client_id"] or "").strip()
            treatment = (record["treatment_flg"] or "").strip()
            outcome = (record["target"] or "").strip()
            if not client_id:
                raise ValueError(f"Missing client_id at line {line_number}")
            if client_id in ids:
                raise ValueError(f"Duplicate client_id at line {line_number}")
            ids.add(client_id)
            if treatment not in ("0", "1") or outcome not in ("0", "1"):
                raise ValueError(f"Nonbinary/missing treatment or target at line {line_number}")
            counts[(treatment, outcome)] += 1

    if rows == 0:
        raise ValueError("Dataset has no observations")
    treatment_counts = {t: sum(counts[(t, y)] for y in ("0", "1")) for t in ("0", "1")}
    if min(treatment_counts.values()) == 0:
        raise ValueError("Both treatment and control observations are required")

    response_rates = {
        t: counts[(t, "1")] / treatment_counts[t] for t in ("0", "1")
    }
    return {
        "dataset": "x5_retailhero_uplift_train",
        "rows": rows,
        "treatment_counts": treatment_counts,
        "response_rates": response_rates,
        "observed_response_difference": response_rates["1"] - response_rates["0"],
        "cells": {f"treatment_{t}_target_{y}": counts[(t, y)] for t in ("0", "1") for y in ("0", "1")},
        "warnings": [
            "Observed differences are not causal effects without justified assignment assumptions.",
            "Review recruitment, treatment assignment, outcome window, and pre-treatment feature timing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate X5 RetailHero uplift_train.csv")
    parser.add_argument("path", type=Path, help="Local path to uplift_train.csv")
    parser.add_argument("--output", type=Path, help="Optional JSON summary output path")
    args = parser.parse_args()
    report = validate_x5(args.path)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

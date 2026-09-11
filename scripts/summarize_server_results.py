"""Aggregate seed-level server evaluation JSON files into a publication table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "mIoU",
    "precision",
    "recall",
    "F1",
    "Pd",
    "Fa_1e6_per_pixel",
    "seconds_per_image",
    "trainable_parameters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RCP-RiRUFold seed-level JSON results")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def aggregate_records(records):
    """Validate and aggregate in-memory seed records (also used by smoke tests)."""
    validated = []
    for source, record in records:
        required = {"dataset", "net_name", "seed", *METRICS}
        missing = sorted(required.difference(record))
        if missing:
            raise ValueError("{} is missing keys: {}".format(source, ", ".join(missing)))
        validated.append(record)

    frame = pd.DataFrame.from_records(validated)
    duplicate = frame.duplicated(["dataset", "net_name", "seed"], keep=False)
    if duplicate.any():
        rows = frame.loc[duplicate, ["dataset", "net_name", "seed"]]
        raise ValueError("Duplicate dataset/model/seed records:\n{}".format(rows.to_string(index=False)))

    grouped = frame.groupby(["dataset", "net_name"], sort=True)
    summary = grouped[METRICS].agg(["mean", "std"])
    summary.columns = ["{}_{}".format(metric, statistic) for metric, statistic in summary.columns]
    summary.insert(0, "seed_count", grouped["seed"].nunique())
    return summary.reset_index()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError("No JSON result files found under {}".format(args.input_root))
    records = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    summary = aggregate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Read {} seed-level records; wrote {}".format(len(records), args.output.resolve()))


if __name__ == "__main__":
    main()

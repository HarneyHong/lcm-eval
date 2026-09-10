#!/usr/bin/env python3
"""Report QPP-Net coverage and verify strict matched prediction files."""

import argparse
import csv
import json
import math
import statistics


def percentile(values, fraction):
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def load_predictions(specification):
    model, path = specification.split("=", 1)
    with open(path, newline="") as prediction_file:
        rows = list(csv.DictReader(prediction_file))
    ids = [row.get("query_id") or row.get("query_index") for row in rows]
    if any(query_id is None for query_id in ids):
        raise ValueError(f"{path} has no query_id/query_index column")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate query IDs")
    qerrors = [float(row["qerror"]) for row in rows]
    return model, path, ids, qerrors


def evaluate(alignment_manifest_path, split_manifest_path, prediction_specs=()):
    with open(alignment_manifest_path) as alignment_file:
        alignment = json.load(alignment_file)
    with open(split_manifest_path) as split_file:
        split = json.load(split_file)

    supported = {
        query_id for query_id in alignment["master_query_ids"]
        if alignment["queries"][query_id].get("qppnet_supported") is True
    }
    matched = {
        name: [query_id for query_id in split[name] if query_id in supported]
        for name in ("train", "validation", "test")
    }
    result = {
        "seed": split["seed"],
        "coverage": {
            name: {
                "supported": len(matched[name]),
                "master": len(split[name]),
                "ratio": len(matched[name]) / len(split[name]) if split[name] else 0,
            }
            for name in matched
        },
        "predictions": {},
    }
    expected_test = set(matched["test"])
    for specification in prediction_specs:
        model, path, ids, qerrors = load_predictions(specification)
        if set(ids) != expected_test:
            missing = sorted(expected_test - set(ids))
            unexpected = sorted(set(ids) - expected_test)
            raise ValueError(
                f"{model} predictions are not the strict matched test set: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        result["predictions"][model] = {
            "path": path,
            "count": len(ids),
            "qerror_50": statistics.median(qerrors),
            "qerror_95": percentile(qerrors, 0.95),
            "qerror_max": max(qerrors),
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment_manifest", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--prediction", action="append", default=[], metavar="MODEL=CSV")
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = evaluate(
        arguments.alignment_manifest,
        arguments.split_manifest,
        arguments.prediction,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if arguments.output:
        with open(arguments.output, "w") as output_file:
            output_file.write(rendered + "\n")

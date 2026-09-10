#!/usr/bin/env python3
"""Build an aligned baseline master and the corresponding JSON master.

The input raw/JSON files must come from ``run_benchmark.py --mode paired``.
The augmented baseline input is a staging artifact produced by the existing
raw -> baseline parse -> sample bitmap pipeline.  QPP-Net support is recorded
after the baseline master is fixed and never changes master membership.
"""

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from training.featurizations import QPPNetFeaturization


JOIN_TYPES = {"Hash Join", "Merge Join", "Nested Loop"}
QPP_SUPPORTED_TYPES = set(QPPNetFeaturization.QPP_NET_OPERATOR_TYPES)
QPP_ALLOWED_MISSING = {
    ("Hash", "Hash Buckets"),
    ("Hash", "Peak Memory Usage"),
    ("Join", "Parent Relationship"),
    ("Sort", "Sort Method"),
}


class InsufficientMasterQueries(RuntimeError):
    pass


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as output_file:
        json.dump(value, output_file)
    os.replace(temp, path)


def _query_map(queries, source_name):
    result = {}
    for query in queries:
        query_id = query.get("query_id")
        if not query_id:
            raise ValueError(f"{source_name} contains a query without query_id")
        if query_id in result:
            raise ValueError(f"{source_name} contains duplicate query_id {query_id}")
        result[query_id] = query
    return result


def _contains_subplan(query):
    lines = []
    for line in query.get("verbose_plan") or []:
        lines.append(line[0] if isinstance(line, (list, tuple)) else str(line))
    text = "\n".join(lines)
    return "InitPlan" in text or "SubPlan" in text


def _filter_count(filter_node):
    if filter_node is None:
        return 0
    count = 1
    for child in filter_node.get("children", []) or []:
        count += _filter_count(child)
    return count


def _queryformer_filter_issue(filter_node, database_statistics, column_statistics):
    operator = filter_node.get("operator")
    if operator not in {"AND", "OR"}:
        column_id = filter_node.get("column")
        if column_id is None:
            return "queryformer_missing_filter_column"
        try:
            column = database_statistics["column_stats"][int(column_id)]
            statistics = column_statistics[column["tablename"]][column["attname"]]
        except (IndexError, KeyError, TypeError, ValueError):
            return "queryformer_unknown_filter_column"
        if statistics.get("datatype") in {"float", "int"}:
            literal = filter_node.get("literal")
            if literal is None:
                return "queryformer_missing_numeric_literal"
            if not isinstance(literal, (int, float)):
                return "queryformer_unsupported_numeric_literal"
    for child in filter_node.get("children", []) or []:
        issue = _queryformer_filter_issue(child, database_statistics, column_statistics)
        if issue:
            return issue
    return None


def _baseline_input_issue(plan, database_statistics, column_statistics,
                          max_nodes=30, max_filters=6, max_joins=5):
    if not plan.get("query_id"):
        return "missing_query_id"
    if len(plan.get("join_conds", [])) > max_joins:
        return "too_many_joins"

    node_count = 0
    stack = [plan]
    while stack:
        node = stack.pop()
        node_count += 1
        parameters = node.get("plan_parameters") or {}
        if not parameters.get("op_name"):
            return "missing_operator_name"
        filter_node = parameters.get("filter_columns")
        if filter_node is not None:
            if _filter_count(filter_node) > max_filters:
                return "too_many_filters"
            if parameters.get("sample_vec") is None:
                return "missing_sample_bitmap"
            issue = _queryformer_filter_issue(filter_node, database_statistics, column_statistics)
            if issue:
                return issue
        stack.extend(node.get("children", []) or [])
    if node_count > max_nodes:
        return "too_many_plan_nodes"
    return None


def _qpp_scan_statistics_issue(node, column_statistics):
    condition = node.get("Filter") or node.get("Recheck Cond")
    if not condition:
        return None
    relation = node.get("Relation Name")
    if relation not in column_statistics:
        return "unknown_scan_relation"
    if not any(column_name in condition for column_name in column_statistics[relation]):
        return "unmapped_scan_filter_column"
    return None


def _qpp_input_issues(node, column_statistics, issues):
    raw_type = node.get("Node Type")
    node_type = "Join" if raw_type in JOIN_TYPES else raw_type
    if node_type not in QPP_SUPPORTED_TYPES:
        issues.append(f"unsupported_operator:{raw_type}")
    else:
        if "Actual Total Time" not in node:
            issues.append(f"missing_feature:{raw_type}:Actual Total Time")
        for feature in QPPNetFeaturization.QPP_NET_OPERATOR_TYPES[node_type]:
            if feature in {"Min", "Max", "Mean"} and "Scan" in node_type:
                issue = _qpp_scan_statistics_issue(node, column_statistics)
                if issue:
                    issues.append(f"{issue}:{raw_type}")
                continue
            if feature not in node and (node_type, feature) not in QPP_ALLOWED_MISSING:
                issues.append(f"missing_feature:{raw_type}:{feature}")
            value = node.get(feature)
            if isinstance(value, list) and feature != "Sort Key" and len(value) != 1:
                issues.append(f"unsupported_feature_arity:{raw_type}:{feature}")

        children = node.get("Plans", []) or []
        if node_type == "Join" and len(children) != 2:
            issues.append(f"unsupported_child_count:{raw_type}:{len(children)}")
        elif node_type == "Bitmap Heap Scan" and len(children) != 1:
            issues.append(f"unsupported_child_count:{raw_type}:{len(children)}")
        elif ("Scan" in node_type or node_type == "Result") and node_type != "Bitmap Heap Scan" and children:
            issues.append(f"unsupported_child_count:{raw_type}:{len(children)}")
        elif node_type not in {"Join", "Bitmap Heap Scan", "Result"} and "Scan" not in node_type \
                and len(children) != 1:
            issues.append(f"unsupported_child_count:{raw_type}:{len(children)}")

    for child in node.get("Plans", []) or []:
        _qpp_input_issues(child, column_statistics, issues)


def qppnet_support(query, column_statistics):
    analyze_plans = query.get("analyze_plans") or []
    if not analyze_plans:
        return False, ["missing_analyze_plan"]
    issues = []
    _qpp_input_issues(analyze_plans[0]["Plan"], column_statistics, issues)
    issues = sorted(set(issues))
    return not issues, issues


def create_split(master_query_ids, seed):
    ordered = sorted(
        master_query_ids,
        key=lambda query_id: hashlib.sha256(f"{seed}:{query_id}".encode("utf-8")).digest(),
    )
    train_end = int(len(ordered) * 0.8)
    validation_end = train_end + int(len(ordered) * 0.1)
    return {
        "seed": seed,
        "master_count": len(ordered),
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def prepare_aligned_workload(
        raw_path, json_path, augmented_baseline_path, baseline_master_path,
        json_master_path, alignment_manifest_path, split_dir,
        column_statistics_path, database, master_count=10000, seeds=(0, 1, 2)):
    with open(raw_path) as raw_file:
        raw_run = json.load(raw_file)
    with open(json_path) as json_file:
        json_run = json.load(json_file)
    with open(augmented_baseline_path) as baseline_file:
        baseline_run = json.load(baseline_file)
    with open(column_statistics_path) as statistics_file:
        column_statistics = json.load(statistics_file)

    for source_name, run in (("raw run", raw_run), ("JSON run", json_run),
                             ("augmented baseline run", baseline_run)):
        if run.get("collection_mode") != "paired":
            raise ValueError(f"{source_name} was not produced from paired collection")
    workload_digest = raw_run.get("source_workload_sha256")
    if not workload_digest or json_run.get("source_workload_sha256") != workload_digest \
            or baseline_run.get("source_workload_sha256") != workload_digest:
        raise ValueError("Raw, JSON and augmented baseline inputs belong to different workloads")
    paired_settings = raw_run.get("paired_collection_settings")
    if not paired_settings or json_run.get("paired_collection_settings") != paired_settings \
            or baseline_run.get("paired_collection_settings") != paired_settings:
        raise ValueError("Raw, JSON and augmented baseline inputs use different paired settings")

    raw_by_id = _query_map(raw_run.get("query_list", []), "raw run")
    json_by_id = _query_map(json_run.get("query_list", []), "JSON run")
    baseline_by_id = _query_map(baseline_run.get("parsed_plans", []), "augmented baseline run")
    if set(raw_by_id) != set(json_by_id):
        raise ValueError("Paired raw and JSON files do not contain identical query_id sets")

    statuses = {}
    eligible = []
    baseline_reasons = Counter()
    for query_id, raw_query in raw_by_id.items():
        json_query = json_by_id[query_id]
        if raw_query.get("sql") != json_query.get("sql"):
            raise ValueError(f"SQL mismatch for paired query_id {query_id}")
        if raw_query.get("workload_index") != json_query.get("workload_index"):
            raise ValueError(f"Workload index mismatch for paired query_id {query_id}")
        for key in ("pair_valid", "pair_invalid_reasons", "raw_runtime_ms", "json_runtime_ms",
                    "raw_plan_fingerprint_sha256", "json_plan_fingerprint_sha256"):
            if raw_query.get(key) != json_query.get(key):
                raise ValueError(f"Paired metadata {key} mismatch for query_id {query_id}")
        pair_valid = bool(raw_query.get("pair_valid")) and bool(json_query.get("pair_valid"))
        if pair_valid and (not raw_query.get("raw_plan_fingerprint_sha256")
                           or raw_query.get("raw_plan_fingerprint_sha256")
                           != raw_query.get("json_plan_fingerprint_sha256")):
            raise ValueError(f"Valid pair has inconsistent plan fingerprints for query_id {query_id}")
        status = {
            "workload_index": raw_query.get("workload_index"),
            "sql_sha256": hashlib.sha256(raw_query["sql"].encode("utf-8")).hexdigest(),
            "pair_valid": pair_valid,
            "pair_invalid_reasons": raw_query.get("pair_invalid_reasons", []),
            "raw_runtime_ms": raw_query.get("raw_runtime_ms"),
            "json_runtime_ms": raw_query.get("json_runtime_ms"),
            "raw_plan_fingerprint_sha256": raw_query.get("raw_plan_fingerprint_sha256"),
            "json_plan_fingerprint_sha256": raw_query.get("json_plan_fingerprint_sha256"),
            "baseline_master": False,
            "baseline_exclusion_reason": None,
            "qppnet_supported": None,
            "qppnet_issues": [],
        }
        if not pair_valid:
            reason = "pair_invalid"
        elif _contains_subplan(raw_query):
            reason = "baseline_parser_unsupported_initplan_or_subplan"
        elif query_id not in baseline_by_id:
            reason = "baseline_parse_failed_or_filtered"
        else:
            reason = _baseline_input_issue(
                baseline_by_id[query_id],
                baseline_run["database_stats"],
                column_statistics,
            )
        if reason is None:
            eligible.append(query_id)
        else:
            status["baseline_exclusion_reason"] = reason
            baseline_reasons[reason] += 1
        statuses[query_id] = status

    eligible.sort(key=lambda query_id: statuses[query_id]["workload_index"])
    selected_ids = eligible[:master_count]
    for query_id in selected_ids:
        statuses[query_id]["baseline_master"] = True
        supported, issues = qppnet_support(json_by_id[query_id], column_statistics)
        statuses[query_id]["qppnet_supported"] = supported
        statuses[query_id]["qppnet_issues"] = issues

    qpp_supported_count = sum(bool(statuses[query_id]["qppnet_supported"]) for query_id in selected_ids)
    manifest = {
        "version": 1,
        "database": database,
        "source_workload_sha256": workload_digest,
        "paired_collection_settings": paired_settings,
        "ready": len(selected_ids) == master_count,
        "requested_master_count": master_count,
        "master_query_ids": selected_ids,
        "summary": {
            "paired_candidates": len(raw_by_id),
            "pair_valid": sum(bool(query.get("pair_valid")) for query in raw_by_id.values()),
            "baseline_eligible": len(eligible),
            "baseline_master": len(selected_ids),
            "qppnet_supported_in_master": qpp_supported_count,
            "qppnet_coverage_in_master": qpp_supported_count / len(selected_ids) if selected_ids else 0,
            "baseline_exclusion_reasons": dict(baseline_reasons),
        },
        "queries": statuses,
    }
    _write_json(alignment_manifest_path, manifest)

    if len(selected_ids) < master_count:
        raise InsufficientMasterQueries(
            f"Only {len(selected_ids)} baseline-master queries are ready; "
            f"collect and preprocess at least {master_count - len(selected_ids)} more"
        )

    baseline_master = copy.deepcopy(baseline_run)
    baseline_master["parsed_plans"] = [baseline_by_id[query_id] for query_id in selected_ids]
    baseline_master["alignment_manifest"] = os.fspath(alignment_manifest_path)
    json_master = copy.deepcopy(json_run)
    json_master["query_list"] = [json_by_id[query_id] for query_id in selected_ids]
    json_master["alignment_manifest"] = os.fspath(alignment_manifest_path)
    _write_json(baseline_master_path, baseline_master)
    _write_json(json_master_path, json_master)

    split_dir = Path(split_dir)
    for seed in seeds:
        _write_json(split_dir / f"seed_{seed}.json", create_split(selected_ids, seed))
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--augmented_baseline", required=True)
    parser.add_argument("--baseline_master", required=True)
    parser.add_argument("--json_master", required=True)
    parser.add_argument("--alignment_manifest", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--column_statistics", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--master_count", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        result = prepare_aligned_workload(
            raw_path=arguments.raw,
            json_path=arguments.json,
            augmented_baseline_path=arguments.augmented_baseline,
            baseline_master_path=arguments.baseline_master,
            json_master_path=arguments.json_master,
            alignment_manifest_path=arguments.alignment_manifest,
            split_dir=arguments.split_dir,
            column_statistics_path=arguments.column_statistics,
            database=arguments.database,
            master_count=arguments.master_count,
            seeds=arguments.seeds,
        )
    except InsufficientMasterQueries as exc:
        raise SystemExit(str(exc))
    print(json.dumps(result["summary"], indent=2))

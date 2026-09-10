"""Paired PostgreSQL workload collection.

Each workload line is executed once with text EXPLAIN and once with JSON
EXPLAIN.  The two normal run files remain the source artifacts; shared IDs and
pair status make them auditable without introducing a third copy of the plans.
"""

import copy
import hashlib
import json
import math
import os
import shutil
import time
from json.decoder import JSONDecodeError

from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.load_database import create_db_conn
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_raw_plan
from cross_db_benchmark.benchmark_tools.postgres.plan_fingerprint import (
    json_plan_fingerprint,
    raw_plan_fingerprint,
)
from cross_db_benchmark.benchmark_tools.utils import load_json


def workload_sha256(workload_path):
    digest = hashlib.sha256()
    with open(workload_path, "rb") as workload_file:
        for block in iter(lambda: workload_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_query_id(db_name, workload_digest, workload_index, sql):
    payload = f"{db_name}\0{workload_digest}\0{workload_index}\0{sql.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _execution_issue(result):
    if result.get("query_error"):
        return "query_error"
    if result.get("timeout"):
        return "timeout"
    if not result.get("analyze_plans"):
        return "missing_analyze_plan"
    return None


def _raw_metrics(result):
    runtimes = []
    root = None
    for analyze_plan in result["analyze_plans"]:
        parsed, runtime, _ = parse_raw_plan(copy.deepcopy(analyze_plan), analyze=True, parse=True)
        runtimes.append(runtime)
        if root is None:
            root = parsed
    root.parse_lines_recursively()
    return sum(runtimes) / len(runtimes), root.min_card()


def _json_min_cardinality(node):
    cards = []
    if "Actual Rows" in node:
        cards.append(float(node["Actual Rows"]))
    for child in node.get("Plans", []) or []:
        cards.append(_json_min_cardinality(child))
    return min(cards) if cards else float("inf")


def _json_metrics(result):
    plans = result["analyze_plans"]
    runtime = sum(float(plan["Execution Time"]) for plan in plans) / len(plans)
    return runtime, _json_min_cardinality(plans[0]["Plan"])


def evaluate_pair(raw_result, json_result, min_runtime=100, max_runtime=30000):
    reasons = []
    raw_issue = _execution_issue(raw_result)
    json_issue = _execution_issue(json_result)
    if raw_issue:
        reasons.append(f"raw_{raw_issue}")
    if json_issue:
        reasons.append(f"json_{json_issue}")

    raw_runtime = None
    json_runtime = None
    raw_cardinality = None
    json_cardinality = None
    raw_fingerprint_hash = None
    json_fingerprint_hash = None
    if raw_issue is None:
        try:
            raw_runtime, raw_cardinality = _raw_metrics(raw_result)
        except Exception as exc:
            reasons.append(f"raw_plan_error:{type(exc).__name__}")
    if json_issue is None:
        try:
            json_runtime, json_cardinality = _json_metrics(json_result)
        except Exception as exc:
            reasons.append(f"json_plan_error:{type(exc).__name__}")

    if raw_runtime is not None:
        if raw_cardinality is None or not math.isfinite(raw_cardinality):
            reasons.append("raw_missing_cardinality")
        elif raw_cardinality == 0:
            reasons.append("raw_zero_cardinality")
        if not min_runtime <= raw_runtime <= max_runtime:
            reasons.append("raw_runtime_out_of_range")
    if json_runtime is not None:
        if json_cardinality is None or not math.isfinite(json_cardinality):
            reasons.append("json_missing_cardinality")
        elif json_cardinality == 0:
            reasons.append("json_zero_cardinality")
        if not min_runtime <= json_runtime <= max_runtime:
            reasons.append("json_runtime_out_of_range")

    if raw_issue is None and json_issue is None:
        try:
            raw_fingerprint = raw_plan_fingerprint(raw_result["analyze_plans"][0])
            json_fingerprint = json_plan_fingerprint(json_result["analyze_plans"][0])
            raw_fingerprint_hash = hashlib.sha256(
                json.dumps(raw_fingerprint, sort_keys=True).encode("utf-8")
            ).hexdigest()
            json_fingerprint_hash = hashlib.sha256(
                json.dumps(json_fingerprint, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if raw_fingerprint != json_fingerprint:
                reasons.append("physical_plan_mismatch")
        except Exception as exc:
            reasons.append(f"plan_fingerprint_error:{type(exc).__name__}")

    return {
        "pair_valid": not reasons,
        "pair_invalid_reasons": reasons,
        "raw_runtime_ms": raw_runtime,
        "json_runtime_ms": json_runtime,
        "raw_plan_fingerprint_sha256": raw_fingerprint_hash,
        "json_plan_fingerprint_sha256": json_fingerprint_hash,
    }


def _save_run(run, target_path):
    target_path = os.fspath(target_path)
    target_dir = os.path.dirname(target_path) or "."
    os.makedirs(target_dir, exist_ok=True)
    temp_path = os.path.join(target_dir, f"{os.path.basename(target_path)}_temp")
    with open(temp_path, "w") as output_file:
        json.dump(run, output_file)
    shutil.move(temp_path, target_path)


def _load_existing_pair(raw_target_path, json_target_path, expected_workload_digest,
                        expected_settings):
    if not os.path.exists(raw_target_path) and not os.path.exists(json_target_path):
        return [], [], 0
    if not os.path.exists(raw_target_path) or not os.path.exists(json_target_path):
        raise ValueError("Paired resume requires both raw and JSON target files")
    try:
        raw_run = load_json(raw_target_path, namespace=False)
        json_run = load_json(json_target_path, namespace=False)
    except JSONDecodeError as exc:
        raise ValueError("Could not resume paired collection from invalid JSON") from exc
    for name, run in (("raw", raw_run), ("JSON", json_run)):
        if run.get("collection_mode") != "paired":
            raise ValueError(f"Existing {name} target was not produced by paired collection")
        if run.get("source_workload_sha256") != expected_workload_digest:
            raise ValueError(f"Existing {name} target belongs to a different workload file")
        if run.get("paired_collection_settings") != expected_settings:
            raise ValueError(f"Existing {name} target used different paired collection settings")

    raw_by_id = {query.get("query_id"): query for query in raw_run.get("query_list", [])}
    json_by_id = {query.get("query_id"): query for query in json_run.get("query_list", [])}
    if None in raw_by_id or None in json_by_id:
        raise ValueError("Existing paired targets predate query_id; use new target files")
    if len(raw_by_id) != len(raw_run.get("query_list", [])) or \
            len(json_by_id) != len(json_run.get("query_list", [])):
        raise ValueError("Existing paired targets contain duplicate query IDs")
    common_ids = set(raw_by_id) & set(json_by_id)
    if len(common_ids) != len(raw_by_id) or len(common_ids) != len(json_by_id):
        print("Warning: dropping incomplete trailing paired records while resuming")
    ordered_ids = sorted(common_ids, key=lambda query_id: raw_by_id[query_id]["workload_index"])
    for query_id in ordered_ids:
        raw_query = raw_by_id[query_id]
        json_query = json_by_id[query_id]
        if raw_query.get("sql") != json_query.get("sql") or \
                raw_query.get("workload_index") != json_query.get("workload_index"):
            raise ValueError(f"Existing paired record mismatch for query_id {query_id}")
    raw_queries = [raw_by_id[query_id] for query_id in ordered_ids]
    json_queries = [json_by_id[query_id] for query_id in ordered_ids]
    elapsed = max(raw_run.get("total_time_secs", 0), json_run.get("total_time_secs", 0))
    return raw_queries, json_queries, elapsed


def run_pg_paired_workload(
        workload_path, database, db_name, database_conn_args, database_kwarg_dict,
        raw_target_path, json_target_path, run_kwargs, repetitions_per_query,
        timeout_sec, cap_workload=None, random_hints=None, with_indexes=False,
        min_runtime=100, max_runtime=30000, index_modifier=None):
    if repetitions_per_query != 1:
        raise ValueError("Paired collection requires repetitions_per_query=1")

    with open(workload_path) as workload_file:
        sql_queries = [line.strip() for line in workload_file]
    workload_digest = workload_sha256(workload_path)

    hints = ["" for _ in sql_queries]
    if random_hints is not None:
        with open(random_hints) as hints_file:
            hints = [line.strip() for line in hints_file]
    if len(hints) != len(sql_queries):
        raise ValueError("The hints file must contain one line per SQL query")

    paired_settings = {
        "database": str(database),
        "db_name": db_name,
        "repetitions_per_query": repetitions_per_query,
        "timeout_sec": timeout_sec,
        "min_runtime_ms": min_runtime,
        "max_runtime_ms": max_runtime,
        "with_indexes": with_indexes,
        "hints_sha256": workload_sha256(random_hints) if random_hints is not None else None,
        "execution_order_policy": "alternate_raw_json_by_workload_index",
    }

    if os.path.abspath(raw_target_path) == os.path.abspath(json_target_path):
        raise ValueError("Raw and JSON paired targets must be different files")
    raw_queries, json_queries, time_offset = _load_existing_pair(
        raw_target_path,
        json_target_path,
        workload_digest,
        paired_settings,
    )
    seen_ids = {query["query_id"] for query in raw_queries}
    pair_valid_count = sum(bool(query.get("pair_valid")) for query in raw_queries)
    if cap_workload is not None and pair_valid_count >= cap_workload:
        print(f"Existing paired files already contain {pair_valid_count} valid pairs")
        return

    db_conn = create_db_conn(database, db_name, database_conn_args, database_kwarg_dict)
    database_stats = db_conn.collect_db_statistics()
    db_conn.set_statement_timeout(timeout_sec)
    existing_indexes = {}
    if with_indexes:
        db_conn.remove_remaining_fk_indexes()

    start = time.perf_counter()
    try:
        for workload_index, sql in enumerate(tqdm(sql_queries)):
            query_id = make_query_id(db_name, workload_digest, workload_index, sql)
            if query_id in seen_ids:
                continue
            if with_indexes and index_modifier is not None:
                index_modifier(db_conn, sql, existing_indexes, timeout_sec)

            execution_order = [ExecutionMode.RAW_OUTPUT, ExecutionMode.JSON_OUTPUT]
            if workload_index % 2:
                execution_order.reverse()
            results = {}
            for execution_mode in execution_order:
                results[execution_mode] = db_conn.run_query_collect_statistics(
                    sql=sql,
                    mode=execution_mode,
                    repetitions=repetitions_per_query,
                    prefix=hints[workload_index],
                    hint_validation=False,
                    include_hint_notices=False,
                    explain_only=False,
                )

            pair = evaluate_pair(
                results[ExecutionMode.RAW_OUTPUT],
                results[ExecutionMode.JSON_OUTPUT],
                min_runtime=min_runtime,
                max_runtime=max_runtime,
            )
            metadata = {
                "query_id": query_id,
                "workload_index": workload_index,
                "sql": sql,
                "hint": hints[workload_index],
                "execution_order": execution_order,
                **pair,
            }
            raw_query = {**results[ExecutionMode.RAW_OUTPUT], **metadata}
            json_query = {**results[ExecutionMode.JSON_OUTPUT], **metadata}
            raw_queries.append(raw_query)
            json_queries.append(json_query)
            seen_ids.add(query_id)
            if pair["pair_valid"]:
                pair_valid_count += 1

            elapsed = time_offset + time.perf_counter() - start
            collection_metadata = {
                "collection_mode": "paired",
                "source_workload_sha256": workload_digest,
                "paired_collection_settings": paired_settings,
                "database_stats": database_stats,
                "run_kwargs": run_kwargs,
                "total_time_secs": elapsed,
            }
            if len(raw_queries) % 50 == 0:
                _save_run({**collection_metadata, "query_list": raw_queries}, raw_target_path)
                _save_run({**collection_metadata, "query_list": json_queries}, json_target_path)

            print(f"Valid paired queries {pair_valid_count}/{cap_workload or 'all'}")
            if cap_workload is not None and pair_valid_count >= cap_workload:
                break
    finally:
        if with_indexes:
            db_conn.remove_remaining_fk_indexes()

    elapsed = time_offset + time.perf_counter() - start
    collection_metadata = {
        "collection_mode": "paired",
        "source_workload_sha256": workload_digest,
        "paired_collection_settings": paired_settings,
        "database_stats": database_stats,
        "run_kwargs": run_kwargs,
        "total_time_secs": elapsed,
    }
    _save_run({**collection_metadata, "query_list": raw_queries}, raw_target_path)
    _save_run({**collection_metadata, "query_list": json_queries}, json_target_path)
    print(f"Collected {pair_valid_count} valid pairs in {elapsed:.2f}s")

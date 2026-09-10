import json
import csv
from types import SimpleNamespace

from evaluation.aligned_metrics import evaluate
from prepare_aligned_workload import _baseline_input_issue, create_split, prepare_aligned_workload
from training.dataset.aligned_splits import create_aligned_datasets


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def json_query(query_id, index, node_type="Seq Scan"):
    if node_type == "Seq Scan":
        plan = {
            "Node Type": "Seq Scan",
            "Relation Name": "t",
            "Plan Rows": 1,
            "Plan Width": 4,
            "Total Cost": 10,
            "Actual Rows": 1,
            "Actual Total Time": 120,
        }
    else:
        plan = {
            "Node Type": node_type,
            "Plan Rows": 1,
            "Plan Width": 4,
            "Total Cost": 10,
            "Actual Rows": 1,
            "Actual Total Time": 120,
            "Plans": [],
        }
    return {
        "query_id": query_id,
        "workload_index": index,
        "sql": f"SELECT {index}",
        "pair_valid": index != 3,
        "pair_invalid_reasons": [] if index != 3 else ["raw_timeout"],
        "raw_plan_fingerprint_sha256": "same" if index != 3 else None,
        "json_plan_fingerprint_sha256": "same" if index != 3 else None,
        "analyze_plans": [{"Plan": plan, "Execution Time": 120}],
    }


def baseline_plan(query_id, index):
    return {
        "query_id": query_id,
        "workload_index": index,
        "sql": f"SELECT {index}",
        "join_conds": [],
        "plan_runtime": 120,
        "plan_parameters": {"op_name": "Seq Scan"},
        "children": [],
    }


def test_master_is_baseline_defined_and_qpp_support_is_only_a_mask(tmp_path):
    raw_queries = [json_query(f"q{i}", i) for i in range(4)]
    raw_queries[1]["verbose_plan"] = [["InitPlan 1"]]
    json_queries = [json_query(f"q{i}", i, "Memoize" if i == 2 else "Seq Scan") for i in range(4)]
    baseline_plans = [baseline_plan("q0", 0), baseline_plan("q2", 2), baseline_plan("q3", 3)]
    root = {
        "collection_mode": "paired",
        "source_workload_sha256": "workload-digest",
        "paired_collection_settings": {"repetitions_per_query": 1},
        "database_stats": {},
        "run_kwargs": {},
    }

    raw_path = tmp_path / "raw.json"
    json_path = tmp_path / "json.json"
    baseline_path = tmp_path / "augmented.json"
    stats_path = tmp_path / "column_statistics.json"
    baseline_master_path = tmp_path / "baseline_master.json"
    json_master_path = tmp_path / "json_master.json"
    manifest_path = tmp_path / "alignment" / "manifest.json"
    split_dir = tmp_path / "alignment" / "splits"
    write_json(raw_path, {**root, "query_list": raw_queries})
    write_json(json_path, {**root, "query_list": json_queries})
    write_json(baseline_path, {**root, "parsed_plans": baseline_plans})
    write_json(stats_path, {"t": {"x": {"datatype": "int", "num_unique": 2}}})

    manifest = prepare_aligned_workload(
        raw_path, json_path, baseline_path, baseline_master_path, json_master_path,
        manifest_path, split_dir, stats_path, database="test", master_count=2, seeds=(0,),
    )
    assert manifest["master_query_ids"] == ["q0", "q2"]
    assert manifest["queries"]["q1"]["baseline_exclusion_reason"] == \
        "baseline_parser_unsupported_initplan_or_subplan"
    assert manifest["queries"]["q2"]["qppnet_supported"] is False
    assert "unsupported_operator:Memoize" in manifest["queries"]["q2"]["qppnet_issues"]
    assert [q["query_id"] for q in json.loads(json_master_path.read_text())["query_list"]] == ["q0", "q2"]


def test_manifest_split_is_shared_and_matched_never_reassigns(tmp_path):
    query_ids = [f"q{i}" for i in range(10)]
    split = create_split(query_ids, seed=0)
    split_path = tmp_path / "split.json"
    alignment_path = tmp_path / "alignment.json"
    write_json(split_path, split)
    write_json(alignment_path, {
        "master_query_ids": query_ids,
        "queries": {
            query_id: {"qppnet_supported": query_id != "q3"}
            for query_id in query_ids
        },
    })
    plans = [SimpleNamespace(query_id=query_id) for query_id in query_ids]

    native = create_aligned_datasets(plans, split_path, alignment_path, False)
    matched = create_aligned_datasets(plans, split_path, alignment_path, True)
    assert [len(dataset) for dataset in native[:3]] == [8, 1, 1]
    assert sum(len(dataset) for dataset in matched[:3]) == 9
    for native_ids, matched_ids in zip(native[3].values(), matched[3].values()):
        assert matched_ids == [query_id for query_id in native_ids if query_id != "q3"]


def test_aligned_metrics_requires_exact_matched_test_ids(tmp_path):
    query_ids = [f"q{i}" for i in range(10)]
    split = create_split(query_ids, seed=0)
    unsupported_id = split["train"][0]
    split_path = tmp_path / "split.json"
    alignment_path = tmp_path / "alignment.json"
    prediction_path = tmp_path / "predictions.csv"
    alignment = {
        "master_query_ids": query_ids,
        "queries": {
            query_id: {"qppnet_supported": query_id != unsupported_id}
            for query_id in query_ids
        },
    }
    write_json(split_path, split)
    write_json(alignment_path, alignment)
    matched_test = list(split["test"])
    with prediction_path.open("w", newline="") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=["query_id", "qerror"])
        writer.writeheader()
        for query_id in matched_test:
            writer.writerow({"query_id": query_id, "qerror": 1.25})

    report = evaluate(alignment_path, split_path, [f"e2e={prediction_path}"])
    assert report["coverage"]["test"]["supported"] == len(matched_test)
    assert report["predictions"]["e2e"]["qerror_50"] == 1.25


def test_queryformer_unsupported_numeric_null_is_excluded_from_master():
    plan = baseline_plan("q0", 0)
    plan["plan_parameters"]["filter_columns"] = {
        "column": 0,
        "operator": "IS NOT NULL",
        "literal": None,
        "children": [],
    }
    plan["plan_parameters"]["sample_vec"] = [1]
    database_statistics = {
        "column_stats": [{"tablename": "t", "attname": "x"}],
    }
    column_statistics = {
        "t": {"x": {"datatype": "int"}},
    }
    assert _baseline_input_issue(plan, database_statistics, column_statistics) == \
        "queryformer_missing_numeric_literal"

import json

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.postgres import paired_workload
from cross_db_benchmark.benchmark_tools.postgres.plan_fingerprint import (
    json_plan_fingerprint,
    raw_plan_fingerprint,
)


def raw_plan(runtime):
    return [
        [f"Seq Scan on t  (cost=0.00..10.00 rows=1 width=4) (actual time=0.010..{runtime:.3f} rows=1 loops=1)"],
        ["Planning Time: 0.100 ms"],
        [f"Execution Time: {runtime:.3f} ms"],
    ]


def json_plan(runtime):
    return {
        "Plan": {
            "Node Type": "Seq Scan",
            "Parallel Aware": False,
            "Relation Name": "t",
            "Plan Rows": 1,
            "Plan Width": 4,
            "Total Cost": 10.0,
            "Actual Rows": 1,
            "Actual Total Time": runtime,
        },
        "Planning Time": 0.1,
        "Execution Time": runtime,
    }


def test_text_and_json_plan_fingerprints_match():
    assert raw_plan_fingerprint(raw_plan(120)) == json_plan_fingerprint(json_plan(121))


def test_pair_diagnostics_keep_the_successful_side_runtime():
    raw_result = {"timeout": True, "query_error": None, "analyze_plans": None}
    json_result = {
        "timeout": False,
        "query_error": None,
        "analyze_plans": [json_plan(120)],
    }
    pair = paired_workload.evaluate_pair(raw_result, json_result)
    assert pair["pair_valid"] is False
    assert pair["pair_invalid_reasons"] == ["raw_timeout"]
    assert pair["raw_runtime_ms"] is None
    assert pair["json_runtime_ms"] == 120


def test_paired_collection_uses_both_runtime_boundaries_and_shared_ids(tmp_path, monkeypatch):
    workload = tmp_path / "workload.sql"
    workload.write_text("SELECT 1;\nSELECT 2;\n")
    raw_target = tmp_path / "raw" / "run.json"
    json_target = tmp_path / "json" / "run.json"
    calls = []

    class FakeConnection:
        def collect_db_statistics(self):
            return {"column_stats": [], "table_stats": []}

        def set_statement_timeout(self, timeout):
            pass

        def run_query_collect_statistics(self, sql, mode, **kwargs):
            calls.append((sql, mode))
            runtimes = {
                "SELECT 1;": {ExecutionMode.RAW_OUTPUT: 98, ExecutionMode.JSON_OUTPUT: 103},
                "SELECT 2;": {ExecutionMode.RAW_OUTPUT: 120, ExecutionMode.JSON_OUTPUT: 121},
            }
            runtime = runtimes[sql][mode]
            analyze = raw_plan(runtime) if mode == ExecutionMode.RAW_OUTPUT else json_plan(runtime)
            return {
                "analyze_plans": [analyze],
                "verbose_plan": [],
                "timeout": False,
                "query_error": None,
                "hint_notices": None,
            }

    monkeypatch.setattr(paired_workload, "create_db_conn", lambda *args, **kwargs: FakeConnection())
    paired_workload.run_pg_paired_workload(
        workload_path=workload,
        database="postgres",
        db_name="test",
        database_conn_args={},
        database_kwarg_dict={},
        raw_target_path=raw_target,
        json_target_path=json_target,
        run_kwargs={},
        repetitions_per_query=1,
        timeout_sec=30,
        cap_workload=1,
    )

    raw_run = json.loads(raw_target.read_text())
    json_run = json.loads(json_target.read_text())
    assert [query["query_id"] for query in raw_run["query_list"]] == [
        query["query_id"] for query in json_run["query_list"]
    ]
    assert [query["pair_valid"] for query in raw_run["query_list"]] == [False, True]
    assert "raw_runtime_out_of_range" in raw_run["query_list"][0]["pair_invalid_reasons"]
    assert calls == [
        ("SELECT 1;", ExecutionMode.RAW_OUTPUT),
        ("SELECT 1;", ExecutionMode.JSON_OUTPUT),
        ("SELECT 2;", ExecutionMode.JSON_OUTPUT),
        ("SELECT 2;", ExecutionMode.RAW_OUTPUT),
    ]


def test_paired_collection_resumes_by_query_id(tmp_path, monkeypatch):
    workload = tmp_path / "workload.sql"
    workload.write_text("SELECT 1;\nSELECT 2;\n")
    raw_target = tmp_path / "raw.json"
    json_target = tmp_path / "json.json"
    calls = []

    class FakeConnection:
        def collect_db_statistics(self):
            return {"column_stats": [], "table_stats": []}

        def set_statement_timeout(self, timeout):
            pass

        def run_query_collect_statistics(self, sql, mode, **kwargs):
            calls.append((sql, mode))
            runtime = 120 if sql == "SELECT 1;" else 140
            analyze = raw_plan(runtime) if mode == ExecutionMode.RAW_OUTPUT else json_plan(runtime)
            return {
                "analyze_plans": [analyze],
                "verbose_plan": [],
                "timeout": False,
                "query_error": None,
                "hint_notices": None,
            }

    monkeypatch.setattr(paired_workload, "create_db_conn", lambda *args, **kwargs: FakeConnection())
    arguments = dict(
        workload_path=workload,
        database="postgres",
        db_name="test",
        database_conn_args={},
        database_kwarg_dict={},
        raw_target_path=raw_target,
        json_target_path=json_target,
        run_kwargs={},
        repetitions_per_query=1,
        timeout_sec=30,
    )
    paired_workload.run_pg_paired_workload(**arguments, cap_workload=1)
    paired_workload.run_pg_paired_workload(**arguments, cap_workload=2)

    raw_run = json.loads(raw_target.read_text())
    json_run = json.loads(json_target.read_text())
    assert len(raw_run["query_list"]) == len(json_run["query_list"]) == 2
    assert calls == [
        ("SELECT 1;", ExecutionMode.RAW_OUTPUT),
        ("SELECT 1;", ExecutionMode.JSON_OUTPUT),
        ("SELECT 2;", ExecutionMode.JSON_OUTPUT),
        ("SELECT 2;", ExecutionMode.RAW_OUTPUT),
    ]


def test_paired_collection_rejects_multiple_repetitions(tmp_path):
    workload = tmp_path / "workload.sql"
    workload.write_text("SELECT 1;\n")
    try:
        paired_workload.run_pg_paired_workload(
            workload_path=workload,
            database="postgres",
            db_name="test",
            database_conn_args={},
            database_kwarg_dict={},
            raw_target_path=tmp_path / "raw.json",
            json_target_path=tmp_path / "json.json",
            run_kwargs={},
            repetitions_per_query=2,
            timeout_sec=30,
        )
    except ValueError as exc:
        assert str(exc) == "Paired collection requires repetitions_per_query=1"
    else:
        raise AssertionError("Multiple paired repetitions should be rejected")

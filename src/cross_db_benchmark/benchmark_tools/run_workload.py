from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.postgres.paired_workload import run_pg_paired_workload
from cross_db_benchmark.benchmark_tools.postgres.run_workload import modify_indexes
from cross_db_benchmark.benchmark_tools.postgres.run_workload import run_pg_workload


def run_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path, run_kwargs,
                 repetitions_per_query, timeout_sec, mode, hints=None, with_indexes=False, cap_workload=None, explain_only: bool = False,
                 min_runtime=100, max_runtime=30000, raw_target_path=None, json_target_path=None):
    if database == DatabaseSystem.POSTGRES:
        if mode == "paired":
            if explain_only:
                raise ValueError("Paired collection requires EXPLAIN ANALYZE")
            if raw_target_path is None or json_target_path is None:
                raise ValueError("Paired collection requires --raw_target and --json_target")
            run_pg_paired_workload(
                workload_path, database, db_name, database_conn_args, database_kwarg_dict,
                raw_target_path, json_target_path, run_kwargs, repetitions_per_query,
                timeout_sec, cap_workload=cap_workload, random_hints=hints,
                with_indexes=with_indexes, min_runtime=min_runtime,
                max_runtime=max_runtime, index_modifier=modify_indexes)
        else:
            run_pg_workload(workload_path, database, db_name, database_conn_args, database_kwarg_dict, target_path,
                            run_kwargs, repetitions_per_query, timeout_sec, random_hints=hints, with_indexes=with_indexes,
                            cap_workload=cap_workload, min_runtime=min_runtime, mode=mode, explain_only=explain_only)
    else:
        raise NotImplementedError

import argparse

from cross_db_benchmark.benchmark_tools.compare_runs import compare_runs
from cross_db_benchmark.benchmark_tools.create_fk_indexes import create_fk_indexes
from cross_db_benchmark.benchmark_tools.database import DatabaseSystem
from cross_db_benchmark.benchmark_tools.drop_db import drop_db
from cross_db_benchmark.benchmark_tools.generate_column_stats import generate_column_statistics
from cross_db_benchmark.benchmark_tools.generate_string_statistics import generate_string_stats
from cross_db_benchmark.benchmark_tools.generate_workload import generate_workload
from cross_db_benchmark.benchmark_tools.join_conditions import check_join_conditions
from cross_db_benchmark.benchmark_tools.load_database import load_database
from cross_db_benchmark.benchmark_tools.parse_run import parse_run
from cross_db_benchmark.benchmark_tools.run_workload import run_workload
from cross_db_benchmark.meta_tools.inflate_cardinality_errors import inflate_cardinality_errors
from cross_db_benchmark.meta_tools.scale_dataset import scale_up_dataset
from cross_db_benchmark.meta_tools.slice_no_tables import slice_by_table_no


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--database', default=DatabaseSystem.POSTGRES, type=DatabaseSystem,
                        choices=list(DatabaseSystem))
    parser.add_argument('--db_name', default=None)
    parser.add_argument("--database_conn", dest='database_conn_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--database_kwargs", dest='database_kwarg_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    parser.add_argument("--run_kwargs", dest='run_kwargs_dict', action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")

    # workload generation
    parser.add_argument('--workload_max_no_joins', default=3, type=int)
    parser.add_argument('--workload_max_no_predicates', default=3, type=int)
    parser.add_argument('--workload_max_no_aggregates', default=3, type=int)
    parser.add_argument('--workload_max_no_group_by', default=0, type=int)
    parser.add_argument('--workload_max_cols_per_agg', default=1, type=int)
    parser.add_argument('--workload_num_queries', default=100, type=int)
    parser.add_argument('--workload_left_outer_join_ratio', default=0., type=float)
    parser.add_argument('--workload_groupby_limit_prob', default=0., type=float)
    parser.add_argument('--workload_groupby_having_prob', default=0., type=float)
    parser.add_argument('--workload_exists_predicate_prob', default=0., type=float)
    parser.add_argument('--workload_max_no_exists', default=0, type=int)
    parser.add_argument('--workload_outer_groupby_prob', default=0., type=float)
    parser.add_argument('--query_timeout', default=30, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--target', default=None)
    parser.add_argument('--raw_target', default=None)
    parser.add_argument('--json_target', default=None)
    parser.add_argument('--workload_slice_stats', default=None)
    parser.add_argument('--source', default=None)
    parser.add_argument('--hints', default=None)
    parser.add_argument('--second_source', default=None)
    parser.add_argument('--repetitions_per_query', default=1, type=int)
    parser.add_argument('--scale', default=None, type=int)
    parser.add_argument('--card_error_factor', default=1, type=float)
    parser.add_argument('--autoscale', default=2000000, type=int)
    parser.add_argument('--no_prev_replications', default=0, type=int)
    parser.add_argument('--cap_workload', default=None, type=int)

    # query parsing
    parser.add_argument('--min_query_ms', default=100, type=int)
    parser.add_argument('--max_query_ms', default=30000, type=int)
    parser.add_argument('--min_no_tables', default=None, type=int)
    parser.add_argument('--max_no_tables', default=None, type=int)
    parser.add_argument('--with_indexes', action='store_true')

    # Benchmark Steps
    parser.add_argument('--generate_column_statistics', action='store_true')
    parser.add_argument('--create_fk_indexes', action='store_true')
    parser.add_argument('--load_database', action='store_true')
    parser.add_argument('--generate_workload', action='store_true')
    parser.add_argument('--check_join_conditions', action='store_true')
    parser.add_argument('--run_workload', action='store_true')
    parser.add_argument('--mode', choices=["json", "raw", "paired"], default="json"),
    parser.add_argument('--run_explain_only', action='store_true')
    parser.add_argument('--parse_explain_only', action='store_true')
    parser.add_argument('--parse_run', action='store_true')
    parser.add_argument('--scale_dataset', action='store_true')
    parser.add_argument('--compare_runs', action='store_true')
    parser.add_argument('--generate_string_statistics', action='store_true')
    parser.add_argument('--complex_predicates', action='store_true')
    parser.add_argument('--parse_baseline', action='store_true')
    parser.add_argument('--parse_join_conds', action='store_true')
    parser.add_argument('--scale_in_db', action='store_true')
    parser.add_argument('--drop_db', action='store_true')
    parser.add_argument('--scale_adaptively', action='store_true')
    parser.add_argument('--include_zero_card', action='store_true')
    parser.add_argument('--slice_no_tables', action='store_true')
    parser.add_argument('--inflate_cardinality_errors', action='store_true')
    parser.add_argument('--cap_queries', default=None, type=int)

    args = parser.parse_args()

    if args.database_kwarg_dict is None:
        args.database_kwarg_dict = dict()

    force = True

    if args.generate_column_statistics:
        generate_column_statistics(args.data_dir, args.dataset, force=force)

    if args.generate_string_statistics:
        generate_string_stats(args.data_dir, args.dataset, force=force)

    if args.scale_dataset:
        scale_up_dataset(args.dataset, args.data_dir, args.target, scale=args.scale, autoscale_tuples=args.autoscale)

    if args.load_database:
        load_database(args.data_dir, args.dataset, args.database, args.db_name, args.database_conn_dict,
                      args.database_kwarg_dict, force=force)
    if args.drop_db:
        drop_db(args.database, args.db_name, args.database_conn_dict, args.database_kwarg_dict)

    if args.generate_workload:
        generate_workload(args.dataset, args.target, num_queries=args.workload_num_queries,
                          max_no_joins=args.workload_max_no_joins, max_no_predicates=args.workload_max_no_predicates,
                          max_no_aggregates=args.workload_max_no_aggregates,
                          max_no_group_by=args.workload_max_no_group_by,
                          max_cols_per_agg=args.workload_max_cols_per_agg, seed=args.seed,
                          complex_predicates=args.complex_predicates,
                          left_outer_join_ratio=args.workload_left_outer_join_ratio,
                          groupby_limit_prob=args.workload_groupby_limit_prob,
                          groupby_having_prob=args.workload_groupby_having_prob,
                          exists_predicate_prob=args.workload_exists_predicate_prob,
                          max_no_exists=args.workload_max_no_exists,
                          outer_groupby_prob=args.workload_outer_groupby_prob)

    if args.check_join_conditions:
        check_join_conditions(args.dataset, args.database, args.db_name, args.database_conn_dict,
                              args.database_kwarg_dict)

    if args.create_fk_indexes:
        create_fk_indexes(args.dataset, args.database, args.db_name, args.database_conn_dict,
                          args.database_kwarg_dict)

    if args.run_workload:
        run_workload(args.source, args.database, args.db_name, args.database_conn_dict, args.database_kwarg_dict,
                     args.target, args.run_kwargs_dict, args.repetitions_per_query, args.query_timeout, mode=args.mode,
                     with_indexes=args.with_indexes, cap_workload=args.cap_workload, min_runtime=args.min_query_ms,
                     max_runtime=args.max_query_ms, raw_target_path=args.raw_target,
                     json_target_path=args.json_target, hints=args.hints, explain_only=args.run_explain_only)

    if args.parse_run:
        parse_run(args.source, args.target, args.database, min_query_ms=args.min_query_ms,
                  max_query_ms=args.max_query_ms, parse_baseline=args.parse_baseline, cap_queries=args.cap_queries,
                  parse_join_conds=args.parse_join_conds, include_zero_card=args.include_zero_card,
                  explain_only=args.parse_explain_only)

    if args.compare_runs:
        compare_runs(args.source, args.second_source, args.database)

    if args.slice_no_tables:
        slice_by_table_no(args.source, args.target, args.min_no_tables, args.max_no_tables, args.workload_slice_stats)

    if args.inflate_cardinality_errors:
        inflate_cardinality_errors(args.source, args.target, args.card_error_factor, args.database)

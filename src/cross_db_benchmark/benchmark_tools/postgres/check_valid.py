from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.postgres.json_plan import operator_tree_from_json
from cross_db_benchmark.benchmark_tools.postgres.parse_plan import parse_raw_plan
import copy
import traceback


def check_valid(mode: ExecutionMode, curr_statistics: dict, min_runtime: int = 100, verbose=True) -> bool:
    query_error = curr_statistics.get('query_error')
    if query_error:
        if verbose:
            print(f"Invalid because query execution failed "
                  f"({query_error.get('error_type', 'database error')})")
        return False

    # Timeouts are also a valid signal in learning
    if 'timeout' in curr_statistics and curr_statistics['timeout']:
        if verbose:
            print("Invalid since it ran into a timeout")
        return False

    try:
        analyze_plans = curr_statistics['analyze_plans']

        if analyze_plans is None or len(analyze_plans) == 0:
            if verbose:
                print("Invalid because no analyze plans are available")
            return False

        if mode == ExecutionMode.JSON_OUTPUT:
            analyze_plan = copy.deepcopy(analyze_plans[0])
            analyze_plan = operator_tree_from_json(analyze_plan)
            runtime = analyze_plan.runtime * 1000
            cardinality = analyze_plan.min_cardinality()

        else:
            analyze_plan = analyze_plans[0]
            analyze_plan, runtime, _ = parse_raw_plan(analyze_plan, analyze=True, parse=True)
            analyze_plan.parse_lines_recursively()
            cardinality = analyze_plan.min_card()

        if cardinality == 0:
            if verbose:
                print("Invalid because of zero cardinality")
            return False

        if runtime < min_runtime:
            if verbose:
                print("Invalid because of too short runtime")
            return False

        return True
    except Exception as e:
        if verbose:
            print("Invalid due to error" + traceback.format_exc())
        return False

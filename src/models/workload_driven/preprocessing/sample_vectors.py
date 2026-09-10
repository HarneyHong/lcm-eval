import json
import os

import pandas as pd
from tqdm import tqdm

from cross_db_benchmark.benchmark_tools.parse_run import dumper
from cross_db_benchmark.benchmark_tools.utils import load_json, load_schema_json


def construct_filter_sample(table_samples, col_stats, filter_column):
    sample_vec = None
    if filter_column.operator in {'=', '>=', '<=', '!=', 'NOT LIKE', 'LIKE', 'IN', 'IS NOT NULL', 'IS NULL'}:
        col_name = col_stats[filter_column.column].attname
        table_name = col_stats[filter_column.column].tablename

        base_sample = table_samples[table_name][col_name]

        if filter_column.operator == '=':
            sample_vec = base_sample == filter_column.literal
        elif filter_column.operator == '>=':
            sample_vec = base_sample >= filter_column.literal
        elif filter_column.operator == '<=':
            sample_vec = base_sample <= filter_column.literal
        elif filter_column.operator == '!=':
            sample_vec = base_sample != filter_column.literal
        elif filter_column.operator in {'NOT LIKE', 'LIKE'}:
            # replace wildcards to obtain valid regex
            regex = filter_column.literal
            for esc_char in ['.', '(', ')', '?']:
                regex = regex.replace(esc_char, f'\{esc_char}')
            regex = regex.replace('%', '.*?')
            regex = f'^{regex}$'
            sample_vec = base_sample.str.contains(regex)
            if filter_column.operator == 'NOT LIKE':
                sample_vec[sample_vec.isna()] = False
                sample_vec = ~sample_vec
        elif filter_column.operator == 'IN':
            assert isinstance(filter_column.literal, list)
            sample_vec = base_sample.isin(filter_column.literal)
        elif filter_column.operator == 'IS NOT NULL':
            sample_vec = ~base_sample.isna()
        elif filter_column.operator == 'IS NULL':
            sample_vec = base_sample.isna()

        # handle nans
        sample_vec[sample_vec.isna()] = False

    elif filter_column.operator in {'AND', 'OR'}:
        children_vec = [construct_filter_sample(table_samples, col_stats, c) for c in filter_column.children]
        sample_vec = children_vec[0]
        for c_vec in children_vec[1:]:
            if filter_column.operator == 'AND':
                sample_vec &= c_vec
            elif filter_column.operator == 'OR':
                sample_vec ^= c_vec
            else:
                raise NotImplementedError

    else:
        print(filter_column.operator)
        raise NotImplementedError

    assert sample_vec is not None
    return sample_vec


def augment_sample(table_samples, col_stats, p_node):
    filter_columns = vars(p_node.plan_parameters).get('filter_columns')
    if filter_columns is not None:
        sample_vec = construct_filter_sample(table_samples, col_stats, filter_columns)
        # convert to binary
        sample_vec = [1 if s_i else 0 for s_i in sample_vec]
        p_node.plan_parameters.sample_vec = sample_vec

    for c in p_node.children:
        augment_sample(table_samples, col_stats, c)


def augment_sample_vectors(dataset, data_dir, plan_path, target_path, no_samples=1000, force=False):
    print("Augment Sample Vectors")
    if os.path.exists(target_path) and not force:
        print(f'Skip for {target_path}')
        return
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    run = load_json(plan_path)
    col_stats = run.database_stats.column_stats

    # create sample per table
    schema = load_schema_json(dataset)

    # read individual table csvs and derive statistics
    table_samples = dict()
    for t in schema.tables:
        table_dir = os.path.join(data_dir, f'{t}.csv')
        assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"
        print(f"Generating sample for {t}")

        # df_sample = pd.read_csv(table_dir, **vars(schema.csv_kwargs), nrows=1000)
        df_sample = pd.read_csv(table_dir, **vars(schema.csv_kwargs))
        if len(df_sample) > no_samples:
            df_sample = df_sample.sample(random_state=0, n=no_samples)
        table_samples[t] = df_sample

    for p in tqdm(run.parsed_plans):
        augment_sample(table_samples, col_stats, p)

    with open(target_path, 'w') as outfile:
        json.dump(run, outfile, default=dumper)

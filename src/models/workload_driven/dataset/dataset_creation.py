import functools
from pathlib import Path

from gensim.models import KeyedVectors
from torch.utils.data import DataLoader

from models.query_former.dataloader import query_former_plan_collator
from models.workload_driven.dataset.mscn_batching import mscn_plan_collator
from models.workload_driven.dataset.plan_tree_batching import baseline_plan_collator, extract_dimensions
from classes.classes import ModelConfig, DataLoaderOptions, TPoolModelConfig, TlstmModelConfig, \
    TestSumModelConfig, MSCNModelConfig, QueryFormerModelConfig
from classes.workload_runs import WorkloadRuns
from cross_db_benchmark.benchmark_tools.utils import load_json
from training.dataset.aligned_splits import create_aligned_datasets
from training.dataset.dataset_creation import create_datasets, read_workload_runs


class InputDims:
    input_dim_plan: int
    input_dim_pred: int


class PlanModelInputDims(InputDims):
    def __init__(self, feature_statistics, dim_word_embedding, dim_word_hash, dim_bitmaps):
        dim_cols, dim_ops, dim_tables, dim_pred_op = extract_dimensions(feature_statistics)
        self.input_dim_pred = dim_cols + dim_pred_op + dim_word_embedding + dim_word_hash
        self.input_dim_plan = dim_cols + dim_tables + dim_ops + dim_bitmaps


class MSCNInputDims(InputDims):
    def __init__(self, feature_statistics, dim_word_embedding, dim_word_hash, dim_bitmaps):
        dim_cols, dim_ops, dim_tables, dim_pred_op, dim_joins, dim_aggs = \
            extract_dimensions(feature_statistics, extended=True)
        self.input_dim_table = dim_tables + dim_bitmaps
        self.input_dim_pred = dim_cols + dim_pred_op + dim_word_embedding + dim_word_hash
        self.input_dim_join = dim_joins
        self.input_dim_agg = dim_aggs + dim_cols


def make_filters_consistent(fc, db_col_mapping):
    if fc.column is not None:
        fc.column = db_col_mapping[fc.column]

    for c in fc.children:
        make_filters_consistent(c, db_col_mapping)

    return fc


def make_pg_plan_consistent(p, db_col_mapping, db_table_mapping):
    params = p.plan_parameters
    if hasattr(params, 'table'):
        params.table = db_table_mapping[params.table]

    if hasattr(params, 'output_columns'):
        for oc in params.output_columns:
            if len(oc.columns) > 0:
                oc.columns = [db_col_mapping[c] for c in oc.columns]

    if hasattr(params, 'filter_columns'):
        fc = params.filter_columns
        make_filters_consistent(fc, db_col_mapping)

    for c in p.children:
        make_pg_plan_consistent(c, db_col_mapping, db_table_mapping)


def make_datasets_consistent(col_id, table_id, test_database_statistics, test_dataset):
    col_mapping = dict()
    table_mapping = dict()
    for db_id, test_stat in test_database_statistics.items():
        db_col_mapping = dict()
        for i, test_c in enumerate(test_stat.column_stats):
            mapped_id = col_id.get((test_c.attname, test_c.tablename))
            if mapped_id is not None:
                db_col_mapping[i] = mapped_id

        db_table_mapping = dict()
        for i, test_c in enumerate(test_stat.table_stats):
            mapped_id = table_id.get(test_c.relname)
            if mapped_id is not None:
                db_table_mapping[i] = mapped_id

        col_mapping[db_id] = db_col_mapping
        table_mapping[db_id] = db_table_mapping

    # There might be two plans pointing to the same object in memory. We do not want to permute those plans twice.
    # Hence, we keep track of which plans were already permuted
    for p in test_dataset.plans:
        if not hasattr(p, 'permuted'):
            p.permuted = False

    # adapt plans to refer to the actual columns (traverse plans and replace tables, columns)
    for p in test_dataset.plans:
        if p.permuted:
            continue
        p.permuted = True
        db_id = p.database_id
        db_col_mapping = col_mapping[db_id]
        db_table_mapping = table_mapping[db_id]
        make_pg_plan_consistent(p, db_col_mapping, db_table_mapping)


def assert_db_stats_consistence(database_statistics, test_database_statistics, test_dataset):
    """
    Makes sure one-hot encoded columns and tables refer to the same database
    :param database_statistics:
    :param test_database_statistics:
    :param test_dataset:
    :return:
    """
    train_stats = database_statistics[0]

    # gold standard mapping
    col_id = {(train_c.attname, train_c.tablename): i for i, train_c in enumerate(train_stats.column_stats)}
    table_id = {train_t.relname: i for i, train_t in enumerate(train_stats.table_stats)}

    # map every other dataset to the gold standard
    make_datasets_consistent(col_id, table_id, test_database_statistics, test_dataset)

    print("Successfully checked consistency of database statistics")


def get_collate_func(model_config: ModelConfig, database_statistics: dict, feature_statistics: dict,
                     column_statistics: dict,
                     word_embeddings: KeyedVectors,
                     dim_word_hash: int, dim_word_emdb: int, dim_bitmaps: int):
    if isinstance(model_config, (TPoolModelConfig, TlstmModelConfig, TestSumModelConfig)):
        collate_kwargs = dict(db_statistics=database_statistics,
                              feature_statistics=feature_statistics,
                              column_statistics=column_statistics,
                              word_embeddings=word_embeddings,
                              dim_word_hash=dim_word_hash,
                              dim_word_emdb=dim_word_emdb,
                              dim_bitmaps=dim_bitmaps)
        plan_collator_func = baseline_plan_collator
        input_dims: InputDims = PlanModelInputDims(feature_statistics, dim_word_emdb, dim_word_hash, dim_bitmaps)

    elif isinstance(model_config, MSCNModelConfig):
        collate_kwargs = dict(db_statistics=database_statistics,
                              feature_statistics=feature_statistics,
                              column_statistics=column_statistics,
                              word_embeddings=word_embeddings,
                              dim_word_hash=dim_word_hash,
                              dim_word_emdb=dim_word_emdb,
                              dim_bitmaps=dim_bitmaps)
        plan_collator_func = mscn_plan_collator
        input_dims: InputDims = MSCNInputDims(feature_statistics, dim_word_emdb, dim_word_hash, dim_bitmaps)

    elif isinstance(model_config, QueryFormerModelConfig):
        collate_kwargs = dict(db_statistics=database_statistics,
                              feature_statistics=feature_statistics,
                              column_statistics=column_statistics,
                              word_embeddings=word_embeddings,
                              dim_word_hash=dim_word_hash,
                              dim_word_embedding=dim_word_emdb,
                              histogram_bin_size = model_config.histogram_bin_number,
                              max_num_filters = model_config.max_num_filters,
                              dim_bitmaps=dim_bitmaps)
        plan_collator_func = query_former_plan_collator
        input_dims: InputDims = PlanModelInputDims(feature_statistics, dim_word_emdb, dim_word_hash, dim_bitmaps)

    else:
        raise NotImplementedError

    collate_fn = functools.partial(plan_collator_func, **collate_kwargs)
    return collate_fn, input_dims


def create_baseline_dataloader(workload_runs: WorkloadRuns,
                               statistics_file: Path,
                               column_statistics,
                               word_embeddings,
                               model_config: ModelConfig,
                               data_loader_options: DataLoaderOptions) -> (
        dict, DataLoader, DataLoader, DataLoader, InputDims):
    train_loader, val_loader, test_loaders = None, None, None

    dim_bitmaps = 1000
    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"
    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=data_loader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=data_loader_options.pin_memory)

    column_statistics = load_json(column_statistics, namespace=False)
    word_embeddings = KeyedVectors.load(str(word_embeddings), mmap='r')
    dim_word_emdb = word_embeddings.vector_size
    dim_word_hash = dim_word_emdb

    if workload_runs.train_workload_runs:
        assert workload_runs.test_workload_runs == [], ("Unseen Test workload runs are not allowed when training "
                                                        "workload driven models")
        print("Create dataloader for training, validation and test data")

        if workload_runs.split_manifest is not None:
            if workload_runs.experiment_protocol not in {"baseline_native", "matched"}:
                raise ValueError("Baseline models require baseline_native or matched protocol")
            plans, database_statistics = read_workload_runs(
                workload_run_paths=workload_runs.train_workload_runs,
                execution_mode=model_config.execution_mode,
            )
            train_dataset, val_dataset, test_dataset, _ = create_aligned_datasets(
                plans,
                workload_runs.split_manifest,
                workload_runs.alignment_manifest,
                apply_qppnet_support=workload_runs.experiment_protocol == "matched",
            )
        else:
            label_norm, train_dataset, val_dataset, database_statistics = \
                create_datasets(workload_run_paths=workload_runs.train_workload_runs,
                                model_config=model_config,
                                val_ratio=data_loader_options.val_ratio)
            test_dataset, val_dataset = val_dataset.split(0.5)
        print(
            f"Created datasets of size: train {len(train_dataset)}, validation: {len(val_dataset)}, test: {len(test_dataset)}")
        assert_db_stats_consistence(database_statistics, database_statistics, train_dataset)
        assert_db_stats_consistence(database_statistics, database_statistics, val_dataset)
        assert_db_stats_consistence(database_statistics, database_statistics, test_dataset)

        # plan_collator does the heavy lifting of creating the graphs and extracting the features and thus requires both
        # database statistics but also feature statistics

        train_collate_fn, input_dims = get_collate_func(model_config=model_config,
                                                        database_statistics=database_statistics,
                                                        feature_statistics=feature_statistics,
                                                        column_statistics=column_statistics,
                                                        word_embeddings=word_embeddings,
                                                        dim_word_hash=dim_word_hash,
                                                        dim_word_emdb=dim_word_emdb,
                                                        dim_bitmaps=dim_bitmaps)
        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        test_loaders = [DataLoader(test_dataset, **dataloader_args)]

    if workload_runs.test_workload_runs:
        test_loaders = []
        for test_workload in workload_runs.test_workload_runs:
            _, _, test_dataset, database_statistics = \
                create_datasets(workload_run_paths=[test_workload],
                                model_config=model_config,
                                val_ratio=1.0)

            assert_db_stats_consistence(database_statistics, database_statistics, test_dataset)

            test_collate_fn, input_dims = get_collate_func(model_config=model_config,
                                                           database_statistics=database_statistics,
                                                           feature_statistics=feature_statistics,
                                                           column_statistics=column_statistics,
                                                           word_embeddings=word_embeddings,
                                                           dim_word_hash=dim_word_hash,
                                                           dim_word_emdb=dim_word_emdb,
                                                           dim_bitmaps=dim_bitmaps)

            dataloader_args.update(collate_fn=test_collate_fn)
            test_loaders.append(DataLoader(test_dataset, **dataloader_args))

    return None, feature_statistics, train_loader, val_loader, test_loaders, input_dims

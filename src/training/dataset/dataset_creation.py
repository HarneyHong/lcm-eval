import functools
from json import JSONDecodeError
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from cross_db_benchmark.benchmark_tools.database import ExecutionMode
from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import ZeroShotModelConfig, DataLoaderOptions, ModelConfig
from classes.workload_runs import WorkloadRuns
from training.dataset.plan_dataset import PlanDataset
from models.zeroshot.postgres_plan_batching import postgres_plan_collator


def read_workload_runs(workload_run_paths: List[Path],
                       execution_mode: ExecutionMode = ExecutionMode.RAW_OUTPUT,
                       limit_queries: bool = None,
                       limit_queries_affected_wl: bool = None):
    # reads several workload runs
    plans = []
    database_statistics = dict()

    for path_index, path in enumerate(workload_run_paths):
        try:
            run = load_json(path)
        except JSONDecodeError:
            raise ValueError(f"Error reading {path}")
        database_statistics[path_index] = run.database_stats
        if hasattr(run, 'run_kwargs'):
         database_statistics[path_index].run_kwars = run.run_kwargs

        limit_per_ds = None
        if limit_queries is not None:
            if path_index >= len(workload_run_paths) - limit_queries_affected_wl:
                limit_per_ds = limit_queries // limit_queries_affected_wl
                print(f"Capping workload {path} after {limit_per_ds} queries")

        if execution_mode == ExecutionMode.RAW_OUTPUT:
            for p_id, plan in enumerate(run.parsed_plans):
                plan.database_id = path_index
                plans.append(plan)
                if limit_per_ds is not None and p_id > limit_per_ds:
                    print("Stopping now")
                    break
        else:
            for plan in run.query_list:
                if plan.analyze_plans:
                    analyze_plans = plan.analyze_plans
                    # assert len(analyze_plans) == 1, "Multiple plans found"
                    analyze_plan = analyze_plans[0]
                    for metadata_key in ("query_id", "workload_index", "sql"):
                        if hasattr(plan, metadata_key):
                            setattr(analyze_plan, metadata_key, getattr(plan, metadata_key))
                    plans.append(analyze_plan)

    #print(f"No of Plans: {len(plans)} for {workload_run_paths}")
    return plans, database_statistics


def _inv_log1p(x):
    return np.exp(x) - 1


def create_datasets(workload_run_paths,
                    model_config: ModelConfig,
                    val_ratio=0.15,
                    shuffle_before_split=True) -> (Pipeline, PlanDataset, PlanDataset, PlanDataset, dict):

    plans, database_statistics = read_workload_runs(workload_run_paths=workload_run_paths,
                                                    limit_queries=model_config.limit_queries,
                                                    limit_queries_affected_wl=model_config.limit_queries_affected_wl,
                                                    execution_mode=model_config.execution_mode)

    no_plans = len(plans)
    plan_indexes = list(range(no_plans))
    if shuffle_before_split:
        np.random.shuffle(plan_indexes)

    train_ratio = 1 - val_ratio
    split_train = int(no_plans * train_ratio)
    train_indexes = plan_indexes[:split_train]
    # Limit number of training samples. To have comparable batch sizes, replicate remaining indexes.
    if model_config.cap_training_samples is not None:
        print(f'Limiting dataset to {model_config.cap_training_samples}')
        prev_train_length = len(train_indexes)
        train_indexes = train_indexes[:model_config.cap_training_samples]
        replicate_factor = max(prev_train_length // len(train_indexes), 1)
        train_indexes = train_indexes * replicate_factor

    train_dataset = PlanDataset([plans[i] for i in train_indexes], train_indexes)

    val_dataset = None
    if val_ratio > 0:
        val_indexes = plan_indexes[split_train:]
        val_dataset = PlanDataset([plans[i] for i in val_indexes], val_indexes)

    # derive label normalization
    runtimes = np.array([p.plan_runtime / 1000 for p in plans])
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    return label_norm, train_dataset, val_dataset, database_statistics


def derive_label_normalizer(loss_class_name, y) -> Pipeline:
    if loss_class_name == "MSELoss":
        log_transformer = preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True)
        scale_transformer = preprocessing.MinMaxScaler()
        pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))

    elif loss_class_name == "QLoss":
        scale_transformer = preprocessing.MinMaxScaler(feature_range=(1e-2, 1))
        pipeline = Pipeline([("scale", scale_transformer)])
        pipeline.fit(y.reshape(-1, 1))

    else:
        pipeline = None

    return pipeline


def create_zeroshot_dataloader(workload_runs: WorkloadRuns,
                               statistics_file: Path,
                               model_config: ZeroShotModelConfig,
                               data_loader_options: DataLoaderOptions) \
        -> (Optional[Pipeline], dict, Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]):
    # Postgres_plan_collator does the heavy lifting of creating the graphs and extracting the features and thus
    # requires both database statistics but also feature statistics
    label_norm = Optional[Pipeline]
    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"
    plan_collator = postgres_plan_collator

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=data_loader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=data_loader_options.pin_memory)

    if workload_runs.train_workload_runs:
        # Split plans into train/test/validation set
        print("Creating dataloader for training and validation data")
        label_norm, train_dataset, val_dataset, database_statistics \
            = create_datasets(workload_run_paths=workload_runs.train_workload_runs,
                              model_config=model_config,
                              val_ratio=data_loader_options.val_ratio)

        train_collate_fn = functools.partial(plan_collator,
                                             db_statistics=database_statistics,
                                             feature_statistics=feature_statistics,
                                             plan_featurization=model_config.featurization)

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader: DataLoader = DataLoader(train_dataset, **dataloader_args)
        val_loader: DataLoader = DataLoader(val_dataset, **dataloader_args)

    if workload_runs.test_workload_runs:
        # For each test workload run create a distinct test loader
        print("Creating dataloader for test data")
        test_loaders = []
        for test_path in workload_runs.test_workload_runs:
            _, test_dataset, _, test_database_statistics = create_datasets([test_path],
                                                                           model_config=model_config,
                                                                           shuffle_before_split=False,
                                                                           val_ratio=0.0)
            # test dataset
            test_collate_fn = functools.partial(plan_collator,
                                                db_statistics=test_database_statistics,
                                                feature_statistics=feature_statistics,
                                                plan_featurization=model_config.featurization)

            # previously shuffle=False but this resulted in bugs
            dataloader_args.update(collate_fn=test_collate_fn)
            test_loader = DataLoader(test_dataset, **dataloader_args)
            test_loaders.append(test_loader)

    return label_norm, feature_statistics, train_loader, val_loader, test_loaders

import copy
import functools
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from cross_db_benchmark.benchmark_tools.postgres.json_plan import operator_tree_from_json, OperatorTree
from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import DataLoaderOptions, QPPNetModelConfig, ModelConfig
from classes.workload_runs import WorkloadRuns
from training.dataset.dataset_creation import read_workload_runs, derive_label_normalizer
from training.dataset.plan_dataset import PlanDataset
from training.dataset.aligned_splits import create_aligned_datasets


def create_qppnet_dataloader(workload_runs: WorkloadRuns,
                             statistics_file: Path,
                             model_config: QPPNetModelConfig,
                             column_statistics: Path,
                             data_loader_options: DataLoaderOptions) \
        -> (Optional[Pipeline], dict, Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]):

    feature_statistics = load_json(statistics_file, namespace=False)
    assert feature_statistics != {}, "Feature statistics file is empty!"

    train_loader, val_loader, test_loaders = Optional[DataLoader], Optional[DataLoader], List[Optional[DataLoader]]

    plan_collator = qppnet_collator

    dataloader_args = dict(batch_size=model_config.batch_size,
                           shuffle=data_loader_options.shuffle,
                           num_workers=model_config.num_workers,
                           pin_memory=data_loader_options.pin_memory)

    column_statistics = load_json(column_statistics, namespace=False)

    if workload_runs.train_workload_runs:
        assert workload_runs.test_workload_runs == [], ("Unseen Test workload runs are not allowed when training "
                                                        "workload driven models")
        print("Create dataloader for training, validation and test data")

        if workload_runs.split_manifest is not None:
            if workload_runs.experiment_protocol not in {"qpp_native", "matched"}:
                raise ValueError("QPP-Net requires qpp_native or matched protocol")
            plans, database_statistics = read_workload_runs(
                workload_run_paths=workload_runs.train_workload_runs,
                execution_mode=model_config.execution_mode,
            )
            train_dataset, val_dataset, test_dataset, _ = create_aligned_datasets(
                plans,
                workload_runs.split_manifest,
                workload_runs.alignment_manifest,
                apply_qppnet_support=True,
            )
        else:
            label_norm, train_dataset, val_dataset, database_statistics \
                = create_datasets(workload_run_paths=workload_runs.train_workload_runs,
                                  model_config=model_config,
                                  val_ratio=data_loader_options.val_ratio)
            test_dataset, val_dataset = val_dataset.split(0.5)
        print(f"Created datasets of size: "
              f"train {len(train_dataset)}, "
              f"validation: {len(val_dataset)}, t"
              f"est: {len(test_dataset)}")

        train_collate_fn = functools.partial(plan_collator,
                                                 db_statistics=database_statistics,
                                                 feature_statistics=feature_statistics,
                                                 column_statistics=column_statistics,
                                                 plan_featurization=model_config.featurization)

        dataloader_args.update(collate_fn=train_collate_fn)
        train_loader = DataLoader(train_dataset, **dataloader_args)
        val_loader = DataLoader(val_dataset, **dataloader_args)
        test_loaders = [DataLoader(test_dataset, **dataloader_args)]

    if workload_runs.test_workload_runs:
        test_loaders = []
        for test_workload in workload_runs.test_workload_runs:
            print(f"Create dataloader for test data: {test_workload}")
            _, _, test_dataset, database_statistics = \
                create_datasets(workload_run_paths=[test_workload],
                                model_config=model_config,
                                val_ratio=1.0)

            train_collate_fn = functools.partial(plan_collator,
                                                 db_statistics=database_statistics,
                                                 feature_statistics=feature_statistics,
                                                 column_statistics=column_statistics,
                                                 plan_featurization=model_config.featurization)

            dataloader_args.update(collate_fn=train_collate_fn)
            test_loaders.append(DataLoader(test_dataset, **dataloader_args))

    return None, feature_statistics, train_loader, val_loader, test_loaders


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
    runtimes = np.array([getattr(p, "Execution Time") / 1000 for p in plans])
    label_norm = derive_label_normalizer(model_config.loss_class_name, runtimes)

    return label_norm, train_dataset, val_dataset, database_statistics


def qppnet_collator(plans, feature_statistics: dict = None, db_statistics: dict = None, column_statistics: dict = None, plan_featurization=None):
    labels = []
    query_plans = []

    # iterate over plans and create lists of edges and features per node
    sample_idxs = []
    for sample_idx, p in plans:
        query_id = getattr(p, "query_id", sample_idx)
        try:
            query_plan: OperatorTree = operator_tree_from_json(copy.deepcopy(vars(p)))
            query_plan.encode_recursively(column_statistics, feature_statistics, plan_featurization)
            sample_idxs.append(sample_idx)
            labels.append(query_plan.runtime)
            query_plans.append(query_plan)
        except Exception as e:
            raise ValueError(f"QPP-Net encoding failed for query_id={query_id}") from e
    return query_plans, labels, sample_idxs

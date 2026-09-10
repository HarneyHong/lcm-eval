import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import attrs as attrs
import numpy as np
import torch
import torch.optim as opt
import wandb
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from wandb.apis.public import Run

from models.dace.dace_dataset import create_dace_dataloader
from models.dace.dace_model import DACELora
from models.qppnet.qppnet_dataloader import create_qppnet_dataloader
from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.tabular.train_tabular_baseline import train_tabular_model, predict_tabular_model
from models.workload_driven.dataset.dataset_creation import create_baseline_dataloader
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import ModelConfig, E2EModelConfig, ZeroShotModelConfig, MSCNModelConfig, DataLoaderOptions, \
    TlstmModelConfig, ModelName, FlatModelConfig, ScaledPostgresModelConfig, TabularModelConfig, TPoolModelConfig, \
    QPPNetModelConfig, DACEModelConfig, DACEModelNoCostsConfig, QPPModelNoCostsConfig, FlatModelActCardModelConfig, \
    QPPModelActCardsConfig, ZeroShotModelActCardConfig, DACEModelActCardConfig, QueryFormerModelConfig, \
    ScaledPostgresModelV16Config
from classes.workload_runs import WorkloadRuns
from training import featurizations
from training.dataset.dataset_creation import create_zeroshot_dataloader
from training.training.checkpoint import load_checkpoint
from training.training.metrics import RMSE, MAPE, QError, Metric
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel
from predict import predict_model
from train import train_model
from attrs import evolve


def init_model(statistics_file: Path,
               column_statistics: Path,
               config: ModelConfig,
               workload_runs: WorkloadRuns,
               word_embeddings: Path) \
        -> (Optional[nn.Module], DataLoader, DataLoader, List[DataLoader]):

    if isinstance(config, ZeroShotModelConfig):
        label_norm, feature_statistics, train_loader, val_loader, test_loaders = \
            create_zeroshot_dataloader(workload_runs=workload_runs,
                                       statistics_file=statistics_file,
                                       model_config=config,
                                       data_loader_options=DataLoaderOptions())

        model: nn.Module = PostgresZeroShotModel(model_config=config,
                                                 feature_statistics=feature_statistics,
                                                 label_norm=label_norm)

    elif isinstance(config, (E2EModelConfig, MSCNModelConfig, TlstmModelConfig, QueryFormerModelConfig)):
        _, feature_statistics, train_loader, val_loader, test_loaders, input_dims = \
            create_baseline_dataloader(workload_runs=workload_runs,
                                       statistics_file=statistics_file,
                                       column_statistics=column_statistics,
                                       word_embeddings=word_embeddings,
                                       model_config=config,
                                       data_loader_options=DataLoaderOptions())

        if isinstance(config, E2EModelConfig):
            model: nn.Module = E2EModel(model_config=config,
                                        input_dims=input_dims)

        elif isinstance(config, MSCNModelConfig):
            model: nn.Module = MSCNModel(model_config=config,
                                         input_dims=input_dims)

        elif isinstance(config, QueryFormerModelConfig):
            model: nn.Module = QueryFormer(config=config,
                                           input_dims=input_dims,
                                           label_norm=None,
                                           feature_statistics=feature_statistics)

    elif isinstance(config, QPPNetModelConfig):
        label_norm, feature_statistics, train_loader, val_loader, test_loaders = \
            create_qppnet_dataloader(workload_runs=workload_runs,
                                     statistics_file=statistics_file,
                                     column_statistics=column_statistics,
                                     model_config=config,
                                     data_loader_options=DataLoaderOptions())

        model: nn.Module = QPPNet(model_config=config,
                                  workload_runs=workload_runs,
                                  feature_statistics=feature_statistics,
                                  label_norm=label_norm)

    elif isinstance(config, DACEModelConfig):
        feature_statistics, train_loader, val_loader, test_loaders = \
            create_dace_dataloader(workload_runs=workload_runs,
                                   statistics_file=statistics_file,
                                   model_config=config,
                                   dataloader_options=DataLoaderOptions())
        model: nn.Module = DACELora(config=config)

    elif isinstance(config, (ScaledPostgresModelConfig, FlatModelConfig)):
        model = None
        train_loader, val_loader, test_loaders = None, None, None
        feature_statistics = None
    else:
        raise NotImplementedError

    return model, train_loader, val_loader, test_loaders, feature_statistics


def init_metrics(loss_class_name: str) -> List[Metric]:
    if loss_class_name in ['QLoss', 'QPPLoss', 'DaceLoss']:
        metrics: List[Metric] = [RMSE(),
                                 MAPE(),
                                 QError(percentile=50, early_stopping_metric=True),
                                 QError(percentile=95),
                                 QError(percentile=100)]

    elif loss_class_name in ['MSELoss']:
        metrics: List[Metric] = [RMSE(early_stopping_metric=True),
                                 MAPE(),
                                 QError(percentile=50),
                                 QError(percentile=95),
                                 QError(percentile=100)]
    else:
        raise Exception("Loss class not supported")
    return metrics


def readout_hyperparameters(hyperparameter_path: Path) -> dict:
    print(f"Reading hyperparameters from {hyperparameter_path}")
    hyperparams = load_json(hyperparameter_path, namespace=False)

    fc_out_kwargs = dict(activation_class_name='LeakyReLU',
                         activation_class_kwargs={},
                         norm_class_name='Identity',
                         norm_class_kwargs={},
                         residual=hyperparams.pop('residual'),
                         dropout=hyperparams.pop('dropout'),
                         activation=True,
                         inplace=True,
                         p_dropout=hyperparams.pop('p_dropout'))

    final_mlp_kwargs = dict(width_factor=hyperparams.pop('final_width_factor'),
                            n_layers=hyperparams.pop('final_layers'),
                            loss_class_name="QLoss",
                            loss_class_kwargs=dict())

    tree_layer_kwargs = dict(width_factor=hyperparams.pop('tree_layer_width_factor'),
                             n_layers=hyperparams.pop('message_passing_layers'))

    node_type_kwargs = dict(width_factor=hyperparams.pop('node_type_width_factor'),
                            n_layers=hyperparams.pop('node_layers'),
                            one_hot_embeddings=True,
                            max_emb_dim=hyperparams.pop('max_emb_dim'),
                            drop_whole_embeddings=False)

    final_mlp_kwargs.update(**fc_out_kwargs)
    tree_layer_kwargs.update(**fc_out_kwargs)
    node_type_kwargs.update(**fc_out_kwargs)

    plan_featurization_name = hyperparams.pop('plan_featurization_name')
    plan_featurization = featurizations.__dict__[plan_featurization_name]

    model_args = dict(
        optimizer_kwargs=dict(lr=hyperparams.pop('lr')),
        fc_out_kwargs=fc_out_kwargs,
        final_mlp_kwargs=final_mlp_kwargs,
        tree_layer_kwargs=tree_layer_kwargs,
        node_type_kwargs=node_type_kwargs,
        tree_layer_name=hyperparams.pop('tree_layer_name'),
        hidden_dim=hyperparams.pop('hidden_dim'),
        batch_size=hyperparams.pop('batch_size'),
        featurization=plan_featurization)

    assert len(hyperparams) == 0, f"Not all hyperparams were used (not used: {hyperparams.keys()}). " \
                                  f"Hence generation and reading does not seem to fit"
    return model_args


def use_model(mode: str,
              wl_runs: WorkloadRuns,
              wandb_name: str,
              statistics_file: Path,
              column_statistics: Path,
              word_embeddings: Path,
              model_dir: Path,
              target_dir: Path,
              config: ModelConfig,
              wandb_project: str):

    # Seed for reproducibility
    print(f"Found model config {config}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Add test workloads to workload_runs
    wl_runs.update_test_workloads(target_dir, config.seed)

    # Check if training and testing was previously executed
    if wl_runs.check_if_done(config.name.NAME):
        return

    # Do sanity check on the training data paths
    wl_runs.check_model_compability(config, mode)

    # Initialize metrics
    metrics: List[Metric] = init_metrics(config.loss_class_name)

    # Initialize models and data loaders
    model, train_loader, val_loader, test_loaders, feature_stats = init_model(config=config,
                                                                              statistics_file=statistics_file,
                                                                              column_statistics=column_statistics,
                                                                              workload_runs=wl_runs,
                                                                              word_embeddings=word_embeddings)

    if model:
        # Move model to device
        model = model.to(model.device)

        # Init optimizer
        optimizer: Optimizer = opt.__dict__[config.optimizer_class_name](model.parameters(),
                                                                         **config.optimizer_kwargs)

        print(f"Load model with name {config.name.NAME} and seed {config.seed} from {model_dir}")
        csv_stats, epochs_wo_improvement, epoch, model, optimizer, metrics, finished = \
            load_checkpoint(model=model,
                            target_path=model_dir,
                            config=config,
                            optimizer=optimizer,
                            metrics=metrics,
                            filetype='.pt')


    run: Optional[Run] = None
    if wandb_project:
        run = wandb.init(project=wandb_project,
                         name=wandb_name,
                         config=attrs.asdict(config))

    # Train model
    if mode == "retrain":
        epochs_wo_improvement = 0

    if mode in ["train", "retrain"]:
        if isinstance(config, TabularModelConfig):
            train_tabular_model(workload_runs=wl_runs,
                                config=config,
                                statistics_file=statistics_file,
                                model_dir=model_dir)
        else:
            train_model(model_dir=model_dir,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=config,
                        csv_stats=csv_stats,
                        epochs_wo_improvement=epochs_wo_improvement,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        run=run)

        print(f'--Run {config.name.NAME} ended at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} --')

    if mode in ["train", "retrain", "predict"]:
        if isinstance(config, TabularModelConfig):
            predict_tabular_model(config=config,
                                  workload_runs=wl_runs,
                                  statistics_file=statistics_file,
                                  model_dir=model_dir,
                                  target_path=target_dir,
                                  run=run)
        else:
            predict_model(mode=mode,
                          config=config,
                          workload_runs=wl_runs,
                          test_loaders=test_loaders,
                          model_dir=model_dir,
                          epoch=epoch,
                          model=model,
                          metrics=metrics,
                          run=run)

    if wandb_project:
        run.finish()


def get_model_config(model_type: str, m_args: dict) -> ModelConfig:
    model_config_map = {
        ModelName.POSTGRES.NAME: ScaledPostgresModelConfig,
        ModelName.POSTGRES_V16.NAME: ScaledPostgresModelV16Config,
        ModelName.MSCN.NAME: MSCNModelConfig,
        ModelName.ZEROSHOT.NAME: ZeroShotModelConfig,
        ModelName.E2E.NAME: TPoolModelConfig,
        ModelName.QPP_NET.NAME: QPPNetModelConfig,
        ModelName.DACE.NAME: DACEModelConfig,
        ModelName.FLAT.NAME: FlatModelConfig,
        ModelName.FLAT_ACT_CARDS.NAME: FlatModelActCardModelConfig,
        ModelName.QUERY_FORMER.NAME: QueryFormerModelConfig,

        # Ablation models without PG costs
        ModelName.DACE_NO_COSTS.NAME: DACEModelNoCostsConfig,
        ModelName.QPP_NET_NO_COSTS.NAME: QPPModelNoCostsConfig,

        # Act card models
        ModelName.QPP_NET_ACT_CARDS.NAME: QPPModelActCardsConfig,
        ModelName.ZEROSHOT_ACT_CARD.NAME: ZeroShotModelActCardConfig,
        ModelName.DACE_ACT_CARDS.NAME: DACEModelActCardConfig,
    }

    if model_type in model_config_map:
        return model_config_map[model_type](**m_args)

    else:
        raise Exception(f"Model name {model_type} not supported")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["train", "predict", "explain", "retrain"])
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--workload_runs', default=None, nargs='+')
    parser.add_argument('--test_workload_runs', default=None, nargs='+')
    parser.add_argument('--statistics_file', default=None)
    parser.add_argument('--column_statistics', default=None)
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--target_dir', default=None, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--hyperparameter_path', default=None)
    parser.add_argument('--word_embeddings', default=None)
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--cap_training_samples', type=int, default=None)
    parser.add_argument('--wandb_name', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--split_manifest', default=None)
    parser.add_argument('--alignment_manifest', default=None)
    parser.add_argument('--experiment_protocol',
                        choices=["legacy", "baseline_native", "qpp_native", "matched"],
                        default="legacy")
    args = parser.parse_args()

    # Init model configuration by either reading out hyperparameters or using standard config
    model_args = {} if args.hyperparameter_path is None else readout_hyperparameters(Path(args.hyperparameter_path))
    model_args.update(device=args.device,
                      cap_training_samples=args.cap_training_samples,
                      seed=args.seed,
                      num_workers=args.num_workers)

    # Create workload runs
    if args.mode in ["predict", "explain"]:
        runs = WorkloadRuns(
            [],
            [Path(raw_path) for raw_path in args.test_workload_runs],
            split_manifest=Path(args.split_manifest) if args.split_manifest else None,
            alignment_manifest=Path(args.alignment_manifest) if args.alignment_manifest else None,
            experiment_protocol=args.experiment_protocol,
        )
        model_args.update(batch_size=16)

    else:
        train_runs = [Path(raw_path) for raw_path in args.workload_runs]
        if args.test_workload_runs:
            test_runs = [Path(raw_path) for raw_path in args.test_workload_runs]
        else:
            test_runs = []
        runs = WorkloadRuns(
            train_workload_runs=train_runs,
            test_workload_runs=test_runs,
            split_manifest=Path(args.split_manifest) if args.split_manifest else None,
            alignment_manifest=Path(args.alignment_manifest) if args.alignment_manifest else None,
            experiment_protocol=args.experiment_protocol,
        )

    model_config = get_model_config(model_type=args.model_type, m_args=model_args)

    # If retraining, set specific model parameters
    if args.mode == "retrain":
        model_config = evolve(model_config, epochs=model_config.epochs + 100)
        # Disable penalty for small values in case of retraining
        if model_config.loss_class_name == "QLoss":
            final_mlp_kwargs = model_config.final_mlp_kwargs
            loss_class_kwargs = final_mlp_kwargs.pop('loss_class_kwargs')
            loss_class_kwargs.update(penalty_negative=0)
            final_mlp_kwargs.update(loss_class_kwargs=loss_class_kwargs)
            model_config = evolve(model_config, final_mlp_kwargs=final_mlp_kwargs)

    args = dict(
        wandb_project=args.wandb_project,
        mode=args.mode,
        wl_runs=runs,
        wandb_name=args.wandb_name,
        statistics_file=Path(args.statistics_file),
        model_dir=Path(args.model_dir),
        target_dir=Path(args.target_dir),
        config=model_config,
        word_embeddings=Path(args.word_embeddings) if args.word_embeddings else None,
        column_statistics=Path(args.column_statistics) if args.column_statistics else None)

    use_model(**args)

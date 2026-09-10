import os
import time
from copy import copy
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as opt
import wandb
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.apis.public import Run

from models.qppnet.qppnet_model import QPPNet
from models.query_former.model import QueryFormer
from models.workload_driven.model.e2e_model import E2EModel
from models.workload_driven.model.mscn_model import MSCNModel
from cross_db_benchmark.benchmark_tools.utils import load_json
from classes.classes import ModelConfig
from training.dataset.dataset_creation import create_zeroshot_dataloader
from training.training.checkpoint import save_checkpoint, load_checkpoint, save_csv
from training.training.metrics import MAPE, RMSE, QError, Metric
from training.training.utils import flatten_dict, find_early_stopping_metric
from training.batch_to_funcs import batch_to
from models.zeroshot.specific_models.model import zero_shot_models
from models.zeroshot.specific_models.postgres_zero_shot import PostgresZeroShotModel


def train_epoch(epoch_stats: dict, train_loader: DataLoader, model: nn.Module,
                optimizer: Optimizer, max_epoch_tuples: int, custom_batch_to=batch_to,
                profiler: Optional[torch.profiler.profile] = None):
    model.train()

    # run remaining batches
    train_start_t = time.perf_counter()
    losses = []
    errs = []
    for batch_idx, batch in enumerate(tqdm(train_loader)):
        if max_epoch_tuples is not None and batch_idx * train_loader.batch_size > max_epoch_tuples:
            break
        input_model, label, sample_idxs = custom_batch_to(batch, model.device, model.label_norm)

        # Reset gradients first
        if isinstance(model, QPPNet):
            model.zero_grad()
        else:
            optimizer.zero_grad()

        # Forward pass
        output = model(input_model)

        # Loss computation
        loss = model.loss_fxn(output, label)
        if torch.isnan(loss):
            raise ValueError('Loss was NaN')

        # Loss backward step
        loss.backward()

        if isinstance(model, (PostgresZeroShotModel, MSCNModel, E2EModel, QPPNet, QueryFormer)):
            clipping_value = 1  # arbitrary value of your choosing
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        # Optimizer backward step
        if isinstance(model, QPPNet):
            model.backward()
        else:
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        output = output.detach().cpu().numpy().reshape(-1)
        label = label.detach().cpu().numpy().reshape(-1)
        errs = np.concatenate((errs, output - label))
        losses.append(loss)

        if profiler is not None:
            profiler.step()

    mean_loss = np.mean(losses)
    mean_rmse = np.sqrt(np.mean(np.square(errs)))
    # print(f"Train Loss: {mean_loss:.2f}")
    # print(f"Train RMSE: {mean_rmse:.2f}")
    epoch_stats.update(train_time=time.perf_counter() - train_start_t, mean_loss=mean_loss, mean_rmse=mean_rmse)


def validate_model(val_loader: DataLoader,
                   model: nn.Module,
                   config: ModelConfig,
                   target_path: Optional[Path],
                   epoch: int = 0,
                   model_dir: Path = None,
                   epoch_stats: dict = None,
                   metrics: List[Metric] = None,
                   max_epoch_tuples: int = None,
                   custom_batch_to: Callable = batch_to,
                   log_all_queries: bool = False,
                   run: Optional[Run] = None):
    model.eval()

    with torch.autograd.no_grad():
        val_loss = torch.Tensor([0])
        labels = []
        preds = []
        probs = []
        sample_idxs = []

        # evaluate test set using model
        test_start_t = time.perf_counter()
        val_num_tuples = 0
        for batch_idx, batch in enumerate(tqdm(val_loader)):
            if max_epoch_tuples is not None and batch_idx * val_loader.batch_size > max_epoch_tuples:
                break

            val_num_tuples += val_loader.batch_size

            input_model, label, sample_idxs_batch = custom_batch_to(batch, model.device, model.label_norm)
            sample_idxs += sample_idxs_batch
            output = model(input_model)

            # sum up mean batch losses
            val_loss += model.loss_fxn(output, label).cpu()

            # inverse transform the predictions and labels
            curr_pred = output.cpu().numpy()
            curr_label = label.cpu().numpy()
            if model.label_norm is not None:
                curr_pred = model.label_norm.inverse_transform(curr_pred)
                curr_label = model.label_norm.inverse_transform(curr_label.reshape(-1, 1))
                curr_label = curr_label.reshape(-1)

            preds.append(curr_pred.reshape(-1))
            labels.append(curr_label.reshape(-1))

        if epoch_stats is not None:
            epoch_stats.update(val_time=time.perf_counter() - test_start_t)
            epoch_stats.update(val_num_tuples=val_num_tuples)
            val_loss = (val_loss.cpu() / len(val_loader)).item()
            print(f'val_loss epoch {epoch}: {val_loss}')
            epoch_stats.update(val_loss=val_loss)

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        epoch_stats.update(val_std=np.std(labels))

        # save best model for every metric
        any_best_metric = False
        if metrics is not None:
            for metric in metrics:
                best_seen = metric.evaluate(metrics_dict=epoch_stats,
                                            model=model,
                                            labels=labels,
                                            preds=preds,
                                            probs=probs)
                if best_seen and metric.early_stopping_metric:
                    any_best_metric = True
                    print(f"New best model for {metric.metric_name}")

        if log_all_queries:
            rows = to_rows(indexes=sample_idxs,
                           labels=[float(f) for f in labels],
                           predictions=[float(f) for f in preds])

            save_csv(rows, target_csv_path=str(target_path) + "_test_pred.csv")

            if run:
                wandb.log({f"{config.name.NAME}/{model_dir.stem}/{config.seed}/test_scores":
                    wandb.Table(columns=["metric", "value"], data=list(epoch_stats.items()))})
                wandb.log({f"{config.name.NAME}/{model_dir.stem}/{config.seed}/test_preds":
                   wandb.Table(dataframe=pd.DataFrame(rows))})
                #wandb.log(epoch_stats)
    return any_best_metric


def to_rows(indexes, labels, predictions):
    """ Write down test-set predictions to file and to wandb"""
    rows = []
    for (index, label, prediction) in zip(indexes, labels, predictions):
        entry = dict(query_id=index, query_index=index, label=label, prediction=prediction)
        entry.update(qerror=QError().evaluate_metric(label, prediction))
        rows.append(entry)
    return rows


def optuna_intermediate_value(metrics):
    for m in metrics:
        if m.early_stopping_metric:
            assert isinstance(m, QError)
            return m.best_seen_value
    raise ValueError('Metric invalid')

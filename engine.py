import sys
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.optim as optim

import util.misc as utils

def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    experiment=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt="{value:6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = {k: v.to(device) for k, v in batch.items()}

        free_energies = model._compute_objective(batch)
        total_loss = free_energies["total"]

        if not math.isfinite(total_loss):
            print(f"Loss is {total_loss.item()}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        total_loss.backward()
        if max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=total_loss, **free_energies)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if experiment is not None:
        #     experiment.log_metrics(
        #         {f"train_{k}": v.item() for k, v in free_energies.items()})

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if experiment is not None:
        experiment.log_metrics({f"train_avg_loss": avg_stats["total"]}, step=epoch)
    return avg_stats


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, experiment=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    print_freq = 10

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        batch = {k: v.to(device) for k, v in batch.items()}

        free_energies = model._compute_objective(batch)
        total_loss = free_energies["total"]

        metric_logger.update(loss=total_loss, **free_energies)
        # if experiment is not None:
        #     experiment.log_metrics(
        #         {f"val_{k}": v.item() for k, v in free_energies.items()})

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    avg_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if experiment is not None:
        experiment.log_metrics({f"val_avg_loss": avg_stats["total"]}, step=epoch)
    return avg_stats


def encoder_evaluate(
    linear_predictor,
    linear_optimizer,
    train_linpred_datloader,
    val_linpred_dataloader,
    linpred_epochs,
    device,
    epoch,
    experiment=None,
):
    linear_predictor.train()
    for _ in range(linpred_epochs):
        for batch in train_linpred_datloader:
            imgs = batch["target_frame"].to(device)
            lbls = batch["labels"].flatten().to(device)
            pred = linear_predictor(imgs)

            loss = nn.CrossEntropyLoss()(pred, lbls)
            linear_optimizer.zero_grad()
            loss.backward()
            linear_optimizer.step()

    linear_predictor.eval()
    correct = 0
    with torch.no_grad():
        for batch in val_linpred_dataloader:
            imgs = batch["target_frame"].to(device)
            lbls = batch["labels"].flatten().to(device)
            pred = linear_predictor(imgs)

            correct += (torch.argmax(pred, dim=1) == lbls).sum().item()

    acc = correct / len(val_linpred_dataloader.dataset)
    print(f"Linear acc: {acc:.4f}")
    if experiment is not None:
        experiment.log_metrics({f"eval_acc": acc}, step=epoch)
    return acc

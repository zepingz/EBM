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
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt="{value:6f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = 1 # 10

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

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    for batch in metric_logger.log_every(data_loader, 1, header):
        batch = {k: v.to(device) for k, v in batch.items()}

        free_energies = model._compute_objective(batch)
        total_loss = free_energies["total"]

        metric_logger.update(loss=total_loss, **free_energies)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

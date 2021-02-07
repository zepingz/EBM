import os
import copy
import random
import argparse
import numpy as np
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import util.misc as utils
from engine import evaluate, train_one_epoch, encoder_evaluate
from data import build_dataset, add_data_specific_args
from models import build_model, add_model_specific_args
from models.layers import LinearPredictor


def get_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--clip_max_norm", default=0., type=float)
    parser.add_argument("--batch_repeat_step", default=3, type=int)

    # Encoder evaluation
    parser.add_argument("--no_linpred_eval", action="store_true")
    parser.add_argument("--linpred_epochs", default=1, type=int)
    parser.add_argument("--linpred_batch_size", default=512, type=int)
    parser.add_argument("--linpred_lr", default=0.1, type=float)
    parser.add_argument("--linpred_interval", default=5, type=int)

    # General
    parser.add_argument("--counter", default=None, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)

    parser = add_data_specific_args(parser)
    parser = add_model_specific_args(parser)

    args = parser.parse_args()
    return args


def main(args):
    print(args)

    device = args.device
    if args.output_dir:
        save_dir = os.path.join(args.output_dir, f"ebm_{args.counter}")
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    # Build dataloader
    train_dataset, val_dataset, train_linpred_dataset, val_linpred_dataset = build_dataset(args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # drop_last=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # drop_last=True,
        pin_memory=True,
    )
    args.ptp_size = train_dataset._ptp_size

    train_linpred_dataloader = DataLoader(
        train_linpred_dataset,
        batch_size=args.linpred_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        # drop_last=True,
        pin_memory=True,
    )
    val_linpred_dataloader = DataLoader(
        val_linpred_dataset,
        batch_size=args.linpred_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # drop_last=True,
        pin_memory=True,
    )

    # Fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    # if args.distributed:
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    encoder_n_parameters = sum(
        p.numel() for p in model_without_ddp.frame_encoder.parameters())
    decoder_n_parameters = sum(
        p.numel() for p in model_without_ddp.frame_decoder.parameters())
    predictor_n_parameters = sum(
        p.numel() for p in model_without_ddp.hidden_predictor.parameters())
    print((f"Number of params\n"
           f"encoder: {encoder_n_parameters}\n"
           f"decoder: {decoder_n_parameters}\n"
           f"predictor: {predictor_n_parameters}"
    ))

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup comet ml
    api_key = os.environ.get("COMET_API_KEY")
    project_name = os.environ.get("COMET_PROJECT_NAME")
    workspace = os.environ.get("COMET_WORKSPACE")
    do_log = (
        api_key is not None
        and project_name is not None
        and workspace is not None
    )
    if False:# do_log:
        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )
        experiment.set_name(f"ebm_{args.counter}")
    else:
        experiment = None


    print("Start training")
    for epoch in range(args.epochs):
        # Train
        train_stats = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            1 if args.no_latent else args.batch_repeat_step,
            device,
            epoch,
            args.clip_max_norm,
            experiment,
        )
        # lr_scheduler.step()

        # Save model
        if save_dir:
            checkpoint_path = os.path.join(save_dir, f"{epoch}epoch.pth")
            utils.save_on_master({
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }, checkpoint_path)

        # Val
        val_stats = evaluate(
            model, val_dataloader, device, epoch, experiment)

        # Encoder val
        if not args.no_linpred_eval and\
        (epoch % args.linpred_interval == 0 or epoch == args.epochs):
            linear_predictor = LinearPredictor(
                args.embedding_size,
                10 if args.dataset == "moving_mnist" else 8,
                copy.deepcopy(model.frame_encoder),
            ).to(device)
            linear_optimizer = optim.Adam(
                linear_predictor.parameters(), lr=args.linpred_lr)
            linpred_acc = encoder_evaluate(
                linear_predictor,
                linear_optimizer,
                train_linpred_dataloader,
                val_linpred_dataloader,
                args.linpred_epochs,
                device,
                epoch,
                experiment,
            )


if __name__ == "__main__":
    args = get_args()
    main(args)

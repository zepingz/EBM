import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import util.misc as utils
from engine import evaluate, train_one_epoch
from data import build_dataset, add_data_specific_args
from models import build_model, add_model_specific_args


def get_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--clip_max_norm", default=0., type=float)

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
    train_dataset, val_dataset = build_dataset(args)
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

    print("Start training")
    for epoch in range(args.epochs):
        # Train
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, device, epoch, args.clip_max_norm)
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
            model, val_dataloader, device, save_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)

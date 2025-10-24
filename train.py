from enum import Enum
import torch
import logging
import tqdm
import json
import os
import numpy as np
from dataset import RadarDataset
from model import UNet
from loss import cross_entropy_loss, FocalLoss
from torch.utils.data import DataLoader
from config import N_LABELS
import torch.nn as nn

from runner import (
    OptimizerChoice,
    train_step,
    validate_epoch,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)
DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def create_optimizer(model: torch.nn.Module, choice: OptimizerChoice, lr: float):
    if choice is OptimizerChoice.ADAMW:
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif choice is OptimizerChoice.SGD:
        # momentum value from UNET paper
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4
        )
    else:
        raise ValueError(f"Unknown optimizer: {choice}")


if __name__ == "__main__":
    # hyper-parameters
    experiment_name = "test"
    resume_training = False
    initial_epoch = 0
    SEED = 0
    n_epochs = 20
    lr = 1e-4
    batch_size = 4
    save_each = 25
    optimizer_choice = OptimizerChoice.ADAMW
    # criterion = cross_entropy_loss
    criterion = FocalLoss(task_type="multi-class", num_classes=N_LABELS, gamma=2.0)
    # wbce = torch.tensor([0.8], device=DEVICE) # weight of the BCE loss
    chs = [8, 16, 32]
    augment_data = False
    # -------------------------

    # initializing experiment configuration
    config = {
        "exp_name": experiment_name,
        "optimizer_choice": optimizer_choice.value,
        "augmentation": augment_data,
        "criterion": repr(criterion),
        "unet_chs": chs,
    }

    logging.basicConfig(
        filename=f"checkpoints/{experiment_name}.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    train_data_folder = "data/train/input/"
    train_gt_folder = "data/train/gt/"
    train_indices = np.arange(0, 512)
    val_indices = np.arange(512, 640)

    train_data = RadarDataset(train_data_folder, train_gt_folder)
    train_data.subset_on_indices(train_indices)
    val_data = RadarDataset(train_data_folder, train_gt_folder)
    val_data.subset_on_indices(val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = UNet(chs=chs, n_classes=N_LABELS).to(DEVICE)

    optimizer = create_optimizer(model, optimizer_choice, lr)

    for epoch in range(n_epochs):
        global_epoch = initial_epoch + epoch + 1
        print(f"Training local epoch {epoch + 1}/{n_epochs}")

        model.train()
        epoch_tr_loss = 0.0
        for batch in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
            data, label = batch
            data, label = data.to(DEVICE), label.to(DEVICE)
            loss = train_step(model, optimizer, criterion, data, label)
            epoch_tr_loss += loss

        epoch_tr_loss /= len(train_loader)

        epoch_val_loss = validate_epoch(model, val_loader, criterion, DEVICE)

        logging.info(
            f"Global epoch: {global_epoch} -> Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}"
        )
        print(
            f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}"
        )
        print("-----------------------------------------------------------------------")

    save_checkpoint(
        model,
        optimizer,
        global_epoch,
        f"checkpoints/{experiment_name}_ep{global_epoch}.pth",
    )

import torch
import logging
import tqdm
import json
import numpy as np
from common import set_seed
from dataset import RadarDataset
from model import UNet
from loss import WeightedCrossEntropyLoss, FocalLoss
from torch.utils.data import DataLoader
from config import DONT_CARE, N_LABELS, label_to_index

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
    experiment_name = "deep2_unet_b4_DoppFilt_WCE01_LN"
    resume_training = False
    initial_epoch = 0
    SEED = 0
    n_epochs = 5
    lr = 3e-4
    batch_size = 4
    save_each = 25
    optimizer_choice = OptimizerChoice.ADAMW
    criterion = WeightedCrossEntropyLoss(
        weight=torch.tensor([1, 1, 1, 1, 1.0, 1], device=DEVICE),
        ignore_index=label_to_index[DONT_CARE],
    )
    scheduler_fn = torch.optim.lr_scheduler.OneCycleLR
    # scheduler_fn = None
    # chs = [16, 32, 64]
    chs = [32, 64, 128, 256]
    augment_data = False
    steps_for_validation = 1000
    # -------------------------

    set_seed(SEED)
    # initializing experiment configuration
    config = {
        "exp_name": experiment_name,
        "optimizer_choice": optimizer_choice.value,
        "augmentation": augment_data,
        "criterion": repr(criterion),
        "scheduler": scheduler_fn.__class__.__name__ if scheduler_fn else None,
        "unet_chs": chs,
    }

    with open(f"checkpoints/{experiment_name}_config.json", "w") as f:
        json.dump(config, f, indent=4)

    logging.basicConfig(
        filename=f"checkpoints/{experiment_name}.log",
        encoding="utf-8",
        level=logging.DEBUG,
    )

    train_data = RadarDataset("data/train/input/", "data/train/gt/")
    val_data = RadarDataset("data/validation/input/", "data/validation/gt/")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"n_train samples: {len(train_data)} | n_val samples: {len(val_data)}")

    model = UNet(chs=chs, n_classes=N_LABELS).to(DEVICE)

    optimizer = create_optimizer(model, optimizer_choice, lr)

    if scheduler_fn:
        scheduler = scheduler_fn(
            optimizer, max_lr=1e-3, total_steps=n_epochs * len(train_loader)
        )

    lower_val_loss = 100
    steps = 0
    epoch_tr_loss = 0.0
    for epoch in range(n_epochs):
        global_epoch = initial_epoch + epoch + 1
        print(f"Training local epoch {epoch + 1}/{n_epochs}")

        model.train()
        with tqdm.tqdm(
            total=len(train_loader), desc=f"Training epoch {epoch + 1}"
        ) as pbar:
            for batch in train_loader:
                data, label, _ = batch
                data, label = data.to(DEVICE), label.to(DEVICE)
                loss = train_step(model, optimizer, criterion, data, label)
                epoch_tr_loss += loss
                steps += 1
                if scheduler_fn:
                    scheduler.step()
                pbar.update()

                if steps % steps_for_validation == 0:
                    epoch_tr_loss /= steps_for_validation

                    epoch_val_loss = validate_epoch(
                        model, val_loader, criterion, DEVICE
                    )
                    model.train()

                    logging.info(
                        f"Global epoch: {global_epoch} -> Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}"
                    )
                    print(
                        f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )
                    print(
                        "-----------------------------------------------------------------------"
                    )
                    epoch_tr_loss = 0.0
                    # pbar.set_postfix(loss=f"{epoch_val_loss:.3f}", refresh=True)
                    pbar.update()

                    if (global_epoch % save_each == 0) | (
                        epoch_val_loss < lower_val_loss
                    ):
                        lower_val_loss = epoch_val_loss
                        save_checkpoint(
                            model,
                            optimizer,
                            epoch,
                            f"checkpoints/{experiment_name}_ep{global_epoch}.pth",
                        )

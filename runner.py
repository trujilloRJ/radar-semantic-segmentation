from enum import Enum
import torch


class OptimizerChoice(Enum):
    ADAMW = "adamw"
    SGD = "sgd"


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)


def train_step(model, optimizer, loss_fn, img, label, wbce=None):
    optimizer.zero_grad(set_to_none=True)
    logits = model(img)
    loss = loss_fn(logits, label.squeeze(1).long())
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_epoch(model, val_loader, loss_fn, device, wbce=None):
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for img, label, _ in val_loader:
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label.squeeze(1).long())
            epoch_val_loss += loss.item()
    return epoch_val_loss / len(val_loader)

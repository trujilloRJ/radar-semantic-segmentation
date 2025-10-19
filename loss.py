import torch
import torch.nn.functional as F
import torch.nn as nn


def dice_loss(pred_bhw, target_bhw, eps=0.001, **kwargs):
    pred_bhw = torch.sigmoid(pred_bhw)
    sum_dim = (-1, -2)  # sum over H, W
    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim)
    dice = (2.0 * intersection + eps) / (
        pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps
    )
    return 1.0 - dice.mean()


def cross_entropy_loss(logits_bhw, label_bhw):
    criterion = nn.CrossEntropyLoss()
    return criterion(logits_bhw, label_bhw.long())


def jaccard_loss(pred_bhw, target_bhw, eps=0.001):
    pred_bhw = torch.sigmoid(pred_bhw)
    sum_dim = (-1, -2)  # sum over H, W
    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim)
    dice = (intersection + eps) / (
        pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps - intersection
    )
    return 1.0 - dice.mean()


def loss_bce_dice(logits_bhw, label_bhw, wbce, alpha=0.5):
    label_bhw = label_bhw.float()
    loss_bce = F.binary_cross_entropy_with_logits(logits_bhw, label_bhw, weight=wbce)
    loss_dice = dice_loss(logits_bhw, label_bhw)
    return loss_bce + loss_dice, loss_dice

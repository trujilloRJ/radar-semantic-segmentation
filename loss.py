import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2,
        alpha=None,
        reduction="sum",
        task_type="binary",
        num_classes=None,
    ):
        # adapted from here: https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if (
            task_type == "multi-class"
            and alpha is not None
            and isinstance(alpha, (list, torch.Tensor))
        ):
            assert num_classes is not None, (
                "num_classes must be specified for multi-class classification"
            )
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha}, reduction='{self.reduction}', task_type='{self.task_type}')"

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == "binary":
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == "multi-class":
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == "multi-label":
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'."
            )

    def binary_focal_loss(self, inputs, targets):
        """Focal loss for binary classification."""
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """Focal loss for multi-class classification."""
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_one_hot = targets_one_hot.permute(
            0, 3, 1, 2
        )  # shape: (batch_size, num_classes)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        weights = torch.tensor([1, 1, 1, 1, 1, 1], device=inputs.device).view(
            1, -1, 1, 1
        )
        p_t = torch.sum(weights * probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """Focal loss for multi-label classification."""
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction="mean", ignore_index=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight, reduction=reduction, ignore_index=ignore_index
        )

    def __repr__(self):
        return f"WeightedCrossEntropyLoss(weight={self.weight}, reduction='{self.reduction}')"

    def forward(self, logits_bhw, label_bhw):
        return self.ce_loss(logits_bhw, label_bhw.long())


def dice_loss(pred_bhw, target_bhw, eps=0.001, **kwargs):
    pred_bhw = torch.sigmoid(pred_bhw)
    sum_dim = (-1, -2)  # sum over H, W
    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim)
    dice = (2.0 * intersection + eps) / (
        pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps
    )
    return 1.0 - dice.mean()


# def cross_entropy_loss(logits_bhw, label_bhw):
#     criterion = nn.CrossEntropyLoss(ignore_index=5, reduction="mean")
#     return criterion(logits_bhw, label_bhw.long())


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

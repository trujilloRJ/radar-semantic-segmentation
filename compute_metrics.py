import torch
import tqdm
import os
from dotenv import load_dotenv
from config import label_to_index, DONT_CARE, N_LABELS, final_labels, label_to_str
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

from dataset import OutGridDataset

load_dotenv()

RESULTS_FOLDER = "results/"


def plot_confusion_matrix(
    ax,
    confusion_matrix: torch.Tensor,
    plot_ylabels: bool = True,
    is_pct: bool = False,
):
    if is_pct:
        text_suffix = "%"
        vmax = 100
    else:
        text_suffix = ""
        vmax = 15000

    ax.imshow(confusion_matrix.numpy(), cmap="YlGn", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(N_LABELS - 1))
    ax.set_yticks(np.arange(N_LABELS - 1))
    ax.set_xticklabels(
        [label_to_str[label] for label in final_labels[:-1]],
        rotation=20,
        ha="right",
    )
    if plot_ylabels:
        ax.set_yticklabels(
            [label_to_str[label] for label in final_labels[:-1]],
            rotation=0,
            ha="right",
        )
    else:
        ax.set_yticklabels([])
    for i in range(N_LABELS - 1):
        for j in range(N_LABELS - 1):
            color = "white" if confusion_matrix[i, j] > vmax / 2 else "black"
            if confusion_matrix.dtype == torch.int64:
                # draw as white text if value is above threshold
                ax.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j].item():d}{text_suffix}",
                    ha="center",
                    va="center",
                    color=color,
                )
            else:
                ax.text(
                    j,
                    i,
                    f"{confusion_matrix[i, j].item():.1f}{text_suffix}",
                    ha="center",
                    va="center",
                    color=color,
                )


def plot_results(confusion_matrix: torch.Tensor, experiment: str):
    confusion_matrix = confusion_matrix[:-1, :-1]
    # represent confusion matrix in percentages
    row_sums = confusion_matrix.float().sum(dim=1, keepdim=True)
    pct_confmat = confusion_matrix / row_sums * 100

    # visualize confusion matrix in matplotlib
    fig, axes = plt.subplots(figsize=(8, 4), ncols=2)
    plot_confusion_matrix(axes[0], confusion_matrix)
    plot_confusion_matrix(axes[1], pct_confmat, plot_ylabels=False, is_pct=True)
    plt.suptitle(f"Experiment: {experiment}")
    plt.tight_layout()
    fig.savefig(f"results/{experiment}.png", dpi=300)


if __name__ == "__main__":
    exp_name = "deep3_unet_b4_DoppFilt_WCE01_LN_ep3"
    results_path = os.path.join(RESULTS_FOLDER, exp_name)
    gt_path = "data/validation/gt"

    pred_dataset = OutGridDataset(results_path)
    gt_dataset = OutGridDataset(gt_path)

    confmat = ConfusionMatrix(
        task="multiclass", num_classes=N_LABELS, ignore_index=label_to_index[DONT_CARE]
    )

    n_examples = len(pred_dataset)

    confusion_matrix = torch.zeros((N_LABELS, N_LABELS), dtype=torch.int64)
    for batch_idx in tqdm.tqdm(range(n_examples)):
        pred_fn = pred_dataset.file_list[batch_idx]
        gt_fn = gt_dataset.file_list[batch_idx]
        assert pred_fn.removesuffix("_PRED.pth") == gt_fn.removesuffix("_fl.pt"), (
            "Prediction and GT filenames do not match!"
        )

        pred_tensor = pred_dataset[batch_idx]
        gt_tensor = gt_dataset[batch_idx]

        predicted_classes = pred_tensor.unsqueeze(0)

        confusion_matrix += confmat(predicted_classes, gt_tensor.unsqueeze(0))

    # torch.save(confusion_matrix, f"results/confusion_matrix_{experiment}.pth")

    plot_results(confusion_matrix, experiment=exp_name)

import torch
import tqdm
import json
import os
from dataset import RadarDataset
from model import UNet
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from config import label_to_index, DONT_CARE, N_LABELS, final_labels, label_to_str
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()

EXP_FOLDER = "checkpoints/"


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
    evaluation_folder = "data/train"
    exp_name = "baseline_unet_noWCE"
    epoch = 12

    # load config
    with open(os.path.join(EXP_FOLDER, f"{exp_name}_config.json")) as f:
        config = json.load(f)

    # load model
    model = UNet(chs=config["unet_chs"], n_classes=N_LABELS)
    checkpoint = torch.load(f"checkpoints/{exp_name}_ep{epoch}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_data = RadarDataset(f"{evaluation_folder}/input", f"{evaluation_folder}/gt")
    confmat = ConfusionMatrix(
        task="multiclass", num_classes=N_LABELS, ignore_index=label_to_index[DONT_CARE]
    )

    eval_loader = DataLoader(eval_data, batch_size=8, shuffle=False)
    n_examples = len(eval_data)

    confusion_matrix = torch.zeros((N_LABELS, N_LABELS), dtype=int)
    for batch in tqdm.tqdm(
        eval_loader,
        desc=f"Evaluating on {n_examples} examples",
        total=n_examples // eval_loader.batch_size,
    ):
        input_tensor, gt_tensor = batch

        with torch.no_grad():
            output = model(input_tensor)
            predicted_classes = torch.argmax(output, dim=1)

        confusion_matrix += confmat(predicted_classes, gt_tensor.squeeze(1))

    experiment = f"{exp_name}_ep{epoch}"
    torch.save(confusion_matrix, f"results/confusion_matrix_{experiment}.pth")

    plot_results(confusion_matrix, experiment=experiment)

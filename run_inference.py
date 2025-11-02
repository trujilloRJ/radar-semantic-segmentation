import torch
import tqdm
import json
import os
from config import N_LABELS
from dataset import RadarDataset
from model import UNet

EXP_FOLDER = "checkpoints/"


if __name__ == "__main__":
    evaluation_folder = "data/validation"
    sequence_id = "sequence_1"
    results_folder = "results"
    model_name = "deep2_unet_b4_DoppFilt_WCE01_LN"
    epoch = 4
    exp_name = f"{model_name}_ep{epoch}"

    results_path = os.path.join(results_folder, exp_name)
    if os.path.exists(results_path) is False:
        os.makedirs(results_path)

    # load config and model
    with open(os.path.join(EXP_FOLDER, f"{model_name}_config.json")) as f:
        config = json.load(f)
    model = UNet(chs=config["unet_chs"], n_classes=N_LABELS)
    checkpoint = torch.load(f"checkpoints/{exp_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # load data
    eval_data = RadarDataset(f"{evaluation_folder}/input", f"{evaluation_folder}/gt")
    # eval_data.subset_on_sequence_id(sequence_id)
    n_frames = len(eval_data)

    for i, batch in enumerate(
        tqdm.tqdm(
            eval_data,
            desc=f"Inference on {n_frames} frames",
        )
    ):
        input_tensor, gt_tensor, ts = batch
        fn, _ = eval_data.data_gt_list[i]
        sequence_name = fn.split("_")[0]

        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))
            predicted_classes = torch.argmax(output, dim=1)

        torch.save(
            predicted_classes,
            os.path.join(results_path, f"{sequence_name}_ts{ts}_PRED.pth"),
        )

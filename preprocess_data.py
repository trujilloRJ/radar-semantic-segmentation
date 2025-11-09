import numpy as np
import h5py
import json
import os
import pandas as pd
from encoder import Grid, get_grid_encoder
from config import SENSOR_FL, labels_map, final_labels
import torch
import tqdm
from common import get_scene
from dotenv import load_dotenv

load_dotenv()

# params
train_seq = [
    8,
    10,
    77,
    91,
    98,
    100,
    102,
    105,
    106,
    107,
    131,
    132,
    134,
    137,
    139,
    141,
    142,
    143,
    144,
    146,
    147,
    148,
]
val_seq = [2, 9, 101, 108]
# ---------------
overwrite_existing = True
# sequences = [f"sequence_{i}" for i in train_seq]
# path = "data/train"
sequences = [f"sequence_{i}" for i in val_seq]
path = "data/validation"
# ----------------------------

if __name__ == "__main__":
    # scene loader
    for sequence_id in sequences:
        print(f"Processing sequence: {sequence_id}...")

        scene_fn = os.path.join(os.getenv("DATA_LOCATION"), sequence_id, "scenes.json")
        detections = get_scene(scene_fn)
        timestamps = np.unique(detections["timestamp"])

        if overwrite_existing is False:
            if os.path.exists(
                os.path.join(path, "input", f"{sequence_id}_ts{timestamps[0]}_fl.pt")
            ):
                print(f"Skipping {sequence_id} as it already exists.")
                continue

        # saving data per cycle
        for ts in tqdm.tqdm(timestamps):
            cur_dets = detections[detections["timestamp"] == ts]

            # creating input and output grids
            grid_fl = get_grid_encoder()
            grid_fl.fill_grid(cur_dets)
            grid_fl.fill_grid(cur_dets, is_output=True)

            filename = f"{sequence_id}_ts{ts}_fl.pt"

            input_data = torch.from_numpy(grid_fl.grid).permute(2, 0, 1)
            out_data = torch.from_numpy(grid_fl.out_grid).permute(2, 0, 1)

            if torch.any(torch.isnan(input_data)) or torch.any(torch.isnan(out_data)):
                print(f"NaN values found in input or output data for {filename}")
                continue

            torch.save(input_data, os.path.join(path, "input", filename))
            torch.save(out_data, os.path.join(path, "gt", filename))

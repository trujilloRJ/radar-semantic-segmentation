import numpy as np
import h5py
import json
import os
import pandas as pd
from encoder import Grid
from config import SENSOR_FL, labels_map, final_labels
import torch
import tqdm
from common import get_scene
from dotenv import load_dotenv

load_dotenv()

# params
# path = "data/train"
# sequences = ["sequence_1", "sequence_2", "sequence_3", "sequence_4", "sequence_5"]
# path = "data/validation"
# sequences = ["sequence_6", "sequence_7"]
path = "data/test"
sequences = ["sequence_8", "sequence_9"]
# ----------------------------

if __name__ == "__main__":
    # scene loader
    for sequence_id in sequences:
        print(f"Processing sequence: {sequence_id}...")
        scene_fn = os.path.join(os.getenv("DATA_LOCATION"), sequence_id, "scenes.json")

        detections = get_scene(scene_fn)

        # saving data per cycle
        timestamps = np.unique(detections["timestamp"])
        for ts in tqdm.tqdm(timestamps):
            cur_dets = detections[detections["timestamp"] == ts]

            # creating input and output grids
            grid_fl = Grid(x_lims=(2, 100), y_lims=(-50, 20), cell_size=0.5)
            grid_fl.fill_grid(cur_dets)
            grid_fl.fill_grid(cur_dets, is_output=True)

            filename = f"{sequence_id}_ts{ts}_fl.pt"

            input_data = torch.from_numpy(grid_fl.grid).permute(2, 0, 1)
            out_data = torch.from_numpy(grid_fl.out_grid).permute(2, 0, 1)

            torch.save(input_data, os.path.join(path, "input", filename))
            torch.save(out_data, os.path.join(path, "gt", filename))

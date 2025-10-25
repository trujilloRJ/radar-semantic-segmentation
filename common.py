import os
import h5py
import pandas as pd
import torch
import numpy as np
import random

from config import SENSOR_FL, labels_map, final_labels


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_scene(scene_fn):
    h5_filename = os.path.join(os.path.dirname(scene_fn), "radar_data.h5")

    with h5py.File(h5_filename, "r") as f:
        radar_data = f["radar_data"][:]
        # odometry_data = f["odometry"][:]

    # filter detections by sensor
    detections = radar_data[radar_data["sensor_id"] == SENSOR_FL]
    detections = detections[
        ["timestamp", "x_cc", "y_cc", "vr_compensated", "rcs", "label_id"]
    ]
    detections = pd.DataFrame(detections)

    # map labels and filtering labels
    for k, v in labels_map.items():
        detections.loc[detections["label_id"] == k, "label_id"] = v
    detections = detections[detections["label_id"].isin(final_labels)]

    return detections

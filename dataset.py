import os
import torch
from torch.utils.data import Dataset
from os import listdir


class OutGridDataset(Dataset):
    def __init__(self, data_folder: str, seq_filter: str = None):
        self.data_folder = data_folder

        self.file_list = (
            [f for f in listdir(data_folder) if seq_filter in f]
            if seq_filter
            else listdir(data_folder)
        )
        self.compute_ts_and_sequences()

    def compute_ts_and_sequences(self):
        self.timestamps = [fn.split("_")[2].replace("ts", "") for fn in self.file_list]
        self.sequences = [fn.rsplit("_", 2)[0] for fn in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_fn = self.file_list[idx]
        data = torch.load(os.path.join(self.data_folder, data_fn))
        return data

    def get_data_by_timestamp(self, timestamp: str):
        return self.__getitem__(self.timestamps.index(timestamp))

    def subset_on_sequence_id(self, sequence_id: str):
        valid_names = [data_fn for data_fn in self.file_list if sequence_id in data_fn]
        if valid_names:
            self.file_list = valid_names
            self.compute_ts_and_sequences()
        else:
            raise ValueError(f"No data found for sequence ID: {sequence_id}")


class RadarDataset(Dataset):
    def __init__(self, data_folder: str, gt_folder: str, augment=False, seed=0):
        self.data_folder = data_folder
        self.gt_folder = gt_folder

        img_name_lists = listdir(data_folder)
        gt_name_lists = listdir(gt_folder)

        self.data_gt_list = []
        for img, gt in zip(img_name_lists, gt_name_lists):
            assert img == gt, f"Image and GT file names do not match: {img} != {gt}"
            self.data_gt_list.append((img, gt))

    def __len__(self):
        return len(self.data_gt_list)

    def __getitem__(self, idx):
        data_fn, gt_fn = self.data_gt_list[idx]
        data = torch.load(os.path.join(self.data_folder, data_fn))
        gt = torch.load(os.path.join(self.gt_folder, gt_fn))

        timestamp = data_fn.split("_")[2].replace("ts", "")

        return data.float(), gt.float(), timestamp

    def subset_on_sequence_id(self, sequence_id: str):
        valid_names = [
            (data_fn, gt_fn)
            for data_fn, gt_fn in self.data_gt_list
            if sequence_id in data_fn
        ]
        if valid_names:
            self.data_gt_list = valid_names
        else:
            raise ValueError(f"No data found for sequence ID: {sequence_id}")

    def subset_on_indices(self, indices: list):
        self.data_gt_list = [self.data_gt_list[i] for i in indices]

import os
import torch
from torch.utils.data import Dataset
from os import listdir


class RadarDataset(Dataset):
    def __init__(self, data_folder: str, gt_folder: str, augment=False, seed=0):
        self.data_folder = data_folder
        self.gt_folder = gt_folder

        img_name_lists = listdir(data_folder)
        gt_name_lists = listdir(gt_folder)
        self.data_gt_list = [
            (img, gt) for img, gt in zip(img_name_lists, gt_name_lists)
        ]

    def __len__(self):
        return len(self.data_gt_list)

    def __getitem__(self, idx):
        data_fn, gt_fn = self.data_gt_list[idx]
        data = torch.load(os.path.join(self.data_folder, data_fn))
        gt = torch.load(os.path.join(self.gt_folder, gt_fn))

        return data.float(), gt.float()

    def subset_on_indices(self, indices: list):
        self.data_gt_list = [self.data_gt_list[i] for i in indices]

from torch.utils.data import Dataset
from os import listdir


# class RadarDataset(Dataset):
#     def __init__(self, data_folder: str, augment=False, seed=0):
#         self.data_folder = data_folder

#         img_name_lists = listdir(data_folder)
#         gt_name_lists = listdir(data_folder)
#         self.img_gt_list = [(img, gt) for img, gt in zip(img_name_lists, gt_name_lists)]

#     def __len__(self):
#         return len(self.img_gt_list)

#     def __getitem__(self, idx):
#         img_fn, gt_fn = self.img_gt_list[idx]
#         img = cv2.imread(f"{self.img_folder}/{img_fn}")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         gt = cv2.imread(f"{self.gt_folder}/{gt_fn}", cv2.IMREAD_GRAYSCALE)

#         # Pad or crop img and gt to IMG_HEIGHT, IMG_WIDTH
#         h, w = img.shape[:2]
#         if h != IMG_HEIGHT or w != IMG_WIDTH:
#             pass
#         pad_h = max(0, IMG_HEIGHT - h)
#         pad_w = max(0, IMG_WIDTH - w)
#         img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
#         img = img[:IMG_HEIGHT, :IMG_WIDTH]

#         # if augmented is False, this will be just cast as tensors
#         augmented = self.transform(image=img, mask=gt)
#         img = augmented["image"]
#         gt = augmented["mask"]

#         return img.float(), gt.float()

#     def get_curr_img_fn(self, idx):
#         img_fn, _ = self.img_gt_list[idx]
#         return img_fn

#     def subset_on_indices(self, indices: list):
#         self.img_gt_list = [self.img_gt_list[i] for i in indices]

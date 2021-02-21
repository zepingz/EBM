import os
import json
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


train_cls_dict = {
    "run_00": 0, #chocolate
    "run_01": 0,
    "run_03": 1, #hammer
    "run_04": 2, #plastic ball?
    "run_05": 3, #cylinder
    "run_06": 4, #gun
    "run_08": 3,
    "run_09": 5, #watermelon
    "run_11": 1,
    "run_12": 1,
    "run_14": 0,
    "run_16": 3,
    "run_17": 1,
    "run_18": 6, #brush
    "run_19": 3,
    "run_20": 5,
    "run_21": 0,
    "run_22": 1,
    "run_23": 2,
    "run_24": 4,
    "run_30": 6,
    "run_31": 5,
    "run_32": 2,
    "run_34": 0,
    "run_35": 1,
    "run_37": 2,
    "run_39": 7, #bottle
    "run_40": 3,
}

val_cls_dict = {
    "run_00": 0,
    "run_01": 6,
    "run_02": 2,
    "run_03": 7,
    "run_04": 4,
    "run_05": 3,
    "run_07": 5,
    "run_09": 1,
    "run_10": 6,
    "run_14": 3,
    "run_16": 1
}


class PokeDataset(Dataset):
    _ptp_size = 4

    def __init__(
        self, data_root, num_conditional_frames, transform, size=64, train=True):
        super().__init__()

        self.num_conditional_frames = num_conditional_frames

        self.resize_transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
        ])
        # self.transform = transform
        self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # Initialize indices
        self.indices = []
        self.img_dict = {}
        self.actions = {}
        folder_paths = sorted(glob(os.path.join(data_root, "train" if train else "test", "*")))
        for folder_path in folder_paths:
            episode_name = os.path.basename(folder_path)

            # Skip scenes with multiple objects
            cls_dict = train_cls_dict if train else val_cls_dict
            if episode_name not in cls_dict:
                continue

            img_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
            self.img_dict[episode_name] = img_paths
            self.indices += [{
                "episode_name": episode_name,
                "frame_idx": i,
            } for i in range(len(img_paths)-self.num_conditional_frames)]

            # Load actions
            self.actions[episode_name] = np.load(os.path.join(folder_path, "actions.npy"))

    def __getitem__(self, index):
        info = self.indices[index]
        episode_name = info["episode_name"]
        frame_idx = info["frame_idx"]

        conditional_frames = []
        for i in range(frame_idx, frame_idx+self.num_conditional_frames):
            conditional_frames.append(
                self.resize_transform(Image.open(self.img_dict[episode_name][i])))

        target_frame = self.resize_transform(
            Image.open(self.img_dict[episode_name][frame_idx+self.num_conditional_frames]))
        target_frame -= conditional_frames[-1]

        conditional_frames = [self.transform(frame) for frame in conditional_frames]
        conditional_frames = torch.stack(conditional_frames)

        action = torch.from_numpy(
            self.actions[episode_name][frame_idx+self.num_conditional_frames-1])[:4].float()

        return {
            "conditional_frames": conditional_frames,
            "PTP": action,
            "target_frame": target_frame,
        }

    def __len__(self):
        return len(self.indices)


class PokeLinpredDataset(Dataset):
    def __init__(self, data_root, transform, train=True):
        super().__init__()

        self.transform = transform
        self.img_paths = []
        self.lbls = []

        if train:
            with open(os.path.join(data_root, "poke_train_linpred_indices.json"), "r") as f:
                img_path_dict = json.load(f)
            for episode_name, path_list in img_path_dict.items():
                self.img_paths += [
                    os.path.join(data_root, "train", episode_name, path) for path in path_list]
                self.lbls += [train_cls_dict[episode_name],] * len(path_list)
        else:
            folder_paths = sorted(glob(os.path.join(data_root, "test", "*")))
            for folder_path in folder_paths:
                episode_name = os.path.basename(folder_path)

                # Skip scenes with multiple objects
                if episode_name not in val_cls_dict:
                    continue

                path_list = sorted(glob(os.path.join(folder_path, "*.jpg")))
                self.img_paths += path_list
                self.lbls += [val_cls_dict[episode_name],] * len(path_list)

        assert len(self.img_paths) == len(self.lbls)

    def __getitem__(self, index):
        img = self.transform(Image.open(self.img_paths[index]))
        lbl = self.lbls[index]

        return {
            "target_frame": img,
            "label": lbl,
        }

    def __len__(self):
        return len(self.img_paths)

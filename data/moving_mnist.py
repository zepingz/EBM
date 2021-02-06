import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

MNIST_ORIGINAL_SIZE = 28


def get_linspace(min_value, max_value, step, determinstic):
    if determinstic:
        start, end = np.random.uniform(min_value, max_value, 2)
        values = np.linspace(start, end, step)
    else:
        values = np.array(sorted(np.random.uniform(min_value, max_value, step)))
        if np.random.random() > 0.5:
            values = np.flip(values)

    return values


class MovingMNISTDataset(Dataset):
    _ptp_size = 5

    def __init__(
        self,
        data_root,
        num_conditional_frames,
        transform,
        train=True,
        height=64,
        width=64,
        num_digits=1,
        determinstic=False,
        angle_range=(-30, 30),
        scale_range=(0.8, 1.2),
        shear_range=(-20, 20),
        dataset_size=20000,
        ptp_type="2",
        linpred=False,
    ):
        self.num_conditional_frames = num_conditional_frames
        self.transform = transform
        self.train = train
        self.height = height
        self.width = width
        self.num_digits = num_digits
        self.determinstic = determinstic
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.dataset_size = dataset_size
        self.ptp_type = ptp_type
        self.linpred = linpred

        # Load mnist digits
        self.data = MNIST(
            data_root,
            train=train,
            download=True,
            transform=None,  # transforms.ToTensor(),
        )
        self.N = len(self.data)

        # Create a fixed dataset if not exists
        if self.linpred:
            split_name = "linpred"
        elif self.train:
            split_name = "train"
        else:
            split_name = "val"
        folder_name = (
            f"{split_name}_{height}x{width}x{num_conditional_frames}_"
            f"{dataset_size}_{num_digits}digit_ptp{ptp_type}"
        )
        if self.determinstic:
            folder_name += "_determinstic"

        self.dataset_folder = os.path.join(data_root, folder_name)
        if not os.path.exists(self.dataset_folder):
            print(f"Creating {self.dataset_folder}")
            os.makedirs(self.dataset_folder)
            for i in tqdm(range(self.dataset_size)):
                imgs, PTP, lbls = self.create_moving_mnist_sequence()
                np.save(os.path.join(self.dataset_folder, f"{i}_img.npy"), imgs.numpy())
                np.save(os.path.join(self.dataset_folder, f"{i}_ptp.npy"), PTP.numpy())
                np.save(os.path.join(self.dataset_folder, f"{i}_lbl.npy"), lbls.numpy())

    def create_moving_mnist_sequence(self):
        total_frames = self.num_conditional_frames + 1

        shear = get_linspace(
            self.shear_range[0],
            self.shear_range[1],
            total_frames,
            self.determinstic
        )
        angle = get_linspace(
            self.angle_range[0],
            self.angle_range[1],
            total_frames,
            self.determinstic
        )
        scale = get_linspace(
            self.scale_range[0],
            self.scale_range[1],
            total_frames,
            self.determinstic
        )

        imgs = torch.zeros(
            (total_frames, 1, self.height, self.width), dtype=torch.float32
        )
        lbls = torch.zeros(self.num_digits, dtype=torch.long)

        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, lbl = self.data[idx]

            lbls[n] = lbl

            sx = np.random.randint(self.width - MNIST_ORIGINAL_SIZE)
            sy = np.random.randint(self.height - MNIST_ORIGINAL_SIZE)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            dxs = []
            dys = []

            for t in range(total_frames):
                if sy < 0:
                    sy = 0
                    if self.determinstic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= self.height - MNIST_ORIGINAL_SIZE:
                    sy = self.width - MNIST_ORIGINAL_SIZE - 1
                    if self.determinstic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.determinstic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= self.width - MNIST_ORIGINAL_SIZE:
                    sx = self.width - MNIST_ORIGINAL_SIZE - 1
                    if self.determinstic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                # Apply affine transformation
                new_digit = transforms.ToTensor()(
                    transforms.functional.affine(
                        digit,
                        angle=angle[t],
                        translate=(0, 0),
                        scale=scale[t],
                        shear=shear[t],
                        resample=Image.BILINEAR,
                    )
                )

                imgs[
                    t,
                    0,
                    sy : sy + MNIST_ORIGINAL_SIZE,
                    sx : sx + MNIST_ORIGINAL_SIZE,
                ] += new_digit.squeeze()
                sy += dy
                sx += dx

                dxs.append(dx)
                dys.append(dy)

        imgs[imgs > 1] = 1.0

        imgs = torch.cat([self.transform(img) for img in imgs]).unsqueeze(1)

        # NOTE: the PTP for moving mnist only contains the difference between
        # x and y of the first digit
        PTP = torch.zeros(16,)

        # have to change scale ratio to log
        if self.ptp_type == "absolute":
            PTP[0] = dxs[-2]
            PTP[1] = dys[-2]
            PTP[2] = angle[-1]
            PTP[3] = scale[-1]
            PTP[4] = shear[-1]
        elif self.ptp_type == "2":
            # ptp2
            PTP[0] = dxs[-2]
            PTP[1] = dys[-2]
            PTP[2] = angle[-1] - angle[-2]
            PTP[3] = scale[-1] / scale[-2]
            PTP[4] = shear[-1] - shear[-2]
        elif self.ptp_type == "1_2":
            # ptp2
            PTP[0] = dxs[-2]
            PTP[1] = dys[-2]
            PTP[2] = angle[-1] - angle[-2]
            PTP[3] = scale[-1] / scale[-2]
            PTP[4] = shear[-1] - shear[-2]

            # ptp1
            PTP[5] = dxs[-3]
            PTP[6] = dys[-3]
            PTP[7] = angle[-2] - angle[-3]
            PTP[8] = scale[-2] / scale[-3]
            PTP[9] = shear[-2] - shear[-3]

        return imgs, PTP, lbls

    def __getitem__(self, index):
        imgs = torch.from_numpy(np.load(
            os.path.join(self.dataset_folder, f"{index}_img.npy"))).float()
        PTP = torch.from_numpy(np.load(
            os.path.join(self.dataset_folder, f"{index}_ptp.npy"))).float()
        lbls = torch.from_numpy(np.load(
            os.path.join(self.dataset_folder, f"{index}_lbl.npy"))).long()

        return {
            "conditional_frames": imgs[:-1],
            "PTP": PTP,
            "target_frame": imgs[-1],
            "labels": lbls,
        }

    def __len__(self):
        return self.dataset_size

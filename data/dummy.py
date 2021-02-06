import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    _ptp_size = 4

    def __init__(self, num_conditional_frames):
        super().__init__()
        self.num_conditional_frames = num_conditional_frames

    def __len__(self):
        return 100

    def __getitem__(self, i):
        return {
            "conditional_frames": torch.randn(
                self.num_conditional_frames, 3, 4, 4),
            "PTP": torch.randn(4),
            "target_frame": torch.randn(3, 4, 4),
        }

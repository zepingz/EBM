import torch.nn as nn

from .layers import ConvBlock


class DummyFrameDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.fc = nn.Linear(5, 48)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 3, 4, 4)
        return x


class TestFrameDecoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 2x2
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 4x4
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 8x8
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 16x16
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 32x32
            ConvBlock(64, 64, 3, 2, 1, transpose=True),  # 64x64
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.view(len(x), -1, 1, 1)
        x = self.layers(x)
        return x


decoder_dict = {
    "dummy": DummyFrameDecoder,
    "test": TestFrameDecoder,
}

def build_decoder(args):
    output_dim = 1 if args.dataset == "moving_mnist" else 3
    return decoder_dict[args.decoder_type](output_dim)

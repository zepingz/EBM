import torch.nn as nn

from .layers import ConvBlock


class DummyFrameEncoder(nn.Module):
    _embedding_size = 5
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(48, 5)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x


class TestFrameEncoder(nn.Module):
    _embedding_size = 64

    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(input_dim, 64, 7, 2, 3),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
            ConvBlock(64, 64, 5, 2, 2),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = self.layers(x)
        x = x.view(bs, -1)
        return x


encoder_dict = {
    "dummy": DummyFrameEncoder,
    "test": TestFrameEncoder,
}

def build_encoder(args):
    input_dim = 1 if args.dataset == "moving_mnist" else 3
    return encoder_dict[args.encoder_type](input_dim)

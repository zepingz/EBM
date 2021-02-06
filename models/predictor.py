import copy

import torch.nn as nn

from .layers import BaselineLayer, MLP


class DummyHiddenPredictor(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()
        self.fc = nn.Linear(num_conditional_frames * 5, 5)
        self.latent_fc = nn.Linear(latent_size, 5)
        self.ptp_fc = nn.Linear(ptp_size, 5)

    def forward(self, x, ptp, latent):
        x = x.view(len(x), -1)
        x = self.fc(x) + self.latent_fc(latent) + self.ptp_fc(ptp)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class BaselineHiddenPredictor(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        input_dim = num_conditional_frames * embedding_size
        baseline_layer = BaselineLayer(
            input_dim, transformer_hparams["dim_feedforward"])
        self.layers = _get_clones(
            baseline_layer, transformer_hparams["layers"])

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x


predictor_dict = {
    "dummy": DummyHiddenPredictor,
    "baseline": BaselineHiddenPredictor,
}

def build_predictor(args):
    transformer_hparams = {
        "layers": args.hidden_predictor_layers,
        "dim_feedforward": args.hidden_predictor_dim_feedforward,
        "nhead": args.hidden_predictor_nhead,
    }
    return predictor_dict[args.predictor_type](
        args.embedding_size,
        args.num_conditional_frames,
        args.latent_size,
        args.ptp_size,
        transformer_hparams,
    )

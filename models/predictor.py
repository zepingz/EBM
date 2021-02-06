import copy

import torch
import torch.nn as nn

from .layers import BaselineLayer, MLP
from .position_encoding import PositionEmbeddingLearned
from .transformer import TransformerEncoderLayer, TransformerEncoder


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


class BaselineHiddenPredictor_ptp(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        input_dim = (num_conditional_frames + 1) * embedding_size
        baseline_layer = BaselineLayer(
            input_dim, transformer_hparams["dim_feedforward"])
        self.layers = _get_clones(
            baseline_layer, transformer_hparams["layers"])

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

        self.ptp_fc = MLP(ptp_size, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        ptp = self.ptp_fc(ptp)
        x = torch.cat((x, ptp), dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x


class BaselineHiddenPredictor_latent(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        input_dim = (num_conditional_frames + 1) * embedding_size
        baseline_layer = BaselineLayer(
            input_dim, transformer_hparams["dim_feedforward"])
        self.layers = _get_clones(
            baseline_layer, transformer_hparams["layers"])

        self.last_mlp = MLP(
            input_dim, embedding_size, embedding_size, 3)

        self.latent_fc = MLP(latent_size, 256, embedding_size, 3)

    def forward(self, x, ptp, latent):
        bs = x.shape[0]
        x = x.view(bs, -1)

        latent = self.latent_fc(latent)
        x = torch.cat((x, latent), dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_hparams["nhead"],
            dim_feedforward=transformer_hparams["dim_feedforward"]
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_hparams["layers"])
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ptp, latent):
        bs = len(x)

        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor1_ptp(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_hparams["nhead"],
            dim_feedforward=transformer_hparams["dim_feedforward"],
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_hparams["layers"])
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.ptp_expander = MLP(
            ptp_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ptp, latent):
        bs = len(x)

        ptp = self.ptp_expander(ptp)

        x = torch.cat((x, self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1), ptp.unsqueeze(1)), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -2]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor1_latent(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_hparams["nhead"],
            dim_feedforward=transformer_hparams["dim_feedforward"],
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_hparams["layers"])
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.latent_expander = MLP(
            latent_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ptp, latent):
        bs = len(x)

        latent = self.latent_expander(latent)

        x = torch.cat((x, self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1), latent.unsqueeze(1)), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -2]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor2_ptp(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_hparams["nhead"],
            dim_feedforward=transformer_hparams["dim_feedforward"],
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_hparams["layers"])
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.ptp_expander = MLP(
            ptp_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ptp, latent):
        bs = len(x)

        ptp = self.ptp_expander(ptp)
        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1) + ptp.unsqueeze(1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


class TransformerHiddenPredictor2_latent(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_conditional_frames,
        latent_size,
        ptp_size,
        transformer_hparams,
    ):
        super().__init__()

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=transformer_hparams["nhead"],
            dim_feedforward=transformer_hparams["dim_feedforward"],
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=transformer_hparams["layers"])
        self.position_encoding = PositionEmbeddingLearned(embedding_size)
        self.query_embed = nn.Embedding(1, embedding_size)

        self.latent_expander = MLP(
            latent_size, embedding_size, embedding_size, 3)
        self.last_mlp = MLP(
            embedding_size, embedding_size, embedding_size, 3)

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, ptp, latent):
        bs = len(x)

        latent = self.latent_expander(latent)
        token = self.query_embed.weight.unsqueeze(1).repeat(bs, 1, 1) + latent.unsqueeze(1)
        x = torch.cat((x, token), dim=1)

        pos = self.position_encoding(x).permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = x + pos
        x = self.transformer_encoder(x)

        x = x.permute(1, 0, 2)[:, -1]
        x = self.last_mlp(x)

        return x


predictor_dict = {
    "dummy": DummyHiddenPredictor,
    "baseline": BaselineHiddenPredictor,
    "baseline_ptp": BaselineHiddenPredictor_ptp,
    "baseline_latent": BaselineHiddenPredictor_latent,
    "transformer": TransformerHiddenPredictor,
    "transformer1_ptp": TransformerHiddenPredictor1_ptp,
    "transformer1_latent": TransformerHiddenPredictor1_latent,
    "transformer2_ptp": TransformerHiddenPredictor2_ptp,
    "transformer2_latent": TransformerHiddenPredictor2_latent,
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

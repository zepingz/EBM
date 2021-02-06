import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class LatentMinimizationEBM(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        predictor,
        num_conditional_frames,
        latent_size,
        no_latent,
        latent_optimizer_type,
        lambda_target_prediction,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.latent_optimizer_type = latent_optimizer_type
        self.no_latent = no_latent

        self.frame_encoder = encoder
        self.frame_decoder = decoder
        self.hidden_predictor = predictor

        self.lambdas = {
            "target_prediction": lambda_target_prediction,
            # "decoding_error": lambda_decoding_error,
        }

    def forward(self, batch):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"]
        bs, seq_len, c, h, w = conditional_frames.shape
        device = conditional_frames.device

        # Encode conditional frames
        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)

        # Predict target frame
        latent = self.sample_latent(bs, device)
        predicted_hidden = self.hidden_predictor(
            encoded_frames, ptp, latent)
        predicted_frame = self.frame_decoder(predicted_hidden)

        return predicted_frame

    def sample_latent(self, bs, device):
        latent = torch.randn(bs, self.latent_size).to(device)
        latent.requires_grad = True
        return latent

    def _compute_objective(self, batch):
        conditional_frames = batch["conditional_frames"]
        ptp = batch["PTP"]
        target_frame = batch["target_frame"]
        bs, seq_len, c, h, w = conditional_frames.shape
        device = conditional_frames.device

        # Encode conditional frames
        conditional_frames = conditional_frames.view(bs * seq_len, c, h, w)
        encoded_frames = self.frame_encoder(conditional_frames)
        encoded_frames = encoded_frames.view(bs, seq_len, -1)

        # Compute optimal latent
        if self.no_latent:
            latent = 0
        else:
            latent = self._compute_optimal_latent(
                encoded_frames.detach(), ptp, target_frame)

        # Predict target frame
        predicted_hidden = self.hidden_predictor(
            encoded_frames, ptp, latent)
        predicted_frame = self.frame_decoder(predicted_hidden)

        # Compute free energies
        free_energies = self.compute_energies(predicted_frame, target_frame)
        return free_energies

    def _compute_optimal_latent(self, encoded_frames, ptp, target_frame):
        # Freeze decoder and predictor
        for param in self.frame_decoder.parameters():
            param.requires_grad = False
        for param in self.hidden_predictor.parameters():
            param.requires_grad = False

        latent = self.sample_latent(len(encoded_frames), encoded_frames.device)
        self.optimize_latent(latent, ptp, encoded_frames, target_frame)

        # Unfreeze decoder and predictor
        for param in self.frame_decoder.parameters():
            param.requires_grad = True
        for param in self.hidden_predictor.parameters():
            param.requires_grad = True

        return latent

    def optimize_latent(self, latent, ptp, encoded_frames, target_frame):
        with torch.enable_grad():
            if self.latent_optimizer_type == "GD":
                latent_optimizer = optim.SGD((latent,), lr=0.1)

                for i in range(10):
                    predicted_hidden = self.hidden_predictor(
                        encoded_frames, ptp, latent)
                    predicted_frame = self.frame_decoder(predicted_hidden)
                    energies = self.compute_energies(
                        predicted_frame,
                        target_frame,
                        latent=latent,
                    )
                    latent_optimizer.zero_grad()
                    energies["total"].backward()
                    latent_optimizer.step()

            if self.latent_optimizer_type == "LBFGS":
                latent_optimizer = optim.LBFGS(
                    (latent,), line_search_fn=None) #"strong_wolfe"

                def closure():
                    predicted_hidden = self.hidden_predictor(
                        encoded_frames, ptp, latent)
                    predicted_frame = self.frame_decoder(predicted_hidden)
                    energies = self.compute_energies(
                        predicted_frame,
                        target_frame,
                        latent=latent,
                    )
                    latent_optimizer.zero_grad()
                    energies["total"].backward()
                    return energies["total"]

                latent_optimizer.step(closure)

    def compute_energies(self, predicted_frame, target_frame, latent=None):
        energies = {}
        energies["target_prediction"] = F.mse_loss(predicted_frame, target_frame)

        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies])
        return energies

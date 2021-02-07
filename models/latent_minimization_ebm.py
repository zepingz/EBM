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
        latent_optimizer_step,
        lambda_target_prediction,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.no_latent = no_latent
        self.latent_optimizer_type = latent_optimizer_type
        self.latent_optimizer_step = latent_optimizer_step

        self.frame_encoder = encoder
        self.frame_decoder = decoder
        self.hidden_predictor = predictor

        self.lambdas = {
            "target_prediction": lambda_target_prediction,
            # "decoding_error": lambda_decoding_error,
        }

    def forward(self, conditional_frames, ptp):
        # Encode conditional frames
        encoded_frames = self.encode_frames(conditional_frames)

        # Predict target frame
        latent = self.sample_latent(len(encoded_frames), encoded_frames.device)
        predicted_hidden = self.hidden_predictor(
            encoded_frames, ptp, latent)
        predicted_frame = self.frame_decoder(predicted_hidden)

        return predicted_frame

    def encode_frames(self, frames):
        if len(frames.shape) == 5:
            bs, seq_len, c, h, w = frames.shape
            frames = frames.view(bs * seq_len, c, h, w)
            encoded_frames = self.frame_encoder(frames)
            encoded_frames = encoded_frames.view(bs, seq_len, -1)
        else:
            encoded_frames = self.frame_encoder(frames)

        return encoded_frames

    def sample_latent(self, bs, device):
        latent = torch.randn(bs, self.latent_size).to(device)
        latent.requires_grad = True
        return latent

    def compute_optimal_latent(self, encoded_frames, ptp, target_frame):
        if self.no_latent:
            return torch.zeros(1)

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

                for _ in range(self.latent_optimizer_step):
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
                    (latent,), max_iter=20, line_search_fn=None) #"strong_wolfe"

                def closure():
                    predicted_hidden = self.hidden_predictor(
                        encoded_frames, ptp, latent)
                    predicted_frame = self.frame_decoder(predicted_hidden)
                    energies = self.compute_energies(
                        predicted_frame,
                        target_frame,
                        latent=latent,
                    )
                    print(f"{energies['total'].item():.5f}")
                    latent_optimizer.zero_grad()
                    energies["total"].backward()
                    return energies["total"]

                for _ in range(self.latent_optimizer_step):
                    latent_optimizer.step(closure)
                    print()
                print("---\n")

    def compute_energies(self, predicted_frame, target_frame, latent=None):
        energies = {}
        energies["target_prediction"] = F.mse_loss(predicted_frame, target_frame)

        energies["total"] = sum(
            [energies[k] * self.lambdas[k] for k in energies])
        return energies

"""Neural network models used in the RAFA experiments.

The architectures follow the NoF 2026 RAFA development notebook:
- fully connected autoencoder (AE)
- fully connected variational autoencoder (VAE)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleAE(nn.Module):
    """Fully connected autoencoder used for reconstruction-based IDS."""

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class SimpleVAE(nn.Module):
    """Fully connected variational autoencoder used in the main experiments."""

    def __init__(self, input_dim: int, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_layer = nn.Linear(32, latent_dim)
        self.logvar_layer = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared_encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        # The development notebook uses stochastic sampling in training mode
        # and the posterior mean in evaluation mode.
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    return nn.functional.mse_loss(x_hat, x, reduction=reduction)


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )

    if reduction == "mean":
        return kl_per_sample.mean()
    if reduction == "sum":
        return kl_per_sample.sum()
    if reduction in {"none", None}:
        return kl_per_sample
    raise ValueError(f"Unsupported reduction: {reduction}")


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = reconstruction_loss(x_hat, x, reduction="mean")
    kl = kl_divergence(mu, logvar, reduction="mean")
    return rec + kl_weight * kl, rec, kl


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    model_type: str,
    input_dim: int,
    latent_dim: int = 16,
    dropout: float = 0.2,
) -> nn.Module:
    model_type = model_type.upper()
    if model_type == "AE":
        return SimpleAE(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    if model_type == "VAE":
        return SimpleVAE(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
    raise ValueError(f"Unknown model type: {model_type}")

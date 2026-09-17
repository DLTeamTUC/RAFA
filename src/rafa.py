"""Reconstruction-Aware Federated Aggregation (RAFA).

This module exposes the paper-facing reconstruction-only AQS formulation.

For client update i:

    AQS_i = exp(
        -alpha * max(
            0,
            (RE_i - RE_0) / (RE_0 + epsilon) - m_rec
        )
    )

Updates with AQS < tau are rejected. Accepted updates are weighted by AQS.

The development notebook also contained inactive KL-score machinery
(beta=0 in the published configuration). It is intentionally not part of
the public paper-facing implementation here.
"""

from __future__ import annotations

import numpy as np
import torch

from src.aggregators import (
    add_update_inplace,
    apply_update_to_model,
    average_updates,
    clone_model,
)


def get_reference_tensor(reference_df, feature_cols) -> torch.Tensor:
    return torch.tensor(
        reference_df[feature_cols].values,
        dtype=torch.float32,
    )


def mean_reconstruction_error(
    model,
    x_tensor: torch.Tensor,
    model_type: str,
) -> float:
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        x = x_tensor.to(device)
        if model_type.upper() == "VAE":
            x_hat, _, _ = model(x)
        else:
            x_hat = model(x)

        rec_per_sample = torch.mean((x_hat - x) ** 2, dim=1)
        return float(rec_per_sample.mean().item())


def compute_aqs(
    global_model,
    client_update,
    reference_tensor: torch.Tensor,
    alpha: float = 15.0,
    m_rec: float = 0.01,
    epsilon: float = 1e-8,
    model_type: str = "VAE",
) -> dict:
    """Compute RAFA's Aggregation Quality Score for one client update."""

    candidate_model = apply_update_to_model(global_model, client_update)

    rec_0 = mean_reconstruction_error(
        global_model,
        reference_tensor,
        model_type,
    )
    rec_i = mean_reconstruction_error(
        candidate_model,
        reference_tensor,
        model_type,
    )

    relative_degradation = (rec_i - rec_0) / (rec_0 + epsilon)
    penalty = max(0.0, relative_degradation - m_rec)
    aqs = float(np.exp(-alpha * penalty))

    return {
        "aqs": aqs,
        "rec_0": float(rec_0),
        "rec_i": float(rec_i),
        "relative_degradation": float(relative_degradation),
        "penalty": float(penalty),
    }


def rafa_aggregate(
    global_model,
    client_updates,
    reference_tensor: torch.Tensor,
    alpha: float = 15.0,
    m_rec: float = 0.01,
    tau: float = 0.2,
    epsilon: float = 1e-8,
    model_type: str = "VAE",
):
    """Apply RAFA screening and AQS-weighted aggregation."""

    aqs_info = [
        compute_aqs(
            global_model=global_model,
            client_update=upd,
            reference_tensor=reference_tensor,
            alpha=alpha,
            m_rec=m_rec,
            epsilon=epsilon,
            model_type=model_type,
        )
        for upd in client_updates
    ]

    scores = np.asarray([item["aqs"] for item in aqs_info], dtype=np.float64)
    accepted = scores >= tau

    if accepted.sum() == 0:
        return clone_model(global_model), aqs_info

    accepted_updates = [
        upd for upd, keep in zip(client_updates, accepted) if keep
    ]
    accepted_weights = scores[accepted]

    avg_update = average_updates(
        accepted_updates,
        weights=accepted_weights,
    )
    new_model = clone_model(global_model)
    add_update_inplace(new_model, avg_update)

    return new_model, aqs_info

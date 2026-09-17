"""Baseline federated aggregation methods used in the RAFA experiments."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import reconstruction_loss, vae_loss


def clone_model(model, device: torch.device | str | None = None):
    cloned = copy.deepcopy(model)
    cloned.load_state_dict(copy.deepcopy(model.state_dict()))
    if device is not None:
        cloned = cloned.to(device)
    return cloned


def get_trainable_param_names(model) -> list[str]:
    return [name for name, _ in model.named_parameters()]


def get_model_update(global_model, local_model) -> dict[str, torch.Tensor]:
    global_params = dict(global_model.named_parameters())
    local_params = dict(local_model.named_parameters())
    return {
        name: (local_params[name].detach() - global_params[name].detach()).clone()
        for name in global_params
    }


def apply_update_to_model(global_model, update: dict[str, torch.Tensor]):
    candidate = clone_model(global_model)
    with torch.no_grad():
        for name, param in candidate.named_parameters():
            if name in update:
                param.add_(update[name].to(param.device))
    return candidate


def update_to_vector(update: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [tensor.detach().flatten().float().cpu() for tensor in update.values()]
    )


def vector_to_update(
    vector: torch.Tensor,
    template_update: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_update = {}
    pointer = 0
    for name, tensor in template_update.items():
        numel = tensor.numel()
        new_update[name] = (
            vector[pointer:pointer + numel]
            .view_as(tensor)
            .to(tensor.device)
        )
        pointer += numel
    return new_update


def average_updates(
    updates: list[dict[str, torch.Tensor]],
    weights=None,
) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("Cannot average an empty update list.")

    if weights is None:
        weights = np.ones(len(updates), dtype=np.float64)

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    avg_update = {}
    for name in updates[0]:
        stacked = torch.stack(
            [u[name].detach().float() for u in updates],
            dim=0,
        )
        w = torch.tensor(
            weights,
            dtype=stacked.dtype,
            device=stacked.device,
        ).view(-1, *([1] * (stacked.ndim - 1)))
        avg_update[name] = (stacked * w).sum(dim=0)

    return avg_update


def add_update_inplace(model, update: dict[str, torch.Tensor], scale: float = 1.0):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in update:
                param.add_(scale * update[name].to(param.device))
    return model


def fedavg_aggregate(global_model, client_updates, client_sizes=None):
    weights = (
        None
        if client_sizes is None
        else np.asarray(client_sizes, dtype=np.float64)
    )
    avg_update = average_updates(client_updates, weights=weights)
    new_model = clone_model(global_model)
    add_update_inplace(new_model, avg_update)
    return new_model, {"selected": len(client_updates)}


def fedavgm_aggregate(
    global_model,
    client_updates,
    velocity_state=None,
    client_sizes=None,
    beta_momentum: float = 0.9,
    server_lr: float = 1.0,
):
    if velocity_state is None:
        velocity_state = {
            name: torch.zeros_like(tensor, dtype=torch.float32)
            for name, tensor in client_updates[0].items()
        }

    avg_update = average_updates(client_updates, weights=client_sizes)

    new_velocity = {}
    momentum_update = {}
    for name in avg_update:
        prev_v = velocity_state[name].to(avg_update[name].device)
        new_v = beta_momentum * prev_v + avg_update[name]
        new_velocity[name] = new_v.detach().clone()
        momentum_update[name] = server_lr * new_v

    new_model = clone_model(global_model)
    add_update_inplace(new_model, momentum_update)

    return new_model, {
        "velocity": new_velocity,
        "selected": len(client_updates),
    }


def multi_krum_aggregate(global_model, client_updates, n_byzantine: int):
    """Krum-style selection used by the development notebook.

    Despite the historical function name ``multi_krum_aggregate``, the
    paper-producing notebook selects one lowest-score update.
    """

    n = len(client_updates)
    vectors = torch.stack([update_to_vector(u) for u in client_updates])

    k_neighbors = max(1, n - n_byzantine - 2)
    distances = torch.cdist(vectors, vectors, p=2) ** 2

    scores = []
    for i in range(n):
        nearest = torch.topk(
            distances[i],
            k=k_neighbors + 1,
            largest=False,
        ).values[1:]
        scores.append(nearest.sum().item())

    selected_idx = int(np.argmin(scores))
    selected_update = client_updates[selected_idx]

    new_model = clone_model(global_model)
    add_update_inplace(new_model, selected_update)

    return new_model, {
        "selected_idx": selected_idx,
        "scores": scores,
    }


def trimmed_mean_aggregate(global_model, client_updates, n_byzantine: int):
    n = len(client_updates)
    trim = min(n_byzantine, (n - 1) // 2)

    avg_update = {}
    for name in client_updates[0]:
        stacked = torch.stack(
            [u[name].detach().float() for u in client_updates],
            dim=0,
        )
        if trim > 0 and n > 2 * trim:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim:n - trim]
        else:
            trimmed = stacked
        avg_update[name] = trimmed.mean(dim=0)

    new_model = clone_model(global_model)
    add_update_inplace(new_model, avg_update)
    return new_model, {"trim": trim, "selected": n - 2 * trim}


def dnc_aggregate(global_model, client_updates, n_byzantine: int):
    n = len(client_updates)
    vectors = torch.stack([update_to_vector(u) for u in client_updates])
    centered = vectors - vectors.mean(dim=0, keepdim=True)

    try:
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        scores = torch.abs(centered @ direction)
    except Exception:
        scores = torch.norm(centered, dim=1)

    keep = max(1, n - n_byzantine)
    selected_idx = (
        torch.topk(scores, k=keep, largest=False)
        .indices.cpu().numpy().tolist()
    )

    selected_updates = [client_updates[i] for i in selected_idx]
    avg_update = average_updates(selected_updates)

    new_model = clone_model(global_model)
    add_update_inplace(new_model, avg_update)

    return new_model, {
        "selected_idx": selected_idx,
        "scores": scores.cpu().numpy().tolist(),
    }


def train_server_root_update(
    global_model,
    reference_tensor: torch.Tensor,
    model_type: str,
    device: torch.device | str,
    lr: float = 1e-3,
    epochs: int = 1,
    batch_size: int = 256,
):
    server_model = clone_model(global_model, device=device)
    server_model.train()

    loader = DataLoader(
        TensorDataset(reference_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(server_model.parameters(), lr=lr)

    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()

            if model_type.upper() == "VAE":
                x_hat, mu, logvar = server_model(x)
                loss, _, _ = vae_loss(x_hat, x, mu, logvar)
            else:
                x_hat = server_model(x)
                loss = reconstruction_loss(x_hat, x)

            loss.backward()
            optimizer.step()

    return get_model_update(global_model, server_model)


def fltrust_ae_aggregate(
    global_model,
    client_updates,
    reference_tensor: torch.Tensor,
    model_type: str,
    device: torch.device | str,
    lr: float = 1e-3,
    batch_size: int = 256,
):
    """FLTrust-inspired AE/VAE baseline used by the development notebook."""

    root_update = train_server_root_update(
        global_model=global_model,
        reference_tensor=reference_tensor,
        model_type=model_type,
        device=device,
        lr=lr,
        epochs=1,
        batch_size=batch_size,
    )

    root_vec = update_to_vector(root_update)
    root_norm = torch.norm(root_vec) + 1e-12

    trust_scores = []
    for upd in client_updates:
        vec = update_to_vector(upd)
        cos = torch.dot(vec, root_vec) / (
            (torch.norm(vec) + 1e-12) * root_norm
        )
        trust_scores.append(max(0.0, float(cos.item())))

    if np.sum(trust_scores) <= 1e-12:
        return clone_model(global_model), {
            "trust_scores": trust_scores,
            "selected": 0,
        }

    avg_update = average_updates(client_updates, weights=trust_scores)
    new_model = clone_model(global_model)
    add_update_inplace(new_model, avg_update)

    return new_model, {
        "trust_scores": trust_scores,
        "selected": int(np.sum(np.asarray(trust_scores) > 0)),
    }

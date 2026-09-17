"""Byzantine attack implementations used in the RAFA experiments."""

from __future__ import annotations

import torch

from src.aggregators import clone_model, get_model_update, update_to_vector
from src.models import kl_divergence, reconstruction_loss, vae_loss


def scale_update(
    update: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    return {name: tensor * scale for name, tensor in update.items()}


def scale_update_to_norm(
    update: dict[str, torch.Tensor],
    target_norm: float,
) -> dict[str, torch.Tensor]:
    vec = update_to_vector(update)
    norm = torch.norm(vec).item()

    if norm <= 1e-12:
        return update

    scale = target_norm / norm
    return {name: tensor * scale for name, tensor in update.items()}


def attack_random_noise(
    genuine_update,
    scale: float = 1.0,
    match_norm: bool = True,
):
    noise_update = {
        name: torch.randn_like(tensor.float()) * scale
        for name, tensor in genuine_update.items()
    }

    if match_norm:
        target_norm = torch.norm(update_to_vector(genuine_update)).item()
        noise_update = scale_update_to_norm(noise_update, target_norm)

    return noise_update


def attack_sign_flip(genuine_update, eta: float = 5.0):
    return {
        name: -eta * tensor.detach().clone()
        for name, tensor in genuine_update.items()
    }


def train_model_for_attack(
    global_model,
    data_loader,
    model_type: str,
    device,
    objective: str = "minimize_re",
    local_epochs: int = 1,
    lr: float = 1e-3,
    max_batches: int | None = None,
):
    attack_model = clone_model(global_model, device=device)
    attack_model.train()

    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    batches_seen = 0

    for _ in range(local_epochs):
        for (x,) in data_loader:
            x = x.to(device)
            optimizer.zero_grad()

            if model_type.upper() == "VAE":
                x_hat, mu, logvar = attack_model(x)
                _, rec, _ = vae_loss(x_hat, x, mu, logvar)
                target = rec
            else:
                x_hat = attack_model(x)
                target = reconstruction_loss(x_hat, x)

            if objective == "maximize_re":
                train_loss = -target
            elif objective == "minimize_re":
                train_loss = target
            else:
                raise ValueError(f"Unknown attack objective: {objective}")

            train_loss.backward()
            optimizer.step()

            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break

        if max_batches is not None and batches_seen >= max_batches:
            break

    return get_model_update(global_model, attack_model)


def attack_recon_inflation(
    global_model,
    target_loader,
    model_type: str,
    device,
    local_epochs: int = 1,
    lr: float = 1e-3,
    max_batches: int = 20,
    match_norm_update=None,
):
    poisoned_update = train_model_for_attack(
        global_model=global_model,
        data_loader=target_loader,
        model_type=model_type,
        device=device,
        objective="maximize_re",
        local_epochs=local_epochs,
        lr=lr,
        max_batches=max_batches,
    )

    if match_norm_update is not None:
        target_norm = torch.norm(update_to_vector(match_norm_update)).item()
        poisoned_update = scale_update_to_norm(poisoned_update, target_norm)

    return poisoned_update


def attack_train_on_attack_traffic(
    global_model,
    attack_loader,
    model_type: str,
    device,
    local_epochs: int = 1,
    lr: float = 1e-3,
    max_batches: int = 20,
    match_norm_update=None,
):
    poisoned_update = train_model_for_attack(
        global_model=global_model,
        data_loader=attack_loader,
        model_type=model_type,
        device=device,
        objective="minimize_re",
        local_epochs=local_epochs,
        lr=lr,
        max_batches=max_batches,
    )

    if match_norm_update is not None:
        target_norm = torch.norm(update_to_vector(match_norm_update)).item()
        poisoned_update = scale_update_to_norm(poisoned_update, target_norm)

    return poisoned_update


def attack_adaptive(
    global_model,
    proxy_loader,
    reference_tensor: torch.Tensor,
    model_type: str,
    device,
    alpha: float,
    tau: float,
    m_rec: float = 0.01,
    lr: float = 1e-3,
    steps: int = 20,
    penalty_weight: float = 10.0,
    match_norm_update=None,
):
    """Adaptive attack against the paper-facing reconstruction-only AQS."""

    attack_model = clone_model(global_model, device=device)
    attack_model.train()

    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)
    proxy_iter = iter(proxy_loader)

    global_model.eval()
    with torch.no_grad():
        x_ref = reference_tensor.to(device)
        if model_type.upper() == "VAE":
            x_ref_hat, _, _ = global_model(x_ref)
        else:
            x_ref_hat = global_model(x_ref)
        rec_0 = torch.mean((x_ref_hat - x_ref) ** 2)

    for _ in range(steps):
        try:
            (x_proxy,) = next(proxy_iter)
        except StopIteration:
            proxy_iter = iter(proxy_loader)
            (x_proxy,) = next(proxy_iter)

        x_proxy = x_proxy.to(device)
        optimizer.zero_grad()

        if model_type.upper() == "VAE":
            x_proxy_hat, _, _ = attack_model(x_proxy)
            x_ref_hat_i, _, _ = attack_model(x_ref)
        else:
            x_proxy_hat = attack_model(x_proxy)
            x_ref_hat_i = attack_model(x_ref)

        rec_proxy = torch.mean((x_proxy_hat - x_proxy) ** 2)
        rec_i = torch.mean((x_ref_hat_i - x_ref) ** 2)

        relative_degradation = (
            (rec_i - rec_0) / torch.clamp(rec_0, min=1e-8)
        )
        penalty = torch.clamp(
            relative_degradation - m_rec,
            min=0.0,
        )
        aqs = torch.exp(-alpha * penalty)

        constraint_penalty = (
            torch.relu(torch.tensor(tau, device=device) - aqs) ** 2
        )
        loss = -rec_proxy + penalty_weight * constraint_penalty

        loss.backward()
        optimizer.step()

    poisoned_update = get_model_update(global_model, attack_model)

    if match_norm_update is not None:
        target_norm = torch.norm(update_to_vector(match_norm_update)).item()
        poisoned_update = scale_update_to_norm(poisoned_update, target_norm)

    return poisoned_update

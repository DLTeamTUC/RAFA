"""Federated training loop used by the RAFA paper reproduction artifact."""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch

from src.aggregators import (
    clone_model,
    dnc_aggregate,
    fedavg_aggregate,
    fedavgm_aggregate,
    fltrust_ae_aggregate,
    get_model_update,
    multi_krum_aggregate,
    trimmed_mean_aggregate,
)
from src.attacks import (
    attack_adaptive,
    attack_random_noise,
    attack_recon_inflation,
    attack_sign_flip,
    attack_train_on_attack_traffic,
    scale_update,
)
from src.data import make_loader_from_df
from src.metrics import (
    calibrate_threshold,
    compute_screening_metrics,
    evaluate_global_model,
    reconstruction_errors,
    summarize_aqs_history,
)
from src.models import reconstruction_loss, vae_loss
from src.rafa import get_reference_tensor, rafa_aggregate


def set_global_seed(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # The paper-producing notebook used deterministic=False and benchmark=True.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def train_local_model(
    global_model,
    train_loader,
    model_type: str,
    device,
    local_epochs: int,
    lr: float,
    max_batches: int | None = None,
):
    """Train one client model.

    Reproduction note: ``max_batches`` is a cap over the complete local
    training call, not a per-epoch cap. The paper-producing experiments used
    ``local_epochs=5`` with ``max_batches=20``.
    """

    local_model = clone_model(global_model, device=device)
    local_model.train()

    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    batches_seen = 0

    for _ in range(local_epochs):
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            if model_type.upper() == "VAE":
                x_hat, mu, logvar = local_model(x)
                loss, _, _ = vae_loss(x_hat, x, mu, logvar)
            else:
                x_hat = local_model(x)
                loss = reconstruction_loss(x_hat, x)

            loss.backward()
            optimizer.step()

            batches_seen += 1
            if max_batches is not None and batches_seen >= max_batches:
                break

        if max_batches is not None and batches_seen >= max_batches:
            break

    return local_model, get_model_update(global_model, local_model)


def select_byzantine_clients(
    n_clients: int,
    n_byzantine: int,
    seed: int,
) -> list[int]:
    if n_byzantine < 0 or n_byzantine > n_clients:
        raise ValueError("n_byzantine must be between 0 and n_clients.")
    if n_byzantine == 0:
        return []

    rng = np.random.default_rng(seed)
    return sorted(
        rng.choice(
            np.arange(n_clients),
            size=n_byzantine,
            replace=False,
        ).tolist()
    )


def make_attack_loader(
    dataset_obj: dict,
    feature_cols,
    batch_size: int = 256,
    max_samples: int = 5000,
    seed: int = 42,
):
    attack_df = dataset_obj["test"].loc[
        dataset_obj["test"]["binary_label"] == 1
    ].copy()

    if len(attack_df) > max_samples:
        attack_df = attack_df.sample(
            n=max_samples,
            random_state=seed,
        ).reset_index(drop=True)

    return make_loader_from_df(
        attack_df,
        feature_cols,
        batch_size=batch_size,
        shuffle=True,
    )


def apply_attack_to_update(
    attack_name: str | None,
    genuine_update,
    global_model,
    model_type: str,
    client_loader,
    attack_loader,
    reference_tensor,
    device,
    lr: float = 1e-3,
    alpha: float = 15.0,
    m_rec: float = 0.01,
    tau: float = 0.2,
    sign_flip_eta: float = 5.0,
    recon_inflation_scale: float = 1.0,
    adaptive_attack_scale: float = 1.0,
    train_on_attack_scale: float = 1.0,
    attack_max_batches: int = 20,
):
    if attack_name in {None, "none"}:
        return genuine_update

    if attack_name == "noise":
        return attack_random_noise(
            genuine_update,
            scale=1.0,
            match_norm=True,
        )

    if attack_name == "sign_flip":
        return attack_sign_flip(
            genuine_update,
            eta=sign_flip_eta,
        )

    if attack_name == "recon_inflation":
        attack_update = attack_recon_inflation(
            global_model=global_model,
            target_loader=client_loader,
            model_type=model_type,
            device=device,
            local_epochs=1,
            lr=lr,
            max_batches=attack_max_batches,
            match_norm_update=genuine_update,
        )
        return scale_update(attack_update, recon_inflation_scale)

    if attack_name == "train_on_attack":
        attack_update = attack_train_on_attack_traffic(
            global_model=global_model,
            attack_loader=attack_loader,
            model_type=model_type,
            device=device,
            local_epochs=1,
            lr=lr,
            max_batches=attack_max_batches,
            match_norm_update=genuine_update,
        )
        return scale_update(attack_update, train_on_attack_scale)

    if attack_name in {"adaptive_score", "adaptive_reference"}:
        # Matches the development notebook:
        # score-aware uses sampled attack traffic as the proxy;
        # reference-aware uses the malicious client's local benign data.
        proxy_loader = (
            attack_loader
            if attack_name == "adaptive_score"
            else client_loader
        )

        attack_update = attack_adaptive(
            global_model=global_model,
            proxy_loader=proxy_loader,
            reference_tensor=reference_tensor,
            model_type=model_type,
            device=device,
            alpha=alpha,
            tau=tau,
            m_rec=m_rec,
            lr=lr,
            steps=20,
            match_norm_update=genuine_update,
        )
        return scale_update(attack_update, adaptive_attack_scale)

    raise ValueError(f"Unknown attack name: {attack_name}")


def aggregate_by_name(
    aggregator: str,
    global_model,
    client_updates,
    client_sizes,
    reference_tensor,
    model_type: str,
    n_byzantine: int,
    state: dict,
    device,
    *,
    alpha: float = 15.0,
    m_rec: float = 0.01,
    tau: float = 0.2,
    epsilon: float = 1e-8,
    server_lr: float = 1.0,
    beta_momentum: float = 0.9,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
):
    if aggregator == "RAFA":
        new_model, info = rafa_aggregate(
            global_model=global_model,
            client_updates=client_updates,
            reference_tensor=reference_tensor,
            alpha=alpha,
            m_rec=m_rec,
            tau=tau,
            epsilon=epsilon,
            model_type=model_type,
        )
        return new_model, info, state

    if aggregator == "FedAvg":
        new_model, info = fedavg_aggregate(
            global_model,
            client_updates,
            client_sizes=client_sizes,
        )
        return new_model, info, state

    if aggregator == "FedAvgM":
        new_model, info = fedavgm_aggregate(
            global_model,
            client_updates,
            velocity_state=state.get("velocity"),
            client_sizes=client_sizes,
            beta_momentum=beta_momentum,
            server_lr=server_lr,
        )
        state["velocity"] = info["velocity"]
        return new_model, info, state

    if aggregator == "Krum":
        new_model, info = multi_krum_aggregate(
            global_model,
            client_updates,
            n_byzantine=n_byzantine,
        )
        return new_model, info, state

    if aggregator == "TrimMean":
        new_model, info = trimmed_mean_aggregate(
            global_model,
            client_updates,
            n_byzantine=n_byzantine,
        )
        return new_model, info, state

    if aggregator == "DnC":
        new_model, info = dnc_aggregate(
            global_model,
            client_updates,
            n_byzantine=n_byzantine,
        )
        return new_model, info, state

    if aggregator == "FLTrust-AE":
        new_model, info = fltrust_ae_aggregate(
            global_model=global_model,
            client_updates=client_updates,
            reference_tensor=reference_tensor,
            model_type=model_type,
            device=device,
            lr=learning_rate,
            batch_size=batch_size,
        )
        return new_model, info, state

    raise ValueError(f"Unknown aggregator: {aggregator}")


def run_federation(
    dataset_obj: dict,
    initial_model,
    *,
    aggregator: str = "RAFA",
    model_type: str = "VAE",
    partition: str = "iid",
    n_clients: int = 10,
    n_byzantine: int = 0,
    attack: str = "none",
    fed_rounds: int = 20,
    local_epochs: int = 5,
    max_local_batches: int | None = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device=None,
    alpha: float = 15.0,
    m_rec: float = 0.01,
    tau: float = 0.2,
    epsilon: float = 1e-8,
    server_lr: float = 1.0,
    beta_momentum: float = 0.9,
    threshold_percentile: float = 95,
    sign_flip_eta: float = 5.0,
    recon_inflation_scale: float = 1.0,
    adaptive_attack_scale: float = 1.0,
    train_on_attack_scale: float = 1.0,
) -> dict:
    """Run one complete federated experiment."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_global_seed(seed)

    feature_cols = dataset_obj["feature_cols"]

    if partition == "iid":
        client_loaders = dataset_obj["iid_loaders"]
        client_parts = dataset_obj["iid_parts"]
    elif partition == "noniid":
        client_loaders = dataset_obj["noniid_loaders"]
        client_parts = dataset_obj["noniid_parts"]
    else:
        raise ValueError(f"Unknown partition: {partition}")

    if len(client_loaders) != n_clients:
        raise ValueError(
            f"Expected {n_clients} client loaders, found {len(client_loaders)}."
        )

    client_sizes = [len(client_parts[cid]) for cid in range(n_clients)]

    global_model = clone_model(initial_model, device=device)
    reference_tensor = get_reference_tensor(
        dataset_obj["reference"],
        feature_cols,
    )
    attack_loader = make_attack_loader(
        dataset_obj,
        feature_cols,
        batch_size=batch_size,
        seed=seed,
    )

    # Matches the paper-producing notebook's deterministic client selection.
    byzantine_ids = select_byzantine_clients(
        n_clients=n_clients,
        n_byzantine=n_byzantine,
        seed=seed + n_byzantine,
    )

    aqs_history = []
    round_re_ref = []
    round_re_test_benign = []
    aggregator_state = {}

    start_time = time.time()

    for _round_idx in range(fed_rounds):
        client_updates = []

        for cid in range(n_clients):
            _, genuine_update = train_local_model(
                global_model=global_model,
                train_loader=client_loaders[cid],
                model_type=model_type,
                device=device,
                local_epochs=local_epochs,
                lr=learning_rate,
                max_batches=max_local_batches,
            )

            if cid in byzantine_ids:
                final_update = apply_attack_to_update(
                    attack_name=attack,
                    genuine_update=genuine_update,
                    global_model=global_model,
                    model_type=model_type,
                    client_loader=client_loaders[cid],
                    attack_loader=attack_loader,
                    reference_tensor=reference_tensor,
                    device=device,
                    lr=learning_rate,
                    alpha=alpha,
                    m_rec=m_rec,
                    tau=tau,
                    sign_flip_eta=sign_flip_eta,
                    recon_inflation_scale=recon_inflation_scale,
                    adaptive_attack_scale=adaptive_attack_scale,
                    train_on_attack_scale=train_on_attack_scale,
                )
            else:
                final_update = genuine_update

            client_updates.append(final_update)

        global_model, agg_info, aggregator_state = aggregate_by_name(
            aggregator=aggregator,
            global_model=global_model,
            client_updates=client_updates,
            client_sizes=client_sizes,
            reference_tensor=reference_tensor,
            model_type=model_type,
            n_byzantine=n_byzantine,
            state=aggregator_state,
            device=device,
            alpha=alpha,
            m_rec=m_rec,
            tau=tau,
            epsilon=epsilon,
            server_lr=server_lr,
            beta_momentum=beta_momentum,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        if aggregator == "RAFA":
            aqs_history.append([item["aqs"] for item in agg_info])

        ref_loader = make_loader_from_df(
            dataset_obj["reference"],
            feature_cols,
            batch_size=1024,
            shuffle=False,
        )
        test_benign_df = dataset_obj["test"].loc[
            dataset_obj["test"]["binary_label"] == 0
        ]
        test_benign_loader = make_loader_from_df(
            test_benign_df,
            feature_cols,
            batch_size=1024,
            shuffle=False,
        )

        ref_errors = reconstruction_errors(
            global_model,
            ref_loader,
            model_type,
            device,
        )
        test_benign_errors = reconstruction_errors(
            global_model,
            test_benign_loader,
            model_type,
            device,
        )

        round_re_ref.append(float(np.nanmean(ref_errors)))
        round_re_test_benign.append(float(np.nanmean(test_benign_errors)))

    threshold = calibrate_threshold(
        model=global_model,
        benign_val_df=dataset_obj["val"],
        feature_cols=feature_cols,
        model_type=model_type,
        device=device,
        percentile=threshold_percentile,
    )

    final_metrics = evaluate_global_model(
        model=global_model,
        test_df=dataset_obj["test"],
        feature_cols=feature_cols,
        model_type=model_type,
        threshold=threshold,
        device=device,
    )

    screening = compute_screening_metrics(
        aqs_history,
        byzantine_ids,
        tau,
    )
    aqs_summary = summarize_aqs_history(
        aqs_history,
        byzantine_ids,
    )

    return {
        "config": {
            "aggregator": aggregator,
            "model_type": model_type,
            "partition": partition,
            "n_clients": n_clients,
            "n_byzantine": n_byzantine,
            "attack": attack,
            "fed_rounds": fed_rounds,
            "local_epochs": local_epochs,
            "max_local_batches": max_local_batches,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "alpha": alpha,
            "m_rec": m_rec,
            "tau": tau,
        },
        "aggregator": aggregator,
        "model_type": model_type,
        "partition": partition,
        "attack": attack,
        "n_byzantine": n_byzantine,
        "byzantine_ids": byzantine_ids,
        "final_model": global_model,
        "final_metrics": final_metrics,
        "screening": screening,
        "aqs_summary": aqs_summary,
        "aqs_history": aqs_history,
        "round_re_ref": round_re_ref,
        "round_re_test_benign": round_re_test_benign,
        "runtime_sec": float(time.time() - start_time),
    }

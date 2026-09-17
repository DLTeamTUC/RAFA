#!/usr/bin/env python3
"""Reproduce the CICDDoS2019 cross-dataset RAFA experiments.

Default usage from the repository root:

    python experiments/reproduce_cicddos2019.py

Run only the main cross-dataset comparison:

    python experiments/reproduce_cicddos2019.py --experiment matched_robustness

Run the adapted reconstruction/update-space comparison:

    python experiments/reproduce_cicddos2019.py --experiment adapted_baselines

The paper-producing notebook used source-specific RAFA settings for the final
MSSQL and DrDoS-DNS runs. Those settings are read from
``configs/paper_cicddos2019.yaml`` and are preserved here explicitly.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.rafa as rafa_module  # noqa: E402
from src.data import (  # noqa: E402
    add_client_partitions,
    common_numeric_features,
    prepare_cicddos_subset,
    refresh_reference_and_partitions,
    scan_cicddos_file,
)
from src.models import build_model  # noqa: E402
from src.training import run_federation, set_global_seed  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_cicddos2019.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "cicddos2019"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the RAFA CICDDoS2019 paper experiments."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--experiment",
        choices=["all", "matched_robustness", "adapted_baselines"],
        default="all",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke-test mode: one subset, one seed, one round and one local "
            "batch. Not a paper reproduction."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files instead of resuming.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_initial_model(dataset: dict, config: dict, seed: int, device):
    model_cfg = config["model"]
    set_global_seed(seed)
    return build_model(
        model_type=model_cfg["type"],
        input_dim=len(dataset["feature_cols"]),
        latent_dim=int(model_cfg["latent_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)


def prepare_selected_subsets(config: dict, quick: bool) -> dict[str, dict]:
    dataset_cfg = config["dataset"]
    root = resolve_repo_path(dataset_cfg["root"])

    subset_items = list(config["selected_subsets"].items())
    if quick:
        subset_items = subset_items[:1]

    scan_infos = []
    item_by_file = {}

    print("Scanning selected CICDDoS2019 files ...", flush=True)
    for subset_name, item in subset_items:
        info = scan_cicddos_file(root, item["file"])
        if not info.get("usable", False):
            raise RuntimeError(
                f"{item['file']} is not usable: {info.get('reason')}"
            )
        scan_infos.append(info)
        item_by_file[item["file"]] = (subset_name, item)

    feature_cols = common_numeric_features(scan_infos)
    print(f"Common numeric features: {len(feature_cols)}", flush=True)

    fed = config["federated"]
    reference_size = int(config["reference"]["size"])
    prepared = {}

    for info in scan_infos:
        subset_name, item = item_by_file[info["file"]]
        dataset = prepare_cicddos_subset(
            scan_info=info,
            feature_cols=feature_cols,
            seed=int(config["reproducibility"]["seeds"][0]),
            reference_size=reference_size,
        )
        dataset = add_client_partitions(
            dataset,
            n_clients=int(fed["n_clients"]),
            batch_size=int(fed["batch_size"]),
            seed=int(config["reproducibility"]["seeds"][0]),
        )
        dataset["dataset_name"] = item["dataset_id"]
        dataset["source_file"] = item["file"]
        dataset["attack_family"] = item.get("attack_family")
        dataset["review_benign_pool"] = pd.concat(
            [dataset["train"], dataset["reference"]],
            axis=0,
        ).reset_index(drop=True)
        prepared[subset_name] = dataset

        print(
            f"Prepared {item['dataset_id']} from {item['file']} "
            f"(train={len(dataset['train'])}, ref={len(dataset['reference'])})",
            flush=True,
        )

    return prepared


def refresh_for_seed(
    dataset: dict,
    config: dict,
    seed: int,
) -> dict:
    fed = config["federated"]
    refreshed = copy.deepcopy(dataset)
    return refresh_reference_and_partitions(
        refreshed,
        seed=seed,
        n_clients=int(fed["n_clients"]),
        batch_size=int(fed["batch_size"]),
        reference_size=int(config["reference"]["size"]),
    )


def common_run_kwargs(config: dict, *, seed: int, device, quick: bool) -> dict:
    fed = config["federated"]
    default = config["rafa_default"]
    evaluation = config["evaluation"]

    return {
        "model_type": config["model"]["type"],
        "n_clients": int(fed["n_clients"]),
        "fed_rounds": 1 if quick else int(fed["rounds"]),
        "local_epochs": int(fed["local_epochs"]),
        "max_local_batches": 1 if quick else int(fed["max_local_batches"]),
        "batch_size": int(fed["batch_size"]),
        "learning_rate": float(fed["learning_rate"]),
        "seed": int(seed),
        "device": device,
        "alpha": float(default["alpha"]),
        "m_rec": float(default["m_rec"]),
        "tau": float(default["tau"]),
        "epsilon": float(default.get("epsilon", 1e-8)),
        "threshold_percentile": float(evaluation["threshold_percentile"]),
    }


def result_to_row(
    result: dict,
    *,
    dataset_name: str,
    source_file: str,
    experiment: str,
    scenario: str,
    method: str,
    seed: int,
    alpha: float,
    m_rec: float,
    tau: float,
) -> dict:
    metrics = result["final_metrics"]
    screening = result["screening"]
    aqs = result["aqs_summary"]
    return {
        "dataset": dataset_name,
        "source_file": source_file,
        "experiment": experiment,
        "scenario": scenario,
        "method": method,
        "aggregator": result["aggregator"],
        "seed": int(seed),
        "alpha": float(alpha),
        "m_rec": float(m_rec),
        "tau": float(tau),
        "n_byzantine": int(result["n_byzantine"]),
        "f1": metrics.get("f1"),
        "auc": metrics.get("auc"),
        "fpr": metrics.get("fpr"),
        "asr": metrics.get("asr"),
        "mdr": screening.get("mdr"),
        "bfrr": screening.get("bfrr"),
        "mean_aqs_benign": aqs.get("mean_aqs_benign"),
        "mean_aqs_malicious": aqs.get("mean_aqs_malicious"),
        "aqs_gap": aqs.get("aqs_gap"),
        "runtime_sec": result.get("runtime_sec"),
        "status": "ok",
        "error_message": "",
    }


def error_row(
    *,
    dataset_name: str,
    source_file: str,
    experiment: str,
    scenario: str,
    method: str,
    seed: int,
    alpha: float,
    m_rec: float,
    tau: float,
    message: str,
) -> dict:
    lower = str(message).lower()
    status = (
        "diverged"
        if any(x in lower for x in ["non-finite", "nonfinite", "nan", "inf", "diverged"])
        else "error"
    )
    return {
        "dataset": dataset_name,
        "source_file": source_file,
        "experiment": experiment,
        "scenario": scenario,
        "method": method,
        "seed": int(seed),
        "alpha": float(alpha),
        "m_rec": float(m_rec),
        "tau": float(tau),
        "status": status,
        "error_message": str(message),
    }


def load_existing(path: Path, overwrite: bool) -> pd.DataFrame:
    if overwrite or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def append_and_save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def completed_keys(df: pd.DataFrame, fields: list[str]) -> set[tuple]:
    if df.empty or not set(fields + ["status"]).issubset(df.columns):
        return set()
    done = df[df["status"].isin(["ok", "diverged"])]
    return {
        tuple(row[field] for field in fields)
        for _, row in done.iterrows()
    }


def scenario_to_attack(scenario: str, config: dict) -> tuple[str, float]:
    exp = config["experiments"]["matched_robustness_comparison"]

    if scenario == "reconstruction_targeted":
        item = exp["reconstruction_targeted"]
        return item["attack"], float(item["attack_scale"])

    if scenario == "sign_flip":
        item = exp["sign_flip"]
        return item["attack"], float(item.get("eta", 5.0))

    raise ValueError(f"Unknown scenario: {scenario}")


def subset_rafa_params(config: dict, subset_name: str) -> dict:
    rafa_cfg = config["selected_subsets"][subset_name]["rafa"]
    return {
        "alpha": float(rafa_cfg["alpha"]),
        "m_rec": float(rafa_cfg["m_rec"]),
        "tau": float(rafa_cfg["tau"]),
    }


def run_matched_robustness(
    prepared: dict[str, dict],
    config: dict,
    output_path: Path,
    device,
    quick: bool,
    overwrite: bool,
) -> None:
    exp_cfg = config["experiments"]["matched_robustness_comparison"]
    seeds = list(config["reproducibility"]["seeds"])
    if quick:
        seeds = seeds[:1]

    existing = load_existing(output_path, overwrite)
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(
        existing,
        ["dataset", "scenario", "method", "seed"],
    )

    subset_names = list(prepared.keys())
    if quick:
        subset_names = subset_names[:1]

    for subset_name in subset_names:
        source_cfg = config["selected_subsets"][subset_name]
        rafa_params = subset_rafa_params(config, subset_name)

        for scenario in ["reconstruction_targeted", "sign_flip"]:
            if scenario == "reconstruction_targeted":
                baseline_methods = list(
                    exp_cfg["reconstruction_targeted"]["baselines"]
                )
            else:
                baseline_methods = list(exp_cfg["sign_flip"]["baselines"])

            methods = ["RAFA"] + baseline_methods
            if quick:
                methods = ["RAFA"]

            attack_name, attack_value = scenario_to_attack(scenario, config)
            frac = float(exp_cfg["byzantine_fraction"])

            for seed in seeds:
                dataset = refresh_for_seed(prepared[subset_name], config, int(seed))

                for method in methods:
                    key = (
                        source_cfg["dataset_id"],
                        scenario,
                        method,
                        int(seed),
                    )
                    if key in done:
                        continue

                    kwargs = common_run_kwargs(
                        config,
                        seed=int(seed),
                        device=device,
                        quick=quick,
                    )

                    if method == "RAFA":
                        kwargs.update(rafa_params)
                    else:
                        # Baselines do not use RAFA parameters, but retaining
                        # the nominal default in the result metadata is useful.
                        default = config["rafa_default"]
                        kwargs.update(
                            {
                                "alpha": float(default["alpha"]),
                                "m_rec": float(default["m_rec"]),
                                "tau": float(default["tau"]),
                            }
                        )

                    model = build_initial_model(
                        dataset,
                        config,
                        int(seed),
                        device,
                    )

                    print(
                        f"{source_cfg['dataset_id']} | {scenario} | "
                        f"{method} | seed={seed}",
                        flush=True,
                    )

                    try:
                        result = run_federation(
                            dataset_obj=dataset,
                            initial_model=model,
                            aggregator=method,
                            partition="iid",
                            n_byzantine=int(
                                round(kwargs["n_clients"] * frac)
                            ),
                            attack=attack_name,
                            recon_inflation_scale=(
                                attack_value
                                if scenario == "reconstruction_targeted"
                                else 1.0
                            ),
                            sign_flip_eta=(
                                attack_value
                                if scenario == "sign_flip"
                                else 5.0
                            ),
                            **kwargs,
                        )
                        row = result_to_row(
                            result,
                            dataset_name=source_cfg["dataset_id"],
                            source_file=source_cfg["file"],
                            experiment="matched_robustness",
                            scenario=scenario,
                            method=method,
                            seed=int(seed),
                            alpha=kwargs["alpha"],
                            m_rec=kwargs["m_rec"],
                            tau=kwargs["tau"],
                        )
                    except Exception as exc:
                        row = error_row(
                            dataset_name=source_cfg["dataset_id"],
                            source_file=source_cfg["file"],
                            experiment="matched_robustness",
                            scenario=scenario,
                            method=method,
                            seed=int(seed),
                            alpha=kwargs["alpha"],
                            m_rec=kwargs["m_rec"],
                            tau=kwargs["tau"],
                            message=str(exc),
                        )

                    rows.append(row)
                    append_and_save(rows, output_path)
                    print(
                        f"    status={row['status']} "
                        f"F1={safe_float(row.get('f1')):.4f} "
                        f"MDR={safe_float(row.get('mdr')):.4f}",
                        flush=True,
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def _update_norm_proxy(client_update) -> float:
    """FedREDefense-style proxy from the development notebook.

    This is intentionally labelled an adapted proxy, not original
    FedREDefense. It maps RMS-like update norm to a score in [0, 1].
    """

    values = []
    for tensor in client_update.values():
        values.append(tensor.detach().float().cpu().reshape(-1).numpy())

    if not values:
        return 1.0

    vector = np.concatenate(values).astype(np.float32)
    norm = float(np.linalg.norm(vector) / max(np.sqrt(vector.size), 1.0))
    return float(np.exp(-norm))


@contextmanager
def adapted_scoring(method: str):
    """Temporarily replace RAFA's per-update score for adapted baselines."""

    original = rafa_module.compute_aqs

    if method == "RAFA":
        yield
        return

    def patched_compute_aqs(
        global_model,
        client_update,
        reference_tensor,
        alpha=15.0,
        m_rec=0.01,
        epsilon=1e-8,
        model_type="VAE",
    ):
        out = original(
            global_model=global_model,
            client_update=client_update,
            reference_tensor=reference_tensor,
            alpha=alpha,
            m_rec=m_rec,
            epsilon=epsilon,
            model_type=model_type,
        )

        if method == "Zeno-RE":
            # Mirrors the final development notebook adaptation: it retains
            # the reconstruction-validation score. It is not a faithful
            # asynchronous Zeno++ implementation.
            new_score = float(out["aqs"])
        elif method == "FedREDefense-style":
            # Update-space norm/outlier proxy used in the final notebook.
            new_score = _update_norm_proxy(client_update)
        else:
            raise ValueError(f"Unknown adapted method: {method}")

        patched = dict(out)
        patched["aqs"] = float(np.clip(new_score, 0.0, 1.0))
        return patched

    rafa_module.compute_aqs = patched_compute_aqs
    try:
        yield
    finally:
        rafa_module.compute_aqs = original


def run_adapted_baselines(
    prepared: dict[str, dict],
    config: dict,
    output_path: Path,
    device,
    quick: bool,
    overwrite: bool,
) -> None:
    """Run the adapted Table-VII-style comparison.

    The notebook explicitly labels Zeno-RE and FedREDefense-style as
    reconstruction-compatible adaptations rather than faithful original
    implementations. This script preserves that labeling.
    """

    exp_cfg = config["experiments"]["adapted_reconstruction_baselines"]
    seeds = list(config["reproducibility"]["seeds"])
    methods = list(exp_cfg["methods"])
    subset_names = list(prepared.keys())

    if quick:
        seeds = seeds[:1]
        methods = ["RAFA"]
        subset_names = subset_names[:1]

    existing = load_existing(output_path, overwrite)
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(
        existing,
        ["dataset", "scenario", "method", "seed"],
    )

    frac = float(exp_cfg["byzantine_fraction"])

    for subset_name in subset_names:
        source_cfg = config["selected_subsets"][subset_name]
        rafa_params = subset_rafa_params(config, subset_name)

        for scenario_name, scenario_cfg in exp_cfg["scenarios"].items():
            for seed in seeds:
                dataset = refresh_for_seed(
                    prepared[subset_name],
                    config,
                    int(seed),
                )

                for method in methods:
                    key = (
                        source_cfg["dataset_id"],
                        scenario_name,
                        method,
                        int(seed),
                    )
                    if key in done:
                        continue

                    kwargs = common_run_kwargs(
                        config,
                        seed=int(seed),
                        device=device,
                        quick=quick,
                    )
                    kwargs.update(rafa_params)

                    model = build_initial_model(
                        dataset,
                        config,
                        int(seed),
                        device,
                    )

                    attack_name = scenario_cfg["attack"]
                    attack_scale = float(
                        scenario_cfg.get(
                            "attack_scale",
                            scenario_cfg.get("eta", 1.0),
                        )
                    )

                    print(
                        f"{source_cfg['dataset_id']} | {scenario_name} | "
                        f"{method} | seed={seed}",
                        flush=True,
                    )

                    try:
                        with adapted_scoring(method):
                            result = run_federation(
                                dataset_obj=dataset,
                                initial_model=model,
                                # All three comparison methods use the RAFA
                                # accept/weight loop with a different score.
                                aggregator="RAFA",
                                partition="iid",
                                n_byzantine=int(
                                    round(kwargs["n_clients"] * frac)
                                ),
                                attack=attack_name,
                                recon_inflation_scale=(
                                    attack_scale
                                    if attack_name == "recon_inflation"
                                    else 1.0
                                ),
                                sign_flip_eta=(
                                    attack_scale
                                    if attack_name == "sign_flip"
                                    else 5.0
                                ),
                                **kwargs,
                            )

                        row = result_to_row(
                            result,
                            dataset_name=source_cfg["dataset_id"],
                            source_file=source_cfg["file"],
                            experiment="adapted_baselines",
                            scenario=scenario_name,
                            method=method,
                            seed=int(seed),
                            alpha=kwargs["alpha"],
                            m_rec=kwargs["m_rec"],
                            tau=kwargs["tau"],
                        )
                    except Exception as exc:
                        row = error_row(
                            dataset_name=source_cfg["dataset_id"],
                            source_file=source_cfg["file"],
                            experiment="adapted_baselines",
                            scenario=scenario_name,
                            method=method,
                            seed=int(seed),
                            alpha=kwargs["alpha"],
                            m_rec=kwargs["m_rec"],
                            tau=kwargs["tau"],
                            message=str(exc),
                        )

                    rows.append(row)
                    append_and_save(rows, output_path)
                    print(
                        f"    status={row['status']} "
                        f"F1={safe_float(row.get('f1')):.4f} "
                        f"MDR={safe_float(row.get('mdr')):.4f}",
                        flush=True,
                    )

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


def summarize_macro(raw_path: Path, summary_path: Path) -> None:
    if not raw_path.exists():
        return

    df = pd.read_csv(raw_path)
    if df.empty:
        return

    metric_cols = [
        "f1",
        "auc",
        "fpr",
        "asr",
        "mdr",
        "bfrr",
        "mean_aqs_benign",
        "mean_aqs_malicious",
        "aqs_gap",
        "runtime_sec",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    summary = (
        ok.groupby(["experiment", "scenario", "method"], dropna=False)
        .agg(
            runs=("f1", "count"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            auc_mean=("auc", "mean"),
            fpr_mean=("fpr", "mean"),
            asr_mean=("asr", "mean"),
            mdr_mean=("mdr", "mean"),
            bfrr_mean=("bfrr", "mean"),
            aqs_gap_mean=("aqs_gap", "mean"),
            runtime_mean=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)


def write_manifest(
    config_path: Path,
    output_dir: Path,
    device,
    quick: bool,
) -> None:
    manifest = {
        "config": str(config_path.resolve()),
        "device": str(device),
        "quick_mode": bool(quick),
        "repository_root": str(REPO_ROOT),
        "adapted_baseline_note": (
            "Zeno-RE and FedREDefense-style are reconstruction-compatible "
            "adaptations from the development notebook, not faithful "
            "implementations of the original methods."
        ),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    device = resolve_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.config, args.output_dir, device, args.quick)

    print("RAFA CICDDoS2019 reproduction")
    print(f"Config : {args.config.resolve()}")
    print(f"Device : {device}")
    print(f"Output : {args.output_dir.resolve()}")
    if args.quick:
        print("Mode   : QUICK SMOKE TEST (not paper reproduction)")

    prepared = prepare_selected_subsets(config, args.quick)

    if args.experiment in {"all", "matched_robustness"}:
        raw_path = args.output_dir / "matched_robustness.csv"
        run_matched_robustness(
            prepared,
            config,
            output_path=raw_path,
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )
        summarize_macro(
            raw_path,
            args.output_dir / "matched_robustness_summary.csv",
        )

    if args.experiment in {"all", "adapted_baselines"}:
        raw_path = args.output_dir / "adapted_baselines.csv"
        run_adapted_baselines(
            prepared,
            config,
            output_path=raw_path,
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )
        summarize_macro(
            raw_path,
            args.output_dir / "adapted_baselines_summary.csv",
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reproduce the CICIoT2023 experiments from the RAFA NoF 2026 paper.

Run from the repository root, for example:

    python experiments/reproduce_ciciot2023.py

or run one experiment group:

    python experiments/reproduce_ciciot2023.py --experiment reconstruction_targeted

The script reads ``configs/paper_ciciot2023.yaml`` by default and writes
incremental CSV files to ``results/ciciot2023/``.

Important reproducibility note
------------------------------
The paper-producing implementation used ``local_epochs=5`` together with a
global cap of ``max_local_batches=20`` for each client training call. This
script preserves that behavior through ``src.training.run_federation``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# Make "src" importable when this file is executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    add_client_partitions,
    clean_and_prepare_dataset,
    refresh_reference_and_partitions,
)
from src.models import build_model  # noqa: E402
from src.training import run_federation, set_global_seed  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_ciciot2023.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "ciciot2023"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the RAFA CICIoT2023 paper experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to paper_ciciot2023.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated CSV/JSON results.",
    )
    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "reconstruction_targeted",
            "untargeted",
            "adaptive",
            "non_iid_benign",
            "alpha_sensitivity",
            "reference_size_sensitivity",
        ],
        default="all",
        help="Experiment group to run.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke-test mode: one round, one local batch, one setting per "
            "selected experiment. Not a paper reproduction."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing result files instead of resuming them.",
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


def result_to_row(
    result: dict,
    *,
    experiment: str,
    scenario: str,
    seed: int,
    attack_scale: float | None = None,
    byzantine_fraction: float | None = None,
    reference_size: int | None = None,
    alpha: float | None = None,
) -> dict:
    metrics = result["final_metrics"]
    screening = result["screening"]
    aqs = result["aqs_summary"]
    cfg = result["config"]

    return {
        "dataset": "CICIoT2023",
        "experiment": experiment,
        "scenario": scenario,
        "aggregator": result["aggregator"],
        "seed": int(seed),
        "partition": result["partition"],
        "attack": result["attack"],
        "attack_scale": attack_scale,
        "byzantine_fraction": byzantine_fraction,
        "n_byzantine": int(result["n_byzantine"]),
        "reference_size": reference_size,
        "alpha": alpha if alpha is not None else cfg.get("alpha"),
        "m_rec": cfg.get("m_rec"),
        "tau": cfg.get("tau"),
        "fed_rounds": cfg.get("fed_rounds"),
        "local_epochs": cfg.get("local_epochs"),
        "max_local_batches": cfg.get("max_local_batches"),
        "f1": metrics.get("f1"),
        "auc": metrics.get("auc"),
        "fpr": metrics.get("fpr"),
        "asr": metrics.get("asr"),
        "mdr": screening.get("mdr"),
        "bfrr": screening.get("bfrr"),
        "mean_aqs_benign": aqs.get("mean_aqs_benign"),
        "mean_aqs_malicious": aqs.get("mean_aqs_malicious"),
        "aqs_gap": aqs.get("aqs_gap"),
        "threshold": metrics.get("threshold"),
        "runtime_sec": result.get("runtime_sec"),
        "status": "ok",
        "error_message": "",
    }


def error_row(
    *,
    experiment: str,
    scenario: str,
    aggregator: str,
    seed: int,
    message: str,
    attack_scale: float | None = None,
    byzantine_fraction: float | None = None,
    reference_size: int | None = None,
    alpha: float | None = None,
) -> dict:
    lower = str(message).lower()
    status = (
        "diverged"
        if any(x in lower for x in ["non-finite", "nonfinite", "nan", "inf", "diverged"])
        else "error"
    )
    return {
        "dataset": "CICIoT2023",
        "experiment": experiment,
        "scenario": scenario,
        "aggregator": aggregator,
        "seed": int(seed),
        "attack_scale": attack_scale,
        "byzantine_fraction": byzantine_fraction,
        "reference_size": reference_size,
        "alpha": alpha,
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


def build_prepared_dataset(config: dict, seed: int) -> dict:
    dataset_cfg = config["dataset"]
    fed = config["federated"]
    ref = config["reference"]

    dataset = clean_and_prepare_dataset(
        dataset_name=dataset_cfg["name"],
        path=resolve_repo_path(dataset_cfg["path"]),
        label_col=dataset_cfg.get("label_column", "Label"),
        forced_drop_cols=[],
        seed=seed,
        reference_size=int(ref["default_size"]),
        heavy_nan_threshold=0.50,
        drop_non_numeric_features=True,
        drop_rows_after_column_cleanup=True,
        expected_features=dataset_cfg.get("expected_features"),
    )
    return add_client_partitions(
        dataset,
        n_clients=int(fed["n_clients"]),
        batch_size=int(fed["batch_size"]),
        seed=seed,
    )


def build_initial_model(dataset: dict, config: dict, seed: int, device: torch.device):
    model_cfg = config["model"]
    set_global_seed(seed)
    return build_model(
        model_type=model_cfg["type"],
        input_dim=len(dataset["feature_cols"]),
        latent_dim=int(model_cfg["latent_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)


def common_run_kwargs(
    config: dict,
    *,
    seed: int,
    device: torch.device,
    quick: bool,
) -> dict:
    fed = config["federated"]
    rafa = config["rafa"]
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
        "alpha": float(rafa["alpha"]),
        "m_rec": float(rafa["m_rec"]),
        "tau": float(rafa["tau"]),
        "epsilon": float(rafa.get("epsilon", 1e-8)),
        "threshold_percentile": float(evaluation["threshold_percentile"]),
    }


def run_standard_grid(
    dataset: dict,
    config: dict,
    *,
    experiment_name: str,
    scenarios: list[dict],
    output_path: Path,
    device: torch.device,
    quick: bool,
    overwrite: bool,
) -> None:
    existing = load_existing(output_path, overwrite)
    rows = existing.to_dict("records") if not existing.empty else []
    key_fields = [
        "scenario",
        "aggregator",
        "seed",
        "attack_scale",
        "byzantine_fraction",
    ]
    done = completed_keys(existing, key_fields)

    aggregators = list(config["aggregators"])
    if quick:
        aggregators = ["RAFA"]

    seed = int(config["reproducibility"]["seed"])
    run_kwargs = common_run_kwargs(config, seed=seed, device=device, quick=quick)
    initial_model = build_initial_model(dataset, config, seed, device)

    total = len(scenarios) * len(aggregators)
    run_no = 0

    for scenario in scenarios:
        for aggregator in aggregators:
            run_no += 1
            key = (
                scenario["name"],
                aggregator,
                seed,
                scenario.get("attack_scale"),
                scenario.get("byzantine_fraction"),
            )
            if key in done:
                continue

            frac = float(scenario.get("byzantine_fraction", 0.0))
            n_byz = int(round(run_kwargs["n_clients"] * frac))

            print(
                f"[{run_no:03d}/{total:03d}] "
                f"{scenario['name']} | {aggregator} | f={frac:.2f}",
                flush=True,
            )

            try:
                result = run_federation(
                    dataset_obj=dataset,
                    initial_model=initial_model,
                    aggregator=aggregator,
                    partition=scenario.get("partition", "iid"),
                    n_byzantine=n_byz,
                    attack=scenario["attack"],
                    sign_flip_eta=float(scenario.get("sign_flip_eta", 5.0)),
                    recon_inflation_scale=float(
                        scenario.get("attack_scale", 1.0)
                    ),
                    adaptive_attack_scale=float(
                        scenario.get("attack_scale", 1.0)
                    ),
                    **run_kwargs,
                )
                row = result_to_row(
                    result,
                    experiment=experiment_name,
                    scenario=scenario["name"],
                    seed=seed,
                    attack_scale=scenario.get("attack_scale"),
                    byzantine_fraction=frac,
                    reference_size=len(dataset["reference"]),
                )
            except Exception as exc:
                row = error_row(
                    experiment=experiment_name,
                    scenario=scenario["name"],
                    aggregator=aggregator,
                    seed=seed,
                    message=str(exc),
                    attack_scale=scenario.get("attack_scale"),
                    byzantine_fraction=frac,
                    reference_size=len(dataset["reference"]),
                )

            rows.append(row)
            append_and_save(rows, output_path)
            print(
                f"    status={row['status']} "
                f"F1={safe_float(row.get('f1')):.4f} "
                f"MDR={safe_float(row.get('mdr')):.4f} "
                f"BFRR={safe_float(row.get('bfrr')):.4f}",
                flush=True,
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def scenarios_reconstruction(config: dict, quick: bool) -> list[dict]:
    exp = config["experiments"]["reconstruction_targeted"]
    scales = list(exp["attack_scales"])
    fractions = list(exp["byzantine_fractions"])
    if quick:
        scales = scales[:1]
        fractions = fractions[:1]

    return [
        {
            "name": f"recon_targeted_s{scale:g}_f{frac:g}",
            "partition": exp["partition"],
            "attack": exp["attack"],
            "attack_scale": float(scale),
            "byzantine_fraction": float(frac),
        }
        for scale in scales
        for frac in fractions
    ]


def scenarios_untargeted(config: dict, quick: bool) -> list[dict]:
    exp = config["experiments"]["untargeted"]
    attacks = list(exp["attacks"])
    fractions = list(exp["byzantine_fractions"])
    if quick:
        attacks = attacks[:1]
        fractions = fractions[:1]

    rows = []
    for attack in attacks:
        for frac in fractions:
            rows.append(
                {
                    "name": f"{attack}_f{frac:g}",
                    "partition": exp["partition"],
                    "attack": attack,
                    "attack_scale": None,
                    "byzantine_fraction": float(frac),
                    "sign_flip_eta": float(
                        exp.get("sign_flip", {}).get("eta", 5.0)
                    ),
                }
            )
    return rows


def scenarios_adaptive(config: dict, quick: bool) -> list[dict]:
    exp = config["experiments"]["adaptive"]
    variants = list(exp["variants"])
    scales = list(exp["attack_scales"])
    fractions = list(exp["byzantine_fractions"])
    if quick:
        variants = variants[:1]
        scales = scales[:1]
        fractions = fractions[:1]

    return [
        {
            "name": f"{variant}_s{scale:g}_f{frac:g}",
            "partition": exp["partition"],
            "attack": variant,
            "attack_scale": float(scale),
            "byzantine_fraction": float(frac),
        }
        for variant in variants
        for scale in scales
        for frac in fractions
    ]


def scenarios_noniid(config: dict) -> list[dict]:
    exp = config["experiments"]["non_iid_benign"]
    return [
        {
            "name": "noniid_no_attack",
            "partition": exp["partition"],
            "attack": exp["attack"],
            "attack_scale": None,
            "byzantine_fraction": 0.0,
        }
    ]


def run_alpha_sensitivity(
    dataset: dict,
    config: dict,
    output_path: Path,
    device: torch.device,
    quick: bool,
    overwrite: bool,
) -> None:
    """Re-run the balanced/conservative alpha comparison.

    The paper reports the sensitivity of RAFA to alpha while holding
    ``m_rec=0.01`` and ``tau=0.2`` fixed. The public configuration stores
    alpha values 10 and 15. This script evaluates them under the strong
    reconstruction-targeted setting (scale=5, f/N=0.4).
    """

    exp = config["experiments"]["alpha_sensitivity"]
    alphas = list(exp["alpha_values"])
    if quick:
        alphas = alphas[:1]

    existing = load_existing(output_path, overwrite)
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(existing, ["alpha", "seed"])

    seed = int(config["reproducibility"]["seed"])
    base_kwargs = common_run_kwargs(config, seed=seed, device=device, quick=quick)
    initial_model = build_initial_model(dataset, config, seed, device)

    for alpha in alphas:
        key = (float(alpha), seed)
        if key in done:
            continue

        kwargs = dict(base_kwargs)
        kwargs["alpha"] = float(alpha)
        kwargs["m_rec"] = float(exp["fixed_m_rec"])
        kwargs["tau"] = float(exp["fixed_tau"])

        try:
            result = run_federation(
                dataset_obj=dataset,
                initial_model=initial_model,
                aggregator="RAFA",
                partition="iid",
                n_byzantine=int(round(kwargs["n_clients"] * 0.4)),
                attack="recon_inflation",
                recon_inflation_scale=5.0,
                **kwargs,
            )
            row = result_to_row(
                result,
                experiment="alpha_sensitivity",
                scenario="recon_targeted_s5_f0.4",
                seed=seed,
                attack_scale=5.0,
                byzantine_fraction=0.4,
                reference_size=len(dataset["reference"]),
                alpha=float(alpha),
            )
        except Exception as exc:
            row = error_row(
                experiment="alpha_sensitivity",
                scenario="recon_targeted_s5_f0.4",
                aggregator="RAFA",
                seed=seed,
                message=str(exc),
                attack_scale=5.0,
                byzantine_fraction=0.4,
                reference_size=len(dataset["reference"]),
                alpha=float(alpha),
            )

        rows.append(row)
        append_and_save(rows, output_path)
        print(
            f"alpha={alpha}: status={row['status']} "
            f"F1={safe_float(row.get('f1')):.4f} "
            f"MDR={safe_float(row.get('mdr')):.4f} "
            f"BFRR={safe_float(row.get('bfrr')):.4f}",
            flush=True,
        )


def run_reference_size_sensitivity(
    base_dataset: dict,
    config: dict,
    output_path: Path,
    device: torch.device,
    quick: bool,
    overwrite: bool,
) -> None:
    exp = config["experiments"]["reference_size_sensitivity"]
    sizes = list(exp["reference_sizes"])
    seeds = list(exp["seeds"])
    if quick:
        sizes = sizes[:1]
        seeds = seeds[:1]

    existing = load_existing(output_path, overwrite)
    rows = existing.to_dict("records") if not existing.empty else []
    done = completed_keys(existing, ["reference_size", "seed"])

    fed = config["federated"]
    base_for_resampling = copy.deepcopy(base_dataset)
    base_for_resampling["review_benign_pool"] = pd.concat(
        [base_for_resampling["train"], base_for_resampling["reference"]],
        axis=0,
    ).reset_index(drop=True)

    for ref_size in sizes:
        for seed in seeds:
            key = (int(ref_size), int(seed))
            if key in done:
                continue

            dataset = copy.deepcopy(base_for_resampling)
            dataset = refresh_reference_and_partitions(
                dataset,
                seed=int(seed),
                n_clients=int(fed["n_clients"]),
                batch_size=int(fed["batch_size"]),
                reference_size=int(ref_size),
            )

            model = build_initial_model(dataset, config, int(seed), device)
            kwargs = common_run_kwargs(
                config,
                seed=int(seed),
                device=device,
                quick=quick,
            )
            kwargs.update(
                {
                    "fed_rounds": 1 if quick else int(exp["rounds"]),
                    "max_local_batches": (
                        1 if quick else int(exp["max_local_batches"])
                    ),
                    "alpha": float(exp["alpha"]),
                    "m_rec": float(exp["m_rec"]),
                    "tau": float(exp["tau"]),
                }
            )

            frac = float(exp["byzantine_fraction"])
            try:
                result = run_federation(
                    dataset_obj=dataset,
                    initial_model=model,
                    aggregator="RAFA",
                    partition=exp["partition"],
                    n_byzantine=int(round(kwargs["n_clients"] * frac)),
                    attack=exp["attack"],
                    recon_inflation_scale=float(exp["attack_scale"]),
                    **kwargs,
                )
                row = result_to_row(
                    result,
                    experiment="reference_size_sensitivity",
                    scenario="recon_targeted_s5_f0.4",
                    seed=int(seed),
                    attack_scale=float(exp["attack_scale"]),
                    byzantine_fraction=frac,
                    reference_size=int(ref_size),
                    alpha=float(exp["alpha"]),
                )
            except Exception as exc:
                row = error_row(
                    experiment="reference_size_sensitivity",
                    scenario="recon_targeted_s5_f0.4",
                    aggregator="RAFA",
                    seed=int(seed),
                    message=str(exc),
                    attack_scale=float(exp["attack_scale"]),
                    byzantine_fraction=frac,
                    reference_size=int(ref_size),
                    alpha=float(exp["alpha"]),
                )

            rows.append(row)
            append_and_save(rows, output_path)
            print(
                f"|R|={ref_size}, seed={seed}: "
                f"status={row['status']} F1={safe_float(row.get('f1')):.4f}",
                flush=True,
            )


def write_manifest(
    config_path: Path,
    output_dir: Path,
    device: torch.device,
    quick: bool,
) -> None:
    manifest = {
        "config": str(config_path.resolve()),
        "device": str(device),
        "quick_mode": bool(quick),
        "repository_root": str(REPO_ROOT),
        "note": (
            "quick_mode=true is a smoke test and does not reproduce "
            "the reported paper results."
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

    print("RAFA CICIoT2023 reproduction")
    print(f"Config : {args.config.resolve()}")
    print(f"Device : {device}")
    print(f"Output : {args.output_dir.resolve()}")
    if args.quick:
        print("Mode   : QUICK SMOKE TEST (not paper reproduction)")

    seed = int(config["reproducibility"]["seed"])
    print("\nPreparing CICIoT2023 ...", flush=True)
    dataset = build_prepared_dataset(config, seed)

    selected = (
        [
            "reconstruction_targeted",
            "untargeted",
            "adaptive",
            "non_iid_benign",
            "alpha_sensitivity",
            "reference_size_sensitivity",
        ]
        if args.experiment == "all"
        else [args.experiment]
    )

    if "reconstruction_targeted" in selected:
        run_standard_grid(
            dataset,
            config,
            experiment_name="reconstruction_targeted",
            scenarios=scenarios_reconstruction(config, args.quick),
            output_path=args.output_dir / "reconstruction_targeted.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    if "untargeted" in selected:
        run_standard_grid(
            dataset,
            config,
            experiment_name="untargeted",
            scenarios=scenarios_untargeted(config, args.quick),
            output_path=args.output_dir / "untargeted.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    if "adaptive" in selected:
        run_standard_grid(
            dataset,
            config,
            experiment_name="adaptive",
            scenarios=scenarios_adaptive(config, args.quick),
            output_path=args.output_dir / "adaptive.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    if "non_iid_benign" in selected:
        run_standard_grid(
            dataset,
            config,
            experiment_name="non_iid_benign",
            scenarios=scenarios_noniid(config),
            output_path=args.output_dir / "non_iid_benign.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    if "alpha_sensitivity" in selected:
        run_alpha_sensitivity(
            dataset,
            config,
            output_path=args.output_dir / "alpha_sensitivity.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    if "reference_size_sensitivity" in selected:
        run_reference_size_sensitivity(
            dataset,
            config,
            output_path=args.output_dir / "reference_size_sensitivity.csv",
            device=device,
            quick=args.quick,
            overwrite=args.overwrite,
        )

    print("\nFinished.")


if __name__ == "__main__":
    main()

"""Dataset loading, preprocessing, and client partitioning for RAFA.

The functions reproduce the preprocessing logic used in the NoF 2026
development notebook while using project-relative paths and explicit arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


CICDDOS_DROP_COLS = {
    "Unnamed: 0",
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Timestamp",
    "SimillarHTTP",
}


def infer_binary_label(label_series: pd.Series) -> pd.Series:
    label_text = label_series.astype(str).str.lower()
    benign_mask = label_text.str.contains("benign|normal", regex=True, na=False)
    return (~benign_mask).astype(int)


def map_ciciot_attack_category(label: object) -> str:
    label = str(label).lower()

    if "benign" in label:
        return "Benign"
    if "ddos" in label:
        return "DDoS"
    if "dos" in label:
        return "DoS"
    if "mirai" in label:
        return "Mirai"
    if "recon" in label:
        return "Reconnaissance"
    if "spoof" in label:
        return "Spoofing"
    if "brute" in label or "password" in label:
        return "Brute Force"
    if "web" in label or "upload" in label or "xss" in label or "sql" in label:
        return "Web-based"
    return "Other"


def map_attack_category(dataset_name: str, label: object) -> str:
    if dataset_name == "CICIoT2023":
        return map_ciciot_attack_category(label)

    label_text = str(label)
    if label_text.lower() in {"benign", "normal"}:
        return "Benign"
    return "Other"


def clean_and_prepare_dataset(
    dataset_name: str,
    path: str | Path,
    label_col: str = "Label",
    forced_drop_cols: Iterable[str] | None = None,
    seed: int = 42,
    reference_size: int = 500,
    heavy_nan_threshold: float = 0.50,
    drop_non_numeric_features: bool = True,
    drop_rows_after_column_cleanup: bool = True,
    expected_features: int | None = None,
) -> dict:
    """Load and preprocess the main CICIoT2023-style CSV dataset."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df_raw = pd.read_csv(path, low_memory=True)
    raw_rows = len(df_raw)

    if label_col not in df_raw.columns:
        raise ValueError(f"Label column '{label_col}' not found in {dataset_name}")

    forced_drop_cols = list(forced_drop_cols or [])
    forced_drop_cols = [
        c for c in forced_drop_cols
        if c in df_raw.columns and c != label_col
    ]

    candidate_feature_cols = [
        c for c in df_raw.columns
        if c != label_col and c not in forced_drop_cols
    ]

    nan_ratio = df_raw[candidate_feature_cols].isna().mean()
    heavy_nan_cols = nan_ratio[nan_ratio > heavy_nan_threshold].index.tolist()

    candidates_after_nan = [
        c for c in candidate_feature_cols if c not in heavy_nan_cols
    ]
    non_numeric_cols = [
        c for c in candidates_after_nan
        if not pd.api.types.is_numeric_dtype(df_raw[c])
    ]

    if drop_non_numeric_features:
        feature_cols = [
            c for c in candidates_after_nan if c not in non_numeric_cols
        ]
    else:
        feature_cols = candidates_after_nan

    excluded_cols = sorted(
        set(forced_drop_cols + heavy_nan_cols + non_numeric_cols)
    )

    df = df_raw[feature_cols + [label_col]].copy()
    del df_raw

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if drop_rows_after_column_cleanup:
        before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
        row_drops = before - len(df)
    else:
        row_drops = 0

    df["binary_label"] = infer_binary_label(df[label_col])
    df["attack_category"] = df[label_col].apply(
        lambda value: map_attack_category(dataset_name, value)
    )

    benign_df = df[df["binary_label"] == 0].copy().reset_index(drop=True)
    attack_df = df[df["binary_label"] == 1].copy().reset_index(drop=True)

    if benign_df.empty:
        raise ValueError(f"No benign samples found in {dataset_name}")
    if attack_df.empty:
        raise ValueError(f"No attack samples found in {dataset_name}")

    benign_train_pool, benign_temp = train_test_split(
        benign_df,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )
    benign_val, benign_test = train_test_split(
        benign_temp,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    benign_train_pool = benign_train_pool.reset_index(drop=True)
    benign_val = benign_val.reset_index(drop=True)
    benign_test = benign_test.reset_index(drop=True)

    ref_size = min(int(reference_size), len(benign_train_pool))
    reference_df = benign_train_pool.sample(n=ref_size, random_state=seed)
    train_df = benign_train_pool.drop(index=reference_df.index).reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)

    test_df = pd.concat([benign_test, attack_df], axis=0).reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])

    def apply_scaler(split_df: pd.DataFrame) -> pd.DataFrame:
        out = split_df.copy()
        out[feature_cols] = scaler.transform(out[feature_cols])
        return out.reset_index(drop=True)

    train_df = apply_scaler(train_df)
    reference_df = apply_scaler(reference_df)
    benign_val = apply_scaler(benign_val)
    test_df = apply_scaler(test_df)

    attack_counts = (
        test_df.loc[test_df["binary_label"] == 1, "attack_category"]
        .value_counts()
        .to_dict()
    )

    return {
        "name": dataset_name,
        "path": str(path),
        "label_col": label_col,
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "expected_features": expected_features,
        "forced_drop_cols": forced_drop_cols,
        "heavy_nan_cols": heavy_nan_cols,
        "non_numeric_cols": non_numeric_cols,
        "excluded_cols": excluded_cols,
        "row_drops": row_drops,
        "scaler": scaler,
        "train": train_df,
        "reference": reference_df,
        "val": benign_val,
        "test": test_df,
        "attack_counts": attack_counts,
        "benign_train_count": len(train_df),
        "benign_ref_count": len(reference_df),
        "benign_val_count": len(benign_val),
        "benign_test_count": int((test_df["binary_label"] == 0).sum()),
        "attack_test_count": int((test_df["binary_label"] == 1).sum()),
        "scaled_train_min": float(np.nanmin(train_df[feature_cols].to_numpy())),
        "scaled_train_max": float(np.nanmax(train_df[feature_cols].to_numpy())),
        "scaled_test_min": float(np.nanmin(test_df[feature_cols].to_numpy())),
        "scaled_test_max": float(np.nanmax(test_df[feature_cols].to_numpy())),
    }


def make_loader_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def iid_partition(
    df: pd.DataFrame,
    n_clients: int,
    seed: int,
) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    splits = np.array_split(indices, n_clients)
    return {
        cid: df.iloc[idx].reset_index(drop=True)
        for cid, idx in enumerate(splits)
    }


def choose_noniid_key(
    df: pd.DataFrame,
    feature_cols: list[str],
    preferred_names: list[str] | None = None,
) -> str:
    preferred_names = preferred_names or [
        "Protocol Type",
        "protocol_type",
        "Protocol",
        "Proto",
        "sTtl",
        "Rate",
    ]

    lower_to_real = {c.lower(): c for c in feature_cols}
    for name in preferred_names:
        if name.lower() in lower_to_real:
            return lower_to_real[name.lower()]

    variances = df[feature_cols].var(numeric_only=True).sort_values(ascending=False)
    if variances.empty:
        raise ValueError("Cannot select a non-IID key from an empty feature set.")
    return str(variances.index[0])


def make_group_labels(values: pd.Series, n_bins: int = 10) -> pd.Series:
    if values.nunique(dropna=True) <= n_bins:
        return values.astype(str)

    ranked = values.rank(method="first")
    return pd.qcut(
        ranked,
        q=n_bins,
        labels=False,
        duplicates="drop",
    ).astype(str)


def balanced_shard_noniid_partition(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clients: int,
    seed: int,
    shards_per_client: int = 3,
    n_bins: int = 10,
) -> tuple[dict[int, pd.DataFrame], str]:
    rng = np.random.default_rng(seed)

    key_col = choose_noniid_key(df, feature_cols)
    group_labels = make_group_labels(df[key_col], n_bins=n_bins)

    work = df.copy()
    work["_noniid_group"] = group_labels.values
    work["_rand"] = rng.random(len(work))
    work = work.sort_values(["_noniid_group", "_rand"]).reset_index(drop=True)

    n_shards = n_clients * shards_per_client
    shard_indices = np.array_split(np.arange(len(work)), n_shards)
    shard_order = np.arange(n_shards)
    rng.shuffle(shard_order)

    client_indices = {cid: [] for cid in range(n_clients)}
    for pos, shard_id in enumerate(shard_order):
        cid = pos % n_clients
        client_indices[cid].extend(shard_indices[shard_id].tolist())

    partitions: dict[int, pd.DataFrame] = {}
    for cid in range(n_clients):
        idx = np.asarray(client_indices[cid], dtype=int)
        rng.shuffle(idx)
        partitions[cid] = (
            work.iloc[idx]
            .drop(columns=["_noniid_group", "_rand"])
            .reset_index(drop=True)
        )

    return partitions, key_col


def summarize_partitions(partitions: dict[int, pd.DataFrame]) -> dict:
    counts = np.array([len(v) for v in partitions.values()])
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "empty_clients": int((counts == 0).sum()),
    }


def summarize_noniid_heterogeneity(
    partitions: dict[int, pd.DataFrame],
    key_col: str,
    n_bins: int = 10,
) -> dict:
    top_shares = []

    for part in partitions.values():
        labels = make_group_labels(part[key_col], n_bins=n_bins)
        shares = labels.value_counts(normalize=True)
        top_shares.append(float(shares.max()))

    top_shares = np.asarray(top_shares)
    return {
        "avg_top_group_share": float(top_shares.mean()),
        "min_top_group_share": float(top_shares.min()),
        "max_top_group_share": float(top_shares.max()),
    }


def add_client_partitions(
    dataset_obj: dict,
    n_clients: int = 10,
    batch_size: int = 256,
    seed: int = 42,
) -> dict:
    """Attach IID and non-IID partitions/loaders to a prepared dataset."""

    feature_cols = dataset_obj["feature_cols"]
    train_df = dataset_obj["train"]

    iid_parts = iid_partition(train_df, n_clients=n_clients, seed=seed)
    noniid_parts, noniid_key_col = balanced_shard_noniid_partition(
        train_df,
        feature_cols=feature_cols,
        n_clients=n_clients,
        seed=seed,
        shards_per_client=3,
        n_bins=n_clients,
    )

    dataset_obj["iid_parts"] = iid_parts
    dataset_obj["noniid_parts"] = noniid_parts
    dataset_obj["iid_loaders"] = {
        cid: make_loader_from_df(part, feature_cols, batch_size, shuffle=True)
        for cid, part in iid_parts.items()
    }
    dataset_obj["noniid_loaders"] = {
        cid: make_loader_from_df(part, feature_cols, batch_size, shuffle=True)
        for cid, part in noniid_parts.items()
    }
    dataset_obj["partition_summary"] = {
        "iid": summarize_partitions(iid_parts),
        "noniid": summarize_partitions(noniid_parts),
        "noniid_key_col": noniid_key_col,
        "noniid_heterogeneity": summarize_noniid_heterogeneity(
            noniid_parts,
            noniid_key_col,
            n_bins=n_clients,
        ),
    }
    return dataset_obj


def refresh_reference_and_partitions(
    dataset_obj: dict,
    seed: int,
    n_clients: int = 10,
    batch_size: int = 256,
    reference_size: int | None = None,
) -> dict:
    """Re-sample the trusted reference set and client partitions for a seed."""

    if "review_benign_pool" not in dataset_obj:
        dataset_obj["review_benign_pool"] = pd.concat(
            [dataset_obj["train"], dataset_obj["reference"]],
            axis=0,
        ).reset_index(drop=True)

    pool = dataset_obj["review_benign_pool"].copy().reset_index(drop=True)
    ref_size = min(
        int(reference_size or len(dataset_obj["reference"])),
        len(pool),
    )

    reference_df = pool.sample(n=ref_size, random_state=int(seed))
    train_df = pool.drop(index=reference_df.index).reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)

    dataset_obj["train"] = train_df
    dataset_obj["reference"] = reference_df
    dataset_obj["benign_train_count"] = len(train_df)
    dataset_obj["benign_ref_count"] = len(reference_df)

    return add_client_partitions(
        dataset_obj,
        n_clients=n_clients,
        batch_size=batch_size,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# CICDDoS2019 helpers
# ---------------------------------------------------------------------------

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_label_col(columns: Iterable[object]) -> object:
    for col in columns:
        if str(col).strip().lower() == "label":
            return col
    raise ValueError("No Label column found.")


def safe_cicddos_dataset_name(csv_name: str) -> str:
    stem = Path(csv_name).stem.replace("DrDoS_", "")
    return f"CICDDoS2019_{stem}"


def _raw_col_map(path: Path) -> dict[str, object]:
    header_raw = pd.read_csv(path, nrows=0, low_memory=True)
    return {str(c).strip(): c for c in header_raw.columns}


def scan_cicddos_file(
    root: str | Path,
    csv_name: str,
    chunksize: int = 100_000,
    min_benign: int = 2_000,
    min_attack: int = 2_000,
) -> dict:
    """Inspect one CICDDoS2019 CSV without loading it fully into memory."""

    root = Path(root)
    path = root / csv_name

    if not path.exists():
        return {
            "file": csv_name,
            "path": str(path),
            "exists": False,
            "usable": False,
            "reason": "missing_file",
        }

    header_raw = pd.read_csv(path, nrows=0, low_memory=True)
    raw_label_col = _find_label_col(header_raw.columns)

    sample = _strip_columns(pd.read_csv(path, nrows=2_000, low_memory=True))
    label_col = "Label"

    candidate_cols = [
        c for c in sample.columns
        if c != label_col and c not in CICDDOS_DROP_COLS
    ]

    numeric_cols = []
    for col in candidate_cols:
        converted = pd.to_numeric(sample[col], errors="coerce")
        if converted.notna().mean() > 0.95:
            numeric_cols.append(col)

    benign_count = 0
    attack_count = 0
    attack_labels: set[str] = set()
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        usecols=[raw_label_col],
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = _strip_columns(chunk)
        labels = chunk[label_col].astype(str).str.strip()
        benign_mask = labels.str.lower().eq("benign")

        benign_count += int(benign_mask.sum())
        attack_count += int((~benign_mask).sum())
        total_rows += len(labels)
        attack_labels.update(labels.loc[~benign_mask].dropna().unique().tolist())

    usable = (
        benign_count >= min_benign
        and attack_count >= min_attack
        and len(numeric_cols) > 0
    )

    if benign_count < min_benign:
        reason = "too_few_benign"
    elif attack_count < min_attack:
        reason = "too_few_attack"
    elif not numeric_cols:
        reason = "no_numeric_features"
    else:
        reason = "selected_candidate"

    return {
        "file": csv_name,
        "path": str(path),
        "exists": True,
        "usable": usable,
        "reason": reason,
        "raw_label_col": raw_label_col,
        "label_col": label_col,
        "rows": int(total_rows),
        "benign": int(benign_count),
        "attack": int(attack_count),
        "attack_labels": sorted(attack_labels),
        "numeric_feature_count": len(numeric_cols),
        "numeric_cols": numeric_cols,
    }


def common_numeric_features(scan_infos: list[dict]) -> list[str]:
    usable = [x for x in scan_infos if x.get("usable", False)]
    if not usable:
        raise ValueError("No usable CICDDoS2019 scan information.")
    common = set(usable[0]["numeric_cols"])
    for info in usable[1:]:
        common &= set(info["numeric_cols"])
    common = sorted(common)
    if not common:
        raise ValueError("Selected CICDDoS2019 subsets have no common numeric features.")
    return common


def _load_capped_cicddos_frame(
    scan_info: dict,
    feature_cols: list[str],
    seed: int,
    max_benign_rows: int = 120_000,
    max_attack_rows: int = 300_000,
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, int, int]:
    path = Path(scan_info["path"])
    raw_label_col = scan_info["raw_label_col"]
    label_col = scan_info["label_col"]

    col_map = _raw_col_map(path)
    raw_feature_cols = [col_map[c] for c in feature_cols if c in col_map]
    usecols = raw_feature_cols + [raw_label_col]

    benign_chunks = []
    attack_chunks = []
    benign_seen = 0
    attack_seen = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = _strip_columns(chunk)
        labels = chunk[label_col].astype(str).str.strip()
        benign_mask = labels.str.lower().eq("benign")

        if benign_seen < max_benign_rows:
            take = chunk.loc[benign_mask]
            remaining = max_benign_rows - benign_seen
            if len(take) > remaining:
                take = take.sample(n=remaining, random_state=seed + benign_seen)
            if len(take):
                benign_chunks.append(take)
                benign_seen += len(take)

        if attack_seen < max_attack_rows:
            take = chunk.loc[~benign_mask]
            remaining = max_attack_rows - attack_seen
            if len(take) > remaining:
                take = take.sample(n=remaining, random_state=seed + attack_seen)
            if len(take):
                attack_chunks.append(take)
                attack_seen += len(take)

        if benign_seen >= max_benign_rows and attack_seen >= max_attack_rows:
            break

    if not benign_chunks:
        raise ValueError(f"No benign rows loaded from {scan_info['file']}")
    if not attack_chunks:
        raise ValueError(f"No attack rows loaded from {scan_info['file']}")

    benign_df = pd.concat(benign_chunks, axis=0).reset_index(drop=True)
    attack_df = pd.concat(attack_chunks, axis=0).reset_index(drop=True)
    return (
        pd.concat([benign_df, attack_df], axis=0).reset_index(drop=True),
        len(benign_df),
        len(attack_df),
    )


def prepare_cicddos_subset(
    scan_info: dict,
    feature_cols: list[str],
    seed: int = 42,
    reference_size: int = 500,
    min_benign: int = 2_000,
    min_attack: int = 2_000,
    max_benign_rows: int = 120_000,
    max_attack_rows: int = 300_000,
    chunksize: int = 100_000,
) -> dict:
    """Prepare one source-specific CICDDoS2019 subset."""

    dataset_name = safe_cicddos_dataset_name(scan_info["file"])
    label_col = scan_info["label_col"]

    df_raw, loaded_benign, loaded_attack = _load_capped_cicddos_frame(
        scan_info,
        feature_cols,
        seed=seed,
        max_benign_rows=max_benign_rows,
        max_attack_rows=max_attack_rows,
        chunksize=chunksize,
    )

    for col in feature_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    labels = df_raw[label_col].astype(str).str.strip()
    df_raw["binary_label"] = (~labels.str.lower().eq("benign")).astype(int)
    df_raw["attack_category"] = np.where(
        df_raw["binary_label"] == 0,
        "Benign",
        labels,
    )

    before = len(df_raw)
    df = (
        df_raw.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + ["binary_label"])
        .reset_index(drop=True)
    )
    row_drops = before - len(df)

    benign_df = df[df["binary_label"] == 0].copy().reset_index(drop=True)
    attack_df = df[df["binary_label"] == 1].copy().reset_index(drop=True)

    if len(benign_df) < min_benign:
        raise ValueError(f"Too few clean benign rows: {len(benign_df)}")
    if len(attack_df) < min_attack:
        raise ValueError(f"Too few clean attack rows: {len(attack_df)}")

    benign_train_pool, benign_temp = train_test_split(
        benign_df,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )
    benign_val, benign_test = train_test_split(
        benign_temp,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    benign_train_pool = benign_train_pool.reset_index(drop=True)
    benign_val = benign_val.reset_index(drop=True)
    benign_test = benign_test.reset_index(drop=True)

    ref_size = min(reference_size, len(benign_train_pool))
    reference_df = benign_train_pool.sample(n=ref_size, random_state=seed)
    train_df = benign_train_pool.drop(index=reference_df.index).reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)
    test_df = pd.concat([benign_test, attack_df], axis=0).reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])

    def apply_scaler(split_df: pd.DataFrame) -> pd.DataFrame:
        out = split_df.copy()
        out[feature_cols] = scaler.transform(out[feature_cols])
        return out.reset_index(drop=True)

    train_df = apply_scaler(train_df)
    reference_df = apply_scaler(reference_df)
    benign_val = apply_scaler(benign_val)
    test_df = apply_scaler(test_df)

    return {
        "name": dataset_name,
        "path": scan_info["path"],
        "label_col": label_col,
        "raw_rows": len(df_raw),
        "clean_rows": len(df),
        "loaded_benign_rows": loaded_benign,
        "loaded_attack_rows": loaded_attack,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "expected_features": None,
        "forced_drop_cols": sorted(CICDDOS_DROP_COLS),
        "heavy_nan_cols": [],
        "non_numeric_cols": [],
        "excluded_cols": sorted(CICDDOS_DROP_COLS),
        "row_drops": row_drops,
        "scaler": scaler,
        "train": train_df,
        "reference": reference_df,
        "val": benign_val,
        "test": test_df,
        "attack_counts": (
            test_df.loc[test_df["binary_label"] == 1, "attack_category"]
            .value_counts()
            .to_dict()
        ),
        "benign_train_count": len(train_df),
        "benign_ref_count": len(reference_df),
        "benign_val_count": len(benign_val),
        "benign_test_count": int((test_df["binary_label"] == 0).sum()),
        "attack_test_count": int((test_df["binary_label"] == 1).sum()),
        "scaled_train_min": float(np.nanmin(train_df[feature_cols].to_numpy())),
        "scaled_train_max": float(np.nanmax(train_df[feature_cols].to_numpy())),
        "scaled_test_min": float(np.nanmin(test_df[feature_cols].to_numpy())),
        "scaled_test_max": float(np.nanmax(test_df[feature_cols].to_numpy())),
    }

"""Evaluation and update-screening metrics for RAFA experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.data import make_loader_from_df


def reconstruction_errors(model, data_loader, model_type: str, device) -> np.ndarray:
    model.eval()
    errors = []

    with torch.no_grad():
        for (x,) in data_loader:
            x = x.to(device)

            if model_type.upper() == "VAE":
                x_hat, _, _ = model(x)
            else:
                x_hat = model(x)

            batch_errors = torch.mean((x_hat - x) ** 2, dim=1)
            errors.append(batch_errors.detach().cpu().numpy())

    return np.concatenate(errors) if errors else np.array([])


def calibrate_threshold(
    model,
    benign_val_df,
    feature_cols,
    model_type: str,
    device,
    percentile: float = 95,
) -> float:
    val_loader = make_loader_from_df(
        benign_val_df,
        feature_cols,
        batch_size=1024,
        shuffle=False,
    )
    val_errors = reconstruction_errors(
        model,
        val_loader,
        model_type,
        device,
    )

    if len(val_errors) == 0:
        raise ValueError("Validation set is empty, cannot calibrate threshold.")

    finite_errors = val_errors[np.isfinite(val_errors)]
    if len(finite_errors) == 0:
        raise ValueError("Validation reconstruction errors are all non-finite.")

    return float(np.percentile(finite_errors, percentile))


def evaluate_global_model(
    model,
    test_df,
    feature_cols,
    model_type: str,
    threshold: float,
    device,
) -> dict:
    test_loader = make_loader_from_df(
        test_df,
        feature_cols,
        batch_size=2048,
        shuffle=False,
    )
    errors = reconstruction_errors(model, test_loader, model_type, device)
    y_true = test_df["binary_label"].values.astype(int)

    if len(errors) != len(y_true):
        raise ValueError(
            f"Error/label length mismatch: {len(errors)} vs {len(y_true)}"
        )

    finite_mask = np.isfinite(errors)
    nonfinite_count = int((~finite_mask).sum())

    if nonfinite_count > 0:
        finite_errors = errors[finite_mask]
        replacement = (
            float(np.nanmax(finite_errors))
            if len(finite_errors)
            else threshold
        )
        errors = np.where(finite_mask, errors, replacement)

    y_pred = (errors > threshold).astype(int)

    f1 = f1_score(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    try:
        auc = roc_auc_score(y_true, errors)
    except ValueError:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn + 1e-12)
    # ASR follows the paper/notebook definition: attack samples whose
    # reconstruction error remains below the anomaly threshold.
    asr = fn / (fn + tp + 1e-12)

    return {
        "f1": float(f1),
        "auc": float(auc),
        "fpr": float(fpr),
        "asr": float(asr),
        "threshold": float(threshold),
        "mean_re_benign": (
            float(errors[y_true == 0].mean())
            if np.any(y_true == 0)
            else np.nan
        ),
        "mean_re_attack": (
            float(errors[y_true == 1].mean())
            if np.any(y_true == 1)
            else np.nan
        ),
        "nonfinite_re_count": nonfinite_count,
    }


def compute_screening_metrics(
    aqs_history,
    byzantine_ids,
    tau: float,
) -> dict:
    if not aqs_history:
        return {"mdr": np.nan, "bfrr": np.nan}

    malicious_flags = []
    benign_flags = []
    byz = set(byzantine_ids)

    for round_scores in aqs_history:
        for cid, score in enumerate(round_scores):
            rejected = score < tau
            if cid in byz:
                malicious_flags.append(rejected)
            else:
                benign_flags.append(rejected)

    mdr = np.mean(malicious_flags) if malicious_flags else np.nan
    bfrr = np.mean(benign_flags) if benign_flags else np.nan

    return {
        "mdr": float(mdr) if not np.isnan(mdr) else np.nan,
        "bfrr": float(bfrr) if not np.isnan(bfrr) else np.nan,
    }


def summarize_aqs_history(aqs_history, byzantine_ids) -> dict:
    if not aqs_history:
        return {
            "mean_aqs_benign": np.nan,
            "mean_aqs_malicious": np.nan,
            "aqs_gap": np.nan,
        }

    byz = set(byzantine_ids)
    benign_scores = []
    malicious_scores = []

    for round_scores in aqs_history:
        for cid, score in enumerate(round_scores):
            if cid in byz:
                malicious_scores.append(score)
            else:
                benign_scores.append(score)

    benign_scores = np.asarray(benign_scores, dtype=float)
    malicious_scores = np.asarray(malicious_scores, dtype=float)

    mean_benign = (
        float(np.nanmean(benign_scores))
        if len(benign_scores)
        else np.nan
    )
    mean_malicious = (
        float(np.nanmean(malicious_scores))
        if len(malicious_scores)
        else np.nan
    )

    return {
        "mean_aqs_benign": mean_benign,
        "mean_aqs_malicious": mean_malicious,
        "aqs_gap": (
            mean_benign - mean_malicious
            if np.isfinite(mean_benign) and np.isfinite(mean_malicious)
            else np.nan
        ),
    }

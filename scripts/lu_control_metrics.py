"""Source-compatible benchmark metrics for paper LU controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

def binary_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= threshold
    matrix = confusion_matrix(labels, prediction, labels=[0, 1]).astype(int)
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
        "AUROC": float(roc_auc_score(labels, probability)),
        "AP_ent": float(average_precision_score(labels, probability)),
        "AP_sep": float(average_precision_score(1 - labels, 1.0 - probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "confusion_matrix_sep_ent": matrix.tolist(),
        "threshold": float(threshold),
    }


def _ranking(probability: np.ndarray, negative: np.ndarray, positive: np.ndarray) -> dict[str, Any] | None:
    if not negative.any() or not positive.any():
        return None
    chosen = negative | positive
    labels = positive[chosen].astype(np.int8)
    return {
        "count": int(chosen.sum()),
        "AUROC": float(roc_auc_score(labels, probability[chosen])),
        "AP_ent": float(average_precision_score(labels, probability[chosen])),
        "AP_sep": float(average_precision_score(1 - labels, 1.0 - probability[chosen])),
    }


def benchmark_metrics(probability: np.ndarray, labels: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    separable = np.asarray(labels["separable"], dtype=bool)
    entangled = np.asarray(labels["entangled"], dtype=bool)
    e2 = np.asarray(labels.get("E2", np.zeros_like(entangled)), dtype=bool)
    e3 = np.asarray(labels.get("E3", np.zeros_like(entangled)), dtype=bool)
    e4 = np.asarray(labels.get("E4", np.zeros_like(entangled)), dtype=bool)
    unknown = np.asarray(labels.get("unknown", ~(separable | entangled)), dtype=bool)
    e_gt_2 = entangled & ~e2
    strict = separable | entangled
    result = binary_metrics(entangled[strict], probability[strict], threshold)
    result.update(
        {
            "total_rows": int(len(probability)),
            "strict_count": int(strict.sum()),
            "separable_count": int(separable.sum()),
            "entangled_count": int(entangled.sum()),
            "unknown_count": int(unknown.sum()),
            "separable_FPR": float(np.mean(probability[separable] >= threshold)),
            "E2_count": int(e2.sum()),
            "E2_recall": float(np.mean(probability[e2] >= threshold)) if e2.any() else None,
            "E_gt_2_count": int(e_gt_2.sum()),
            "E_gt_2_recall": float(np.mean(probability[e_gt_2] >= threshold)) if e_gt_2.any() else None,
            "E3_count": int(e3.sum()),
            "E3_recall": float(np.mean(probability[e3] >= threshold)) if e3.any() else None,
            "E4_count": int(e4.sum()),
            "E4_recall": float(np.mean(probability[e4] >= threshold)) if e4.any() else None,
            "sep_vs_E2": _ranking(probability, separable, e2),
            "sep_vs_E_gt_2": _ranking(probability, separable, e_gt_2),
            "sep_vs_E3E4": _ranking(probability, separable, e3 | e4),
        }
    )
    return result


def load_n16_benchmark(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if not bool(h5.attrs.get("generation_complete", False)):
            raise RuntimeError(f"benchmark is incomplete: {path}")
        separable = h5["labels/certified_separable"][:].astype(bool)
        entangled = h5["labels/certified_entangled"][:].astype(bool)
        if "E2" in h5["labels"]:
            e2 = h5["labels/E2"][:].astype(bool)
            e_gt_2 = h5["labels/E_gt_2"][:].astype(bool)
        else:
            e2 = entangled & h5["cert2/all_pairs/any_npt"][:].astype(bool)
            e_gt_2 = entangled & ~e2
        sample_key = "metadata/sample_id" if "metadata/sample_id" in h5 else "params/sample_id"
        family = h5["metadata/family"].asstr()[:] if "metadata/family" in h5 else np.asarray(["historical"] * len(separable))
        return {
            "sample_id": h5[sample_key][:],
            "separable": separable,
            "entangled": entangled,
            "E2": e2,
            "E3": np.zeros_like(entangled),
            "E4": np.zeros_like(entangled),
            "E_gt_2": e_gt_2,
            "unknown": ~(separable | entangled),
            "family": family,
        }



from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import h5py
import numpy as np


def feature_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "correlation_mean": float(values.mean()),
        "correlation_std": float(values.std()),
        "correlation_l2_norm": float(np.linalg.norm(values)),
        "correlation_abs_max": float(np.abs(values).max()),
    }


def validate_hdf5(
    path: str | Path,
    expected_features: int | None = None,
    *,
    expected_samples: int | None = None,
    expected_per_label: int | None = None,
    feature_names_path: str | Path | None = None,
    strict_certificates: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        X = h5["features/X"][:]
        y = h5["labels/y"][:]
        errors: list[str] = []
        if X.dtype != np.float32:
            errors.append(f"X dtype is {X.dtype}, expected float32")
        if y.dtype != np.int8:
            errors.append(f"y dtype is {y.dtype}, expected int8")
        if X.shape[0] != y.shape[0]:
            errors.append("X/y sample counts differ")
        if expected_samples is not None and X.shape[0] != expected_samples:
            errors.append(f"sample count is {X.shape[0]}, expected {expected_samples}")
        if expected_features is not None and X.shape[1] != expected_features:
            errors.append(f"feature count is {X.shape[1]}, expected {expected_features}")
        if not np.isfinite(X).all():
            errors.append("X contains NaN or infinity")
        if np.abs(X).max(initial=0.0) > 1.0 + 2e-5:
            errors.append("Pauli expectation outside [-1,1]")
        if not set(np.unique(y)).issubset({0, 1}):
            errors.append("labels other than 0/1 found")
        for key, dataset in h5["metadata"].items():
            if dataset.shape[0] != y.shape[0]:
                errors.append(f"metadata/{key} has wrong sample count")
        counts = {int(label): int((y == label).sum()) for label in np.unique(y)}
        if expected_per_label is not None and counts != {0: expected_per_label, 1: expected_per_label}:
            errors.append(f"label counts are {counts}, expected {expected_per_label} per label")
        if strict_certificates:
            if "certificate_type" not in h5["metadata"]:
                errors.append("metadata/certificate_type missing")
            else:
                cert = h5["metadata/certificate_type"].asstr()[:]
                if not np.all(cert[y == 0] == "constructive_fully_separable"):
                    errors.append("some label=0 rows lack a constructive separability certificate")
                if not np.all(cert[y == 1] == "two_site_npt"):
                    errors.append("some label=1 rows lack a two-site NPT certificate")
                if np.any(np.char.find(cert.astype(str), "unknown") >= 0):
                    errors.append("unknown certificate found in accepted dataset")
            if "npt_min_eigenvalue" in h5["metadata"]:
                npt = h5["metadata/npt_min_eigenvalue"][:]
                if not np.all(np.isfinite(npt[y == 1])) or not np.all(npt[y == 1] < 0):
                    errors.append("some label=1 rows lack a finite negative NPT eigenvalue")
        if feature_names_path is not None:
            names = json.loads(Path(feature_names_path).read_text(encoding="utf-8"))
            if len(names) != X.shape[1]:
                errors.append("feature_names length does not match X")
            canonical = json.dumps(names, ensure_ascii=False, separators=(",", ":"))
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            stored = h5.attrs.get("feature_names_sha256", "")
            if stored != digest:
                errors.append("feature_names hash does not match dataset attribute")
    return {
        "valid": not errors,
        "errors": errors,
        "shape": list(X.shape),
        "label_counts": counts,
        "file_size_bytes": path.stat().st_size,
    }

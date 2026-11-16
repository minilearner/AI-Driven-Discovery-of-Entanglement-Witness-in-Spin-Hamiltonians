from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def _metadata_array(values: list[Any]) -> tuple[np.ndarray, Any | None]:
    if all(isinstance(v, (str, type(None))) for v in values):
        return np.asarray(["" if v is None else v for v in values], dtype=object), h5py.string_dtype("utf-8")
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        return np.asarray(values, dtype=np.int8), None
    if all(isinstance(v, (int, np.integer)) for v in values):
        return np.asarray(values, dtype=np.int64), None
    return np.asarray(values, dtype=np.float64), None


def write_dataset(
    output_dir: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    metadata: list[dict[str, Any]],
    feature_names: list[str],
    config: dict[str, Any],
    *,
    connected_X: np.ndarray | None = None,
    connected_names: list[str] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    h5_path = output / "dataset.h5"
    if X.dtype != np.float32 or y.dtype != np.int8:
        raise ValueError("X must be float32 and y must be int8")
    if X.shape[0] != y.shape[0] or len(metadata) != y.shape[0]:
        raise ValueError("sample dimensions do not match")
    with h5py.File(h5_path, "w") as h5:
        features = h5.create_group("features")
        rows_per_chunk = max(1, min(128, X.shape[0]))
        features.create_dataset(
            "X", data=X, compression="gzip", compression_opts=4, shuffle=True, chunks=(rows_per_chunk, X.shape[1])
        )
        if connected_X is not None:
            features.create_dataset(
                "X_connected",
                data=connected_X,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(rows_per_chunk, connected_X.shape[1]),
            )
        h5.create_group("labels").create_dataset("y", data=y)
        meta_group = h5.create_group("metadata")
        for key in sorted(metadata[0]) if metadata else []:
            array, dtype = _metadata_array([row[key] for row in metadata])
            kwargs = {"dtype": dtype} if dtype is not None else {}
            meta_group.create_dataset(key, data=array, **kwargs)
        h5.attrs["format_version"] = "1.0"
        h5.attrs["N"] = int(config["system"]["N"])
        h5.attrs["num_features"] = X.shape[1]
        h5.attrs["labels"] = "0=certified_separable,1=certified_entangled"
        canonical_names = json.dumps(feature_names, ensure_ascii=False, separators=(",", ":"))
        h5.attrs["feature_names_sha256"] = hashlib.sha256(canonical_names.encode("utf-8")).hexdigest()
    (output / "feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    if connected_names is not None:
        (output / "connected_feature_names.json").write_text(
            json.dumps(connected_names, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return h5_path

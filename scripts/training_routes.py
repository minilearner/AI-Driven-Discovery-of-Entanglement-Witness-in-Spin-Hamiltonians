"""Canonical data-source splits used by the retained training pipeline.

This is the split-loading utility formerly imported through exploratory
Siamese training scripts. The same parent-leakage and array-shape checks are
kept here so the final paper pipeline does not depend on those model trials.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ("D_v1", "D_pair", "D_H")
H5_PATHS = {
    "D_v1": ROOT / "data/n16_production_v1/dataset.h5",
    "D_pair": ROOT / "analysis/paired_training_v1/D_pair.h5",
    "D_H": ROOT / "data/random_local_hamiltonian_v1/D_H.h5",
}


def route_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.asarray(array).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def concatenate(parts: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, row, label = [], [], []
    for source_id, rows, labels in parts:
        source.append(np.full(len(rows), source_id, dtype=np.int8))
        row.append(np.asarray(rows, dtype=np.int64))
        label.append(np.asarray(labels, dtype=np.float32))
    return np.concatenate(source), np.concatenate(row), np.concatenate(label)


def load_routes(cache_dir: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    arrays = [np.load(cache_dir / f"{name}_raw_T1T2T3_Robs4_float32.npy", mmap_mode="r") for name in BASE]
    metadata: dict[str, dict[str, Any]] = {}
    parent_audit = {}
    for name, path in H5_PATHS.items():
        with h5py.File(path, "r") as h5:
            labels = h5["labels/y"][:].astype(np.float32)
            split = h5["metadata/split"].asstr()[:]
            parent_key = "parent_state_id" if name == "D_v1" else "pair_id"
            parent = h5[f"metadata/{parent_key}"].asstr()[:]
            generator = h5["metadata/generator_type"].asstr()[:] if "generator_type" in h5["metadata"] else None
            family = h5["metadata/family_id"].asstr()[:] if "family_id" in h5["metadata"] else None
        if arrays[BASE.index(name)].shape != (len(labels), 912):
            raise RuntimeError(f"{name} canonical cache shape changed")
        assignment = {}
        for parent_id, split_name in zip(parent.tolist(), split.tolist()):
            assignment.setdefault(parent_id, set()).add(split_name)
        crossing = [key for key, values in assignment.items() if len(values) != 1]
        if crossing:
            raise RuntimeError(f"{name} parent split leakage: {crossing[:3]}")
        parent_audit[name] = {"parent_key": parent_key, "rows": len(labels), "unique_parents": len(assignment), "parents_crossing_splits": 0}
        metadata[name] = {"y": labels, "split": split, "generator": generator, "family": family}
    routes = {}
    for split_name in ("train", "validation", "test"):
        parts = []
        for source_id, name in enumerate(BASE):
            rows = np.flatnonzero(metadata[name]["split"] == split_name)
            parts.append((source_id, rows, metadata[name]["y"][rows]))
        routes[split_name] = concatenate(parts)
    train = routes["train"]
    v1_rows = train[1][train[0] == 0]
    pair_rows = train[1][train[0] == 1]
    local_ghz = int(np.sum(metadata["D_v1"]["generator"][v1_rows] == "classical_ghz_mixture"))
    local_ghz += int(np.sum(metadata["D_pair"]["family"][pair_rows] == "B_classical_vs_coherent_GHZ_block"))
    composition = {
        "D_v1_rows": int(np.sum(train[0] == 0)), "D_pair_rows": int(np.sum(train[0] == 1)),
        "D_H_rows": int(np.sum(train[0] == 2)), "D_PI_positive_rows": 0,
        "D_PI_related_negative_rows": 0, "positive_count": int(train[2].sum()),
        "negative_count": int(len(train[2]) - train[2].sum()), "local_GHZ_rows": local_ghz,
        "other_rows": int(len(train[2]) - local_ghz),
    }
    return arrays, {
        "metadata": metadata, "routes": routes, "parent_audit": parent_audit,
        "training_composition": composition,
        "route_hash": route_hash(*routes["train"], *routes["validation"], *routes["test"]),
    }

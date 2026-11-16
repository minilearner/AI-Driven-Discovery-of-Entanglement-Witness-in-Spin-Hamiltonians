"""Structured N=32 benchmark reader used by the final LU input control."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from entanglement_data.ml.correlator_transformer import arrays_to_tokens

def load_structured_benchmark(path: Path, group: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, "r") as h5:
        if not bool(h5.attrs.get("generation_complete", False)):
            raise RuntimeError(f"benchmark is incomplete: {path}")
        prefix = "features" if group == "xy" else "raw"
        tokens, masks, sites = arrays_to_tokens(
            h5[f"{prefix}/T1"][:], h5[f"{prefix}/T2"][:], h5[f"{prefix}/T3"][:]
        )
        if group == "xy":
            labels = {
                "sample_id": h5["params/sample_id"][:],
                "separable": h5["labels/strict_separable"][:],
                "entangled": h5["labels/binary_entangled"][:],
                "E2": h5["labels/E2"][:],
                "E3": h5["labels/E3"][:],
                "E4": h5["labels/E4"][:],
                "unknown": h5["labels/unknown"][:],
            }
        else:
            labels = {
                "sample_id": h5["meta/sample_id"][:],
                "separable": h5["labels/strict_separable"][:],
                "entangled": h5["labels/binary_entangled"][:],
                "E2": h5["labels/E2"][:],
                "E3": h5["labels/E3"][:],
                "E4": h5["labels/E4"][:],
                "unknown": h5["labels/unknown"][:],
            }
    return tokens, masks, sites, labels


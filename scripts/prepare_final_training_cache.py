"""Build the 122-channel Dv1 token cache required by final-model training.

Only canonical Pauli moments are read from the input HDF5. The frozen K16
manifest is applied later by the training script; this cache contains every
candidate LU-invariant channel and the five geometry masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.final_entnet import tokens_from_pauli_features  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/n16_production_v1/dataset.h5")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/training_cache")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true", help="Replace an existing cache")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    target = args.cache_dir / "D_v1_Inv123RawT4_Robs4.npy"
    report = args.cache_dir / "D_v1_cache_manifest.json"
    if (target.exists() or report.exists()) and not args.force:
        parser.error(f"cache exists in {args.cache_dir}; choose another --cache-dir or use --force")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as h5:
        if "features/X" not in h5:
            raise ValueError("expected canonical Dv1 HDF5 with features/X")
        rows = h5["features/X"]
        if rows.ndim != 2 or rows.shape[1] != 2559:
            raise ValueError(f"expected N=16 canonical 2559-column Pauli features, got {rows.shape}")
        n_rows = len(rows)
        cache = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32,
                                          shape=(n_rows, 16, 122))
        try:
            for start in range(0, n_rows, args.batch_size):
                stop = min(start + args.batch_size, n_rows)
                cache[start:stop] = tokens_from_pauli_features(rows[start:stop], N=16)
                print(f"\rPrepared {stop}/{n_rows} rows", end="", flush=True)
            print()
            cache.flush()
        except BaseException:
            del cache
            target.unlink(missing_ok=True)
            raise
        del cache
    payload = {"source": str(args.dataset), "source_sha256": sha256(args.dataset),
               "cache": target.name, "cache_sha256": sha256(target),
               "shape": [n_rows, 16, 122], "dtype": "float32",
               "feature_manifest": "configs/joint_K16.json"}
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved verified cache and manifest in {args.cache_dir}")


if __name__ == "__main__":
    main()

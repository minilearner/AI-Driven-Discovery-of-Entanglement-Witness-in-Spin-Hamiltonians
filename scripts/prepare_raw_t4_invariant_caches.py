from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.spatial_invariants import SpatialInvariantSpec
from entanglement_data.ml.invariant_transformer import (
    local_inv123_channel_indices,
    raw_t4_invariant_manifest,
    structured_inv123_raw_t4_local_tokens,
)
from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.lu_invariants import decode_correlators, raw_t4_invariant_blocks
from training_routes import BASE, H5_PATHS, load_routes


EXACT500 = ROOT / "data/xy_n16_pbc_exact500_v1/dataset.h5"
BALANCED = ROOT / "data/exactXY_balanced_N16_factorized_v1/dataset.h5"
N32_XY = ROOT / "data/xy_n32_pbc_matched_exact500_v1/dataset.h5"
N32_DPI = ROOT / "data/dpi_n32_general_v1/D_PI_N32.h5"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_flat_cache(
    h5_path: Path,
    inv123_path: Path,
    output_path: Path,
    *,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    base = np.load(inv123_path, mmap_mode="r")
    indices = local_inv123_channel_indices(16, 4)
    if base.ndim != 3 or base.shape[1] != 16 or len(indices) != 82:
        raise RuntimeError(f"unexpected Inv123 cache layout: {inv123_path} {base.shape}")
    if output_path.exists() and not force:
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape != (len(base), 16, 122):
            raise RuntimeError(f"stale raw-T4 cache shape: {output_path} {existing.shape}")
        return {"status": "reused", "rows": len(base), "shape": list(existing.shape)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(len(base), 16, 122),
    )
    spec = FeatureSpec(16)
    with h5py.File(h5_path, "r") as h5:
        raw = h5["features/X"]
        if raw.shape != (len(base), spec.num_features):
            raise RuntimeError(f"raw features and Inv123 cache differ: {h5_path}")
        for start in range(0, len(base), batch_size):
            stop = min(len(base), start + batch_size)
            old = np.asarray(base[start:stop, :, indices], dtype=np.float32)
            decoded = decode_correlators(raw[start:stop], spec)
            t4 = raw_t4_invariant_blocks(decoded.four, output_dtype=np.float32)
            output[start:stop, :, :77] = old[:, :, :77]
            output[start:stop, :, 77:117] = 0.0
            output[start:stop, :13, 77:117] = t4
            output[start:stop, :, 117:] = old[:, :, 77:]
    output.flush()
    return {"status": "created", "rows": len(base), "shape": list(output.shape)}


def build_structured_cache(
    h5_path: Path,
    output_path: Path,
    *,
    prefix: str,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        existing = np.load(output_path, mmap_mode="r")
        if existing.ndim != 3 or existing.shape[1:] != (32, 122):
            raise RuntimeError(f"stale structured raw-T4 cache: {output_path} {existing.shape}")
        return {"status": "reused", "rows": len(existing), "shape": list(existing.shape)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        one_key = f"{prefix}/T1"
        rows = len(h5[one_key])
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(rows, 32, 122),
        )
        for start in range(0, rows, batch_size):
            stop = min(rows, start + batch_size)
            output[start:stop] = structured_inv123_raw_t4_local_tokens(
                h5[one_key][start:stop],
                h5[f"{prefix}/T2"][start:stop],
                h5["derived/K2"][start:stop],
                h5[f"{prefix}/T3"][start:stop],
                h5["derived/K3"][start:stop],
                h5[f"{prefix}/T4"][start:stop],
                two_mask=h5[f"{prefix}/T2_mask"][start:stop],
                three_mask=h5[f"{prefix}/T3_mask"][start:stop],
                four_mask=h5[f"{prefix}/T4_mask"][start:stop],
                R_obs=4,
                output_dtype=np.float32,
            )
    output.flush()
    return {"status": "created", "rows": rows, "shape": list(output.shape)}


def training_normalization(
    cache_dir: Path,
    route_cache_dir: Path,
    old_normalization_dir: Path,
) -> dict[str, Any]:
    _, route_data = load_routes(route_cache_dir)
    source, rows, _ = route_data["routes"]["train"]
    arrays = [np.load(cache_dir / f"{name}_Inv123RawT4_Robs4.npy", mmap_mode="r") for name in BASE]
    total = np.zeros(40, dtype=np.float64)
    squared = np.zeros(40, dtype=np.float64)
    count = 0
    for source_id in range(len(BASE)):
        selected = rows[source == source_id]
        for start in range(0, len(selected), 2048):
            values = np.asarray(arrays[source_id][selected[start:start + 2048], :13, 77:117], dtype=np.float64)
            flat = values.reshape(-1, 40)
            total += flat.sum(axis=0)
            squared += np.square(flat).sum(axis=0)
            count += len(flat)
    t4_mean = total / count
    t4_variance = np.maximum(squared / count - np.square(t4_mean), 1e-12)
    t4_std = np.sqrt(t4_variance)
    indices = local_inv123_channel_indices(16, 4)
    old_mean = np.load(old_normalization_dir / "local_mean.npy")[indices]
    old_std = np.load(old_normalization_dir / "local_std.npy")[indices]
    mean = np.concatenate((old_mean[:77], t4_mean, old_mean[77:])).astype(np.float32)
    std = np.concatenate((old_std[:77], t4_std, old_std[77:])).astype(np.float32)
    np.save(cache_dir / "input_mean_full.npy", mean)
    np.save(cache_dir / "input_std_full.npy", std)
    return {
        "route_hash": route_data["route_hash"],
        "valid_T4_scalar_anchor_rows": count,
        "minimum_std": float(std.min()),
        "maximum_std": float(std.max()),
    }


def duplicate_audit(cache_dir: Path) -> dict[str, Any]:
    rng = np.random.default_rng(20260903)
    random_values = raw_t4_invariant_blocks(
        rng.normal(size=(256, 3, 3, 3, 3)), output_dtype=np.float64,
    )
    real_parts = []
    for name in BASE:
        values = np.load(cache_dir / f"{name}_Inv123RawT4_Robs4.npy", mmap_mode="r")
        rows = min(len(values), 128)
        real_parts.append(np.asarray(values[:rows, :13, 77:117]).reshape(-1, 40))
    probe = np.concatenate((random_values, *real_parts), axis=0)
    duplicates = []
    for left in range(40):
        for right in range(left + 1, 40):
            maximum = float(np.max(np.abs(probe[:, left] - probe[:, right])))
            if maximum <= 1e-12:
                duplicates.append({"left": left, "right": right, "maximum_absolute_difference": maximum})
    return {
        "candidate_count": 40,
        "probe_rows": len(probe),
        "exact_duplicate_pairs": duplicates,
        "deduplicated_count": 40 - len({row["right"] for row in duplicates}),
        "frobenius_algebraically_redundant_but_retained_as_control": True,
    }


def write_manifest(path: Path) -> None:
    rows = raw_t4_invariant_manifest()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dpi_geometry_audit(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        e4 = h5["labels/E4"][:].astype(bool)
        raw = h5["certificates/subsystem"].asstr()[:]
    supports = [tuple(int(value) for value in ast.literal_eval(text)) for text in raw[e4]]
    spans = np.asarray([max(sites) - min(sites) for sites in supports], dtype=np.int64)
    contiguous = np.asarray([
        tuple(sorted(sites)) == tuple(range(min(sites), max(sites) + 1)) and len(sites) == 4
        for sites in supports
    ])
    values, counts = np.unique(spans, return_counts=True)
    return {
        "E4_count": len(supports),
        "contiguous_count": int(contiguous.sum()),
        "noncontiguous_count": int((~contiguous).sum()),
        "fraction_contiguous": float(contiguous.mean()),
        "span_distribution": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "supports": [list(sites) for sites in supports],
        "contiguous_mask": contiguous.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Local Inv123 + raw-T4 caches without changing source HDF5 files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/invariant_transformer_raw_t4_v1")
    parser.add_argument("--inv123-cache-dir", type=Path, default=ROOT / "analysis/cnn_locality_v1")
    parser.add_argument("--route-cache-dir", type=Path, default=ROOT / "analysis/connected_siamese_v1")
    parser.add_argument("--normalization-dir", type=Path, default=ROOT / "models/cnn_locality_remote_4090_20260813/SpatialMLP_control")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is closed for this experiment")
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    exact_hash_before = sha256(EXACT500)
    reports: dict[str, Any] = {}
    for name in BASE:
        reports[name] = build_flat_cache(
            H5_PATHS[name], args.inv123_cache_dir / f"{name}_Inv123_local.npy",
            cache_dir / f"{name}_Inv123RawT4_Robs4.npy",
            batch_size=args.batch_size, force=args.force,
        )
    reports["exact500"] = build_flat_cache(
        EXACT500, args.inv123_cache_dir / "exact500_Inv123_local.npy",
        cache_dir / "exact500_Inv123RawT4_Robs4.npy",
        batch_size=args.batch_size, force=args.force,
    )
    balanced_inv123 = args.output_dir.parent / "invariant_transformer_v1/cache/XY_factorized_balanced_N16_Inv123_local.npy"
    if not balanced_inv123.exists():
        raise FileNotFoundError(f"frozen balanced Inv123 cache is missing: {balanced_inv123}")
    reports["balanced_N16"] = build_flat_cache(
        BALANCED, balanced_inv123,
        cache_dir / "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy",
        batch_size=args.batch_size, force=args.force,
    )
    reports["N32_XY"] = build_structured_cache(
        N32_XY, cache_dir / "N32_XY_Inv123RawT4_Robs4.npy",
        prefix="features", batch_size=args.batch_size, force=args.force,
    )
    reports["N32_DPI"] = build_structured_cache(
        N32_DPI, cache_dir / "N32_DPI_Inv123RawT4_Robs4.npy",
        prefix="raw", batch_size=args.batch_size, force=args.force,
    )
    normalization = training_normalization(cache_dir, args.route_cache_dir, args.normalization_dir)
    duplicate = duplicate_audit(cache_dir)
    write_manifest(args.output_dir / "raw_t4_invariant_manifest.csv")
    geometry = dpi_geometry_audit(N32_DPI)
    save_json(args.output_dir / "dpi_e4_certificate_geometry.json", geometry)
    exact_hash_after = sha256(EXACT500)
    report = {
        "status": "complete",
        "raw_T4_only": True,
        "K4_or_connected_T4_included": False,
        "C4": duplicate["deduplicated_count"],
        "candidate_manifest": "raw_t4_invariant_manifest.csv",
        "cache_reports": reports,
        "normalization": normalization,
        "duplicate_audit": duplicate,
        "dpi_E4_geometry": {key: value for key, value in geometry.items() if key not in {"supports", "contiguous_mask"}},
        "historical_exact500_sha256_before": exact_hash_before,
        "historical_exact500_sha256_after": exact_hash_after,
        "historical_exact500_unchanged": exact_hash_before == exact_hash_after,
        "N32_used_for_training_or_selection": False,
        "exact5000_accessed": False,
    }
    save_json(args.output_dir / "cache_preparation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

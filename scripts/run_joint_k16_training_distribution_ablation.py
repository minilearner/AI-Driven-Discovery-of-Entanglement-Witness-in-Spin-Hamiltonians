from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    local_inv123_raw_t4_channel_manifest,
    trainable_parameter_count,
)
from evaluate_corrected_pair_range_models import (  # noqa: E402
    build_general_separable_cache,
    general_separable_metrics,
)
from run_raw_t4_invariant_transformer import (  # noqa: E402
    TrainingConfig,
    benchmark_metrics,
    git_hash,
    predict_array,
    predict_loader,
    save_json,
    seed_all,
    write_scores,
)
from train_invariant_transformer import RoutedInvariantTokens, sha256  # noqa: E402


EXPERIMENT = "joint_k16_training_distribution_ablation_v1"
SEEDS = (20260829, 20260830, 20260831)
SOURCE_NAMES = ("D_v1", "D_pair", "D_H")
ROUTES = {
    "Dv1_only": ("D_v1",),
    "Dv1_plus_Dpair": ("D_v1", "D_pair"),
    "Dv1_plus_DH": ("D_v1", "D_H"),
    "historical_full_reference": SOURCE_NAMES,
}
EXPECTED_CHANNELS = [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 44, 83, 91, 51, 3, 59,
                     117, 118, 119, 120, 121]
EXPECTED_PARAMETERS = 483_969


def default_source_root() -> Path:
    # Avoid borrowing data from a private parent archive during preparation.
    return ROOT


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def route_hash(route: tuple[np.ndarray, np.ndarray, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in route:
        digest.update(np.asarray(value).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def h5_paths(source_root: Path) -> dict[str, Path]:
    return {
        "D_v1": source_root / "data/n16_production_v1/dataset.h5",
        "D_pair": source_root / "analysis/paired_training_v1/D_pair.h5",
        "D_H": source_root / "data/random_local_hamiltonian_v1/D_H.h5",
    }


def cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {name: cache_dir / f"{name}_Inv123RawT4_Robs4.npy" for name in SOURCE_NAMES}


def ensure_inputs(source_root: Path, cache_dir: Path, manifest_path: Path,
                  sources: tuple[str, ...] = SOURCE_NAMES) -> None:
    required = [manifest_path, *[h5_paths(source_root)[name] for name in sources],
                *[cache_paths(cache_dir)[name] for name in sources]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required frozen inputs are missing:\n" + "\n".join(missing))


def load_manifest(manifest_path: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = np.asarray(manifest["selected_full_token_channels"], dtype=np.int64)
    checks = {
        "byte_sha256": sha256(manifest_path),
        "semantic_sha256": canonical_json_hash(manifest),
        "selected_channels_exact_expected_order": channels.tolist() == EXPECTED_CHANNELS,
        "physical_channel_count": int(manifest["physical_channel_count"]),
        "mask_channel_count": int(manifest["mask_channel_count"]),
        "input_width": int(manifest["input_width"]),
        "r4_pair_channels": [
            int(row["channel"]) for row in manifest["selected_channels"]
            if row.get("family") == "pair" and row.get("pair_distance") == 4
        ],
    }
    if not checks["selected_channels_exact_expected_order"]:
        raise RuntimeError("historical Joint K16 channel order changed")
    if checks["physical_channel_count"] != 16 or checks["mask_channel_count"] != 5 or len(channels) != 21:
        raise RuntimeError("historical Joint K16 dimensions changed")
    return manifest, channels, checks


def _as_text(group: h5py.Group, name: str, rows: int) -> np.ndarray:
    if name not in group:
        return np.full(rows, "unavailable", dtype=object)
    return group[name].asstr()[:]


def pair_npt_flags(features: np.ndarray, tolerance: float = 1.0e-9, batch_size: int = 128) -> np.ndarray:
    """Reconstruct all 120 two-site RDMs from exact T1/T2 moments and apply PPT."""
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] < 1128:
        raise ValueError("expected legacy N=16 features with all one- and two-body moments")
    n = len(values)
    left = np.asarray([i for i in range(16) for _ in range(i + 1, 16)], dtype=np.int64)
    right = np.asarray([j for i in range(16) for j in range(i + 1, 16)], dtype=np.int64)
    identity = np.eye(2, dtype=np.complex128)
    pauli = np.asarray([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ], dtype=np.complex128)
    left_ops = np.asarray([np.kron(axis, identity) for axis in pauli]) / 4.0
    right_ops = np.asarray([np.kron(identity, axis) for axis in pauli]) / 4.0
    pair_ops = np.asarray([[np.kron(a, b) for b in pauli] for a in pauli]) / 4.0
    base = np.kron(identity, identity) / 4.0
    output = np.zeros(n, dtype=bool)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        block = values[start:stop]
        one = block[:, :48].reshape(-1, 16, 3)
        two = block[:, 48:1128].reshape(-1, 120, 3, 3)
        rho = np.broadcast_to(base, (len(block), 120, 4, 4)).copy()
        rho += np.einsum("rpa,aij->rpij", one[:, left], left_ops)
        rho += np.einsum("rpa,aij->rpij", one[:, right], right_ops)
        rho += np.einsum("rpab,abij->rpij", two, pair_ops)
        partial = rho.reshape(-1, 120, 2, 2, 2, 2).swapaxes(-4, -2).reshape(-1, 120, 4, 4)
        output[start:stop] = np.linalg.eigvalsh(partial)[..., 0].min(axis=1) < -tolerance
    return output


def load_metadata(source_root: Path, audit_dir: Path,
                  sources: tuple[str, ...] = SOURCE_NAMES) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    parent_audit: dict[str, Any] = {}
    pair_flag_path = audit_dir / "D_pair_allpair_npt_flags.npy"
    for source_id, name in enumerate(SOURCE_NAMES):
        if name not in sources:
            continue
        path = h5_paths(source_root)[name]
        with h5py.File(path, "r") as h5:
            y = h5["labels/y"][:].astype(bool)
            split = h5["metadata/split"].asstr()[:]
            rows = len(y)
            parent_key = "parent_state_id" if name == "D_v1" else "pair_id"
            parent = h5[f"metadata/{parent_key}"].asstr()[:]
            certificate = _as_text(h5["metadata"], "certificate_type", rows)
            generator = _as_text(h5["metadata"], "generator_type", rows)
            family = _as_text(h5["metadata"], "family_id", rows)
            if name == "D_v1":
                e2 = certificate == "two_site_npt"
                if "npt_min_eigenvalue" in h5["metadata"]:
                    npt = h5["metadata/npt_min_eigenvalue"][:]
                    if not np.array_equal(e2 & y, (npt < -1.0e-9) & y):
                        raise RuntimeError("Dv1 certificate_type and stored NPT margin disagree")
            elif name == "D_H":
                e2 = h5["metadata/allpair_any_npt"][:].astype(bool)
            else:
                if pair_flag_path.exists():
                    e2 = np.load(pair_flag_path)
                else:
                    e2 = pair_npt_flags(h5["features/X"][:])
                    audit_dir.mkdir(parents=True, exist_ok=True)
                    np.save(pair_flag_path, e2)
        if len(e2) != rows:
            raise RuntimeError(f"{name} E2 audit length mismatch")
        assignments: dict[str, set[str]] = {}
        for parent_id, split_name in zip(parent.tolist(), split.tolist()):
            assignments.setdefault(parent_id, set()).add(split_name)
        crossing = [key for key, values in assignments.items() if len(values) != 1]
        if crossing:
            raise RuntimeError(f"{name} parent leakage: {crossing[:3]}")
        parent_audit[name] = {
            "source_id": source_id,
            "parent_key": parent_key,
            "rows": rows,
            "unique_parents": len(assignments),
            "parents_crossing_splits": 0,
        }
        metadata[name] = {
            "y": y,
            "split": split,
            "certificate": certificate,
            "generator": generator,
            "family": family,
            "e2": e2 & y,
            "known_e_gt_2": y & ~e2,
        }
    return metadata, parent_audit


def build_route(metadata: dict[str, dict[str, Any]], route_name: str, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = ROUTES[route_name]
    source_parts, row_parts, label_parts = [], [], []
    for name in names:
        source_id = SOURCE_NAMES.index(name)
        rows = np.flatnonzero(metadata[name]["split"] == split_name)
        source_parts.append(np.full(len(rows), source_id, dtype=np.int8))
        row_parts.append(rows.astype(np.int64))
        label_parts.append(metadata[name]["y"][rows].astype(np.float32))
    return np.concatenate(source_parts), np.concatenate(row_parts), np.concatenate(label_parts)


def route_audit(metadata: dict[str, dict[str, Any]], route_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"route": route_name, "sources": list(ROUTES[route_name]), "splits": {}}
    for split_name in ("train", "validation", "test"):
        route = build_route(metadata, route_name, split_name)
        source, rows, labels = route
        entry: dict[str, Any] = {
            "total_rows": len(labels),
            "Sep": int((labels == 0).sum()),
            "Ent": int((labels == 1).sum()),
            "route_hash": route_hash(route),
            "by_source": {},
        }
        for name in ROUTES[route_name]:
            source_id = SOURCE_NAMES.index(name)
            chosen = source == source_id
            local_rows = rows[chosen]
            local = metadata[name]
            positives = local["y"][local_rows]
            e2_count = int(local["e2"][local_rows].sum())
            higher_count = int(local["known_e_gt_2"][local_rows].sum())
            family_values = local["generator"][local_rows] if name == "D_v1" else local["family"][local_rows]
            entry["by_source"][name] = {
                "rows": len(local_rows),
                "Sep": int((~positives).sum()),
                "Ent": int(positives.sum()),
                "E2_certifiable_positive_count": e2_count,
                "E2_certifiable_fraction_of_positives": float(e2_count / positives.sum()) if positives.any() else None,
                "known_E_gt_2_positive_count": higher_count,
                "known_E_gt_2_fraction_of_positives": float(higher_count / positives.sum()) if positives.any() else None,
                "generator_or_family_composition": dict(sorted(Counter(family_values.tolist()).items())),
                "certificate_composition": dict(sorted(Counter(local["certificate"][local_rows].tolist()).items())),
            }
        result["splits"][split_name] = entry
    return result


def channel_valid_mask(full_values: np.ndarray, channel_row: dict[str, Any]) -> np.ndarray:
    family = channel_row["family"]
    if family == "one_body":
        return np.ones(full_values.shape[:2], dtype=bool)
    if family == "pair":
        return full_values[:, :, 116 + int(channel_row["pair_distance"])] > 0.5
    if family == "contiguous_three_body":
        return full_values[:, :, 121] > 0.5
    if family == "contiguous_four_body":
        valid = np.zeros(full_values.shape[:2], dtype=bool)
        valid[:, :13] = True
        return valid
    raise ValueError(f"unknown physical family {family}")


def fit_normalization(
    arrays: list[np.ndarray],
    route: tuple[np.ndarray, np.ndarray, np.ndarray],
    channels: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = {int(row["channel"]): row for row in local_inv123_raw_t4_channel_manifest(4)}
    total = np.zeros(len(channels), dtype=np.float64)
    squared = np.zeros(len(channels), dtype=np.float64)
    count = np.zeros(len(channels), dtype=np.int64)
    source, rows, _ = route
    for source_id in np.unique(source):
        selected = rows[source == source_id]
        for start in range(0, len(selected), batch_size):
            full = np.asarray(arrays[int(source_id)][selected[start:start + batch_size]], dtype=np.float64)
            for local_index, channel in enumerate(channels):
                row = manifest[int(channel)]
                if row["mask"]:
                    continue
                values = full[:, :, int(channel)][channel_valid_mask(full, row)]
                total[local_index] += values.sum()
                squared[local_index] += np.square(values).sum()
                count[local_index] += values.size
    mean = np.zeros(len(channels), dtype=np.float64)
    std = np.ones(len(channels), dtype=np.float64)
    physical = count > 0
    mean[physical] = total[physical] / count[physical]
    variance = squared[physical] / count[physical] - np.square(mean[physical])
    std[physical] = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1.0e-8)
    return mean.astype(np.float32), std.astype(np.float32), {
        "algorithm": "translation-tied moments over valid anchors of the actual training route; masks fixed to mean=0,std=1",
        "counts_by_selected_channel": count.tolist(),
        "minimum_std": float(std.min()),
        "maximum_std": float(std.max()),
    }


def checkpoint_audit(source_root: Path, manifest_checks: dict[str, Any]) -> dict[str, Any]:
    expected_training = asdict(TrainingConfig(seed=SEEDS[0]))
    rows = []
    for seed in SEEDS:
        model_dir = source_root / "models/invariant_transformer_raw_t4_v1" / (
            f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}"
        )
        report_path, checkpoint_path = model_dir / "report.json", model_dir / "best.pt"
        if not report_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"historical model is missing: {model_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        matched_training = dict(report["training_config"])
        matched_training["seed"] = SEEDS[0]
        rows.append({
            "seed": seed,
            "report_sha256": sha256(report_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "selected_channels_identical": report["selected_full_token_channels"] == EXPECTED_CHANNELS,
            "parameter_count": report["parameter_count"],
            "architecture": report["architecture_frozen"],
            "training_protocol_except_seed_identical": matched_training == expected_training,
            "checkpoint_selection": report["checkpoint_selection"],
        })
    if not all(row["selected_channels_identical"] and row["parameter_count"] == EXPECTED_PARAMETERS
               and row["training_protocol_except_seed_identical"] for row in rows):
        raise RuntimeError("historical checkpoints no longer match the frozen K16 protocol")
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=21)),
        np.zeros(21, dtype=np.float32), np.ones(21, dtype=np.float32),
    )
    measured = trainable_parameter_count(model)
    if measured != EXPECTED_PARAMETERS:
        raise RuntimeError(f"architecture parameter count {measured} != {EXPECTED_PARAMETERS}")
    return {
        "manifest": manifest_checks,
        "measured_parameter_count": measured,
        "historical_models": rows,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_inputs(args.source_root, args.cache_dir, args.manifest)
    _, channels, manifest_checks = load_manifest(args.manifest)
    metadata, parent = load_metadata(args.source_root, args.analysis_dir)
    arrays = [np.load(cache_paths(args.cache_dir)[name], mmap_mode="r") for name in SOURCE_NAMES]
    full_route = build_route(metadata, "historical_full_reference", "train")
    reproduced_mean, reproduced_std, norm_details = fit_normalization(arrays, full_route, channels)
    historical_mean = np.load(args.cache_dir / "input_mean_full.npy")[channels]
    historical_std = np.load(args.cache_dir / "input_std_full.npy")[channels]
    norm_reproduction = {
        **norm_details,
        "historical_full_route_mean_max_abs_error": float(np.max(np.abs(reproduced_mean - historical_mean))),
        "historical_full_route_std_max_abs_error": float(np.max(np.abs(reproduced_std - historical_std))),
    }
    norm_reproduction["passed"] = (
        norm_reproduction["historical_full_route_mean_max_abs_error"] <= 2.0e-6
        and norm_reproduction["historical_full_route_std_max_abs_error"] <= 2.0e-6
    )
    if not norm_reproduction["passed"]:
        raise RuntimeError("route-specific normalization implementation does not reproduce the historical full route")
    report = {
        "status": "passed",
        "experiment": EXPERIMENT,
        "source_root": str(args.source_root),
        "frozen_protocol": checkpoint_audit(args.source_root, manifest_checks),
        "parent_safe_split_audit": parent,
        "training_routes": {name: route_audit(metadata, name) for name in ROUTES},
        "normalization_reproduction": norm_reproduction,
        "interpretation": {
            "Dv1_positive_pair_NPT_statement": "audited from certificate_type plus stored NPT eigenvalue",
            "Dpair_E2_statement": "recomputed from every two-site RDM using stored one- and two-body moments",
            "known_E_gt_2_definition": "certified positive with no two-site NPT found; not a claim of complete multipartite classification",
        },
        "exact5000_accessed": False,
    }
    save_json(args.analysis_dir / "training_set_and_protocol_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def exact_labels(source_root: Path) -> dict[str, np.ndarray]:
    with h5py.File(source_root / "data/xy_n16_pbc_exact500_v1/dataset.h5", "r") as h5:
        entangled = h5["labels/certified_entangled"][:].astype(bool)
        e2 = entangled & h5["cert2/all_pairs/any_npt"][:].astype(bool)
        return {
            "sample_id": h5["params/sample_id"][:],
            "separable": h5["labels/certified_separable"][:],
            "entangled": entangled,
            "E2": e2,
            "E_gt_2": entangled & ~e2,
            "unknown": h5["labels/unknown"][:],
        }


def balanced_labels(source_root: Path) -> dict[str, np.ndarray]:
    with h5py.File(source_root / "data/exactXY_balanced_N16_factorized_v1/dataset.h5", "r") as h5:
        return {
            "sample_id": h5["metadata/sample_id"][:],
            "origin": h5["metadata/origin"].asstr()[:],
            "separable": h5["labels/certified_separable"][:],
            "entangled": h5["labels/certified_entangled"][:],
            "E2": h5["labels/E2"][:],
            "E_gt_2": h5["labels/E_gt_2"][:],
            "unknown": h5["labels/unknown"][:],
        }


def n32_labels(source_root: Path, group: str) -> dict[str, np.ndarray]:
    path = (source_root / "data/xy_n32_pbc_matched_exact500_v1/dataset.h5" if group == "xy"
            else source_root / "data/dpi_n32_general_v1/D_PI_N32.h5")
    prefix = "params" if group == "xy" else "meta"
    with h5py.File(path, "r") as h5:
        return {
            "sample_id": h5[f"{prefix}/sample_id"][:],
            "separable": h5["labels/strict_separable"][:],
            "entangled": h5["labels/binary_entangled"][:],
            "E2": h5["labels/E2"][:],
            "E3": h5["labels/E3"][:],
            "E4": h5["labels/E4"][:],
            "unknown": h5["labels/unknown"][:],
        }


def evaluate(
    model: nn.Module,
    channels: np.ndarray,
    *,
    source_root: Path,
    cache_dir: Path,
    general_cache: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_names = {
        "N16_exact500": "exact500_Inv123RawT4_Robs4.npy",
        "N16_XY_balanced": "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy",
        "N32_XY_matched": "N32_XY_Inv123RawT4_Robs4.npy",
        "N32_DPI_general": "N32_DPI_Inv123RawT4_Robs4.npy",
    }
    arrays = {name: np.load(cache_dir / filename, mmap_mode="r") for name, filename in cache_names.items()}
    if not general_cache.exists():
        build_general_separable_cache(
            source_root / "data/general_separable_ood_n32_v1/dataset.h5",
            general_cache, r_obs=4, width=122, batch_size=128,
        )
    arrays["N32_general_separable"] = np.load(general_cache, mmap_mode="r")
    probabilities = {
        name: predict_array(model, np.asarray(values)[:, :, channels], torch.device(config_device(model)),
                            config.eval_batch_size)
        for name, values in arrays.items()
    }
    labels = {
        "N16_exact500": exact_labels(source_root),
        "N16_XY_balanced": balanced_labels(source_root),
        "N32_XY_matched": n32_labels(source_root, "xy"),
        "N32_DPI_general": n32_labels(source_root, "dpi"),
    }
    metrics: dict[str, Any] = {}
    for name, label in labels.items():
        metrics[name] = benchmark_metrics(
            probabilities[name], separable=label["separable"], entangled=label["entangled"],
            e2=label["E2"], e3=label.get("E3"), e4=label.get("E4"),
            unknown=label["unknown"], threshold=config.threshold,
        )
        write_scores(output_dir / f"scores_{name}.csv", probabilities[name], label)
    origin = labels["N16_XY_balanced"]["origin"]
    balanced_probability = probabilities["N16_XY_balanced"]
    metrics["N16_XY_balanced"]["old20_FPR"] = float(
        (balanced_probability[(origin == "historical_exact500") & labels["N16_XY_balanced"]["separable"]]
         >= config.threshold).mean()
    )
    metrics["N16_XY_balanced"]["new412_factorization_product_FPR"] = float(
        (balanced_probability[origin == "new_xy_factorized"] >= config.threshold).mean()
    )
    with h5py.File(source_root / "data/general_separable_ood_n32_v1/dataset.h5", "r") as h5:
        family, k_band = h5["meta/family"].asstr()[:], h5["meta/k_band"].asstr()[:]
    metrics["N32_general_separable"] = general_separable_metrics(
        probabilities["N32_general_separable"], config.threshold, family, k_band,
    )
    write_scores(output_dir / "scores_N32_general_separable.csv", probabilities["N32_general_separable"], {
        "strict_separable": np.ones(len(family), dtype=bool), "family": family, "k_band": k_band,
    })
    return metrics, probabilities


def config_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)


def train_route(args: argparse.Namespace) -> dict[str, Any]:
    route_sources = ROUTES[args.route]
    ensure_inputs(args.source_root, args.cache_dir, args.manifest, route_sources)
    _, channels, manifest_checks = load_manifest(args.manifest)
    if args.route == "historical_full_reference":
        raise ValueError("the historical full route is frozen and must not be retrained")
    metadata, _ = load_metadata(args.source_root, args.analysis_dir, route_sources)
    arrays = [np.load(cache_paths(args.cache_dir)[name], mmap_mode="r")
              if name in route_sources else None for name in SOURCE_NAMES]
    train = build_route(metadata, args.route, "train")
    validation = build_route(metadata, args.route, "validation")
    mean, std, norm_details = fit_normalization(arrays, train, channels)
    config = TrainingConfig(
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience,
    )
    name = f"RoPETransformer_JointK16_{args.route}_seed{args.seed}"
    output_dir = args.models_dir / name
    report_path = output_dir / "report.json"
    if report_path.exists() and not args.force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(json.dumps({"status": "reused", "model": name, "report": str(report_path)}), flush=True)
        return report
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    train_ds = RoutedInvariantTokens(arrays, train, channels)
    validation_ds = RoutedInvariantTokens(arrays, validation, channels)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model_config = InvariantTransformerConfig(physical_channels=len(channels))
    model = NormalizedInvariantTransformer(InvariantTokenTransformer(model_config), mean, std).to(device)
    parameter_count = trainable_parameter_count(model)
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"parameter count {parameter_count} != historical {EXPECTED_PARAMETERS}")
    metadata_report = {
        "model_name": name,
        "experiment": EXPERIMENT,
        "training_route": args.route,
        "training_sources": list(ROUTES[args.route]),
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "parameter_count": parameter_count,
        "physical_channels_per_token": 16,
        "mask_channels": 5,
        "input_width": len(channels),
        "selected_full_token_channels": channels.tolist(),
        "selection_manifest": str(args.manifest),
        "selection_manifest_byte_sha256": manifest_checks["byte_sha256"],
        "selection_manifest_semantic_sha256": manifest_checks["semantic_sha256"],
        "implementation_git_commit": git_hash(),
        "train_route_hash": route_hash(train),
        "validation_route_hash": route_hash(validation),
        "training_set_audit": route_audit(metadata, args.route),
        "normalization": norm_details,
        "normalization_fit_sources": list(ROUTES[args.route]),
        "checkpoint_selection": "minimum canonical N16 validation BCE only",
        "architecture_frozen": {"d_model": 128, "layers": 4, "heads": 4, "FFN": 192,
                                "dropout": 0.1, "RoPE": "Q/K", "pooling": "CLS"},
        "DPI_training_rows": 0,
        "XY_training_rows": 0,
        "N32_training_rows": 0,
        "exact500_used_for_checkpoint_selection": False,
        "exact5000_accessed": False,
    }
    save_json(output_dir / "model_config.json", metadata_report)
    np.save(output_dir / "input_mean.npy", mean)
    np.save(output_dir / "input_std.npy", std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience,
    )
    criterion = nn.BCEWithLogitsLoss()
    # Keep compatibility with the frozen remote PyTorch 2.0 environment used by
    # the historical Joint K16 runs.
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss, best_epoch, stale = math.inf, 0, 0
    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_sum = 0.0
        epoch_start = time.perf_counter()
        for tokens, target in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(target)
        validation_probability, validation_labels = predict_loader(model, validation_loader, device)
        eps = 1.0e-8
        validation_loss = float(-np.mean(
            validation_labels * np.log(np.maximum(validation_probability, eps))
            + (1 - validation_labels) * np.log(np.maximum(1 - validation_probability, eps))
        ))
        scheduler.step(validation_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_sum / len(train_ds),
            "validation_loss": validation_loss,
            "validation_AUROC": float(roc_auc_score(validation_labels, validation_probability)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        checkpoint = {
            "model_state_dict": model.state_dict(), "model_config": asdict(model_config),
            "training_config": asdict(config), "metadata": metadata_report,
            "epoch": epoch, "validation_loss": validation_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_loss < best_loss - 1.0e-7:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(checkpoint, checkpoint_path)
        else:
            stale += 1
        print(json.dumps({"model": name, **row, "stale": stale}), flush=True)
        if stale >= config.patience:
            break
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_probability, validation_labels = predict_loader(model, validation_loader, device)
    metrics: dict[str, Any] = {
        "N16_validation": {
            "count": len(validation_labels),
            "BCE": best_loss,
            "AUROC": float(roc_auc_score(validation_labels, validation_probability)),
        }
    }
    probabilities: dict[str, np.ndarray] = {}
    if not args.train_only:
        evaluated, probabilities = evaluate(
            model, channels, source_root=args.source_root, cache_dir=args.cache_dir,
            general_cache=args.general_cache, output_dir=output_dir, config=config,
        )
        metrics.update(evaluated)
    report = {
        **metadata_report,
        "status": "complete_train_only" if args.train_only else "complete",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_wall_seconds": time.perf_counter() - started,
        "mean_epoch_seconds": float(np.mean([row["epoch_seconds"] for row in history])),
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "metrics": metrics,
        "N32_evaluated_zero_shot": not args.train_only,
        "N32_finetuned": False,
    }
    save_json(report_path, report)
    print(json.dumps({"status": report["status"], "model": name, "metrics": metrics}, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def get_metric(report: dict[str, Any], benchmark: str, key: str) -> float:
    value = report["metrics"][benchmark][key]
    if value is None:
        return math.nan
    return float(value)


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def historical_reports(source_root: Path) -> list[dict[str, Any]]:
    reports = []
    for seed in SEEDS:
        path = source_root / "models/invariant_transformer_raw_t4_v1" / (
            f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}/report.json"
        )
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def general_ensemble(probabilities: list[np.ndarray], threshold: float = 0.5) -> dict[str, Any]:
    mean_probability = np.mean(np.stack(probabilities), axis=0)
    fp = int((mean_probability >= threshold).sum())
    return {"false_positives": fp, "total": len(mean_probability), "FPR": fp / len(mean_probability),
            "specificity": 1.0 - fp / len(mean_probability)}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    route_names = [name for name in ("Dv1_only", "Dv1_plus_Dpair", "Dv1_plus_DH")
                   if all((args.models_dir / f"RoPETransformer_JointK16_{name}_seed{seed}/report.json").exists()
                          for seed in SEEDS)]
    all_reports: dict[str, list[dict[str, Any]]] = {"historical_full_reference": historical_reports(args.source_root)}
    general_probabilities: dict[str, list[np.ndarray]] = {}
    for route_name in route_names:
        reports, probabilities = [], []
        for seed in SEEDS:
            model_dir = args.models_dir / f"RoPETransformer_JointK16_{route_name}_seed{seed}"
            reports.append(json.loads((model_dir / "report.json").read_text(encoding="utf-8")))
            score_path = model_dir / "scores_N32_general_separable.csv"
            with score_path.open("r", newline="", encoding="utf-8") as stream:
                probabilities.append(np.asarray([float(row["probability"]) for row in csv.DictReader(stream)]))
        all_reports[route_name] = reports
        general_probabilities[route_name] = probabilities
    historical_general = json.loads((args.source_root / "analysis/general_separable_ood_n32_v1/report.json").read_text(encoding="utf-8"))
    metric_specs = [
        ("exact500_AUROC", "N16_exact500", "AUROC"),
        ("exact500_E2_recall", "N16_exact500", "E2_recall"),
        ("exact500_E_gt_2_recall", "N16_exact500", "E_gt_2_recall"),
        ("N32_XY_AUROC", "N32_XY_matched", "AUROC"),
        ("N32_DPI_AUROC", "N32_DPI_general", "AUROC"),
        ("DPI_E3_recall", "N32_DPI_general", "E3_recall"),
        ("DPI_E4_recall", "N32_DPI_general", "E4_recall"),
        ("GeneralSep_FPR", "N32_general_separable", "FPR"),
    ]
    rows: dict[str, Any] = {}
    for route_name, reports in all_reports.items():
        route_row: dict[str, Any] = {"per_seed": {}}
        for label, benchmark, key in metric_specs:
            if route_name == "historical_full_reference" and benchmark == "N32_general_separable":
                values = [float(row["FPR"]) for row in historical_general["per_seed"]]
            else:
                values = [get_metric(report, benchmark, key) for report in reports]
            route_row[label] = {**aggregate(values), "values": dict(zip(map(str, SEEDS), values))}
        if route_name == "historical_full_reference":
            route_row["GeneralSep_probability_average_ensemble"] = historical_general["ensemble_mean_probability"]
        else:
            route_row["GeneralSep_probability_average_ensemble"] = general_ensemble(general_probabilities[route_name])
        rows[route_name] = route_row
    criterion = None
    if "Dv1_only" in rows:
        dv1, historical = rows["Dv1_only"], rows["historical_full_reference"]
        delta_auc = dv1["exact500_AUROC"]["mean"] - historical["exact500_AUROC"]["mean"]
        delta_higher = dv1["exact500_E_gt_2_recall"]["mean"] - historical["exact500_E_gt_2_recall"]["mean"]
        ensemble_fpr = dv1["GeneralSep_probability_average_ensemble"]["FPR"]
        criterion = {
            "exact500_AUROC_delta": delta_auc,
            "exact500_AUROC_delta_pass": delta_auc >= -0.02,
            "exact500_E_gt_2_recall_delta": delta_higher,
            "exact500_E_gt_2_recall_delta_pass": delta_higher >= -0.05,
            "GeneralSep_ensemble_FPR": ensemble_fpr,
            "GeneralSep_FPR_pass": ensemble_fpr <= 0.01,
        }
        criterion["Dv1_only_works"] = all(value for key, value in criterion.items() if key.endswith("_pass"))
    report = {
        "status": "complete" if route_names else "historical_reference_only",
        "experiment": EXPERIMENT,
        "completed_new_routes": route_names,
        "three_seed_results": rows,
        "predefined_Dv1_only_criterion": criterion,
        "next_action": ("stop_no_auxiliary_route_needed" if criterion and criterion["Dv1_only_works"]
                        else "run_Dv1_plus_Dpair_and_Dv1_plus_DH" if criterion else "train_Dv1_only"),
        "exact5000_accessed": False,
    }
    save_json(args.analysis_dir / "aggregate_report.json", report)
    lines = [
        "# Joint K16 training-distribution ablation",
        "",
        "Feature selection is frozen to the historical Joint K16 manifest. This is a development ablation; exact5000 was not accessed.",
        "",
        "| route | exact500 AUROC | E2 recall | E>2 recall | N32 XY AUROC | N32 DPI AUROC | DPI E3 | DPI E4 | General-Sep FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route_name, row in rows.items():
        fields = [route_name] + [
            f"{row[label]['mean']:.4f} ± {row[label]['sample_std']:.4f}" for label, _, _ in metric_specs
        ]
        lines.append("| " + " | ".join(fields) + " |")
    if criterion:
        lines += ["", "## Predefined Dv1-only decision", "", "```json",
                  json.dumps(criterion, indent=2), "```", "", f"Next action: `{report['next_action']}`."]
    (args.analysis_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen historical Joint K16 training-distribution ablation")
    parser.add_argument("command", choices=("audit", "train", "report"))
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/joint_K16.json")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / f"analysis/{EXPERIMENT}")
    parser.add_argument("--models-dir", type=Path, default=ROOT / f"models/{EXPERIMENT}")
    parser.add_argument("--general-cache", type=Path)
    parser.add_argument("--route", choices=tuple(ROUTES), default="Dv1_only")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    args.cache_dir = (args.cache_dir or args.source_root / "analysis/invariant_transformer_raw_t4_v1/cache").resolve()
    args.general_cache = (args.general_cache or args.source_root / "analysis/general_separable_ood_n32_v1/Inv123RawT4_Robs4.npy").resolve()
    return args


def main() -> None:
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is closed for this development ablation")
    args = parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.command == "audit":
        run_audit(args)
    elif args.command == "train":
        train_route(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()

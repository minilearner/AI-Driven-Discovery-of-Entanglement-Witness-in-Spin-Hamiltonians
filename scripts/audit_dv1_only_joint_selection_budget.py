"""Audit a Dv1-only selector and retrain frozen-ranking K budgets.

The selector is diagnostic only: budget models always use the pre-existing,
frozen mixed-source consensus ranking. Historical outputs are never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import local_inv123_raw_t4_channel_manifest
from run_raw_t4_invariant_transformer import (
    SELECTOR_SEEDS,
    TRAINING_SEEDS,
    SelectorConfig,
    TrainingConfig,
    consensus_selector,
    load_cached_inputs,
    save_json,
    selection_payload,
    train_model,
    train_selector,
)
from training_routes import load_routes, route_hash


SOURCE = ROOT / "analysis/invariant_transformer_raw_t4_v1"
OUT = ROOT / "analysis/dv1_only_joint_selection_budget_audit_v1"
MODEL_OUT = ROOT / "models/dv1_only_joint_budget_audit_v1"
BUDGETS = (4, 8, 12, 16, 20, 24, 28, 32)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dv1_route() -> dict:
    _, original = load_routes(ROOT / "analysis/connected_siamese_v1")
    routes = {}
    for split in ("train", "validation", "test"):
        source, rows, labels = original["routes"][split]
        keep = source == 0
        routes[split] = (source[keep], rows[keep], labels[keep])
    counts = {key: len(value[1]) for key, value in routes.items()}
    if counts != {"train": 35000, "validation": 7500, "test": 7500}:
        raise RuntimeError(f"unexpected Dv1-only route: {counts}")
    return {
        "routes": routes,
        "route_hash": route_hash(*routes["train"], *routes["validation"], *routes["test"]),
        "training_composition": {"D_v1_rows": counts["train"], "D_pair_rows": 0, "D_H_rows": 0},
        "source_parent_split_audit": original["parent_audit"]["D_v1"],
    }


def train_only_statistics(dv1: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation-tied moments on valid supports of Dv1 training rows only."""
    manifest = local_inv123_raw_t4_channel_manifest(4)
    if len(manifest) != 122:
        raise RuntimeError("117 physical channels plus five masks expected")
    groups: dict[int, list[int]] = {}
    for channel, item in enumerate(manifest[:117]):
        family = item["family"]
        if family == "one_body":
            length = 16
        elif family == "pair":
            length = 16 - int(item["pair_distance"])
        elif family == "contiguous_three_body":
            length = 14
        elif family == "contiguous_four_body":
            length = 13
        else:
            raise RuntimeError(f"unknown physical family: {family}")
        groups.setdefault(length, []).append(channel)
    total = np.zeros(122, np.float64)
    squares = np.zeros(122, np.float64)
    counts = np.zeros(122, np.int64)
    for start in range(0, len(rows), 1024):
        batch = np.asarray(dv1[rows[start:start + 1024]], np.float64)
        for length, channels in groups.items():
            values = batch[:, :length, channels].reshape(-1, len(channels))
            total[channels] += values.sum(axis=0)
            squares[channels] += np.square(values).sum(axis=0)
            counts[channels] += len(values)
    mean = np.zeros(122, np.float32)
    std = np.ones(122, np.float32)
    mean[:117] = (total[:117] / counts[:117]).astype(np.float32)
    variance = squares[:117] / counts[:117] - np.square(mean[:117].astype(np.float64))
    std[:117] = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise RuntimeError("invalid Dv1-only input moments")
    return mean, std


def prepare(route_data: dict) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
    source_cache = SOURCE / "cache"
    arrays = load_cached_inputs(source_cache)
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in (
        "exact500_Inv123RawT4_Robs4.npy",
        "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy",
        "N32_XY_Inv123RawT4_Robs4.npy",
        "N32_DPI_Inv123RawT4_Robs4.npy",
    ):
        link = cache / name
        if not link.exists():
            os.symlink(source_cache / name, link)
    mean, std = train_only_statistics(arrays[0], route_data["routes"]["train"][1])
    np.save(cache / "input_mean_full.npy", mean)
    np.save(cache / "input_std_full.npy", std)
    frozen = json.loads((SOURCE / "joint_channel_selection_summary.json").read_text(encoding="utf-8"))
    ranking = [int(x) for x in frozen["consensus_ranking_best_to_worst_full_channels"]]
    if sorted(ranking) != list(range(117)):
        raise RuntimeError("frozen ranking is not a permutation of 117 channels")
    manifest = local_inv123_raw_t4_channel_manifest(4)
    for k in BUDGETS:
        physical = ranking[:k]
        payload = selection_payload(
            family="joint", budget=k, physical=physical,
            selected_rows=[manifest[index] for index in physical],
        )
        save_json(OUT / "selection_manifests" / f"joint_K{k}.json", payload)
    geometry = SOURCE / "dpi_e4_certificate_geometry.json"
    target_geometry = OUT / geometry.name
    if not target_geometry.exists():
        os.symlink(geometry, target_geometry)
    save_json(OUT / "audit_manifest.json", {
        "status": "prepared", "training_sources": ["D_v1"],
        "train_validation_test_counts": {k: len(v[1]) for k, v in route_data["routes"].items()},
        "Dv1_only_route_hash": route_data["route_hash"],
        "historical_mixed_route_hash": "d8e1050d8b9cda6459935a8480aa2c1bc50ee9dc82f194ebc549957492e8c819",
        "frozen_ranking_source": str(SOURCE / "joint_channel_selection_summary.json"),
        "frozen_ranking_sha256": file_sha256(SOURCE / "joint_channel_selection_summary.json"),
        "frozen_top16": ranking[:16],
        "input_moments": "Dv1 training rows, valid anchors only; five masks mean 0/std 1",
        "selector_is_diagnostic_only": True,
        "budget_training_seeds": list(TRAINING_SEEDS),
        "selector_seeds": list(SELECTOR_SEEDS),
    })
    return arrays, mean, std, ranking


def selector_audit(arrays: list[np.ndarray], route_data: dict,
                   mean: np.ndarray, std: np.ndarray, frozen: list[int], device: torch.device) -> None:
    config = SelectorConfig()
    reports = [
        train_selector("joint", seed, arrays=arrays, route_data=route_data,
            full_mean=mean, full_std=std, old16=tuple(frozen[:16]),
            output_dir=MODEL_OUT / f"selectors/joint_seed{seed}", device=device, config=config)
        for seed in SELECTOR_SEEDS
    ]
    summary = consensus_selector("joint", reports, old16=tuple(frozen[:16]),
                                 analysis_dir=OUT / "diagnostic_selector")
    new_rank = [int(x) for x in summary["consensus_ranking_best_to_worst_full_channels"]]
    old16 = set(frozen[:16]); new16 = set(new_rank[:16])
    save_json(OUT / "selector_comparison.json", {
        "training_sources": ["D_v1"], "diagnostic_only": True,
        "frozen_top16": frozen[:16], "Dv1_only_top16": new_rank[:16],
        "overlap_count": len(old16 & new16),
        "retained_full_channel_indices": sorted(old16 & new16),
        "new_only_full_channel_indices": sorted(new16 - old16),
        "frozen_only_full_channel_indices": sorted(old16 - new16),
        "frozen_channels_new_ranks": {str(ch): new_rank.index(ch) + 1 for ch in frozen[:16]},
    })


def budget_training(arrays: list[np.ndarray], route_data: dict, device: torch.device,
                    budgets: tuple[int, ...]) -> None:
    for k in budgets:
        for seed in TRAINING_SEEDS:
            train_model("joint", k, seed, arrays=arrays, route_data=route_data,
                cache_dir=OUT / "cache", analysis_dir=OUT, models_dir=MODEL_OUT,
                device=device, config=TrainingConfig(seed=seed), force=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "selector", "budget", "all"), default="all")
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    args = parser.parse_args()
    if not set(args.budgets) <= set(BUDGETS):
        raise ValueError(f"budgets must be drawn from {BUDGETS}")
    if args.mode != "prepare" and not torch.cuda.is_available():
        raise RuntimeError("GPU required for selector and budget audit")
    torch.set_num_threads(2)
    route_data = dv1_route()
    arrays, mean, std, ranking = prepare(route_data)
    if args.mode == "prepare":
        print(json.dumps({"status": "prepared", "route_hash": route_data["route_hash"]}), flush=True)
        return
    device = torch.device("cuda")
    if args.mode in ("selector", "all"):
        selector_audit(arrays, route_data, mean, std, ranking, device)
    if args.mode in ("budget", "all"):
        budget_training(arrays, route_data, device, tuple(args.budgets))


if __name__ == "__main__":
    main()

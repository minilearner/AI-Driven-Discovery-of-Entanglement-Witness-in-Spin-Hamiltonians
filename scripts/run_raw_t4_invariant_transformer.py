from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import (
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    SelectiveChannelGatedInvariantTransformer,
    local_inv123_raw_t4_channel_manifest,
    raw_t4_invariant_manifest,
    trainable_parameter_count,
)
from training_routes import BASE, load_routes
from train_invariant_transformer import (
    RoutedInvariantTokens,
    benchmark_metrics,
    exact500_labels,
    git_hash,
    predict_array,
    predict_loader,
    seed_all,
    write_scores,
)


SELECTOR_SEEDS = (2026090301, 2026090302, 2026090303, 2026090304, 2026090305)
TRAINING_SEEDS = (20260829, 20260830, 20260831)
FIXED_ADDITIONS = (2, 4, 8, 16, 40)
JOINT_BUDGETS = (16, 20, 24, 32, 48)


@dataclass(frozen=True)
class SelectorConfig:
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 160
    patience: int = 20
    scheduler_patience: int = 5
    gradient_clip: float = 5.0
    warmup_epochs: int = 25
    sparsity_penalty: float = 0.01
    budget_penalty: float = 20.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    max_epochs: int = 120
    patience: int = 15
    scheduler_patience: int = 4
    gradient_clip: float = 5.0
    threshold: float = 0.5


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_cached_inputs(cache_dir: Path) -> list[np.ndarray]:
    arrays = [np.load(cache_dir / f"{name}_Inv123RawT4_Robs4.npy", mmap_mode="r") for name in BASE]
    if any(array.ndim != 3 or array.shape[1:] != (16, 122) for array in arrays):
        raise RuntimeError("raw-T4 training cache shape changed")
    return arrays


def old16_channels(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = tuple(int(value) for value in payload["selected_physical_token_channels"])
    if len(channels) != 16 or max(channels) >= 77:
        raise RuntimeError("frozen old K=16 channel set is invalid")
    return channels


def selector_layout(kind: str, old16: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...], int]:
    if kind == "t4":
        candidate_full = tuple(range(77, 117))
        selected_full = np.asarray((*old16, *candidate_full, *range(117, 122)), dtype=np.int64)
        gate_positions = tuple(range(16, 56))
        target = 16
    elif kind == "joint":
        candidate_full = tuple(range(117))
        selected_full = np.arange(122, dtype=np.int64)
        gate_positions = tuple(range(117))
        target = 32
    else:
        raise ValueError(kind)
    return selected_full, gate_positions, candidate_full, target


@torch.no_grad()
def selector_validation(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities, labels, loss = [], [], 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for tokens, target in loader:
        tokens, target = tokens.to(device, non_blocking=True), target.to(device, non_blocking=True)
        logits = model(tokens)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())
        loss += float(criterion(logits, target))
    return np.concatenate(probabilities), np.concatenate(labels), loss / len(loader.dataset)


def train_selector(
    kind: str,
    seed: int,
    *,
    arrays: list[np.ndarray],
    route_data: dict[str, Any],
    full_mean: np.ndarray,
    full_std: np.ndarray,
    old16: tuple[int, ...],
    output_dir: Path,
    device: torch.device,
    config: SelectorConfig,
) -> dict[str, Any]:
    report_path = output_dir / "channel_ranking.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    selected_full, gate_positions, candidate_full, target_count = selector_layout(kind, old16)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], selected_full)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], selected_full)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=config.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )
    model = SelectiveChannelGatedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=len(selected_full))),
        full_mean[selected_full], full_std[selected_full], gate_positions,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.scheduler_patience, factor=0.5, min_lr=1.0e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    candidate_count = len(candidate_full)
    best_objective, best_state, best_epoch, stale = math.inf, None, 0, 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        ramp = min(1.0, epoch / config.warmup_epochs)
        train_total = train_bce = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                classification = criterion(logits, labels)
                expected = model.expected_gated_channels()
                sparse = (
                    config.sparsity_penalty * expected / candidate_count
                    + config.budget_penalty * ((expected - target_count) / candidate_count).square()
                )
                loss = classification + ramp * sparse
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * len(labels)
            train_bce += float(classification.detach()) * len(labels)
        probability, labels, validation_bce = selector_validation(model, validation_loader, device)
        expected_count = float(model.expected_gated_channels().detach())
        objective = validation_bce + 0.02 * abs(expected_count - target_count) / target_count
        scheduler.step(objective)
        row = {
            "epoch": epoch,
            "train_total_loss": train_total / len(train_ds),
            "train_BCE": train_bce / len(train_ds),
            "validation_BCE": validation_bce,
            "validation_AUROC": float(roc_auc_score(labels, probability)),
            "expected_gated_channels": expected_count,
            "selection_objective": objective,
            "budget_ramp": ramp,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if epoch >= config.warmup_epochs and objective < best_objective - 1e-8:
            best_objective, best_epoch, stale = objective, epoch, 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        elif epoch >= config.warmup_epochs:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale >= config.patience:
            print(json.dumps({"selector": kind, "seed": seed, **row, "stale": stale}), flush=True)
        if epoch >= config.warmup_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("selector produced no post-warmup checkpoint")
    model.load_state_dict(best_state)
    probability, labels, validation_bce = selector_validation(model, validation_loader, device)
    log_alpha = model.gates.log_alpha.detach().cpu().numpy()
    gate_probability = model.gates.nonzero_probability().detach().cpu().numpy()
    order = np.lexsort((np.arange(candidate_count), -log_alpha))
    rank = np.empty(candidate_count, dtype=np.int64)
    rank[order] = np.arange(1, candidate_count + 1)
    combined_manifest = local_inv123_raw_t4_channel_manifest(4)
    channels = []
    for candidate, full_channel in enumerate(candidate_full):
        channels.append({
            **combined_manifest[full_channel],
            "candidate_channel": candidate,
            "full_token_channel": full_channel,
            "rank": int(rank[candidate]),
            "gate_probability": float(gate_probability[candidate]),
            "log_alpha": float(log_alpha[candidate]),
        })
    report = {
        "status": "complete",
        "selector_kind": kind,
        "selector_seed": seed,
        "implementation_git_commit": git_hash(),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_wall_seconds": time.perf_counter() - started,
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "parameter_count": trainable_parameter_count(model),
        "route_hash": route_data["route_hash"],
        "candidate_count": candidate_count,
        "target_gated_channels": target_count,
        "fixed_old16_context": list(old16) if kind == "t4" else [],
        "ranking_best_to_worst_full_channels": [int(candidate_full[index]) for index in order],
        "validation": {"BCE": validation_bce, "AUROC": float(roc_auc_score(labels, probability))},
        "channels": channels,
        "config": asdict(config),
        "N32_used_for_selection": False,
        "exact500_used_for_selection": False,
        "exact5000_accessed": False,
    }
    torch.save({"model_state_dict": best_state, "report": report}, output_dir / "best.pt")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    save_json(report_path, report)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def consensus_selector(
    kind: str,
    reports: list[dict[str, Any]],
    *,
    old16: tuple[int, ...],
    analysis_dir: Path,
) -> dict[str, Any]:
    candidate_full = tuple(range(77, 117)) if kind == "t4" else tuple(range(117))
    ranks = np.empty((len(reports), len(candidate_full)), dtype=np.float64)
    log_alpha = np.empty_like(ranks)
    for seed_index, report in enumerate(reports):
        for row in report["channels"]:
            candidate = int(row["candidate_channel"])
            ranks[seed_index, candidate] = int(row["rank"])
            log_alpha[seed_index, candidate] = float(row["log_alpha"])
    rank_mean = ranks.mean(axis=0)
    score_mean = log_alpha.mean(axis=0)
    order = np.lexsort((np.arange(len(candidate_full)), -score_mean, rank_mean))
    ranking = [int(candidate_full[index]) for index in order]
    combined_manifest = local_inv123_raw_t4_channel_manifest(4)
    summary = {
        "status": "complete",
        "selector_kind": kind,
        "selection_method": "five-seed consensus mean-rank; mean log-alpha and channel index break ties",
        "selector_seeds": [int(report["selector_seed"]) for report in reports],
        "candidate_count": len(candidate_full),
        "consensus_ranking_best_to_worst_full_channels": ranking,
        "channels": [
            {
                **combined_manifest[full_channel],
                "full_token_channel": full_channel,
                "rank_mean": float(rank_mean[candidate]),
                "rank_sample_std": float(ranks[:, candidate].std(ddof=1)),
                "log_alpha_mean": float(score_mean[candidate]),
                "consensus_rank": ranking.index(full_channel) + 1,
            }
            for candidate, full_channel in enumerate(candidate_full)
        ],
        "N32_used_for_selection": False,
        "exact500_used_for_selection": False,
        "exact5000_accessed": False,
    }
    save_json(analysis_dir / f"{kind}_channel_selection_summary.json", summary)
    manifest_dir = analysis_dir / "selection_manifests"
    if kind == "t4":
        for addition in (0, *FIXED_ADDITIONS):
            physical = [*old16, *ranking[:addition]]
            payload = selection_payload(
                family="fixed", budget=addition, physical=physical,
                selected_rows=[combined_manifest[index] for index in physical],
            )
            save_json(manifest_dir / f"fixed16_addT4_{addition}.json", payload)
    else:
        for budget in (*JOINT_BUDGETS, 117):
            physical = ranking[:budget]
            payload = selection_payload(
                family="joint", budget=budget, physical=physical,
                selected_rows=[combined_manifest[index] for index in physical],
            )
            save_json(manifest_dir / f"joint_K{budget}.json", payload)
    return summary


def selection_payload(
    *, family: str, budget: int, physical: list[int], selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    t4_count = sum(index >= 77 for index in physical)
    return {
        "status": "complete",
        "family": family,
        "budget": budget,
        "physical_channel_count": len(physical),
        "T4_channel_count": t4_count,
        "mask_channel_count": 5,
        "input_width": len(physical) + 5,
        "selected_physical_full_channels": physical,
        "selected_full_token_channels": [*physical, *range(117, 122)],
        "selected_channels": selected_rows,
        "N32_used_for_selection": False,
        "exact500_used_for_selection": False,
        "exact5000_accessed": False,
    }


def n16_balanced_labels() -> dict[str, np.ndarray]:
    with h5py.File(ROOT / "data/exactXY_balanced_N16_factorized_v1/dataset.h5", "r") as h5:
        return {
            "sample_id": h5["metadata/sample_id"][:],
            "origin": h5["metadata/origin"].asstr()[:],
            "separable": h5["labels/certified_separable"][:],
            "entangled": h5["labels/certified_entangled"][:],
            "E2": h5["labels/E2"][:],
            "E_gt_2": h5["labels/E_gt_2"][:],
            "unknown": h5["labels/unknown"][:],
        }


def n32_labels(group: str) -> dict[str, np.ndarray]:
    path = (
        ROOT / "data/xy_n32_pbc_matched_exact500_v1/dataset.h5"
        if group == "xy" else ROOT / "data/dpi_n32_general_v1/D_PI_N32.h5"
    )
    with h5py.File(path, "r") as h5:
        prefix = "params" if group == "xy" else "meta"
        return {
            "sample_id": h5[f"{prefix}/sample_id"][:],
            "separable": h5["labels/strict_separable"][:],
            "entangled": h5["labels/binary_entangled"][:],
            "E2": h5["labels/E2"][:],
            "E3": h5["labels/E3"][:],
            "E4": h5["labels/E4"][:],
            "unknown": h5["labels/unknown"][:],
        }


def geometry_metrics(
    probability: np.ndarray, labels: dict[str, np.ndarray], geometry_path: Path, threshold: float,
) -> dict[str, Any]:
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    e4_indices = np.flatnonzero(labels["E4"])
    contiguous_values = np.asarray(geometry["contiguous_mask"], dtype=bool)
    if len(e4_indices) != len(contiguous_values):
        raise RuntimeError("DPI E4 geometry rows no longer match labels")
    result = {}
    for name, local_mask in (("contiguous", contiguous_values), ("noncontiguous", ~contiguous_values)):
        positive = np.zeros_like(labels["E4"], dtype=bool)
        positive[e4_indices[local_mask]] = True
        selected = labels["separable"] | positive
        if positive.any():
            auc = float(roc_auc_score(positive[selected].astype(np.int8), probability[selected]))
            recall = float((probability[positive] >= threshold).mean())
        else:
            auc = recall = None
        result[name] = {
            "E4_count": int(positive.sum()),
            "recall": recall,
            "Sep_vs_E4_AUROC": auc,
        }
    return result


def model_job(family: str, budget: int, analysis_dir: Path) -> tuple[str, Path]:
    if family == "full":
        return "RawT4_Full", analysis_dir / "selection_manifests/joint_K117.json"
    if family == "fixed":
        return f"Fixed16_AddT4_{budget}", analysis_dir / f"selection_manifests/fixed16_addT4_{budget}.json"
    if family == "joint":
        return f"JointRawT4_K{budget}", analysis_dir / f"selection_manifests/joint_K{budget}.json"
    raise ValueError(family)


def train_model(
    family: str,
    budget: int,
    seed: int,
    *,
    arrays: list[np.ndarray],
    route_data: dict[str, Any],
    cache_dir: Path,
    analysis_dir: Path,
    models_dir: Path,
    device: torch.device,
    config: TrainingConfig,
    force: bool,
) -> dict[str, Any]:
    prefix, manifest_path = model_job(family, budget, analysis_dir)
    name = f"RoPETransformer_LocalInv123_RawT4_{prefix}_seed{seed}"
    output_dir = models_dir / name
    report_path = output_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = np.asarray(selection["selected_full_token_channels"], dtype=np.int64)
    full_mean = np.load(cache_dir / "input_mean_full.npy")
    full_std = np.load(cache_dir / "input_std_full.npy")
    mean, std = full_mean[channels], full_std[channels]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], channels)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], channels)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model_config = InvariantTransformerConfig(physical_channels=len(channels))
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(model_config), mean, std,
    ).to(device)
    metadata = {
        "model_name": name,
        "family": family,
        "budget": budget,
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "parameter_count": trainable_parameter_count(model),
        "physical_channels_per_token": int(selection["physical_channel_count"]),
        "T4_channels_per_token": int(selection["T4_channel_count"]),
        "mask_channels": 5,
        "input_width": len(channels),
        "selected_full_token_channels": channels.tolist(),
        "selection_manifest": str(manifest_path),
        "implementation_git_commit": git_hash(),
        "route_hash": route_data["route_hash"],
        "checkpoint_selection": "minimum canonical N16 validation BCE only",
        "architecture_frozen": {"d_model": 128, "layers": 4, "heads": 4, "FFN": 192, "dropout": 0.1, "RoPE": "Q/K", "pooling": "CLS"},
        "raw_T4_only": True,
        "K4_or_connected_T4_included": False,
        "DPI_training_rows": 0,
        "N32_training_rows": 0,
        "N32_used_for_selection": False,
        "exact500_used_for_checkpoint_selection": False,
        "exact5000_accessed": False,
    }
    save_json(output_dir / "model_config.json", metadata)
    np.save(output_dir / "input_mean.npy", mean)
    np.save(output_dir / "input_std.npy", std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience,
    )
    criterion = nn.BCEWithLogitsLoss()
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
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(labels)
        validation_probability, validation_labels = predict_loader(model, validation_loader, device)
        eps = 1e-8
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
            "training_config": asdict(config), "metadata": metadata, "epoch": epoch,
            "validation_loss": validation_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(checkpoint, checkpoint_path)
        else:
            stale += 1
        print(json.dumps({"model": name, **row, "stale": stale}), flush=True)
        if stale >= config.patience:
            break
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    exact_values = np.load(cache_dir / "exact500_Inv123RawT4_Robs4.npy", mmap_mode="r")
    balanced_values = np.load(cache_dir / "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy", mmap_mode="r")
    xy_values = np.load(cache_dir / "N32_XY_Inv123RawT4_Robs4.npy", mmap_mode="r")
    dpi_values = np.load(cache_dir / "N32_DPI_Inv123RawT4_Robs4.npy", mmap_mode="r")
    exact_probability = predict_array(model, np.asarray(exact_values)[:, :, channels], device, config.eval_batch_size)
    balanced_probability = predict_array(model, np.asarray(balanced_values)[:, :, channels], device, config.eval_batch_size)
    xy_probability = predict_array(model, np.asarray(xy_values)[:, :, channels], device, config.eval_batch_size)
    dpi_probability = predict_array(model, np.asarray(dpi_values)[:, :, channels], device, config.eval_batch_size)
    exact_label = exact500_labels()
    balanced_label = n16_balanced_labels()
    xy_label = n32_labels("xy")
    dpi_label = n32_labels("dpi")
    metrics = {
        "N16_validation": {
            "count": len(validation_labels), "BCE": best_loss,
            "AUROC": float(roc_auc_score(validation_labels, validation_probability)),
        },
        "N16_exact500": benchmark_metrics(
            exact_probability, separable=exact_label["separable"], entangled=exact_label["entangled"],
            e2=exact_label["E2"], unknown=exact_label["unknown"], threshold=config.threshold,
        ),
        "XY_factorized_balanced_N16": benchmark_metrics(
            balanced_probability, separable=balanced_label["separable"], entangled=balanced_label["entangled"],
            e2=balanced_label["E2"], unknown=balanced_label["unknown"], threshold=config.threshold,
        ),
        "N32_XY_matched": benchmark_metrics(
            xy_probability, separable=xy_label["separable"], entangled=xy_label["entangled"],
            e2=xy_label["E2"], e3=xy_label["E3"], e4=xy_label["E4"],
            unknown=xy_label["unknown"], threshold=config.threshold,
        ),
        "N32_DPI_general": benchmark_metrics(
            dpi_probability, separable=dpi_label["separable"], entangled=dpi_label["entangled"],
            e2=dpi_label["E2"], e3=dpi_label["E3"], e4=dpi_label["E4"],
            unknown=dpi_label["unknown"], threshold=config.threshold,
        ),
    }
    metrics["N32_DPI_E4_geometry"] = geometry_metrics(
        dpi_probability, dpi_label, analysis_dir / "dpi_e4_certificate_geometry.json", config.threshold,
    )
    write_scores(output_dir / "scores_N16_exact500.csv", exact_probability, exact_label)
    write_scores(output_dir / "scores_XY_factorized_balanced_N16.csv", balanced_probability, balanced_label)
    write_scores(output_dir / "scores_N32_XY_matched.csv", xy_probability, xy_label)
    write_scores(output_dir / "scores_N32_DPI_general.csv", dpi_probability, dpi_label)
    report = {
        **metadata,
        "status": "complete",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_wall_seconds": time.perf_counter() - started,
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "metrics": metrics,
        "N32_evaluated_zero_shot": True,
        "N32_finetuned": False,
    }
    save_json(report_path, report)
    print(json.dumps({"status": "complete", "model": name, "metrics": metrics}, indent=2), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw-T4-only Local Inv123 Transformer experiment.")
    parser.add_argument("--mode", choices=("selectors", "summarize", "train", "all"), default="all")
    parser.add_argument("--selector-kind", choices=("t4", "joint", "both"), default="both")
    parser.add_argument("--family", choices=("full", "fixed", "joint", "all"), default="all")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAINING_SEEDS)
    parser.add_argument("--selector-seeds", nargs="+", type=int, default=SELECTOR_SEEDS)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis/invariant_transformer_raw_t4_v1")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/invariant_transformer_raw_t4_v1")
    parser.add_argument("--route-cache-dir", type=Path, default=ROOT / "analysis/connected_siamese_v1")
    parser.add_argument("--old16-manifest", type=Path, default=ROOT / "analysis/invariant_transformer_channel_sparsity_window_v1/selection_manifests/K16.json")
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--selector-max-epochs", type=int, default=SelectorConfig.max_epochs)
    parser.add_argument("--selector-patience", type=int, default=SelectorConfig.patience)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is closed for this experiment")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    cache_dir = args.analysis_dir / "cache"
    arrays = load_cached_inputs(cache_dir)
    _, route_data = load_routes(args.route_cache_dir)
    full_mean = np.load(cache_dir / "input_mean_full.npy")
    full_std = np.load(cache_dir / "input_std_full.npy")
    frozen_old16 = old16_channels(args.old16_manifest)
    kinds = ("t4", "joint") if args.selector_kind == "both" else (args.selector_kind,)
    if args.mode in {"selectors", "all"}:
        selector_config = SelectorConfig(
            max_epochs=args.selector_max_epochs, patience=args.selector_patience,
        )
        for kind in kinds:
            reports = [
                train_selector(
                    kind, seed, arrays=arrays, route_data=route_data,
                    full_mean=full_mean, full_std=full_std, old16=frozen_old16,
                    output_dir=args.models_dir / f"selectors/{kind}_seed{seed}",
                    device=device, config=selector_config,
                )
                for seed in args.selector_seeds
            ]
            consensus_selector(kind, reports, old16=frozen_old16, analysis_dir=args.analysis_dir)
    if args.mode == "summarize":
        for kind in kinds:
            reports = [
                json.loads((args.models_dir / f"selectors/{kind}_seed{seed}/channel_ranking.json").read_text(encoding="utf-8"))
                for seed in args.selector_seeds
            ]
            consensus_selector(kind, reports, old16=frozen_old16, analysis_dir=args.analysis_dir)
    if args.mode in {"train", "all"}:
        config_factory = lambda seed: TrainingConfig(
            seed=seed, max_epochs=args.max_epochs, patience=args.patience,
        )
        jobs: list[tuple[str, int]] = []
        if args.family in {"full", "all"}:
            jobs.append(("full", 117))
        if args.family in {"fixed", "all"}:
            budgets = (args.budget,) if args.budget is not None else FIXED_ADDITIONS
            jobs.extend(("fixed", int(value)) for value in budgets)
        if args.family in {"joint", "all"}:
            budgets = (args.budget,) if args.budget is not None else JOINT_BUDGETS
            jobs.extend(("joint", int(value)) for value in budgets)
        completed = []
        for family, budget in jobs:
            for seed in args.seeds:
                report = train_model(
                    family, budget, seed, arrays=arrays, route_data=route_data,
                    cache_dir=cache_dir, analysis_dir=args.analysis_dir,
                    models_dir=args.models_dir, device=device,
                    config=config_factory(seed), force=args.force,
                )
                completed.append({
                    "model": report["model_name"],
                    "N16_AUROC": report["metrics"]["N16_exact500"]["AUROC"],
                    "N32_DPI_AUROC": report["metrics"]["N32_DPI_general"]["AUROC"],
                    "N32_DPI_E4_recall": report["metrics"]["N32_DPI_general"]["E4_recall"],
                    "N32_DPI_Sep_vs_E4_AUROC": report["metrics"]["N32_DPI_general"]["Sep_vs_E4"]["AUROC"],
                })
        print(json.dumps({"status": "complete", "runs": completed}, indent=2), flush=True)


if __name__ == "__main__":
    main()

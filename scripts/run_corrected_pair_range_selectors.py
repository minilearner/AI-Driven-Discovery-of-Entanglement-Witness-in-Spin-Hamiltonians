from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    SelectiveChannelGatedInvariantTransformer,
    trainable_parameter_count,
)
from run_pair_range_r8_models import (  # noqa: E402
    cutoff_layout,
    load_arrays,
    semantic_key,
    sha256,
)
from run_raw_t4_invariant_transformer import (  # noqa: E402
    SelectorConfig,
    TrainingConfig,
    save_json,
    selector_validation,
)
from training_routes import load_routes  # noqa: E402
from train_invariant_transformer import (  # noqa: E402
    RoutedInvariantTokens,
    git_hash,
    predict_loader,
    seed_all,
)


EXPERIMENT = "corrected_pair_range_selector_v1"
# Use the exact historical Joint T1-T4 selector seeds so the rmax=4 gate is a
# paired algorithmic regression rather than a conflation of code and RNG drift.
SELECTOR_SEEDS = (2026090301, 2026090302, 2026090303, 2026090304, 2026090305)
TRAINING_SEEDS = (20260829, 20260830, 20260831)
PRIMARY_CUTOFFS = (6, 8)
PRIMARY_BUDGETS = (12, 16, 20, 24, 32)
SELECTOR_TARGET = 32
WARMUP_EPOCHS = 25
ACTIVE_TOLERANCE_FRACTION = 0.20
ACTIVE_TOLERANCE_MINIMUM = 2.0


def selector_stem(cutoff: int, target_k: int, seed: int) -> str:
    return f"rmax{cutoff}_K{target_k}_seed{seed}"


def selection_paths(analysis_dir: Path, cutoff: int, target_k: int) -> tuple[Path, Path, Path]:
    ranking = analysis_dir / f"channel_importance_r{cutoff}_K{target_k}.csv"
    consensus_path = analysis_dir / f"consensus_rmax{cutoff}_K{target_k}.json"
    manifest = analysis_dir / f"selection_manifests/rmax{cutoff}_K{target_k}.json"
    return ranking, consensus_path, manifest


def allowed_active_deviation(target_k: int) -> float:
    return max(ACTIVE_TOLERANCE_MINIMUM, ACTIVE_TOLERANCE_FRACTION * target_k)


def full_selection_objective(
    validation_bce: float,
    expected_count: float,
    target_k: int,
    candidate_count: int,
    config: SelectorConfig,
) -> tuple[float, float, float]:
    """Return validation BCE plus the same sparsity and budget terms as training."""
    sparsity = config.sparsity_penalty * expected_count / candidate_count
    budget = config.budget_penalty * ((expected_count - target_k) / candidate_count) ** 2
    return validation_bce + sparsity + budget, sparsity, budget


def selector_control_transition(
    *,
    epoch: int,
    warmup_epochs: int,
    objective: float,
    best_objective: float,
    stale: int,
    patience: int,
) -> tuple[bool, int, bool]:
    """Return (save_checkpoint, next_stale, stop) with strict warmup protection."""
    if epoch < warmup_epochs:
        return False, stale, False
    improved = objective < best_objective - 1.0e-8
    next_stale = 0 if improved else stale + 1
    return improved, next_stale, next_stale >= patience


def exact_topk_feature_gates(
    model: SelectiveChannelGatedInvariantTransformer,
    target_k: int,
) -> torch.Tensor:
    """Return an exact-K binary gate with all ungated context channels enabled.

    Hard-concrete ``nonzero_probability`` and its deterministic stretched gate
    are different quantities. At small budgets the deterministic gate can be
    zero for most channels while the expected nonzero count is still above K.
    Selector validation must therefore measure the actual top-K subset that is
    handed to the compact model.
    """
    count = len(model.gate_positions)
    if not 1 <= target_k <= count:
        raise ValueError(f"target K={target_k} is invalid for {count} gates")
    scores = model.gates.log_alpha.detach().cpu().numpy().astype(np.float64)
    order = np.lexsort((np.arange(count), -scores))[:target_k]
    gates = torch.ones(
        model.model.config.physical_channels,
        dtype=model.input_mean.dtype,
        device=model.input_mean.device,
    )
    gates[model.gate_positions] = 0.0
    selected = model.gate_positions[
        torch.as_tensor(order, dtype=torch.long, device=model.gate_positions.device)
    ]
    gates[selected] = 1.0
    return gates


@torch.no_grad()
def selector_topk_validation(
    model: SelectiveChannelGatedInvariantTransformer,
    loader: DataLoader,
    device: torch.device,
    target_k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the exact target-K hard subset, leaving geometry masks on."""
    model.eval()
    gates = exact_topk_feature_gates(model, target_k).reshape(1, 1, -1)
    probabilities, labels = [], []
    loss = 0.0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    for tokens, target in loader:
        tokens, target = tokens.to(device, non_blocking=True), target.to(device, non_blocking=True)
        normalized = (tokens - model.input_mean) / model.input_std
        logits = model.model(normalized * gates)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.cpu().numpy())
        loss += float(criterion(logits, target))
    return np.concatenate(probabilities), np.concatenate(labels), loss / len(loader.dataset)


def train_selector(
    cutoff: int,
    target_k: int,
    seed: int,
    *,
    arrays: list[np.ndarray],
    route_data: dict[str, Any],
    full_mean: np.ndarray,
    full_std: np.ndarray,
    output_dir: Path,
    device: torch.device,
    config: SelectorConfig,
    force: bool,
) -> dict[str, Any]:
    report_path = output_dir / "channel_ranking.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    physical, masks, indices = cutoff_layout(cutoff)
    if not 1 <= target_k <= len(physical):
        raise ValueError(f"target K={target_k} is invalid for {len(physical)} candidates")
    mean, std = full_mean[indices], full_std[indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], indices)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], indices)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = SelectiveChannelGatedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=len(indices))),
        mean,
        std,
        tuple(range(len(physical))),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=1.0e-6,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_objective, best_epoch, stale = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    candidate_count = len(physical)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_total = train_bce = 0.0
        ramp = min(1.0, epoch / config.warmup_epochs)
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                classification = criterion(logits, labels)
                expected = model.expected_gated_channels()
                sparse = (
                    config.sparsity_penalty * expected / candidate_count
                    + config.budget_penalty * ((expected - float(target_k)) / candidate_count).square()
                )
                loss = classification + ramp * sparse
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * len(labels)
            train_bce += float(classification.detach()) * len(labels)
        # Checkpoint and LR decisions use the exact target-K subset that the
        # downstream compact model will receive. The expected gate count stays
        # in the full objective and is audited independently.
        probability, labels, validation_bce = selector_topk_validation(
            model, validation_loader, device, target_k
        )
        expected_count = float(model.expected_gated_channels().detach())
        objective, validation_sparsity, validation_budget = full_selection_objective(
            validation_bce, expected_count, target_k, candidate_count, config
        )
        scheduler.step(objective)
        row = {
            "epoch": epoch,
            "train_total_loss": train_total / len(train_ds),
            "train_BCE": train_bce / len(train_ds),
            "validation_BCE": validation_bce,
            "validation_AUROC": float(roc_auc_score(labels, probability)),
            "expected_active_channels": expected_count,
            "active_minus_target": expected_count - target_k,
            "selection_objective": objective,
            "validation_sparsity_term": validation_sparsity,
            "validation_budget_term": validation_budget,
            "validation_gate_mode": "exact_topK_binary",
            "budget_ramp": ramp,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        # Historical protection: neither checkpointing nor stale accumulation is
        # permitted before the sparsity/budget warmup has fully completed.
        improved, stale, should_stop = selector_control_transition(
            epoch=epoch,
            warmup_epochs=config.warmup_epochs,
            objective=objective,
            best_objective=best_objective,
            stale=stale,
            patience=config.patience,
        )
        if improved:
            best_objective, best_epoch, stale = objective, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps({"selector": selector_stem(cutoff, target_k, seed), **row, "stale": stale}), flush=True)
        if should_stop:
            break
    if best_state is None or best_epoch < config.warmup_epochs:
        raise RuntimeError("selector produced no valid post-warmup checkpoint")
    model.load_state_dict(best_state)
    probability, labels, validation_bce = selector_topk_validation(
        model, validation_loader, device, target_k
    )
    _, _, relaxed_validation_bce = selector_validation(model, validation_loader, device)
    expected_count = float(model.expected_gated_channels().detach())
    active_delta = expected_count - target_k
    tolerance = allowed_active_deviation(target_k)
    valid_selector = bool(abs(active_delta) <= tolerance)
    torch.save(
        {
            "model_state_dict": best_state,
            "cutoff": cutoff,
            "target_K": target_k,
            "seed": seed,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "route_hash": route_data["route_hash"],
        },
        output_dir / "best.pt",
    )
    log_alpha = model.gates.log_alpha.detach().cpu().numpy().astype(np.float64)
    gate_probability = model.gates.nonzero_probability().detach().cpu().numpy().astype(np.float64)
    order = np.lexsort((np.arange(candidate_count), -log_alpha))
    ranks = np.empty(candidate_count, dtype=np.int64)
    ranks[order] = np.arange(1, candidate_count + 1)
    channels = []
    for local_index, channel in enumerate(physical):
        channels.append(
            {
                "candidate_channel": local_index,
                "full_r8_channel": int(channel["channel"]),
                "semantic_key": semantic_key(channel),
                "rank": int(ranks[local_index]),
                "gate_probability": float(gate_probability[local_index]),
                "log_alpha": float(log_alpha[local_index]),
                **{
                    key: channel.get(key)
                    for key in (
                        "family",
                        "raw_or_connected",
                        "pair_distance",
                        "invariant_mathematical_name",
                        "svd_based",
                    )
                },
                "singular_value_index": (channel.get("t4_manifest") or {}).get("singular_value_index"),
            }
        )
    report = {
        "status": "complete" if valid_selector else "failed_active_count",
        "experiment": EXPERIMENT,
        "r_max": cutoff,
        "target_K": target_k,
        "seed": seed,
        "candidate_count": candidate_count,
        "geometry_masks": len(masks),
        "total_input_width": len(indices),
        "best_epoch": best_epoch,
        "warmup_epochs": config.warmup_epochs,
        "best_after_warmup": bool(best_epoch >= config.warmup_epochs),
        "expected_active_channels": expected_count,
        "active_minus_target": active_delta,
        "allowed_active_deviation": tolerance,
        "selection_objective": best_objective,
        "validation_bce": validation_bce,
        "relaxed_deterministic_validation_bce": relaxed_validation_bce,
        "validation_gate_mode": "exact_topK_binary",
        "validation_auroc": float(roc_auc_score(labels, probability)),
        "epochs_completed": len(history),
        "early_stopped": bool(len(history) < config.max_epochs),
        "valid_selector": valid_selector,
        "config": asdict(config),
        "route_hash": route_data["route_hash"],
        "channels": channels,
        "ranking_best_to_worst_full_r8_channels": [int(physical[index]["channel"]) for index in order],
        "checkpoint_sha256": sha256(output_dir / "best.pt"),
        "DPI_training_rows": 0,
        "N32_training_rows": 0,
        "N32_used_for_selection": False,
        "exact500_accessed": False,
        "exact5000_accessed": False,
    }
    save_json(report_path, report)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def build_consensus(
    cutoff: int,
    target_k: int,
    reports: list[dict[str, Any]],
    analysis_dir: Path,
) -> dict[str, Any]:
    if len(reports) != len(SELECTOR_SEEDS):
        raise RuntimeError("consensus requires all five paired selector seeds")
    invalid = [report["seed"] for report in reports if not report.get("valid_selector", False)]
    if invalid:
        raise RuntimeError(f"invalid selectors cannot enter consensus: {invalid}")
    physical, masks, _ = cutoff_layout(cutoff)
    count = len(physical)
    ranks = np.asarray([[row["rank"] for row in report["channels"]] for report in reports], dtype=np.float64)
    scores = np.asarray([[row["log_alpha"] for row in report["channels"]] for report in reports], dtype=np.float64)
    probabilities = np.asarray(
        [[row["gate_probability"] for row in report["channels"]] for report in reports], dtype=np.float64
    )
    rank_mean, rank_std = ranks.mean(axis=0), ranks.std(axis=0, ddof=1)
    score_mean, score_std = scores.mean(axis=0), scores.std(axis=0, ddof=1)
    probability_mean = probabilities.mean(axis=0)
    order = np.lexsort((np.arange(count), -score_mean, rank_mean))
    consensus_rank = np.empty(count, dtype=np.int64)
    consensus_rank[order] = np.arange(1, count + 1)
    rows: list[dict[str, Any]] = []
    for index, channel in enumerate(physical):
        name = channel["invariant_mathematical_name"]
        sv_index = (channel.get("t4_manifest") or {}).get("singular_value_index")
        if sv_index is None and "sv_" in name:
            try:
                sv_index = int(name.rsplit("sv_", 1)[1].split("_", 1)[0])
            except ValueError:
                pass
        row: dict[str, Any] = {
            "channel_id": int(channel["channel"]),
            "semantic_key": semantic_key(channel),
            "order": {
                "one_body": 1,
                "pair": 2,
                "contiguous_three_body": 3,
                "contiguous_four_body": 4,
            }[channel["family"]],
            "family": channel["family"],
            "raw_or_connected": channel["raw_or_connected"],
            "r": channel["pair_distance"],
            "SVD_index": sv_index,
            "invariant_scalar_name": name,
            "SVD_based": bool(channel["svd_based"]),
            "mean_score": float(score_mean[index]),
            "std_score": float(score_std[index]),
            "mean_gate_probability": float(probability_mean[index]),
            "mean_rank": float(rank_mean[index]),
            "rank_std": float(rank_std[index]),
            "selection_frequency_topK": float(np.mean(ranks[:, index] <= target_k)),
            "consensus_rank": int(consensus_rank[index]),
            "normalized_rank": float((consensus_rank[index] - 1) / max(count - 1, 1)),
            "percentile_rank": float(100 * (1 - (consensus_rank[index] - 1) / max(count - 1, 1))),
        }
        for seed_index in range(len(reports)):
            row[f"log_alpha_seed{seed_index + 1}"] = float(scores[seed_index, index])
            row[f"rank_seed{seed_index + 1}"] = int(ranks[seed_index, index])
            row[f"gate_probability_seed{seed_index + 1}"] = float(probabilities[seed_index, index])
        rows.append(row)
    rows.sort(key=lambda row: row["consensus_rank"])
    ranking_path, consensus_path, manifest_path = selection_paths(analysis_dir, cutoff, target_k)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with ranking_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "r_max": cutoff,
        "target_K": target_k,
        "candidate_count": count,
        "selector_seeds": [int(report["seed"]) for report in reports],
        "valid_selectors": len(reports),
        "ordering": [row["channel_id"] for row in rows],
        "channels": rows,
        "selection_method": "five-seed mean rank, then mean log-alpha, then channel id",
        "selection_protocol": "one converged target-32 selector ranking per rmax followed by nested Top-K truncation",
        "mathematically_optimal_for_each_K_claimed": False,
    }
    save_json(consensus_path, payload)
    selected = rows[:target_k]
    mask_indices = [int(row["channel"]) for row in masks]
    manifest = {
        "status": "frozen_from_valid_consensus",
        "experiment": EXPERIMENT,
        "name": f"corrected_rmax{cutoff}_K{target_k}",
        "r_max": cutoff,
        "target_K": target_k,
        "physical_channel_count": target_k,
        "geometry_mask_count": len(mask_indices),
        "total_input_width": target_k + len(mask_indices),
        "selected_physical_token_channels": [row["channel_id"] for row in selected],
        "selected_mask_token_channels": mask_indices,
        "selected_full_token_channels": [row["channel_id"] for row in selected] + mask_indices,
        "channels": selected,
        "selection_source": str(ranking_path),
        "exact500_used_for_selection": False,
        "N32_used_for_selection": False,
    }
    save_json(manifest_path, manifest)
    return payload


def write_topk_truncation_manifests(
    cutoff: int,
    consensus: dict[str, Any],
    budgets: tuple[int, ...],
    analysis_dir: Path,
) -> None:
    """Freeze nested Top-K views of one converged target-32 ranking."""
    if int(consensus["target_K"]) != SELECTOR_TARGET:
        raise RuntimeError("Top-K truncation requires a target-32 consensus")
    _, masks, _ = cutoff_layout(cutoff)
    mask_indices = [int(row["channel"]) for row in masks]
    ranking_path, _, _ = selection_paths(analysis_dir, cutoff, SELECTOR_TARGET)
    for budget in budgets:
        if not 1 <= budget <= SELECTOR_TARGET:
            raise ValueError(f"Top-K budget {budget} is outside 1..{SELECTOR_TARGET}")
        selected = []
        for source in consensus["channels"][:budget]:
            row = dict(source)
            seed_ranks = [
                int(row[f"rank_seed{index}"])
                for index in range(1, len(consensus["selector_seeds"]) + 1)
            ]
            row["selection_frequency_target32"] = row["selection_frequency_topK"]
            row["selection_frequency_topK"] = float(np.mean(np.asarray(seed_ranks) <= budget))
            selected.append(row)
        _, _, manifest_path = selection_paths(analysis_dir, cutoff, budget)
        manifest = {
            "status": "frozen_topK_truncation",
            "experiment": EXPERIMENT,
            "name": f"corrected_rmax{cutoff}_target32_topK{budget}",
            "r_max": cutoff,
            "target_K": budget,
            "selector_target": SELECTOR_TARGET,
            "physical_channel_count": budget,
            "geometry_mask_count": len(mask_indices),
            "total_input_width": budget + len(mask_indices),
            "selected_physical_token_channels": [row["channel_id"] for row in selected],
            "selected_mask_token_channels": mask_indices,
            "selected_full_token_channels": [row["channel_id"] for row in selected] + mask_indices,
            "channels": selected,
            "selection_source": str(ranking_path),
            "selection_protocol": "Top-K truncation of a converged target-32 selector ranking",
            "mathematically_optimal_for_this_K_claimed": False,
            "exact500_used_for_selection": False,
            "N32_used_for_selection": False,
        }
        save_json(manifest_path, manifest)


def write_selector_health(
    cutoffs: tuple[int, ...],
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    models_dir: Path,
    analysis_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for cutoff in cutoffs:
        tested_budgets = (16,) if cutoff == 4 else (SELECTOR_TARGET,)
        for target_k in tested_budgets:
            for seed in seeds:
                path = models_dir / f"selectors/{selector_stem(cutoff, target_k, seed)}/channel_ranking.json"
                report = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        key: report[key]
                        for key in (
                            "r_max",
                            "target_K",
                            "seed",
                            "best_epoch",
                            "warmup_epochs",
                            "best_after_warmup",
                            "expected_active_channels",
                            "active_minus_target",
                            "allowed_active_deviation",
                            "selection_objective",
                            "validation_bce",
                            "relaxed_deterministic_validation_bce",
                            "validation_gate_mode",
                            "early_stopped",
                            "valid_selector",
                            "checkpoint_sha256",
                        )
                    }
                )
    path = analysis_dir / "selector_health_report.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        analysis_dir / "selector_health_summary.json",
        {
            "status": "passed" if all(row["valid_selector"] for row in rows) else "failed",
            "valid_selectors": sum(bool(row["valid_selector"]) for row in rows),
            "total_selectors": len(rows),
            "all_best_after_warmup": all(row["best_after_warmup"] for row in rows),
        },
    )
    return rows


def compact_name(cutoff: int, target_k: int, seed: int) -> str:
    return f"RoPETransformer_CorrectedR{cutoff}_K{target_k}_seed{seed}"


def train_compact(
    cutoff: int,
    target_k: int,
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
    _, _, manifest_path = selection_paths(analysis_dir, cutoff, target_k)
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = np.asarray(selection["selected_full_token_channels"], dtype=np.int64)
    name = compact_name(cutoff, target_k, seed)
    output_dir = models_dir / name
    report_path = output_dir / "report.json"
    if report_path.exists() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    full_mean = np.load(cache_dir / "input_mean_full.npy")
    full_std = np.load(cache_dir / "input_std_full.npy")
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=len(channels))),
        full_mean[channels],
        full_std[channels],
    ).to(device)
    np.save(output_dir / "input_mean.npy", full_mean[channels])
    np.save(output_dir / "input_std.npy", full_std[channels])
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], channels)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], channels)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss, best_epoch, stale = math.inf, 0, 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_sum = 0.0
        epoch_started = time.perf_counter()
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
        probability, labels = predict_loader(model, validation_loader, device)
        eps = 1.0e-8
        validation_loss = float(
            -np.mean(
                labels * np.log(np.maximum(probability, eps))
                + (1 - labels) * np.log(np.maximum(1 - probability, eps))
            )
        )
        scheduler.step(validation_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_sum / len(train_ds),
            "validation_loss": validation_loss,
            "validation_AUROC": float(roc_auc_score(labels, probability)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.model.config),
            "training_config": asdict(config),
            "epoch": epoch,
            "validation_loss": validation_loss,
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
    report = {
        "status": "trained_internal_only",
        "experiment": EXPERIMENT,
        "model_name": name,
        "r_max": cutoff,
        "physical_K": target_k,
        "geometry_masks": int(selection["geometry_mask_count"]),
        "total_input_width": len(channels),
        "parameter_count": trainable_parameter_count(model),
        "model_config": asdict(model.model.config),
        "training_config": asdict(config),
        "selected_full_token_channels": channels.tolist(),
        "selection_manifest": str(manifest_path),
        "implementation_git_commit": git_hash(),
        "route_hash": route_data["route_hash"],
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_validation_AUROC": float(
            next(row["validation_AUROC"] for row in history if row["epoch"] == best_epoch)
        ),
        "training_wall_seconds": time.perf_counter() - started,
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_selection": "minimum parent-safe canonical N16 validation BCE only",
        "OOD_evaluation_performed": False,
        "exact500_accessed": False,
        "exact5000_accessed": False,
        "N32_training_rows": 0,
        "DPI_training_rows": 0,
    }
    save_json(report_path, report)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def select_representative_k(
    rows: list[dict[str, Any]],
    *,
    tolerance: float = 0.001,
    preferred_k: int = 16,
) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Apply the preregistered K=16 preference using development AUROC only."""
    best_auroc = max(row["mean_validation_AUROC"] for row in rows)
    candidates = [
        row for row in rows if row["mean_validation_AUROC"] >= best_auroc - tolerance
    ]
    preferred = next((row for row in candidates if row["K"] == preferred_k), None)
    chosen = preferred or min(
        candidates, key=lambda row: (row["K"], row["mean_validation_BCE"])
    )
    return chosen, best_auroc, candidates


def freeze_best_k(
    cutoffs: tuple[int, ...],
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    models_dir: Path,
    analysis_dir: Path,
) -> dict[str, Any]:
    choices: dict[str, Any] = {}
    for cutoff in cutoffs:
        rows = []
        tested = (16,) if cutoff == 4 else budgets
        for target_k in tested:
            reports = [
                json.loads((models_dir / compact_name(cutoff, target_k, seed) / "report.json").read_text(encoding="utf-8"))
                for seed in seeds
            ]
            if any(report.get("OOD_evaluation_performed", False) for report in reports):
                raise RuntimeError("OOD results were accessed before best-K freezing")
            losses = np.asarray([report["best_validation_loss"] for report in reports], dtype=np.float64)
            aurocs = np.asarray([report["best_validation_AUROC"] for report in reports], dtype=np.float64)
            rows.append(
                {
                    "K": target_k,
                    "mean_validation_BCE": float(losses.mean()),
                    "std_validation_BCE": float(losses.std(ddof=1)),
                    "per_seed_validation_BCE": losses.tolist(),
                    "mean_validation_AUROC": float(aurocs.mean()),
                    "std_validation_AUROC": float(aurocs.std(ddof=1)),
                    "per_seed_validation_AUROC": aurocs.tolist(),
                }
            )
        chosen, best_auroc, candidates = select_representative_k(rows)
        choices[str(cutoff)] = {
            "best_K": int(chosen["K"]),
            "rule": (
                "highest mean parent-safe validation AUROC; explicitly prefer K=16 when within "
                "0.001 absolute AUROC, otherwise choose the smallest K within tolerance"
            ),
            "best_mean_validation_AUROC": best_auroc,
            "internal_validation_cannot_resolve_K": len(candidates) > 1,
            "tested": rows,
        }
    payload = {
        "status": "frozen_from_parent_safe_N16_development_only",
        "experiment": EXPERIMENT,
        "frozen_at_unix": time.time(),
        "choices": choices,
        "forbidden_selection_sources": ["exact500", "N32_XY", "N32_DPI", "general_separable_N32"],
        "exact500_accessed": False,
        "exact5000_accessed": False,
    }
    save_json(analysis_dir / "frozen_best_k.json", payload)
    return payload


def historical_r4_regression_audit(models_dir: Path, analysis_dir: Path) -> dict[str, Any]:
    _, _, manifest_path = selection_paths(analysis_dir, 4, 16)
    corrected = json.loads(manifest_path.read_text(encoding="utf-8"))
    historical_path = ROOT / "analysis/invariant_transformer_raw_t4_v1/selection_manifests/joint_K16.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_keys = {
        (
            row["family"],
            row["raw_or_connected"],
            row.get("pair_distance"),
            row["invariant_mathematical_name"],
        )
        for row in historical["selected_channels"]
    }
    corrected_rows = corrected["channels"]
    corrected_keys = {
        (row["family"], row["raw_or_connected"], row.get("r"), row["invariant_scalar_name"])
        for row in corrected_rows
    }
    reports = [
        json.loads((models_dir / compact_name(4, 16, seed) / "report.json").read_text(encoding="utf-8"))
        for seed in TRAINING_SEEDS
    ]
    losses = np.asarray([report["best_validation_loss"] for report in reports], dtype=np.float64)
    consensus = json.loads((analysis_dir / "consensus_rmax4_K16.json").read_text(encoding="utf-8"))
    pair_distances = [int(row["r"]) for row in corrected_rows if row["family"] == "pair"]
    r1_ranks = [
        int(row["consensus_rank"])
        for row in consensus["channels"]
        if row["family"] == "pair" and int(row["r"]) == 1
    ]
    best_r1_rank = min(r1_ranks)
    top_quartile_boundary = math.ceil(int(consensus["candidate_count"]) / 4)
    t3_spectral = sum(row["family"] == "contiguous_three_body" and row["SVD_based"] for row in corrected_rows)
    t4_spectral = sum(row["family"] == "contiguous_four_body" and row["SVD_based"] for row in corrected_rows)
    checks = {
        "all_five_selectors_valid": all(
            json.loads(
                (models_dir / f"selectors/{selector_stem(4, 16, seed)}/channel_ranking.json").read_text(
                    encoding="utf-8"
                )
            )["valid_selector"]
            for seed in SELECTOR_SEEDS
        ),
        # The historical baseline optimized target=32 and only then truncated
        # to 16, whereas the corrected protocol directly optimizes target=16.
        # Requiring literal top-16 identity would defeat that requested change.
        # We require the historical r=1 family to remain important (top
        # candidate quartile), and separately disclose whether it enters K16.
        "r1_pair_recovered_in_consensus_top_quartile": best_r1_rank <= top_quartile_boundary,
        "contains_r4_pair": 4 in pair_distances,
        "at_least_two_T3_spectral": t3_spectral >= 2,
        "at_least_four_T4_spectral": t4_spectral >= 4,
        "historical_top16_overlap_at_least_8": len(historical_keys & corrected_keys) >= 8,
        "mean_internal_validation_BCE_below_0p005": float(losses.mean()) < 0.005,
    }
    payload = {
        "status": "passed" if all(checks.values()) else "failed_stop_before_r6_r8",
        "predeclared_checks": checks,
        "historical_top16_semantic_overlap": len(historical_keys & corrected_keys),
        "corrected_pair_distances": pair_distances,
        "corrected_r1_in_final_top16": 1 in pair_distances,
        "corrected_r1_best_consensus_rank": best_r1_rank,
        "r1_top_quartile_boundary": top_quartile_boundary,
        "corrected_T3_spectral_count": t3_spectral,
        "corrected_T4_spectral_count": t4_spectral,
        "compact_validation_BCE_mean": float(losses.mean()),
        "compact_validation_BCE_std": float(losses.std(ddof=1)),
        "OOD_accessed": False,
    }
    save_json(analysis_dir / "historical_r4_regression_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One corrected target-32 ranking per pair range plus nested compact Top-K models."
    )
    parser.add_argument("--mode", choices=("selectors", "consensus", "health", "train", "freeze", "r4-audit"), required=True)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=PRIMARY_CUTOFFS)
    parser.add_argument("--budgets", nargs="+", type=int, default=PRIMARY_BUDGETS)
    parser.add_argument("--selector-seeds", nargs="+", type=int, default=SELECTOR_SEEDS)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=TRAINING_SEEDS)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / f"analysis/{EXPERIMENT}")
    parser.add_argument("--models-dir", type=Path, default=ROOT / f"models/{EXPERIMENT}")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "analysis/pair_range_r8_invariant_attention_v1/cache")
    parser.add_argument("--route-dir", type=Path, default=ROOT / "analysis/connected_siamese_v1")
    parser.add_argument("--selector-max-epochs", type=int, default=SelectorConfig.max_epochs)
    parser.add_argument("--selector-patience", type=int, default=SelectorConfig.patience)
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is forbidden in the selection experiment")
    cutoffs = tuple(args.cutoffs)
    budgets = tuple(args.budgets)
    selector_seeds = tuple(args.selector_seeds)
    training_seeds = tuple(args.training_seeds)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "health":
        write_selector_health(cutoffs, budgets, selector_seeds, args.models_dir, args.analysis_dir)
        return
    if args.mode == "r4-audit":
        audit = historical_r4_regression_audit(args.models_dir, args.analysis_dir)
        print(json.dumps(audit, indent=2), flush=True)
        if audit["status"] != "passed":
            raise SystemExit(2)
        return
    if args.mode == "freeze":
        payload = freeze_best_k(cutoffs, budgets, training_seeds, args.models_dir, args.analysis_dir)
        print(json.dumps(payload, indent=2), flush=True)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    arrays = load_arrays(args.cache_dir)
    _, route_data = load_routes(args.route_dir)
    full_mean = np.load(args.cache_dir / "input_mean_full.npy")
    full_std = np.load(args.cache_dir / "input_std_full.npy")
    selector_config = SelectorConfig(
        max_epochs=args.selector_max_epochs,
        patience=args.selector_patience,
        warmup_epochs=WARMUP_EPOCHS,
    )
    if args.mode == "selectors":
        for cutoff in cutoffs:
            for seed in selector_seeds:
                train_selector(
                    cutoff,
                    SELECTOR_TARGET,
                    seed,
                    arrays=arrays,
                    route_data=route_data,
                    full_mean=full_mean,
                    full_std=full_std,
                    output_dir=args.models_dir / f"selectors/{selector_stem(cutoff, SELECTOR_TARGET, seed)}",
                    device=device,
                    config=selector_config,
                    force=args.force,
                )
    elif args.mode == "consensus":
        for cutoff in cutoffs:
            reports = [
                json.loads(
                    (
                        args.models_dir
                        / f"selectors/{selector_stem(cutoff, SELECTOR_TARGET, seed)}/channel_ranking.json"
                    ).read_text(encoding="utf-8")
                )
                for seed in selector_seeds
            ]
            consensus = build_consensus(cutoff, SELECTOR_TARGET, reports, args.analysis_dir)
            write_topk_truncation_manifests(cutoff, consensus, budgets, args.analysis_dir)
    elif args.mode == "train":
        for cutoff in cutoffs:
            for target_k in budgets:
                for seed in training_seeds:
                    train_compact(
                        cutoff,
                        target_k,
                        seed,
                        arrays=arrays,
                        route_data=route_data,
                        cache_dir=args.cache_dir,
                        analysis_dir=args.analysis_dir,
                        models_dir=args.models_dir,
                        device=device,
                        config=TrainingConfig(
                            seed=seed,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                        ),
                        force=args.force,
                    )


if __name__ == "__main__":
    main()

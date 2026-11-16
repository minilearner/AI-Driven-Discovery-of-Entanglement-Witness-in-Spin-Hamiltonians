from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
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
    local_inv123_raw_t4_channel_manifest,
    trainable_parameter_count,
)
from run_raw_t4_invariant_transformer import (  # noqa: E402
    SelectorConfig,
    TrainingConfig,
    geometry_metrics,
    n16_balanced_labels,
    n32_labels,
    save_json,
    selector_validation,
)
from training_routes import BASE, load_routes  # noqa: E402
from train_invariant_transformer import (  # noqa: E402
    RoutedInvariantTokens,
    benchmark_metrics,
    exact500_labels,
    git_hash,
    predict_array,
    predict_loader,
    seed_all,
    write_scores,
)


SELECTOR_SEEDS = (2026091001, 2026091002, 2026091003, 2026091004, 2026091005)
TRAINING_SEEDS = (20260829, 20260830, 20260831)
CUTOFFS = (2, 4, 6, 8)
BUDGETS = (8, 12, 16, 20, 24, 32, 48, 173)
R8 = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_manifest() -> list[dict[str, Any]]:
    return local_inv123_raw_t4_channel_manifest(R8)


def semantic_key(row: dict[str, Any]) -> str:
    if row["mask"]:
        if row["pair_distance"] is None:
            return "mask/triple_contiguous_valid"
        return f"mask/pair_r{row['pair_distance']}_valid"
    if row["family"] == "pair":
        return f"pair/r{row['pair_distance']}/{row['raw_or_connected']}/{row['invariant_mathematical_name']}"
    return f"{row['family']}/{row['raw_or_connected']}/{row['invariant_mathematical_name']}"


def cutoff_layout(cutoff: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    if cutoff not in CUTOFFS:
        raise ValueError(f"unsupported cutoff {cutoff}")
    manifest = full_manifest()
    physical = [row for row in manifest if not row["mask"] and
                (row["family"] != "pair" or int(row["pair_distance"]) <= cutoff)]
    masks = [row for row in manifest if row["mask"] and
             (row["pair_distance"] is None or int(row["pair_distance"]) <= cutoff)]
    expected = 61 + 14 * cutoff
    if len(physical) != expected or len(masks) != cutoff + 1:
        raise RuntimeError(f"R{cutoff} layout mismatch: physical={len(physical)} masks={len(masks)}")
    indices = np.asarray([row["channel"] for row in (*physical, *masks)], dtype=np.int64)
    return physical, masks, indices


def load_arrays(cache_dir: Path) -> list[np.ndarray]:
    arrays = [np.load(cache_dir / f"{name}_Inv123RawT4_Robs8.npy", mmap_mode="r") for name in BASE]
    if any(array.ndim != 3 or array.shape[1:] != (16, 182) for array in arrays):
        raise RuntimeError("R8 training cache shape changed")
    return arrays


def train_selector(
    cutoff: int,
    seed: int,
    *,
    arrays: list[np.ndarray],
    route_data: dict[str, Any],
    full_mean: np.ndarray,
    full_std: np.ndarray,
    output_dir: Path,
    device: torch.device,
    config: SelectorConfig,
) -> dict[str, Any]:
    report_path = output_dir / "channel_ranking.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    physical, masks, indices = cutoff_layout(cutoff)
    mean, std = full_mean[indices], full_std[indices]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], indices)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], indices)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model = SelectiveChannelGatedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=len(indices))),
        mean, std, tuple(range(len(physical))),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience,
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_objective, best_epoch, stale = math.inf, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    candidate_count = len(physical)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens)
                bce = criterion(logits, labels)
                expected = model.expected_gated_channels()
                warmup = min(1.0, epoch / config.warmup_epochs)
                loss = bce + warmup * (
                    config.sparsity_penalty * expected / candidate_count
                    + config.budget_penalty * ((expected - 32.0) / candidate_count).square()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(labels)
        probability, labels, validation_bce = selector_validation(model, validation_loader, device)
        expected = float(model.expected_gated_channels().detach())
        objective = validation_bce + 0.02 * abs(expected - 32.0) / 32.0
        scheduler.step(validation_bce)
        row = {
            "epoch": epoch, "train_objective": loss_sum / len(train_ds),
            "validation_BCE": validation_bce,
            "validation_AUROC": float(roc_auc_score(labels, probability)),
            "expected_channels": expected, "selection_objective": objective,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if objective < best_objective - 1e-7:
            best_objective, best_epoch, stale = objective, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        print(json.dumps({"selector_rmax": cutoff, "seed": seed, **row, "stale": stale}), flush=True)
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("selector produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state, "cutoff": cutoff, "seed": seed,
        "config": asdict(config), "best_epoch": best_epoch,
        "route_hash": route_data["route_hash"],
    }, output_dir / "best.pt")
    log_alpha = model.gates.log_alpha.detach().cpu().numpy().astype(np.float64)
    probability_gate = model.gates.nonzero_probability().detach().cpu().numpy().astype(np.float64)
    order = np.lexsort((np.arange(candidate_count), -log_alpha))
    ranks = np.empty(candidate_count, dtype=np.int64)
    ranks[order] = np.arange(1, candidate_count + 1)
    channels = []
    for local_index, row in enumerate(physical):
        channels.append({
            "candidate_channel": local_index,
            "full_r8_channel": int(row["channel"]),
            "semantic_key": semantic_key(row),
            "rank": int(ranks[local_index]),
            "normalized_rank": float((ranks[local_index] - 1) / max(candidate_count - 1, 1)),
            "percentile_rank": float(100 * (1 - (ranks[local_index] - 1) / max(candidate_count - 1, 1))),
            "gate_probability": float(probability_gate[local_index]),
            "log_alpha": float(log_alpha[local_index]),
            **{key: row.get(key) for key in (
                "family", "raw_or_connected", "pair_distance",
                "invariant_mathematical_name", "svd_based",
            )},
            "singular_value_index": (row.get("t4_manifest") or {}).get("singular_value_index"),
        })
    report = {
        "status": "complete", "r_max": cutoff, "seed": seed,
        "selector_protocol": "unit cost per shared scalar token channel",
        "candidate_count": candidate_count, "geometry_masks": len(masks),
        "total_input_width": len(indices), "target_expected_channels": 32,
        "config": asdict(config), "epochs_completed": len(history),
        "best_epoch": best_epoch, "best_selection_objective": best_objective,
        "training_wall_seconds": time.perf_counter() - started,
        "route_hash": route_data["route_hash"], "channels": channels,
        "ranking_best_to_worst_full_r8_channels": [int(physical[index]["channel"]) for index in order],
        "checkpoint_sha256": sha256(output_dir / "best.pt"),
        "DPI_training_rows": 0, "N32_training_rows": 0,
        "N32_used_for_selection": False, "exact5000_accessed": False,
    }
    save_json(report_path, report)
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def consensus(cutoff: int, reports: list[dict[str, Any]], analysis_dir: Path) -> dict[str, Any]:
    physical, masks, _ = cutoff_layout(cutoff)
    count = len(physical)
    ranks = np.asarray([[row["rank"] for row in report["channels"]] for report in reports], dtype=np.float64)
    scores = np.asarray([[row["log_alpha"] for row in report["channels"]] for report in reports], dtype=np.float64)
    rank_mean, rank_std = ranks.mean(axis=0), ranks.std(axis=0, ddof=1)
    score_mean, score_std = scores.mean(axis=0), scores.std(axis=0, ddof=1)
    order = np.lexsort((np.arange(count), -score_mean, rank_mean))
    consensus_rank = np.empty(count, dtype=np.int64)
    consensus_rank[order] = np.arange(1, count + 1)
    rows = []
    for index, channel in enumerate(physical):
        name = channel["invariant_mathematical_name"]
        sv_index = (channel.get("t4_manifest") or {}).get("singular_value_index")
        if sv_index is None and "sv_" in name:
            try: sv_index = int(name.rsplit("sv_", 1)[1].split("_", 1)[0])
            except ValueError: pass
        row: dict[str, Any] = {
            "channel_id": int(channel["channel"]), "semantic_key": semantic_key(channel),
            "order": {"one_body": 1, "pair": 2, "contiguous_three_body": 3,
                      "contiguous_four_body": 4}[channel["family"]],
            "family": channel["family"], "raw_or_connected": channel["raw_or_connected"],
            "r": channel["pair_distance"], "SVD_index": sv_index,
            "invariant_scalar_name": name, "SVD_based": bool(channel["svd_based"]),
            "mean_score": float(score_mean[index]), "std_score": float(score_std[index]),
            "mean_rank": float(rank_mean[index]), "rank_std": float(rank_std[index]),
            "consensus_rank": int(consensus_rank[index]),
            "normalized_rank": float((consensus_rank[index] - 1) / max(count - 1, 1)),
            "percentile_rank": float(100 * (1 - (consensus_rank[index] - 1) / max(count - 1, 1))),
        }
        for seed_index, report in enumerate(reports, 1):
            row[f"selector_score_seed{seed_index}"] = float(scores[seed_index - 1, index])
            row[f"rank_seed{seed_index}"] = int(ranks[seed_index - 1, index])
        for budget in (8, 16, 24, 32):
            row[f"selection_frequency_top{budget}"] = float(np.mean(ranks[:, index] <= budget))
        rows.append(row)
    rows.sort(key=lambda row: row["consensus_rank"])
    csv_path = analysis_dir / f"channel_importance_r{cutoff}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "status": "complete", "r_max": cutoff, "candidate_count": count,
        "geometry_masks": len(masks), "total_input_width": count + len(masks),
        "selector_seeds": [report["seed"] for report in reports],
        "ordering": [row["channel_id"] for row in rows], "channels": rows,
        "raw_scores_cross_pool_warning": "compare percentile or normalized rank across candidate pools",
    }
    save_json(analysis_dir / f"consensus_rmax{cutoff}.json", payload)
    mask_indices = [row["channel"] for row in masks]
    selected16 = [row["channel_id"] for row in rows[:16]]
    save_json(analysis_dir / f"selection_manifests/rmax{cutoff}_K16.json", {
        "name": f"pair_r{cutoff}_matched_K16", "r_max": cutoff,
        "physical_channel_count": 16, "geometry_mask_count": len(mask_indices),
        "total_input_width": 16 + len(mask_indices),
        "selected_physical_token_channels": selected16,
        "selected_mask_token_channels": mask_indices,
        "selected_full_token_channels": selected16 + mask_indices,
        "channels": rows[:16], "selection_source": str(csv_path),
    })
    if cutoff == 8:
        for budget in BUDGETS:
            selected = [row["channel_id"] for row in rows[:budget]]
            manifest = {
                "name": f"pair_r8_joint_K{budget}", "r_max": 8,
                "physical_channel_count": budget, "geometry_mask_count": len(mask_indices),
                "total_input_width": budget + len(mask_indices),
                "selected_physical_token_channels": selected,
                "selected_mask_token_channels": mask_indices,
                "selected_full_token_channels": selected + mask_indices,
                "channels": rows[:budget], "selection_source": str(csv_path),
            }
            save_json(analysis_dir / f"selection_manifests/joint_K{budget}.json", manifest)
    return payload


def cutoff_robustness(analysis_dir: Path) -> None:
    payloads = {cutoff: json.loads((analysis_dir / f"consensus_rmax{cutoff}.json").read_text(encoding="utf-8"))
                for cutoff in CUTOFFS}
    by_cutoff = {cutoff: {row["semantic_key"]: row for row in payload["channels"]}
                 for cutoff, payload in payloads.items()}
    shared = set(by_cutoff[4]) & set(by_cutoff[8])
    rank4 = np.asarray([by_cutoff[4][key]["consensus_rank"] for key in sorted(shared)])
    rank8 = np.asarray([by_cutoff[8][key]["consensus_rank"] for key in sorted(shared)])
    distance_rows = []
    for cutoff in CUTOFFS:
        ordered = payloads[cutoff]["channels"]
        for budget in (8, 16, 24, 32):
            top = ordered[:budget]
            for distance in range(1, cutoff + 1):
                distance_rows.append({"r_max": cutoff, "K": budget, "r_selected": distance,
                                      "number_selected": sum(row["r"] == distance for row in top)})
    with (analysis_dir / "selected_pair_distance_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(distance_rows[0]))
        writer.writeheader(); writer.writerows(distance_rows)
    overlap = {}
    top4 = [row["semantic_key"] for row in payloads[4]["channels"]]
    top8 = [row["semantic_key"] for row in payloads[8]["channels"]]
    for budget in (8, 16, 24, 32):
        left, right = set(top4[:budget]), set(top8[:budget])
        overlap[str(budget)] = {"intersection": len(left & right),
                                "jaccard": len(left & right) / len(left | right)}
    old16 = set(top4[:16])
    report = {
        "status": "complete", "shared_channel_count": len(shared),
        "spearman_rank_correlation_r4_vs_r8": float(spearmanr(rank4, rank8).statistic),
        "kendall_tau_r4_vs_r8": float(kendalltau(rank4, rank8).statistic),
        "topK_overlap": overlap,
        "old_top16_retained_new_top16": len(old16 & set(top8[:16])),
        "old_top16_retained_new_top24": len(old16 & set(top8[:24])),
    }
    save_json(analysis_dir / "cutoff_robustness.json", report)
    rows = []
    for key in top4[:24]:
        old, new = by_cutoff[4][key], by_cutoff[8][key]
        rows.append({"feature": key, "old_rank": old["consensus_rank"], "new_rank": new["consensus_rank"],
                     "rank_shift_new_minus_old": new["consensus_rank"] - old["consensus_rank"],
                     "old_percentile": old["percentile_rank"], "new_percentile": new["percentile_rank"],
                     "still_selected_K16": new["consensus_rank"] <= 16,
                     "still_selected_K24": new["consensus_rank"] <= 24})
    with (analysis_dir / "old_top24_rank_shifts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_substitution_manifests(analysis_dir: Path) -> dict[str, Any]:
    ranking = json.loads((analysis_dir / "consensus_rmax8.json").read_text(encoding="utf-8"))["channels"]
    top16 = ranking[:16]
    pair_count = sum(row["family"] == "pair" for row in top16)
    core = [row for row in ranking if row["family"] != "pair"][:16 - pair_count]
    _, masks, _ = cutoff_layout(8)
    mask_indices = [row["channel"] for row in masks]
    manifests = {}
    for distance in range(1, 9):
        pair = [row for row in ranking if row["family"] == "pair" and row["r"] == distance][:pair_count]
        if len(pair) != pair_count:
            raise RuntimeError(f"distance {distance} has fewer than {pair_count} candidate pair channels")
        selected = core + pair
        payload = {
            "name": f"pair_distance_substitution_r{distance}", "r_max": 8,
            "substituted_pair_distance": distance, "pair_channel_count": pair_count,
            "non_pair_core_count": len(core), "physical_channel_count": 16,
            "geometry_mask_count": 9, "total_input_width": 25,
            "selected_physical_token_channels": [row["channel_id"] for row in selected],
            "selected_mask_token_channels": mask_indices,
            "selected_full_token_channels": [row["channel_id"] for row in selected] + mask_indices,
            "channels": selected,
            "non_pair_core_semantic_keys": [row["semantic_key"] for row in core],
        }
        path = analysis_dir / f"selection_manifests/pair_distance_r{distance}_K16.json"
        save_json(path, payload)
        manifests[str(distance)] = str(path)
    summary = {"status": "complete", "pair_channel_count_q": pair_count,
               "non_pair_core_count": len(core), "manifests": manifests}
    save_json(analysis_dir / "pair_distance_substitution_design.json", summary)
    return summary


def train_model(
    budget: int,
    seed: int,
    *,
    arrays: list[np.ndarray], route_data: dict[str, Any], cache_dir: Path,
    analysis_dir: Path, models_dir: Path, device: torch.device,
    config: TrainingConfig, force: bool,
    manifest_path: Path | None = None,
    model_name: str | None = None,
    experiment_family: str = "joint_K_curve",
) -> dict[str, Any]:
    manifest_path = manifest_path or analysis_dir / f"selection_manifests/joint_K{budget}.json"
    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = np.asarray(selection["selected_full_token_channels"], dtype=np.int64)
    name = model_name or f"RoPETransformer_PairR8_JointK{budget}_seed{seed}"
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
        full_mean[channels], full_std[channels],
    ).to(device)
    train_ds = RoutedInvariantTokens(arrays, route_data["routes"]["train"], channels)
    validation_ds = RoutedInvariantTokens(arrays, route_data["routes"]["validation"], channels)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0,
                              pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_ds, batch_size=config.eval_batch_size, shuffle=False,
                                   num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=config.scheduler_patience)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_loss, best_epoch, stale = math.inf, 0, 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.max_epochs + 1):
        model.train(); train_sum = 0.0; epoch_start = time.perf_counter()
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(tokens); loss = criterion(logits, labels)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer); scaler.update()
            train_sum += float(loss.detach()) * len(labels)
        validation_probability, validation_labels = predict_loader(model, validation_loader, device)
        eps = 1e-8
        validation_loss = float(-np.mean(validation_labels * np.log(np.maximum(validation_probability, eps))
                          + (1 - validation_labels) * np.log(np.maximum(1 - validation_probability, eps))))
        scheduler.step(validation_loss)
        row = {"epoch": epoch, "train_loss": train_sum / len(train_ds),
               "validation_loss": validation_loss,
               "validation_AUROC": float(roc_auc_score(validation_labels, validation_probability)),
               "learning_rate": float(optimizer.param_groups[0]["lr"]),
               "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(row)
        metadata = {"model_name": name, "experiment_family": experiment_family,
                    "r_max": int(selection["r_max"]),
                    "physical_K": int(selection["physical_channel_count"]),
                    "geometry_masks": int(selection["geometry_mask_count"]),
                    "total_input_width": len(channels),
                    "parameter_count": trainable_parameter_count(model),
                    "model_config": asdict(model.model.config), "training_config": asdict(config),
                    "selected_full_token_channels": channels.tolist(), "selection_manifest": str(manifest_path),
                    "implementation_git_commit": git_hash(), "route_hash": route_data["route_hash"],
                    "checkpoint_selection": "minimum canonical N16 validation BCE only",
                    "DPI_training_rows": 0, "N32_training_rows": 0,
                    "N32_used_for_selection": False, "exact5000_accessed": False}
        checkpoint = {"model_state_dict": model.state_dict(), "model_config": asdict(model.model.config),
                      "training_config": asdict(config), "metadata": metadata, "epoch": epoch,
                      "validation_loss": validation_loss}
        torch.save(checkpoint, output_dir / "last.pt")
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(checkpoint, checkpoint_path)
        else: stale += 1
        print(json.dumps({"model": name, **row, "stale": stale}), flush=True)
        if stale >= config.patience: break
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    exact = np.load(cache_dir / "exact500_Inv123RawT4_Robs8.npy", mmap_mode="r")
    balanced = np.load(cache_dir / "XY_factorized_balanced_N16_Inv123RawT4_Robs8.npy", mmap_mode="r")
    xy = np.load(cache_dir / "N32_XY_Inv123RawT4_Robs8.npy", mmap_mode="r")
    dpi = np.load(cache_dir / "N32_DPI_Inv123RawT4_Robs8.npy", mmap_mode="r")
    probabilities = {name_: predict_array(model, np.asarray(values)[:, :, channels], device, config.eval_batch_size)
                     for name_, values in (("exact", exact), ("balanced", balanced), ("xy", xy), ("dpi", dpi))}
    labels = {"exact": exact500_labels(), "balanced": n16_balanced_labels(),
              "xy": n32_labels("xy"), "dpi": n32_labels("dpi")}
    metrics = {
        "N16_exact500": benchmark_metrics(probabilities["exact"], separable=labels["exact"]["separable"],
            entangled=labels["exact"]["entangled"], e2=labels["exact"]["E2"],
            unknown=labels["exact"]["unknown"], threshold=config.threshold),
        "XY_factorized_balanced_N16": benchmark_metrics(probabilities["balanced"],
            separable=labels["balanced"]["separable"], entangled=labels["balanced"]["entangled"],
            e2=labels["balanced"]["E2"], unknown=labels["balanced"]["unknown"], threshold=config.threshold),
        "N32_XY_matched": benchmark_metrics(probabilities["xy"], separable=labels["xy"]["separable"],
            entangled=labels["xy"]["entangled"], e2=labels["xy"]["E2"], e3=labels["xy"]["E3"],
            e4=labels["xy"]["E4"], unknown=labels["xy"]["unknown"], threshold=config.threshold),
        "N32_DPI_general": benchmark_metrics(probabilities["dpi"], separable=labels["dpi"]["separable"],
            entangled=labels["dpi"]["entangled"], e2=labels["dpi"]["E2"], e3=labels["dpi"]["E3"],
            e4=labels["dpi"]["E4"], unknown=labels["dpi"]["unknown"], threshold=config.threshold),
    }
    metrics["N32_DPI_E4_geometry"] = geometry_metrics(
        probabilities["dpi"], labels["dpi"],
        ROOT / "analysis/invariant_transformer_raw_t4_v1/dpi_e4_certificate_geometry.json", config.threshold)
    for key, filename in (("exact", "N16_exact500"), ("balanced", "XY_factorized_balanced_N16"),
                          ("xy", "N32_XY_matched"), ("dpi", "N32_DPI_general")):
        write_scores(output_dir / f"scores_{filename}.csv", probabilities[key], labels[key])
    metadata = checkpoint["metadata"]
    report = {**metadata, "status": "complete", "epochs_completed": len(history),
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "training_wall_seconds": time.perf_counter() - started,
              "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
              "checkpoint_sha256": sha256(checkpoint_path), "metrics": metrics,
              "N32_evaluated_zero_shot": True, "N32_finetuned": False}
    save_json(report_path, report)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Pair-range R<=8 selector sweep and compact models.")
    parser.add_argument("--mode", choices=("selectors", "summarize", "train", "substitution", "all"), default="all")
    parser.add_argument("--cutoffs", nargs="+", type=int, default=CUTOFFS)
    parser.add_argument("--selector-seeds", nargs="+", type=int, default=SELECTOR_SEEDS)
    parser.add_argument("--budgets", nargs="+", type=int, default=BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=TRAINING_SEEDS)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis/pair_range_r8_invariant_attention_v1")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/pair_range_r8_invariant_attention_v1")
    parser.add_argument("--route-dir", type=Path, default=ROOT / "analysis/connected_siamese_v1")
    parser.add_argument("--selector-max-epochs", type=int, default=SelectorConfig.max_epochs)
    parser.add_argument("--selector-patience", type=int, default=SelectorConfig.patience)
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is closed for this model-selection experiment")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2); torch.set_float32_matmul_precision("high")
    cache_dir = args.analysis_dir / "cache"
    arrays = load_arrays(cache_dir)
    _, route_data = load_routes(args.route_dir)
    full_mean = np.load(cache_dir / "input_mean_full.npy")
    full_std = np.load(cache_dir / "input_std_full.npy")
    if args.mode in {"selectors", "all"}:
        config = SelectorConfig(max_epochs=args.selector_max_epochs, patience=args.selector_patience)
        for cutoff in args.cutoffs:
            reports = [train_selector(cutoff, seed, arrays=arrays, route_data=route_data,
                        full_mean=full_mean, full_std=full_std,
                        output_dir=args.models_dir / f"selectors/rmax{cutoff}_seed{seed}",
                        device=device, config=config) for seed in args.selector_seeds]
            consensus(cutoff, reports, args.analysis_dir)
    if args.mode == "summarize":
        for cutoff in args.cutoffs:
            reports = [json.loads((args.models_dir / f"selectors/rmax{cutoff}_seed{seed}/channel_ranking.json").read_text(encoding="utf-8"))
                       for seed in args.selector_seeds]
            consensus(cutoff, reports, args.analysis_dir)
    if all((args.analysis_dir / f"consensus_rmax{cutoff}.json").exists() for cutoff in CUTOFFS):
        cutoff_robustness(args.analysis_dir)
    if (args.analysis_dir / "consensus_rmax8.json").exists():
        write_substitution_manifests(args.analysis_dir)
    if args.mode in {"train", "all"}:
        for budget in args.budgets:
            for seed in args.seeds:
                train_model(budget, seed, arrays=arrays, route_data=route_data,
                            cache_dir=cache_dir, analysis_dir=args.analysis_dir,
                            models_dir=args.models_dir, device=device,
                            config=TrainingConfig(seed=seed, max_epochs=args.max_epochs, patience=args.patience),
                            force=args.force)
        matched_r4 = args.analysis_dir / "selection_manifests/rmax4_K16.json"
        for seed in args.seeds:
            train_model(16, seed, arrays=arrays, route_data=route_data,
                        cache_dir=cache_dir, analysis_dir=args.analysis_dir,
                        models_dir=args.models_dir, device=device,
                        config=TrainingConfig(seed=seed, max_epochs=args.max_epochs, patience=args.patience),
                        force=args.force, manifest_path=matched_r4,
                        model_name=f"RoPETransformer_PairR4_MatchedK16_seed{seed}",
                        experiment_family="matched_r4_reference")
    if args.mode in {"substitution", "all"}:
        write_substitution_manifests(args.analysis_dir)
        for distance in range(1, 9):
            manifest_path = args.analysis_dir / f"selection_manifests/pair_distance_r{distance}_K16.json"
            for seed in args.seeds:
                train_model(16, seed, arrays=arrays, route_data=route_data,
                            cache_dir=cache_dir, analysis_dir=args.analysis_dir,
                            models_dir=args.models_dir, device=device,
                            config=TrainingConfig(seed=seed, max_epochs=args.max_epochs, patience=args.patience),
                            force=args.force, manifest_path=manifest_path,
                            model_name=f"RoPETransformer_PairR8_PairOnlyDistance_r{distance}_seed{seed}",
                            experiment_family="matched_pair_distance_substitution")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.spatial_invariants import SpatialInvariantSpec, organize_local_invariants
from entanglement_data.ml.invariant_transformer import (
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    local_inv123_channel_indices,
    local_inv123_channel_manifest,
    trainable_parameter_count,
)
from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.lu_invariants import compute_inv123_features
from training_routes import BASE, load_routes


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_hash() -> str:
    override = os.environ.get("ENTANGLEMENT_GIT_COMMIT", "").strip()
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_scores(path: Path, probability: np.ndarray, labels: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["row", "probability", *labels]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in range(len(probability)):
            writer.writerow({
                "row": row,
                "probability": float(probability[row]),
                **{key: value[row].item() if isinstance(value[row], np.generic) else value[row] for key, value in labels.items()},
            })


def ranking(negative: np.ndarray, positive: np.ndarray, probability: np.ndarray) -> dict[str, float | int | None]:
    chosen = np.asarray(negative, bool) | np.asarray(positive, bool)
    labels = np.asarray(positive, bool)[chosen].astype(np.int8)
    if len(np.unique(labels)) < 2:
        return {"count": int(chosen.sum()), "AUROC": None, "AP": None}
    return {
        "count": int(chosen.sum()),
        "AUROC": float(roc_auc_score(labels, probability[chosen])),
        "AP": float(average_precision_score(labels, probability[chosen])),
    }


def benchmark_metrics(
    probability: np.ndarray,
    *,
    separable: np.ndarray,
    entangled: np.ndarray,
    e2: np.ndarray,
    e3: np.ndarray | None = None,
    e4: np.ndarray | None = None,
    unknown: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    probability = np.asarray(probability, dtype=np.float64)
    separable = np.asarray(separable, dtype=bool)
    entangled = np.asarray(entangled, dtype=bool)
    e2 = np.asarray(e2, dtype=bool)
    e3 = np.zeros_like(e2) if e3 is None else np.asarray(e3, dtype=bool)
    e4 = np.zeros_like(e2) if e4 is None else np.asarray(e4, dtype=bool)
    unknown = ~(separable | entangled) if unknown is None else np.asarray(unknown, dtype=bool)
    strict = separable | entangled
    beyond = entangled & ~e2
    labels = entangled[strict].astype(np.int8)
    pred = probability[strict] >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    result: dict[str, Any] = {
        "strict_count": int(strict.sum()),
        "AUROC": float(roc_auc_score(labels, probability[strict])),
        "AP_ent": float(average_precision_score(labels, probability[strict])),
        "AP_sep": float(average_precision_score(1 - labels, 1.0 - probability[strict])),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "separable_count": int(separable.sum()),
        "Sep_FPR": float((probability[separable] >= threshold).mean()),
        "E2_count": int(e2.sum()),
        "E2_recall": float((probability[e2] >= threshold).mean()) if e2.any() else None,
        "E_gt_2_count": int(beyond.sum()),
        "E_gt_2_recall": float((probability[beyond] >= threshold).mean()) if beyond.any() else None,
        "E3_count": int(e3.sum()),
        "E3_recall": float((probability[e3] >= threshold).mean()) if e3.any() else None,
        "E4_count": int(e4.sum()),
        "E4_recall": float((probability[e4] >= threshold).mean()) if e4.any() else None,
        "unknown_count": int(unknown.sum()),
        "Sep_vs_E2": ranking(separable, e2, probability),
        "Sep_vs_E3": ranking(separable, e3, probability),
        "Sep_vs_E4": ranking(separable, e4, probability),
        "Sep_vs_E_gt_2": ranking(separable, beyond, probability),
    }
    return result


class RoutedInvariantTokens(Dataset):
    def __init__(
        self,
        arrays: list[np.ndarray],
        route: tuple[np.ndarray, np.ndarray, np.ndarray],
        channel_indices: np.ndarray,
    ):
        self.arrays = arrays
        self.source, self.rows, self.labels = route
        self.channel_indices = np.asarray(channel_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = int(self.source[index])
        row = int(self.rows[index])
        values = np.asarray(self.arrays[source][row])[:, self.channel_indices]
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)), torch.tensor(
            self.labels[index], dtype=torch.float32
        )


@torch.no_grad()
def predict_array(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(values), batch_size):
        block = torch.as_tensor(np.array(values[start : start + batch_size], copy=True), device=device)
        output.append(torch.sigmoid(model(block)).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability, labels = [], []
    for values, target in loader:
        probability.append(torch.sigmoid(model(values.to(device, non_blocking=True))).cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(probability), np.concatenate(labels).astype(np.int8)


def prepare_balanced_cache(cache_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    source_path = ROOT / "data/exactXY_balanced_N16_factorized_v1/dataset.h5"
    if not cache_path.exists():
        with h5py.File(source_path, "r") as h5:
            inv123 = compute_inv123_features(
                h5["features/X"][:], FeatureSpec(16), output_dtype=np.float32
            )
        spatial = organize_local_invariants(inv123, SpatialInvariantSpec("Inv123", 16))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, spatial)
    values = np.load(cache_path, mmap_mode="r")
    with h5py.File(source_path, "r") as h5:
        labels = {
            "sample_id": h5["metadata/sample_id"][:],
            "origin": h5["metadata/origin"].asstr()[:],
            "separable": h5["labels/certified_separable"][:],
            "entangled": h5["labels/certified_entangled"][:],
            "E2": h5["labels/E2"][:],
            "E_gt_2": h5["labels/E_gt_2"][:],
            "unknown": h5["labels/unknown"][:],
        }
    return values, labels


def exact500_labels() -> dict[str, np.ndarray]:
    with h5py.File(ROOT / "data/xy_n16_pbc_exact500_v1/dataset.h5", "r") as h5:
        entangled = h5["labels/certified_entangled"][:]
        e2 = entangled & h5["cert2/all_pairs/any_npt"][:]
        return {
            "sample_id": h5["params/sample_id"][:],
            "separable": h5["labels/certified_separable"][:],
            "entangled": entangled,
            "E2": e2,
            "E_gt_2": entangled & ~e2,
            "unknown": h5["labels/unknown"][:],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train exact-LU-invariant RoPE Transformers.")
    parser.add_argument("--variant", choices=("full", "robs4", "compact"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "analysis/cnn_locality_v1")
    parser.add_argument("--route-cache-dir", type=Path, default=ROOT / "analysis/connected_siamese_v1")
    parser.add_argument("--normalization-dir", type=Path, default=ROOT / "models/cnn_locality_remote_4090_20260813/SpatialMLP_control")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/invariant_transformer_v1")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis/invariant_transformer_v1")
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--max-epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is closed for this model-selection stage")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = TrainingConfig(seed=args.seed, max_epochs=args.max_epochs, patience=args.patience)
    selection_payload = None
    if args.variant == "compact":
        selection_path = args.selection_manifest or (args.analysis_dir / "selected_sparse_manifest.json")
        if not selection_path.exists():
            raise FileNotFoundError(selection_path)
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        channel_budget = int(selection_payload.get("physical_channel_budget", 200))
        name = f"CompactRoPETransformer_Inv_K{channel_budget}_seed{args.seed}"
    else:
        name = f"RoPETransformer_Inv123_{'Full' if args.variant == 'full' else 'Robs4'}_seed{args.seed}"
    output_dir = args.models_dir / name
    if (output_dir / "report.json").exists() and not args.force:
        print(json.dumps({"status": "skip_complete", "model": name}))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    # load_routes is the frozen Raw Transformer split/protocol audit. Its raw
    # matrices are intentionally discarded; only parent-safe routes are reused.
    _, route_data = load_routes(args.route_cache_dir)
    routes = route_data["routes"]
    local_arrays = [np.load(args.cache_dir / f"{source}_Inv123_local.npy", mmap_mode="r") for source in BASE]
    exact_full = np.load(args.cache_dir / "exact500_Inv123_local.npy", mmap_mode="r")
    full_spec = SpatialInvariantSpec("Inv123", 16)
    if args.variant == "full":
        channel_indices = np.arange(full_spec.local_channels, dtype=np.int64)
    else:
        local_indices = local_inv123_channel_indices(16, 4)
        if args.variant == "robs4":
            selected_local_channels = list(range(82))
        else:
            assert selection_payload is not None
            selected_local_channels = [int(value) for value in selection_payload["selected_local_token_channels"]]
            if len(selected_local_channels) != len(set(selected_local_channels)) or selected_local_channels[-5:] != list(range(77, 82)):
                raise RuntimeError("compact selection must contain unique physical channels followed by all five masks")
        channel_indices = local_indices[np.asarray(selected_local_channels, dtype=np.int64)]
    physical_channels = len(channel_indices)
    mean = np.load(args.normalization_dir / "local_mean.npy")[channel_indices]
    std = np.load(args.normalization_dir / "local_std.npy")[channel_indices]
    model_config = InvariantTransformerConfig(physical_channels=physical_channels)
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(model_config), mean, std
    ).to(device)
    parameters = trainable_parameter_count(model)
    train_ds = RoutedInvariantTokens(local_arrays, routes["train"], channel_indices)
    validation_ds = RoutedInvariantTokens(local_arrays, routes["validation"], channel_indices)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=config.eval_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    provenance = {
        "created_at": utc_now(),
        "git_commit": git_hash(),
        "random_seed": args.seed,
        "training_sources": list(BASE),
        "route_hash": route_data["route_hash"],
        "parent_split_audit": route_data["parent_audit"],
        "training_composition": route_data["training_composition"],
        "DPI_training_rows": 0,
        "N32_training_rows": 0,
        "exact5000_accessed": False,
        "feature_builder": "historical Inv123 cache + SpatialInvariantSpec mapping",
        "feature_manifest": "reports/inv123_feature_definition_manifest.csv",
        "normalization_source": str(args.normalization_dir),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }
    metadata = {
        "model_name": name,
        "variant": args.variant,
        "model_config": asdict(model_config),
        "training_config": asdict(config),
        "parameter_count": parameters,
        "token_shape_N16": [16, physical_channels],
        "physical_inv123_scalar_occurrences_N16": (
            1976 if args.variant == "full" else 1052 if args.variant == "robs4"
            else int(selection_payload.get("N16_physical_scalar_count", -1))
        ),
        "physical_channels_per_token": (
            None if args.variant == "full" else 77 if args.variant == "robs4"
            else int(selection_payload.get("physical_channel_budget", len(selected_local_channels) - 5))
        ),
        "mask_channels": 16 if args.variant == "full" else 5,
        "checkpoint_selection": "minimum N16 validation BCE only",
        "Siamese_loss": False,
        "orbit_averaging": False,
        "LU_augmentation": False,
        "from_scratch_compact_retraining": args.variant == "compact",
        "initialized_from_selector_classifier": False,
        "selection_manifest": str(args.selection_manifest or (args.analysis_dir / "selected_sparse_manifest.json")) if args.variant == "compact" else None,
        "selected_local_token_channels": selected_local_channels if args.variant != "full" else None,
        "provenance": provenance,
    }
    save_json(output_dir / "model_config.json", metadata)
    np.save(output_dir / "input_mean.npy", mean)
    np.save(output_dir / "input_std.npy", std)
    if args.variant == "robs4":
        save_json(args.analysis_dir / "local_inv123_manifest.json", {
            "schema": "existing Inv123 families, R_obs<=4, contiguous triples, OBC masks",
            "channel_count": physical_channels,
            "length_independent": True,
            "channels": local_inv123_channel_manifest(4),
        })

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, float | int]] = []
    best_loss, best_epoch, stale = math.inf, 0, 0
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_sum = 0.0
        epoch_start = time.perf_counter()
        for tokens, labels in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
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
    validation_probability, validation_labels = predict_loader(model, validation_loader, device)
    exact_values = np.asarray(exact_full)[:, :, channel_indices]
    exact_probability = predict_array(model, exact_values, device, config.eval_batch_size)
    exact_labels = exact500_labels()
    balanced_full, balanced_labels = prepare_balanced_cache(
        args.analysis_dir / "cache/XY_factorized_balanced_N16_Inv123_local.npy"
    )
    balanced_values = np.asarray(balanced_full)[:, :, channel_indices]
    balanced_probability = predict_array(model, balanced_values, device, config.eval_batch_size)
    exact_metrics = benchmark_metrics(
        exact_probability,
        separable=exact_labels["separable"], entangled=exact_labels["entangled"],
        e2=exact_labels["E2"], unknown=exact_labels["unknown"], threshold=config.threshold,
    )
    balanced_metrics = benchmark_metrics(
        balanced_probability,
        separable=balanced_labels["separable"], entangled=balanced_labels["entangled"],
        e2=balanced_labels["E2"], unknown=balanced_labels["unknown"], threshold=config.threshold,
    )
    old20 = balanced_labels["separable"] & (balanced_labels["origin"] == "historical_exact500")
    new412 = balanced_labels["separable"] & (balanced_labels["origin"] == "new_xy_factorized")
    balanced_metrics["old20"] = {
        "count": int(old20.sum()), "FPR": float((balanced_probability[old20] >= config.threshold).mean()),
        "mean_probability": float(balanced_probability[old20].mean()),
    }
    balanced_metrics["new412"] = {
        "count": int(new412.sum()), "FPR": float((balanced_probability[new412] >= config.threshold).mean()),
        "mean_probability": float(balanced_probability[new412].mean()),
    }
    validation_metrics = {
        "count": int(len(validation_labels)),
        "AUROC": float(roc_auc_score(validation_labels, validation_probability)),
        "AP_ent": float(average_precision_score(validation_labels, validation_probability)),
        "BCE": float(best_loss),
    }
    write_scores(output_dir / "scores_N16_exact500.csv", exact_probability, exact_labels)
    write_scores(output_dir / "scores_XY_factorized_balanced_N16.csv", balanced_probability, balanced_labels)
    report = {
        **metadata,
        "status": "complete_N16",
        "completed_at": utc_now(),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "training_wall_seconds": time.perf_counter() - started,
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics": {
            "N16_validation": validation_metrics,
            "N16_exact500": exact_metrics,
            "XY_factorized_balanced_N16": balanced_metrics,
        },
        "N32_finetuned": False,
        "N32_evaluated": False,
        "exact5000_accessed": False,
    }
    save_json(output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "model": name, "metrics": report["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

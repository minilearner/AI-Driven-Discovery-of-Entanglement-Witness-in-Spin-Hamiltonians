from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import (
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    structured_inv123_raw_t4_local_tokens,
)
from train_invariant_transformer import predict_array


SEEDS = (20260829, 20260830, 20260831)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probability_summary(values: np.ndarray) -> dict[str, float | int]:
    q = np.quantile(values, (0.5, 0.9, 0.95, 0.99))
    return {
        "count": len(values), "min": float(values.min()), "median": float(q[0]),
        "q90": float(q[1]), "q95": float(q[2]), "q99": float(q[3]),
        "max": float(values.max()), "mean": float(values.mean()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def build_cache(h5_path: Path, cache_path: Path, batch_size: int) -> np.ndarray:
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if cached.ndim != 3 or cached.shape[1:] != (32, 122):
            raise RuntimeError(f"invalid cache shape {cached.shape}")
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as h5:
        if not bool(h5.attrs.get("generation_complete", False)):
            raise RuntimeError("benchmark generation is not complete")
        rows = len(h5["raw/T1"])
        output = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(rows, 32, 122))
        for start in range(0, rows, batch_size):
            stop = min(rows, start + batch_size)
            output[start:stop] = structured_inv123_raw_t4_local_tokens(
                h5["raw/T1"][start:stop], h5["raw/T2"][start:stop], h5["derived/K2"][start:stop],
                h5["raw/T3"][start:stop], h5["derived/K3"][start:stop], h5["raw/T4"][start:stop],
                two_mask=h5["raw/T2_mask"][start:stop], three_mask=h5["raw/T3_mask"][start:stop],
                four_mask=h5["raw/T4_mask"][start:stop], R_obs=4, output_dtype=np.float32,
            )
        output.flush()
    return np.load(cache_path, mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Joint K16 FPR on the general-separable N32 OOD benchmark.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/general_separable_ood_n32_v1/dataset.h5")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models/invariant_transformer_raw_t4_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/general_separable_ood_n32_v1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is not part of this diagnostic")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dataset_hash = sha256(args.dataset)
    cache = build_cache(args.dataset, args.output_dir / "Inv123RawT4_Robs4.npy", args.batch_size)
    with h5py.File(args.dataset, "r") as h5:
        family = h5["meta/family"].asstr()[:]
        k_band = h5["meta/k_band"].asstr()[:]
        sample_id = h5["meta/sample_id"][:]
        if not np.all(h5["labels/strict_separable"][:]) or np.any(h5["labels/binary_entangled"][:]):
            raise RuntimeError("benchmark labels are not strictly separable")

    results, score_columns = [], {"sample_id": sample_id, "family": family, "k_band": k_band}
    checkpoint_hashes_before: dict[str, str] = {}
    for seed in SEEDS:
        name = f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}"
        model_dir = args.models_dir / name
        report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
        if report["budget"] != 16 or report["family"] != "joint" or report["DPI_training_rows"] != 0:
            raise RuntimeError(f"checkpoint is not the frozen Joint K16 model: {name}")
        checkpoint_path = model_dir / "best.pt"
        checkpoint_hashes_before[name] = sha256(checkpoint_path)
        channels = np.asarray(report["selected_full_token_channels"], dtype=np.int64)
        config = InvariantTransformerConfig(**report["model_config"])
        model = NormalizedInvariantTransformer(
            InvariantTokenTransformer(config), np.load(model_dir / "input_mean.npy"), np.load(model_dir / "input_std.npy"),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        probability = predict_array(model, np.asarray(cache)[:, :, channels], device, args.batch_size)
        threshold = float(report["training_config"]["threshold"])
        predicted = probability >= threshold
        row: dict[str, Any] = {
            "model": name, "seed": seed, "threshold": threshold,
            "false_positives": int(predicted.sum()), "total": len(probability),
            "FPR": float(predicted.mean()), "FPR_Wilson_95": wilson_interval(int(predicted.sum()), len(probability)),
            "probability": probability_summary(probability), "per_family": {}, "per_K_band": {},
        }
        for label, values in (("per_family", family), ("per_K_band", k_band)):
            for value in np.unique(values):
                mask = values == value
                fp = int(predicted[mask].sum())
                row[label][str(value)] = {
                    "count": int(mask.sum()), "false_positives": fp, "FPR": float(predicted[mask].mean()),
                    "FPR_Wilson_95": wilson_interval(fp, int(mask.sum())),
                    "probability": probability_summary(probability[mask]),
                }
        results.append(row)
        score_columns[f"probability_seed{seed}"] = probability
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = np.mean(np.column_stack([score_columns[f"probability_seed{seed}"] for seed in SEEDS]), axis=1)
    ensemble_predicted = ensemble >= 0.5
    score_columns["probability_ensemble_mean"] = ensemble
    score_columns["predicted_ensemble_mean"] = ensemble_predicted.astype(np.int8)
    score_path = args.output_dir / "scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        headers = list(score_columns)
        writer.writerow(headers)
        writer.writerows(zip(*(score_columns[key] for key in headers)))

    checkpoint_hashes_after = {name: sha256(args.models_dir / name / "best.pt") for name in checkpoint_hashes_before}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError("a frozen checkpoint changed during evaluation")
    if dataset_hash != sha256(args.dataset):
        raise RuntimeError("benchmark changed during evaluation")
    fprs = np.asarray([row["FPR"] for row in results])
    report = {
        "status": "complete", "benchmark": str(args.dataset.resolve()), "benchmark_sha256": dataset_hash,
        "model_family": "Joint K16", "training_performed": False, "optimizer_steps": 0,
        "checkpoint_selection_performed": False, "threshold_changed": False, "seeds": list(SEEDS),
        "per_seed": results,
        "FPR_across_seeds": {"mean": float(fprs.mean()), "sample_std": float(fprs.std(ddof=1)), "min": float(fprs.min()), "max": float(fprs.max())},
        "pooled_seed_decisions": {
            "false_positives": int(sum(row["false_positives"] for row in results)),
            "total": int(sum(row["total"] for row in results)),
            "descriptive_FPR": float(sum(row["false_positives"] for row in results) / sum(row["total"] for row in results)),
            "note": "descriptive only; the three predictions for each physical state are not independent observations",
        },
        "ensemble_mean_probability": {
            "false_positives": int(ensemble_predicted.sum()), "total": len(ensemble),
            "FPR": float(ensemble_predicted.mean()),
            "FPR_Wilson_95": wilson_interval(int(ensemble_predicted.sum()), len(ensemble)),
            "probability": probability_summary(ensemble),
        },
        "checkpoint_sha256_before": checkpoint_hashes_before,
        "checkpoint_sha256_after": checkpoint_hashes_after,
        "checkpoints_unchanged": True, "N32_finetuned": False, "DPI_training_rows": 0,
        "model_scores_used_for_benchmark_generation": False, "exact5000_accessed": False,
        "scores_csv": str(score_path.resolve()),
    }
    save_json(args.output_dir / "report.json", report)
    print(json.dumps({
        "status": "complete_frozen_evaluation", "per_seed_FPR": {str(row["seed"]): row["FPR"] for row in results},
        "ensemble_FPR": report["ensemble_mean_probability"]["FPR"],
    }, indent=2))


if __name__ == "__main__":
    main()

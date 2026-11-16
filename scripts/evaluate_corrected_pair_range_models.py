from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    structured_inv123_raw_t4_local_tokens,
)
from run_corrected_pair_range_selectors import (  # noqa: E402
    EXPERIMENT,
    TRAINING_SEEDS,
    compact_name,
)
from run_pair_range_r8_models import sha256  # noqa: E402
from run_raw_t4_invariant_transformer import (  # noqa: E402
    geometry_metrics,
    n16_balanced_labels,
    n32_labels,
)
from train_invariant_transformer import benchmark_metrics, exact500_labels, predict_array, write_scores  # noqa: E402


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def probability_summary(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, (0.5, 0.9, 0.95, 0.99))
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def pair_moments_from_separable_proof(
    weights: np.ndarray,
    bloch_vectors: np.ndarray,
    r_obs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly reconstruct open-chain T1, T2, C2 from a saved product-mixture proof."""
    weights = np.asarray(weights, dtype=np.float64)
    bloch_vectors = np.asarray(bloch_vectors, dtype=np.float64)
    if weights.ndim != 2 or bloch_vectors.ndim != 4:
        raise ValueError("expected weights [B,K] and Bloch vectors [B,K,N,3]")
    if weights.shape[:2] != bloch_vectors.shape[:2] or bloch_vectors.shape[-1] != 3:
        raise ValueError("separable proof shapes are inconsistent")
    batch, _, sites, _ = bloch_vectors.shape
    if not 1 <= r_obs < sites:
        raise ValueError(f"r_obs must lie in 1..{sites - 1}")
    one = np.einsum("bk,bkia->bia", weights, bloch_vectors, optimize=True)
    two = np.zeros((batch, sites, r_obs, 3, 3), dtype=np.float64)
    connected = np.zeros_like(two)
    mask = np.zeros((batch, sites, r_obs), dtype=bool)
    for offset in range(1, r_obs + 1):
        valid_sites = sites - offset
        moment = np.einsum(
            "bk,bkia,bkic->biac",
            weights,
            bloch_vectors[:, :, :valid_sites],
            bloch_vectors[:, :, offset:],
            optimize=True,
        )
        two[:, :valid_sites, offset - 1] = moment
        connected[:, :valid_sites, offset - 1] = (
            moment
            - one[:, :valid_sites, :, None] * one[:, offset:, None, :]
        )
        mask[:, :valid_sites, offset - 1] = True
    return one, two, connected, mask


def build_general_separable_cache(
    h5_path: Path,
    cache_path: Path,
    *,
    r_obs: int,
    width: int,
    batch_size: int,
) -> np.ndarray:
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, mmap_mode="r")
        if (
            metadata.get("status") != "complete_exact_from_constructive_proof"
            or cached.ndim != 3
            or cached.shape[1:] != (32, width)
        ):
            raise RuntimeError(f"invalid existing general-separable cache shape {cached.shape}")
        return cached
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = cache_path.with_name(cache_path.stem + ".partial.npy")
    partial_path.unlink(missing_ok=True)
    maximum_t1_error = 0.0
    maximum_t2_error = 0.0
    maximum_c2_error = 0.0
    masks_match = True
    with h5py.File(h5_path, "r") as h5:
        if not bool(h5.attrs.get("generation_complete", False)):
            raise RuntimeError("general-separable benchmark is incomplete")
        if "proof/weights" not in h5 or "proof/bloch_vectors" not in h5:
            raise RuntimeError("longer-range pairs require the saved constructive separability proof")
        rows = len(h5["raw/T1"])
        stored_pair_range = int(h5["raw/T2"].shape[2])
        output = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(rows, 32, width)
        )
        for start in range(0, rows, batch_size):
            stop = min(rows, start + batch_size)
            exact_t1, exact_t2, exact_c2, exact_two_mask = pair_moments_from_separable_proof(
                h5["proof/weights"][start:stop],
                h5["proof/bloch_vectors"][start:stop],
                r_obs,
            )
            prefix = min(stored_pair_range, r_obs)
            maximum_t1_error = max(
                maximum_t1_error,
                float(np.max(np.abs(exact_t1 - h5["raw/T1"][start:stop]))),
            )
            maximum_t2_error = max(
                maximum_t2_error,
                float(np.max(np.abs(
                    exact_t2[:, :, :prefix] - h5["raw/T2"][start:stop, :, :prefix]
                ))),
            )
            maximum_c2_error = max(
                maximum_c2_error,
                float(np.max(np.abs(
                    exact_c2[:, :, :prefix] - h5["derived/K2"][start:stop, :, :prefix]
                ))),
            )
            masks_match = masks_match and bool(np.array_equal(
                exact_two_mask[:, :, :prefix], h5["raw/T2_mask"][start:stop, :, :prefix]
            ))
            output[start:stop] = structured_inv123_raw_t4_local_tokens(
                exact_t1,
                exact_t2,
                exact_c2,
                h5["raw/T3"][start:stop],
                h5["derived/K3"][start:stop],
                h5["raw/T4"][start:stop],
                two_mask=exact_two_mask,
                three_mask=h5["raw/T3_mask"][start:stop],
                four_mask=h5["raw/T4_mask"][start:stop],
                R_obs=r_obs,
                output_dtype=np.float32,
            )
            print(json.dumps({"cache": f"general_sep_R{r_obs}", "completed": stop, "total": rows}), flush=True)
        output.flush()
    if max(maximum_t1_error, maximum_t2_error, maximum_c2_error) > 5.0e-6 or not masks_match:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            "constructive-proof pair reconstruction failed its stored-R4 regression check: "
            f"T1={maximum_t1_error}, T2={maximum_t2_error}, C2={maximum_c2_error}, "
            f"masks_match={masks_match}"
        )
    os.replace(partial_path, cache_path)
    save_json(metadata_path, {
        "status": "complete_exact_from_constructive_proof",
        "source": str(h5_path),
        "r_obs": r_obs,
        "shape": [rows, 32, width],
        "pair_boundary": "open chain; site i paired with i+r only when i+r<N",
        "formula": "T2_ij=sum_mu w_mu r_mu_i outer r_mu_j; C2_ij=T2_ij-m_i outer m_j",
        "stored_R4_regression": {
            "maximum_T1_absolute_error": maximum_t1_error,
            "maximum_T2_absolute_error": maximum_t2_error,
            "maximum_C2_absolute_error": maximum_c2_error,
            "masks_match": masks_match,
            "tolerance": 5.0e-6,
        },
        "historical_HDF5_modified": False,
    })
    return np.load(cache_path, mmap_mode="r")


def load_model(
    model_dir: Path,
    report: dict[str, Any],
    device: torch.device,
) -> tuple[NormalizedInvariantTransformer, np.ndarray, float]:
    channels = np.asarray(report["selected_full_token_channels"], dtype=np.int64)
    config = InvariantTransformerConfig(**report["model_config"])
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(config),
        np.load(model_dir / "input_mean.npy"),
        np.load(model_dir / "input_std.npy"),
    ).to(device)
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, channels, float(report["training_config"]["threshold"])


def general_separable_metrics(
    probability: np.ndarray,
    threshold: float,
    family: np.ndarray,
    k_band: np.ndarray,
) -> dict[str, Any]:
    predicted = probability >= threshold
    result: dict[str, Any] = {
        "false_positives": int(predicted.sum()),
        "total": len(predicted),
        "FPR": float(predicted.mean()),
        "specificity": float(1.0 - predicted.mean()),
        "Wilson_95": wilson_interval(int(predicted.sum()), len(predicted)),
        "probability": probability_summary(probability),
        "per_family": {},
        "per_K_band": {},
    }
    for name, values in (("per_family", family), ("per_K_band", k_band)):
        for value in np.unique(values):
            mask = values == value
            fp = int(predicted[mask].sum())
            result[name][str(value)] = {
                "count": int(mask.sum()),
                "false_positives": fp,
                "FPR": float(predicted[mask].mean()),
                "specificity": float(1.0 - predicted[mask].mean()),
                "Wilson_95": wilson_interval(fp, int(mask.sum())),
            }
    return result


def balanced_product_metrics(
    probability: np.ndarray,
    threshold: float,
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    origins = np.asarray(labels["origin"])
    separable = np.asarray(labels["separable"], dtype=bool)
    for origin in np.unique(origins[separable]):
        mask = separable & (origins == origin)
        predicted = probability[mask] >= threshold
        result[str(origin)] = {
            "count": int(mask.sum()),
            "false_positives": int(predicted.sum()),
            "FPR": float(predicted.mean()),
        }
    return result


def evaluate_one_family(
    *,
    label: str,
    cutoff: int,
    target_k: int,
    model_dirs: list[Path],
    reports: list[dict[str, Any]],
    cache: dict[str, np.ndarray],
    general_cache: np.ndarray,
    labels: dict[str, dict[str, np.ndarray]],
    general_family: np.ndarray,
    general_k_band: np.ndarray,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    per_seed = []
    probabilities_by_seed: dict[int, dict[str, np.ndarray]] = {}
    checkpoint_hashes_before = {model_dir.name: sha256(model_dir / "best.pt") for model_dir in model_dirs}
    for model_dir, report in zip(model_dirs, reports):
        seed = int(report["training_config"]["seed"])
        model, channels, threshold = load_model(model_dir, report, device)
        probability = {
            key: predict_array(model, np.asarray(values)[:, :, channels], device, batch_size)
            for key, values in cache.items()
        }
        probability["general_sep"] = predict_array(
            model, np.asarray(general_cache)[:, :, channels], device, batch_size
        )
        probabilities_by_seed[seed] = probability
        metrics = {
            "N16_exact500": benchmark_metrics(
                probability["exact"],
                separable=labels["exact"]["separable"],
                entangled=labels["exact"]["entangled"],
                e2=labels["exact"]["E2"],
                unknown=labels["exact"]["unknown"],
                threshold=threshold,
            ),
            "N16_XY_balanced": benchmark_metrics(
                probability["balanced"],
                separable=labels["balanced"]["separable"],
                entangled=labels["balanced"]["entangled"],
                e2=labels["balanced"]["E2"],
                unknown=labels["balanced"]["unknown"],
                threshold=threshold,
            ),
            "N32_XY_matched": benchmark_metrics(
                probability["xy"],
                separable=labels["xy"]["separable"],
                entangled=labels["xy"]["entangled"],
                e2=labels["xy"]["E2"],
                e3=labels["xy"]["E3"],
                e4=labels["xy"]["E4"],
                unknown=labels["xy"]["unknown"],
                threshold=threshold,
            ),
            "N32_DPI_general": benchmark_metrics(
                probability["dpi"],
                separable=labels["dpi"]["separable"],
                entangled=labels["dpi"]["entangled"],
                e2=labels["dpi"]["E2"],
                e3=labels["dpi"]["E3"],
                e4=labels["dpi"]["E4"],
                unknown=labels["dpi"]["unknown"],
                threshold=threshold,
            ),
            "N16_product_stress": balanced_product_metrics(
                probability["balanced"], threshold, labels["balanced"]
            ),
            "N32_general_separable": general_separable_metrics(
                probability["general_sep"], threshold, general_family, general_k_band
            ),
        }
        metrics["N32_DPI_E4_geometry"] = geometry_metrics(
            probability["dpi"],
            labels["dpi"],
            ROOT / "analysis/invariant_transformer_raw_t4_v1/dpi_e4_certificate_geometry.json",
            threshold,
        )
        score_dir = output_dir / "scores" / label / f"seed{seed}"
        score_dir.mkdir(parents=True, exist_ok=True)
        for key, benchmark_name in (
            ("exact", "N16_exact500"),
            ("balanced", "N16_XY_balanced"),
            ("xy", "N32_XY_matched"),
            ("dpi", "N32_DPI_general"),
        ):
            write_scores(score_dir / f"scores_{benchmark_name}.csv", probability[key], labels[key])
        per_seed.append(
            {
                "seed": seed,
                "threshold": threshold,
                "checkpoint_sha256": checkpoint_hashes_before[model_dir.name],
                "metrics": metrics,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    ensemble_probability = {
        key: np.mean(
            np.column_stack([
                probabilities_by_seed[seed][key] for seed in sorted(probabilities_by_seed)
            ]),
            axis=1,
        )
        for key in ("exact", "balanced", "xy", "dpi", "general_sep")
    }
    ensemble_threshold = 0.5
    ensemble_metrics = {
        "N16_exact500": benchmark_metrics(
            ensemble_probability["exact"],
            separable=labels["exact"]["separable"],
            entangled=labels["exact"]["entangled"],
            e2=labels["exact"]["E2"],
            unknown=labels["exact"]["unknown"],
            threshold=ensemble_threshold,
        ),
        "N16_XY_balanced": benchmark_metrics(
            ensemble_probability["balanced"],
            separable=labels["balanced"]["separable"],
            entangled=labels["balanced"]["entangled"],
            e2=labels["balanced"]["E2"],
            unknown=labels["balanced"]["unknown"],
            threshold=ensemble_threshold,
        ),
        "N32_XY_matched": benchmark_metrics(
            ensemble_probability["xy"],
            separable=labels["xy"]["separable"],
            entangled=labels["xy"]["entangled"],
            e2=labels["xy"]["E2"],
            e3=labels["xy"]["E3"],
            e4=labels["xy"]["E4"],
            unknown=labels["xy"]["unknown"],
            threshold=ensemble_threshold,
        ),
        "N32_DPI_general": benchmark_metrics(
            ensemble_probability["dpi"],
            separable=labels["dpi"]["separable"],
            entangled=labels["dpi"]["entangled"],
            e2=labels["dpi"]["E2"],
            e3=labels["dpi"]["E3"],
            e4=labels["dpi"]["E4"],
            unknown=labels["dpi"]["unknown"],
            threshold=ensemble_threshold,
        ),
        "N16_product_stress": balanced_product_metrics(
            ensemble_probability["balanced"], ensemble_threshold, labels["balanced"]
        ),
        "N32_general_separable": general_separable_metrics(
            ensemble_probability["general_sep"], ensemble_threshold, general_family, general_k_band
        ),
    }
    ensemble_metrics["N32_DPI_E4_geometry"] = geometry_metrics(
        ensemble_probability["dpi"],
        labels["dpi"],
        ROOT / "analysis/invariant_transformer_raw_t4_v1/dpi_e4_certificate_geometry.json",
        ensemble_threshold,
    )
    ensemble_score_dir = output_dir / "scores" / label / "probability_average_ensemble"
    ensemble_score_dir.mkdir(parents=True, exist_ok=True)
    for key, benchmark_name in (
        ("exact", "N16_exact500"),
        ("balanced", "N16_XY_balanced"),
        ("xy", "N32_XY_matched"),
        ("dpi", "N32_DPI_general"),
    ):
        write_scores(
            ensemble_score_dir / f"scores_{benchmark_name}.csv",
            ensemble_probability[key],
            labels[key],
        )
    checkpoint_hashes_after = {model_dir.name: sha256(model_dir / "best.pt") for model_dir in model_dirs}
    if checkpoint_hashes_before != checkpoint_hashes_after:
        raise RuntimeError(f"frozen checkpoint changed during evaluation: {label}")
    return {
        "label": label,
        "r_max": cutoff,
        "K": target_k,
        "per_seed": per_seed,
        "probability_average_ensemble": {
            "threshold": ensemble_threshold,
            "metrics": ensemble_metrics,
        },
        # Retained as an explicit convenience alias for downstream historical reports.
        "general_separable_probability_average_ensemble": ensemble_metrics["N32_general_separable"],
        "checkpoint_sha256_before": checkpoint_hashes_before,
        "checkpoint_sha256_after": checkpoint_hashes_after,
        "checkpoints_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-freeze evaluation of corrected pair-range models.")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / f"analysis/{EXPERIMENT}")
    parser.add_argument("--models-dir", type=Path, default=ROOT / f"models/{EXPERIMENT}")
    parser.add_argument("--r8-cache-dir", type=Path, default=ROOT / "analysis/pair_range_r8_invariant_attention_v1/cache")
    parser.add_argument("--r4-cache-dir", type=Path, default=ROOT / "analysis/invariant_transformer_raw_t4_v1/cache")
    parser.add_argument("--historical-models-dir", type=Path, default=ROOT / "models/invariant_transformer_raw_t4_v1")
    parser.add_argument("--general-separable", type=Path, default=ROOT / "data/general_separable_ood_n32_v1/dataset.h5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if any("exact5000" in value.lower() for value in sys.argv):
        raise RuntimeError("exact5000 is not part of this corrected selector study")
    frozen_path = args.analysis_dir / "frozen_best_k.json"
    if not frozen_path.exists():
        raise RuntimeError("best-K config must be frozen before OOD evaluation")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_from_parent_safe_N16_development_only":
        raise RuntimeError("invalid frozen best-K status")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = sha256(args.general_separable)
    general_r8 = build_general_separable_cache(
        args.general_separable,
        args.analysis_dir / "cache/general_separable_R8_exact_from_proof.npy",
        r_obs=8,
        width=182,
        batch_size=args.batch_size,
    )
    general_r4 = build_general_separable_cache(
        args.general_separable,
        args.analysis_dir / "cache/general_separable_R4_exact_from_proof.npy",
        r_obs=4,
        width=122,
        batch_size=args.batch_size,
    )
    with h5py.File(args.general_separable, "r") as h5:
        family = h5["meta/family"].asstr()[:]
        k_band = h5["meta/k_band"].asstr()[:]
        if not np.all(h5["labels/strict_separable"][:]) or np.any(h5["labels/binary_entangled"][:]):
            raise RuntimeError("general-separable labels changed")
    labels = {
        "exact": exact500_labels(),
        "balanced": n16_balanced_labels(),
        "xy": n32_labels("xy"),
        "dpi": n32_labels("dpi"),
    }
    r8_cache = {
        "exact": np.load(args.r8_cache_dir / "exact500_Inv123RawT4_Robs8.npy", mmap_mode="r"),
        "balanced": np.load(args.r8_cache_dir / "XY_factorized_balanced_N16_Inv123RawT4_Robs8.npy", mmap_mode="r"),
        "xy": np.load(args.r8_cache_dir / "N32_XY_Inv123RawT4_Robs8.npy", mmap_mode="r"),
        "dpi": np.load(args.r8_cache_dir / "N32_DPI_Inv123RawT4_Robs8.npy", mmap_mode="r"),
    }
    r4_cache = {
        "exact": np.load(args.r4_cache_dir / "exact500_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "balanced": np.load(args.r4_cache_dir / "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "xy": np.load(args.r4_cache_dir / "N32_XY_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "dpi": np.load(args.r4_cache_dir / "N32_DPI_Inv123RawT4_Robs4.npy", mmap_mode="r"),
    }
    jobs = []
    historical_dirs = [
        args.historical_models_dir / f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}"
        for seed in TRAINING_SEEDS
    ]
    historical_reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in historical_dirs]
    jobs.append(
        {
            "label": "Historical_Joint_K16_rmax4",
            "cutoff": 4,
            "K": 16,
            "dirs": historical_dirs,
            "reports": historical_reports,
            "cache": r4_cache,
            "general": general_r4,
        }
    )
    for cutoff in (6, 8):
        target_k = int(frozen["choices"][str(cutoff)]["best_K"])
        model_dirs = [args.models_dir / compact_name(cutoff, target_k, seed) for seed in TRAINING_SEEDS]
        reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in model_dirs]
        if any(report.get("OOD_evaluation_performed", False) for report in reports):
            raise RuntimeError("corrected checkpoint report already claims premature OOD access")
        jobs.append(
            {
                "label": f"Corrected_rmax{cutoff}_K{target_k}",
                "cutoff": cutoff,
                "K": target_k,
                "dirs": model_dirs,
                "reports": reports,
                "cache": r8_cache,
                "general": general_r8,
            }
        )
    started = time.time()
    results = [
        evaluate_one_family(
            label=job["label"],
            cutoff=job["cutoff"],
            target_k=job["K"],
            model_dirs=job["dirs"],
            reports=job["reports"],
            cache=job["cache"],
            general_cache=job["general"],
            labels=labels,
            general_family=family,
            general_k_band=k_band,
            output_dir=args.analysis_dir,
            device=device,
            batch_size=args.batch_size,
        )
        for job in jobs
    ]
    if dataset_hash != sha256(args.general_separable):
        raise RuntimeError("general-separable benchmark changed during evaluation")
    payload = {
        "status": "complete_post_freeze_evaluation",
        "experiment": EXPERIMENT,
        "frozen_best_k_sha256": sha256(frozen_path),
        "general_separable_sha256": dataset_hash,
        "general_separable_pair_extension": {
            "R4": json.loads(
                (args.analysis_dir / "cache/general_separable_R4_exact_from_proof.json").read_text(
                    encoding="utf-8"
                )
            ),
            "R8": json.loads(
                (args.analysis_dir / "cache/general_separable_R8_exact_from_proof.json").read_text(
                    encoding="utf-8"
                )
            ),
        },
        "results": results,
        "evaluation_wall_seconds": time.time() - started,
        "N32_finetuned": False,
        "DPI_training_rows": 0,
        "exact5000_accessed": False,
        "selection_protocol": "Top-K truncation of one converged target-32 ranking per rmax",
        "mathematically_optimal_for_each_K_claimed": False,
    }
    save_json(args.analysis_dir / "post_freeze_evaluation.json", payload)
    print(json.dumps({"status": payload["status"], "models": [row["label"] for row in results]}, indent=2))


if __name__ == "__main__":
    main()

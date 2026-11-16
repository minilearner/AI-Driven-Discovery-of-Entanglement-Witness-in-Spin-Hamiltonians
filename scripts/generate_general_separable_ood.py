from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.general_separable_ood import (
    FAMILIES,
    K_BANDS,
    direction_diagnostics,
    generate_sample,
    product_mixture_local_raw,
    production_manifest,
)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    q05, q25, median, q75, q95 = np.quantile(array, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "count": len(array), "min": float(array.min()), "q05": float(q05),
        "q25": float(q25), "median": float(median), "q75": float(q75),
        "q95": float(q95), "max": float(array.max()), "mean": float(array.mean()),
    }


def create_dataset(h5: h5py.File, name: str, shape: tuple[int, ...], dtype: Any) -> h5py.Dataset:
    chunks = (1, *shape[1:]) if len(shape) > 1 else True
    return h5.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, compression="gzip", shuffle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a model-independent N=32 general-separable OOD benchmark.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/general_separable_ood_n32_v1")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--N", type=int, default=32)
    parser.add_argument("--master-seed", type=int, default=2026091001)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.N != 32:
        raise ValueError("v1 is frozen as an N=32 zero-shot benchmark")
    manifest = production_manifest(args.samples, master_seed=args.master_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dataset.h5"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {output}")

    config = {
        "name": "general_separable_ood_n32_v1", "N": args.N, "samples": args.samples,
        "master_seed": args.master_seed, "R_obs": 4, "families": list(FAMILIES),
        "K_bands": [list(band) for band in K_BANDS], "max_K": 32,
        "direction_sampler": "iid normalized Gaussian independently for every (mu,i)",
        "selection_uses_model_scores": False, "all_samples_constructively_strict_separable": True,
    }
    save_json(args.output_dir / "config.json", config)
    started = time.perf_counter()
    string = h5py.string_dtype("utf-8")
    diagnostics: dict[str, list[float]] = {
        "mean_abs_cosine_mu0_mu1": [], "minimum_site_direction_rank": [],
        "maximum_bloch_norm": [], "effective_components": [],
    }
    family_values, band_values, k_values, alpha_values = [], [], [], []
    with h5py.File(output, "w") as h5:
        h5.attrs.update({
            "benchmark_name": config["name"], "generation_complete": False,
            "model_independent_generation": True, "constructive_separability": True,
            "direction_sampling": config["direction_sampler"],
        })
        shapes = {
            "raw/T1": (args.samples, args.N, 3),
            "raw/T2": (args.samples, args.N, 4, 3, 3),
            "raw/T3": (args.samples, args.N, 3, 3, 3),
            "raw/T4": (args.samples, args.N, 3, 3, 3, 3),
            "derived/K2": (args.samples, args.N, 4, 3, 3),
            "derived/K3": (args.samples, args.N, 3, 3, 3),
        }
        arrays = {name: create_dataset(h5, name, shape, np.float32) for name, shape in shapes.items()}
        masks = {
            "raw/T2_mask": create_dataset(h5, "raw/T2_mask", (args.samples, args.N, 4), np.bool_),
            "raw/T3_mask": create_dataset(h5, "raw/T3_mask", (args.samples, args.N), np.bool_),
            "raw/T4_mask": create_dataset(h5, "raw/T4_mask", (args.samples, args.N), np.bool_),
        }
        weights_ds = create_dataset(h5, "proof/weights", (args.samples, 32), np.float64)
        bloch_ds = create_dataset(h5, "proof/bloch_vectors", (args.samples, 32, args.N, 3), np.float64)
        component_mask = create_dataset(h5, "proof/component_mask", (args.samples, 32), np.bool_)
        completed = create_dataset(h5, "progress/completed", (args.samples,), np.bool_)
        meta = {
            "sample_id": h5.create_dataset("meta/sample_id", shape=(args.samples,), dtype=np.int32),
            "seed": h5.create_dataset("meta/seed", shape=(args.samples,), dtype=np.uint64),
            "K": h5.create_dataset("meta/K", shape=(args.samples,), dtype=np.int16),
            "alpha": h5.create_dataset("meta/dirichlet_alpha", shape=(args.samples,), dtype=np.float64),
            "family": h5.create_dataset("meta/family", shape=(args.samples,), dtype=string),
            "k_band": h5.create_dataset("meta/k_band", shape=(args.samples,), dtype=string),
            "constructive_proof": h5.create_dataset("meta/constructive_proof", shape=(args.samples,), dtype=string),
        }
        h5.create_dataset("labels/strict_separable", data=np.ones(args.samples, dtype=np.bool_))
        h5.create_dataset("labels/binary_entangled", data=np.zeros(args.samples, dtype=np.bool_))
        for index, row in enumerate(manifest):
            sample = generate_sample(row["family"], row["seed"], tuple(row["k_band"]), N=args.N)
            raw = product_mixture_local_raw(sample)
            for key in ("T1", "T2", "T3", "T4", "K2", "K3"):
                target = f"derived/{key}" if key.startswith("K") else f"raw/{key}"
                arrays[target][index] = raw[key].astype(np.float32)
            for key in ("T2_mask", "T3_mask", "T4_mask"):
                masks[f"raw/{key}"][index] = raw[key]
            weights_ds[index] = 0.0
            bloch_ds[index] = 0.0
            component_mask[index] = False
            weights_ds[index, : sample.K] = sample.weights
            bloch_ds[index, : sample.K] = sample.bloch
            component_mask[index, : sample.K] = True
            meta["sample_id"][index] = index
            meta["seed"][index] = sample.seed
            meta["K"][index] = sample.K
            meta["alpha"][index] = sample.alpha
            meta["family"][index] = sample.family
            meta["k_band"][index] = f"{sample.k_band[0]}-{sample.k_band[1]}"
            meta["constructive_proof"][index] = f"explicit_convex_mixture_of_{sample.K}_product_qubit_states"
            diagnostic = direction_diagnostics(sample)
            for key in diagnostics:
                diagnostics[key].append(float(diagnostic[key]))
            family_values.append(sample.family)
            band_values.append(f"{sample.k_band[0]}-{sample.k_band[1]}")
            k_values.append(sample.K)
            alpha_values.append(sample.alpha)
            completed[index] = True
            if args.progress_every and (index + 1) % args.progress_every == 0:
                h5.flush()
                print(json.dumps({"completed": index + 1, "total": args.samples}), flush=True)
        h5.attrs["generation_complete"] = True

    rank_values = np.asarray(diagnostics["minimum_site_direction_rank"], dtype=np.int64)
    expected_ranks = np.minimum(np.asarray(k_values), 3)
    if not np.array_equal(rank_values, expected_ranks):
        raise RuntimeError("direction-rank audit failed: a component/site direction set is unexpectedly collinear")
    mean_cosine = float(np.mean(diagnostics["mean_abs_cosine_mu0_mu1"]))
    if not 0.47 <= mean_cosine <= 0.53:
        raise RuntimeError(f"aggregate iid direction audit failed: mean |cos|={mean_cosine}")
    report = {
        "status": "complete", "dataset": str(output.resolve()), "dataset_sha256": sha256(output),
        "dataset_bytes": output.stat().st_size, "runtime_seconds": time.perf_counter() - started,
        "configuration": config, "family_counts": dict(Counter(family_values)),
        "K_band_counts": dict(Counter(band_values)), "K_summary": summary([float(value) for value in k_values]),
        "dirichlet_alpha_summary": summary(alpha_values),
        "direction_independence_audit": {
            "all_site_ranks_equal_min_K_3": True,
            "mean_abs_cosine_mu0_mu1": mean_cosine,
            "per_state_mean_abs_cosine_summary": summary(diagnostics["mean_abs_cosine_mu0_mu1"]),
            "fixed_axis_sign_flip_rows": 0,
        },
        "maximum_bloch_norm": max(diagnostics["maximum_bloch_norm"]),
        "effective_components_summary": summary(diagnostics["effective_components"]),
        "constructive_proof_saved": True, "model_scores_used_for_generation": False,
        "model_files_accessed_during_generation": False, "implementation_git_commit": git_hash(),
        "historical_datasets_modified": False, "exact5000_accessed": False,
    }
    save_json(args.output_dir / "report.json", report)
    (args.output_dir / "README.md").write_text(
        "# General separable OOD N32 v1\n\n"
        "This model-independent benchmark contains only explicitly constructed convex mixtures of product "
        "qubit states. Every Bloch-vector direction is sampled independently for each mixture component "
        "and site `(mu,i)`; there are no fixed-axis sign-flip rows and no model-score filtering. Full "
        "weights and one-qubit Bloch vectors are stored under `proof/`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "dataset": str(output), "sha256": report["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

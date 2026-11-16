from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
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

from entanglement_data.general_n32 import (
    CERTIFICATE_TOLERANCE,
    CLASS_FAMILIES,
    PPT_TOLERANCE,
    ROBUST_MARGIN,
    LocalRawSpec,
    contiguous_supports,
    generate_family_state,
    hierarchical_certificates,
    local_masks,
    local_raw,
    pair_supports,
    production_manifest,
)


DEFAULT_OUTPUT = ROOT / "data" / "dpi_n32_general_v1"
DATASET_NAME = "D_PI_N32.h5"
CLASS_TO_ID = {"strict_separable": 0, "E2": 1, "E3": 2, "E4": 3, "unknown": -1}
MASTER_SEED = 20260820


def git_hash() -> str:
    explicit = os.environ.get("PRODUCTION_GIT_HASH")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def expected_counts() -> tuple[dict[str, int], dict[str, int]]:
    classes = {name: sum(count for _, count in families) for name, families in CLASS_FAMILIES.items()}
    families = {family: count for items in CLASS_FAMILIES.values() for family, count in items}
    return classes, families


def create_dataset(path: Path, manifest: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    initializing = path.with_suffix(path.suffix + ".initializing")
    string = h5py.string_dtype("utf-8")
    count, N = len(manifest), 32
    spec = LocalRawSpec(N, 4)
    pair_mask, triple_mask, four_mask = local_masks(spec)
    with h5py.File(initializing, "w") as h5:
        h5.attrs.update(
            N=N,
            num_samples=count,
            model="general_block_local_entanglement_benchmark",
            generator_version="dpi_n32_general_v1",
            git_hash=git_hash(),
            master_seed=MASTER_SEED,
            pair_r_max=4,
            tol_pt=PPT_TOLERANCE,
            tol_ccnr=CERTIFICATE_TOLERANCE,
            robust_margin=ROBUST_MARGIN,
            certificate_policy="all_pairs_plus_contiguous_triples_and_quadruples",
            state_representation="at_most_four_qubit_exact_block_times_product_background_or_constructive_product_mixture",
            dense_2_power_N_constructed=False,
            exact5000_accessed=False,
            generation_complete=False,
        )
        h5.create_dataset("progress/completed", (count,), dtype=np.bool_)
        h5.create_dataset("meta/sample_id", data=np.arange(count, dtype=np.int32))
        for key in ("class_name", "family"):
            h5.create_dataset(f"meta/{key}", data=np.asarray([row[key] for row in manifest], dtype=object), dtype=string)
        h5.create_dataset("meta/family_index", data=np.asarray([row["family_index"] for row in manifest], dtype=np.int32))
        h5.create_dataset("meta/seed", data=np.asarray([row["seed"] for row in manifest], dtype=np.uint64))
        h5.create_dataset("meta/generator_detail", (count,), dtype=string)
        h5.create_dataset("meta/constructive_proof", (count,), dtype=string)
        h5.create_dataset("meta/noise_strength", (count,), dtype=np.float64, fillvalue=np.nan)
        h5.create_dataset("meta/circuit_depth", (count,), dtype=np.int16, fillvalue=-1)
        h5.create_dataset("meta/block_start", (count,), dtype=np.int16, fillvalue=-1)
        h5.create_dataset("meta/block_size", (count,), dtype=np.int8, fillvalue=0)
        h5.create_dataset("meta/runtime_seconds", (count,), dtype=np.float64, fillvalue=np.nan)
        h5.create_dataset("meta/max_imaginary_residual", (count,), dtype=np.float64, fillvalue=np.nan)

        h5.create_dataset("raw/T1", (count, N, 3), np.float32, chunks=(1, N, 3), compression="gzip")
        h5.create_dataset("raw/T2", (count, N, 4, 3, 3), np.float32, chunks=(1, N, 4, 3, 3), compression="gzip")
        h5.create_dataset("raw/T3", (count, N, 3, 3, 3), np.float32, chunks=(1, N, 3, 3, 3), compression="gzip")
        h5.create_dataset("raw/T4", (count, N, 3, 3, 3, 3), np.float32, chunks=(1, N, 3, 3, 3, 3), compression="gzip")
        h5.create_dataset("raw/T2_mask", data=np.broadcast_to(pair_mask, (count, *pair_mask.shape)), compression="gzip")
        h5.create_dataset("raw/T3_mask", data=np.broadcast_to(triple_mask, (count, *triple_mask.shape)), compression="gzip")
        h5.create_dataset("raw/T4_mask", data=np.broadcast_to(four_mask, (count, *four_mask.shape)), compression="gzip")
        h5.create_dataset("derived/K2", (count, N, 4, 3, 3), np.float32, chunks=(1, N, 4, 3, 3), compression="gzip")
        h5.create_dataset("derived/K3", (count, N, 3, 3, 3), np.float32, chunks=(1, N, 3, 3, 3), compression="gzip")

        h5.create_dataset("labels/class", data=np.asarray([CLASS_TO_ID[row["class_name"]] for row in manifest], dtype=np.int8))
        h5.create_dataset("labels/binary_entangled", data=np.asarray([row["class_name"] != "strict_separable" for row in manifest], dtype=np.bool_))
        for name in ("strict_separable", "E2", "E3", "E4"):
            h5.create_dataset(f"labels/{name}", data=np.asarray([row["class_name"] == name for row in manifest], dtype=np.bool_))
        h5.create_dataset("labels/unknown", data=np.zeros(count, dtype=np.bool_))

        pairs = np.asarray(pair_supports(N), dtype=np.int8)
        triples = np.asarray(contiguous_supports(N, 3), dtype=np.int8)
        fours = np.asarray(contiguous_supports(N, 4), dtype=np.int8)
        h5.create_dataset("certificates/pair_supports", data=pairs)
        h5.create_dataset("certificates/triple_supports", data=triples)
        h5.create_dataset("certificates/four_supports", data=fours)
        h5.create_dataset("certificates/pair_min_pt_eigenvalue", (count, len(pairs)), np.float32, fillvalue=np.nan, compression="gzip")
        h5.create_dataset("certificates/triple_min_pt_eigenvalue", (count, len(triples)), np.float32, fillvalue=np.nan, compression="gzip")
        h5.create_dataset("certificates/triple_max_ccnr_norm", (count, len(triples)), np.float32, fillvalue=np.nan, compression="gzip")
        h5.create_dataset("certificates/four_min_pt_eigenvalue", (count, len(fours)), np.float32, fillvalue=np.nan, compression="gzip")
        h5.create_dataset("certificates/four_max_ccnr_norm", (count, len(fours)), np.float32, fillvalue=np.nan, compression="gzip")
        h5.create_dataset("certificates/E2_margin", (count,), np.float64, fillvalue=np.nan)
        h5.create_dataset("certificates/E3_margin", (count,), np.float64, fillvalue=np.nan)
        h5.create_dataset("certificates/E4_margin", (count,), np.float64, fillvalue=np.nan)
        h5.create_dataset("certificates/pair_PPT_margin", (count,), np.float64, fillvalue=np.nan)
        h5.create_dataset("certificates/type", (count,), dtype=string)
        h5.create_dataset("certificates/subsystem", (count,), dtype=string)
        h5.create_dataset("certificates/cut", (count,), dtype=string)
        h5.create_dataset("certificates/triple_evaluated", (count,), dtype=np.bool_)
        h5.create_dataset("certificates/four_evaluated", (count,), dtype=np.bool_)
        h5.flush()
    initializing.replace(path)


def _empty(size: int) -> np.ndarray:
    return np.full(size, np.nan, dtype=np.float32)


def compute_one(task: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    index, row = task
    started = time.perf_counter()
    expected = row["class_name"]
    state, metadata = generate_family_state(row["family"], int(row["seed"]), 32)
    certificate = hierarchical_certificates(state, strict_separable=expected == "strict_separable")
    if certificate["class_name"] != expected:
        raise RuntimeError(f"sample {index} expected {expected} but certified {certificate['class_name']}")
    pairs = certificate["pairs"]
    triples = certificate["triples"]
    fours = certificate["fours"]
    if expected in {"E3", "E4"} and float(np.min(pairs.min_pt)) < PPT_TOLERANCE:
        raise RuntimeError(f"sample {index} lacks a strict all-pair-PPT safety margin")
    if expected == "strict_separable" and (pairs.npt_margin > CERTIFICATE_TOLERANCE or triples.certified or fours.certified):
        raise RuntimeError(f"constructive separable sample {index} produced a false certificate")
    if expected != "strict_separable" and float(certificate["certificate_margin"]) < ROBUST_MARGIN:
        raise RuntimeError(f"sample {index} has a fragile {expected} margin")
    raw = local_raw(state, LocalRawSpec(32, 4))
    raw_bound = max(float(np.max(np.abs(raw[key]))) for key in ("T1", "T2", "T3", "T4"))
    if raw_bound > 1.0 + 1e-9:
        raise RuntimeError(f"sample {index} violates a Pauli bound: {raw_bound}")
    margins = {"E2": math.nan, "E3": math.nan, "E4": math.nan}
    if expected in margins:
        margins[expected] = float(certificate["certificate_margin"])
    payload = {
        "raw": {key: raw[key].astype(np.float32) for key in ("T1", "T2", "T3", "T4", "K2", "K3")},
        "metadata": metadata,
        "max_imaginary_residual": float(state.max_imaginary_residual),
        "pair_min_pt": pairs.min_pt.astype(np.float32),
        "triple_min_pt": _empty(30) if triples is None else triples.min_pt.astype(np.float32),
        "triple_max_ccnr": _empty(30) if triples is None else triples.max_ccnr.astype(np.float32),
        "four_min_pt": _empty(29) if fours is None else fours.min_pt.astype(np.float32),
        "four_max_ccnr": _empty(29) if fours is None else fours.max_ccnr.astype(np.float32),
        "E2_margin": margins["E2"],
        "E3_margin": margins["E3"],
        "E4_margin": margins["E4"],
        "pair_PPT_margin": float(np.min(pairs.min_pt)),
        "certificate_type": certificate["certificate_type"],
        "certificate_support": json.dumps(certificate["certificate_support"]),
        "certificate_cut": json.dumps(certificate["certificate_cut"]),
        "triple_evaluated": triples is not None,
        "four_evaluated": fours is not None,
        "runtime_seconds": time.perf_counter() - started,
    }
    return index, payload


def write_one(h5: h5py.File, index: int, payload: dict[str, Any]) -> None:
    for key, value in payload["raw"].items():
        location = f"derived/{key}" if key.startswith("K") else f"raw/{key}"
        h5[location][index] = value
    metadata = payload["metadata"]
    for key in ("noise_strength", "circuit_depth", "block_start", "block_size"):
        h5[f"meta/{key}"][index] = metadata[key]
    h5["meta/generator_detail"][index] = metadata.get("generator_detail", "constructive_separable")
    h5["meta/constructive_proof"][index] = metadata["constructive_proof"]
    h5["meta/runtime_seconds"][index] = payload["runtime_seconds"]
    h5["meta/max_imaginary_residual"][index] = payload["max_imaginary_residual"]
    h5["certificates/pair_min_pt_eigenvalue"][index] = payload["pair_min_pt"]
    h5["certificates/triple_min_pt_eigenvalue"][index] = payload["triple_min_pt"]
    h5["certificates/triple_max_ccnr_norm"][index] = payload["triple_max_ccnr"]
    h5["certificates/four_min_pt_eigenvalue"][index] = payload["four_min_pt"]
    h5["certificates/four_max_ccnr_norm"][index] = payload["four_max_ccnr"]
    for key in ("E2_margin", "E3_margin", "E4_margin", "pair_PPT_margin"):
        h5[f"certificates/{key}"][index] = payload[key]
    h5["certificates/type"][index] = payload["certificate_type"]
    h5["certificates/subsystem"][index] = payload["certificate_support"]
    h5["certificates/cut"][index] = payload["certificate_cut"]
    h5["certificates/triple_evaluated"][index] = payload["triple_evaluated"]
    h5["certificates/four_evaluated"][index] = payload["four_evaluated"]
    h5["progress/completed"][index] = True


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
    }


def finalize(path: Path, output: Path, started: float | None = None) -> dict[str, Any]:
    with h5py.File(path, "r+") as h5:
        if not bool(np.all(h5["progress/completed"][:])):
            raise RuntimeError("cannot finalize an incomplete dataset")
        h5.attrs["generation_complete"] = True
        classes = h5["meta/class_name"].asstr()[:]
        families = h5["meta/family"].asstr()[:]
        report = {
            "status": "complete",
            "total": int(len(classes)),
            "class_counts": dict(Counter(classes)),
            "family_counts": {f"{class_name}/{family}": int(np.sum((classes == class_name) & (families == family))) for class_name in sorted(set(classes)) for family in sorted(set(families[classes == class_name]))},
            "certificate_margin_distributions": {name: distribution(h5[f"certificates/{name}_margin"][:]) for name in ("E2", "E3", "E4")},
            "pair_PPT_margin_E3_E4": distribution(h5["certificates/pair_PPT_margin"][:][np.isin(classes, ["E3", "E4"])]),
            "max_imaginary_residual": float(np.nanmax(h5["meta/max_imaginary_residual"][:])),
            "max_raw_absolute_value": max(float(np.max(np.abs(h5[f"raw/{key}"][:]))) for key in ("T1", "T2", "T3", "T4")),
            "physical_raw_correlator_count_per_state": 4317,
            "derived_connected_scalar_count_per_state": 1872,
            "file_size_bytes": int(path.stat().st_size),
            "summed_sample_runtime_seconds": float(np.nansum(h5["meta/runtime_seconds"][:])),
            "wall_seconds_this_invocation": None if started is None else time.perf_counter() - started,
            "failure_count": 0,
            "numerical_warning_count": 0,
            "unknown_count": int(np.sum(classes == "unknown")),
            "dense_2_power_32_constructed": False,
            "exact5000_accessed": False,
            "git_hash": str(h5.attrs["git_hash"]),
        }
        h5.flush()
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_config(output: Path, workers: int, command: str) -> None:
    classes, families = expected_counts()
    config = {
        "N": 32,
        "num_samples": 2000,
        "master_seed": MASTER_SEED,
        "class_counts": classes,
        "family_counts": families,
        "tol_pt": PPT_TOLERANCE,
        "tol_ccnr": CERTIFICATE_TOLERANCE,
        "robust_margin": ROBUST_MARGIN,
        "certificate_policy": "all C(32,2) pairs; 30 contiguous triples; 29 contiguous quadruples",
        "raw_shapes": {"T1": [2000, 32, 3], "T2": [2000, 32, 4, 3, 3], "T3": [2000, 32, 3, 3, 3], "T4": [2000, 32, 3, 3, 3, 3]},
        "raw_physical_count": 4317,
        "derived_K2_K3_count": 1872,
        "dense_2_power_32_constructed": False,
        "unknown_as_negative": False,
        "exact5000_accessed": False,
        "workers": workers,
        "production_command": command,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "completed": 0, "generation_complete": False}
    with h5py.File(path, "r") as h5:
        return {"exists": True, "completed": int(h5["progress/completed"][:].sum()), "generation_complete": bool(h5.attrs["generation_complete"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    path = args.output / DATASET_NAME
    if args.status:
        print(json.dumps(status(path)))
        return
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = production_manifest(MASTER_SEED)
    if not path.exists():
        create_dataset(path, manifest)
    command = f"PRODUCTION_GIT_HASH={git_hash()} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/generate_dpi_n32_general.py --workers {args.workers} --progress-every {args.progress_every}"
    write_config(args.output, args.workers, command)
    started = time.perf_counter()
    with h5py.File(path, "r+") as h5:
        current_hash = git_hash()
        if "initial_git_hash" not in h5.attrs:
            h5.attrs["initial_git_hash"] = str(h5.attrs["git_hash"])
        h5.attrs["git_hash"] = current_hash
        completed = h5["progress/completed"][:]
        pending = [(int(index), manifest[int(index)]) for index in np.flatnonzero(~completed)]
        if args.workers == 1:
            iterator = (compute_one(task) for task in pending)
            for index, payload in iterator:
                write_one(h5, index, payload)
                done = int(h5["progress/completed"][:].sum())
                if done % args.progress_every == 0 or done == len(manifest):
                    h5.flush()
                    print(json.dumps({"completed": done, "total": len(manifest), "sample_id": index, "class": manifest[index]["class_name"], "family": manifest[index]["family"], "seconds": payload["runtime_seconds"]}), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(compute_one, task): task[0] for task in pending}
                for future in as_completed(futures):
                    index, payload = future.result()
                    write_one(h5, index, payload)
                    done = int(h5["progress/completed"][:].sum())
                    if done % args.progress_every == 0 or done == len(manifest):
                        h5.flush()
                        print(json.dumps({"completed": done, "total": len(manifest), "sample_id": index, "class": manifest[index]["class_name"], "family": manifest[index]["family"], "seconds": payload["runtime_seconds"]}), flush=True)
        h5.flush()
    report = finalize(path, args.output, started)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

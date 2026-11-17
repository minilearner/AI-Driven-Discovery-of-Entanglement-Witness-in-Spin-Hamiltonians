"""Directly infer displayed XY thermal windows with Dv1+R2Support.

The lower and upper windows match the broken-axis ranges in the paper figure.
Every plotted cell is evaluated from exact XY correlators and the final three
checkpoints. Per-field cache files make an interrupted run resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from entanglement_data.control_entnet import load_control_ensemble
from entanglement_data.ml.invariant_transformer import structured_inv123_raw_t4_local_tokens
from entanglement_data.xy.free_fermion_exact import XYFreeFermionExactBackend
from entanglement_data.xy.size_transfer import LocalRawSpec, thermal_local_raw

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260829, 20260830, 20260831)
MODELS = None
CHANNELS = None


def init_worker() -> None:
    global MODELS, CHANNELS
    threadpool_limits(limits=1)
    torch.set_num_threads(1)
    MODELS = load_control_ensemble(ROOT, device="cpu")
    CHANNELS = json.loads((ROOT / "configs/joint_K16.json").read_text())[
        "selected_full_token_channels"]


def one_field(index: int, field: float, temperatures: np.ndarray) -> tuple[int, np.ndarray]:
    backend = XYFreeFermionExactBackend(16, 0.0, float(field), "PBC")
    tokens = []
    for temperature in temperatures:
        raw = thermal_local_raw(backend, 1.0 / float(temperature), LocalRawSpec(16, 4))
        full = structured_inv123_raw_t4_local_tokens(
            raw["T1"][None], raw["T2"][None], raw["K2"][None],
            raw["T3"][None], raw["K3"][None], raw["T4"][None],
            two_mask=raw["T2_mask"][None], three_mask=raw["T3_mask"][None],
            four_mask=raw["T4_mask"][None], R_obs=4, output_dtype=np.float32,
        )
        tokens.append(full[0][:, CHANNELS])
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(np.stack(tokens), dtype=np.float32))
        probabilities = np.column_stack([torch.sigmoid(model(x)).numpy() for model in MODELS])
    return index, probabilities.astype(np.float32)


def compute_panel(name: str, fields: np.ndarray, temperatures: np.ndarray,
                  cache_root: Path, output: Path, workers: int) -> dict:
    cache = cache_root / name
    cache.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(fields.tobytes() + temperatures.tobytes()).hexdigest()[:16]
    result = np.empty((len(fields), len(temperatures), 3), dtype=np.float32)
    pending = []
    for index, field in enumerate(fields):
        path = cache / f"{fingerprint}_{index:03d}.npy"
        if path.exists():
            row = np.load(path)
            if row.shape == (len(temperatures), 3) and np.isfinite(row).all():
                result[index] = row
                continue
        pending.append((index, field, path))
    print(f"{name}: {len(fields)} fields x {len(temperatures)} temperatures; "
          f"{len(pending)} fields to compute", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
            futures = {pool.submit(one_field, i, float(h), temperatures): (i, path)
                       for i, h, path in pending}
            done = 0
            for future in as_completed(futures):
                i, path = futures[future]
                returned_i, row = future.result()
                if returned_i != i or row.shape != (len(temperatures), 3):
                    raise RuntimeError("field result has wrong index or shape")
                np.save(path, row)
                result[i] = row
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"{name}: {done}/{len(pending)} new fields complete", flush=True)
    probability = result.mean(axis=2)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability grid is invalid")
    disagreement = np.any(result >= 0.5, axis=2) & np.any(result < 0.5, axis=2)
    np.savez_compressed(output / f"new_dense_{name}.npz", h=fields,
                        temperature=temperatures, p=probability, p_seed=result,
                        seed_sign_disagreement=disagreement)
    return {"name": name, "h_count": len(fields), "temperature_count": len(temperatures),
            "point_count": int(probability.size), "seed_disagreement_fraction": float(disagreement.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/thermal")
    parser.add_argument("--cache", type=Path, default=ROOT / "outputs/thermal/cache")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Replace completed dense-grid outputs")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.force and any((args.output / name).exists() for name in
                              ("new_dense_lower.npz", "new_dense_upper.npz", "dense_grid_report.json")):
        parser.error(f"output exists in {args.output}; choose --output elsewhere or --force")
    args.output.mkdir(parents=True, exist_ok=True)
    # Dense in the two *visible* windows; the hidden middle is not resampled.
    lower_h = np.linspace(1.0, 1.5, 201)
    lower_t = np.geomspace(0.01, 10 ** -0.5, 151)
    upper_h = np.linspace(0.6, 1.5, 181)
    upper_t = np.geomspace(10.0, 100.0, 101)
    panels = [compute_panel("lower", lower_h, lower_t, args.cache, args.output, args.workers),
              compute_panel("upper", upper_h, upper_t, args.cache, args.output, args.workers)]
    report = {"route": "Dv1+R2Support", "model": "Distance-two-augmented EntNet",
              "N": 16, "gamma": 0, "boundary_condition": "PBC",
              "panels": panels, "total_direct_grid_points": sum(p["point_count"] for p in panels),
              "source": "direct exact XY features and three frozen checkpoints; no interpolation"}
    (args.output / "dense_grid_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

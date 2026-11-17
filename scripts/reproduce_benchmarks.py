"""Recompute six paper benchmarks from public HDF5 data and frozen weights.

Input schemas and hashes come from data/benchmark_manifest.json. Unknown states
never enter labeled metrics; the decision threshold is fixed at p=0.5.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.final_entnet import load_final_ensemble  # noqa: E402
from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    structured_inv123_raw_t4_local_tokens,
)

SEEDS = (20260829, 20260830, 20260831)
MANIFEST = ROOT / "data/benchmark_manifest.json"


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or len(payload.get("files", [])) != 6:
        raise ValueError(f"invalid public benchmark manifest: {path}")
    return payload["files"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_labels(h5: h5py.File, kind: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if "labels/strict_separable" not in h5:
        raise ValueError("unsupported HDF5 schema: missing labels/strict_separable; use a published benchmark")
    group = h5["labels"]
    sep = group["strict_separable"][:].astype(bool)
    classes = {key: group[key][:].astype(bool) if key in group else np.zeros(len(sep), bool)
               for key in ("E2", "E3", "E4")}
    unknown = group["unknown"][:].astype(bool) if "unknown" in group else np.zeros(len(sep), bool)
    flags = sep.astype(int) + unknown.astype(int)
    for values in classes.values():
        flags += values.astype(int)
    if not np.all(flags == 1):
        raise ValueError(f"{kind}: labels are not an exclusive Sep/E2/E3/E4/Unknown partition")
    if kind == "xy32":
        ids = h5["params/sample_id"][:]
    elif kind in ("synthetic32", "separable32"):
        ids = h5["meta/sample_id"][:]
    else:
        ids = np.arange(len(sep))
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{kind}: sample IDs are not unique")
    return sep, {**classes, "unknown": unknown}, ids


def selected_batches(h5: h5py.File, kind: str, channels: list[int], batch_size: int):
    if kind == "adaptive":
        if "features/Joint_K16_inputs" not in h5:
            raise ValueError("adaptive benchmark requires features/Joint_K16_inputs")
        data = h5["features/Joint_K16_inputs"]
        if data.shape[1:] != (16, 21):
            raise ValueError(f"unexpected adaptive token shape {data.shape}")
        for first in range(0, len(data), batch_size):
            yield data[first:first + batch_size]
        return
    raw = "features" if kind == "xy32" else "raw"
    required = [f"{raw}/{key}" for key in ("T1", "T2", "T3", "T4", "T2_mask", "T3_mask", "T4_mask")]
    required += ["derived/K2", "derived/K3"]
    missing = [key for key in required if key not in h5]
    if missing:
        raise ValueError(f"{kind}: missing input fields {missing}; check the published benchmark schema")
    n = len(h5[f"{raw}/T1"])
    for first in range(0, n, batch_size):
        sl = slice(first, first + batch_size)
        tokens = structured_inv123_raw_t4_local_tokens(
            h5[f"{raw}/T1"][sl], h5[f"{raw}/T2"][sl], h5["derived/K2"][sl],
            h5[f"{raw}/T3"][sl], h5["derived/K3"][sl], h5[f"{raw}/T4"][sl],
            two_mask=h5[f"{raw}/T2_mask"][sl],
            three_mask=h5[f"{raw}/T3_mask"][sl],
            four_mask=h5[f"{raw}/T4_mask"][sl], R_obs=4, output_dtype=np.float32,
        )
        yield tokens[:, :, channels]


def score(models: list[torch.nn.Module], h5: h5py.File, kind: str, channels: list[int], batch_size: int) -> np.ndarray:
    blocks = []
    with torch.inference_mode():
        for values in selected_batches(h5, kind, channels, batch_size):
            if not np.isfinite(values).all():
                raise ValueError(f"{kind}: nonfinite selected tokens")
            x = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
            blocks.append(np.column_stack([
                torch.sigmoid(model(x)).cpu().numpy() for model in models
            ]))
    return np.concatenate(blocks) if blocks else np.empty((0, len(models)), dtype=np.float32)


def evaluate(entry: dict, data_root: Path, output: Path, models: list[torch.nn.Module],
             channels: list[int], batch_size: int) -> dict:
    name, kind = entry["id"], entry["schema"]
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe benchmark path: {relative}")
    path = data_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{name}: {path} is missing; run scripts/download_benchmarks.py")
    if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
        raise ValueError(f"{name}: HDF5 size/SHA-256 does not match data/benchmark_manifest.json")
    with h5py.File(path, "r") as h5:
        if kind != "adaptive" and not bool(h5.attrs.get("generation_complete", False)):
            raise ValueError(f"{name}: dataset generation not complete")
        sep, classes, ids = get_labels(h5, kind)
        scores = score(models, h5, kind, channels, batch_size)
    if len(scores) != len(ids) or len(ids) != entry["samples"] or scores.shape[1] != len(SEEDS):
        raise ValueError(f"{name}: prediction/sample count mismatch")
    p = scores.mean(axis=1).astype(np.float64)
    known = sep | classes["E2"] | classes["E3"] | classes["E4"]
    ent = classes["E2"] | classes["E3"] | classes["E4"]
    if sep.any() and ent.any():
        auroc = float(roc_auc_score(ent[known], p[known]))
    else:
        auroc = None
    metrics = {"AUROC": auroc, "TNR": float(np.mean(p[sep] < 0.5)) if sep.any() else None}
    for key in ("E2", "E3", "E4"):
        metrics[f"{key}_recall"] = float(np.mean(p[classes[key]] >= 0.5)) if classes[key].any() else None
    higher = classes["E3"] | classes["E4"]
    metrics["E_gt_2_recall"] = float(np.mean(p[higher] >= 0.5)) if higher.any() else None
    counts = {"total": len(p), "Sep": int(sep.sum()),
              **{key: int(classes[key].sum()) for key in ("E2", "E3", "E4")},
              "Unknown": int(classes["unknown"].sum())}
    expected = entry["labels"]
    for key, label in (("Sep", "strict_separable"), ("Unknown", "unknown"),
                       ("E2", "E2"), ("E3", "E3"), ("E4", "E4")):
        if counts[key] != expected[label]:
            raise ValueError(f"{name}: {key} count differs from public data manifest")
    paper = entry["paper_metrics"]
    differences = {key: float(metrics[key] - value) for key, value in paper.items()
                   if metrics.get(key) is not None}
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class", "ensemble_probability", *[f"probability_seed{s}" for s in SEEDS]])
        for i in range(len(p)):
            label = "Sep" if sep[i] else next((key for key in ("E2", "E3", "E4") if classes[key][i]), "Unknown")
            writer.writerow([int(ids[i]), label, float(p[i]), *[float(x) for x in scores[i]]])
    return {"benchmark": name, "model": "Baseline EntNet (Dv1-only Joint-K16)",
            "dataset": relative.as_posix(), "dataset_sha256": entry["sha256"],
            "counts": counts, "metrics": metrics, "paper_metrics": paper,
            "difference_from_paper": differences,
            "unknown_excluded": True, "threshold": 0.5,
            "checkpoint_seeds": list(SEEDS), "feature_manifest": "configs/joint_K16.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmarks")
    parser.add_argument("--benchmark", action="append", help="Run one benchmark by manifest ID; repeatable")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("--batch-size and --threads must be positive")
    all_entries = load_manifest(args.manifest)
    requested = set(args.benchmark or [entry["id"] for entry in all_entries])
    valid_names = {entry["id"] for entry in all_entries}
    if requested - valid_names:
        parser.error(f"unknown benchmark IDs: {sorted(requested - valid_names)}; use {sorted(valid_names)}")
    entries = [entry for entry in all_entries if entry["id"] in requested]
    expected_outputs = [args.output / f"{entry['id']}_scores.csv" for entry in entries]
    expected_outputs += [args.output / "benchmark_reproduction.json"]
    if not args.force and any(path.exists() for path in expected_outputs):
        parser.error(f"output already exists in {args.output}; use --output elsewhere or --force")
    for entry in entries:
        path = args.data_root / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"{entry['id']}: {path} is missing; run scripts/download_benchmarks.py")
    torch.set_num_threads(args.threads)
    models, channels = load_final_ensemble(ROOT)
    reports = []
    for entry in entries:
        result = evaluate(entry, args.data_root, args.output, models, channels, args.batch_size)
        reports.append(result)
        print(entry["id"], result["counts"], result["metrics"], flush=True)
    (args.output / "benchmark_reproduction.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(reports)} benchmark reports to {args.output}")


if __name__ == "__main__":
    main()

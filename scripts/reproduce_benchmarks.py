"""Recompute the paper's six transfer benchmarks from frozen HDF5 inputs.

The HDF5 files remain in the private local archive; this script writes only
compact predictions and reports inside the release candidate. Unknown states
never enter classification metrics. The old 312-row OOD datasets are excluded.
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
SPECS = {
    "XXZ": ("data/adaptive_gibbs_ood_n16_v1/xxz/dataset.h5", "adaptive"),
    "J1-J2": ("data/adaptive_gibbs_ood_n16_v1/j1j2/dataset.h5", "adaptive"),
    "Random-local-H": ("data/adaptive_gibbs_ood_n16_v1/dh_family/dataset.h5", "adaptive"),
    "N32-XY": ("data/xy_n32_pbc_matched_exact500_v1/dataset.h5", "xy32"),
    "N32-synthetic": ("data/dpi_n32_general_v1/D_PI_N32.h5", "synthetic32"),
    "General-Sep": ("data/general_separable_ood_n32_v1/dataset.h5", "separable32"),
}
PAPER = {
    "XXZ": {"AUROC": 1.0, "E2_recall": 0.9903, "E3_recall": 1.0, "E4_recall": 1.0, "TNR": 1.0},
    "J1-J2": {"AUROC": 1.0, "E2_recall": 0.8645, "E3_recall": 1.0, "E4_recall": 1.0, "TNR": 1.0},
    "Random-local-H": {"AUROC": 1.0, "E2_recall": 1.0, "E3_recall": 1.0, "E4_recall": 1.0, "TNR": 1.0},
    "N32-XY": {"AUROC": 1.0, "E2_recall": 1.0, "E3_recall": 1.0, "E4_recall": 1.0, "TNR": 1.0},
    "N32-synthetic": {"AUROC": 0.99984375, "E2_recall": 0.9875, "E3_recall": 0.9666666666666667, "E4_recall": 1.0, "TNR": 1.0},
    "General-Sep": {"TNR": 0.997},
}
PREVIOUS = {
    "N32-XY": "analysis/dv1_raw_t4_lu_control_v2/scores/joint_k16_N32_XY.csv",
    "N32-synthetic": "analysis/dv1_raw_t4_lu_control_v2/scores/joint_k16_N32_DPI.csv",
    "General-Sep": "analysis/dv1_raw_t4_lu_control_v2/scores/joint_k16_N32_GeneralSep.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hashes(archive: Path) -> dict[str, str]:
    folder = archive / "reports/remote_preservation_20261106"
    hashes = {}
    for filename in ("downloaded_manifest.json", "existing_core_verified.json"):
        for row in json.loads((folder / filename).read_text(encoding="utf-8")):
            hashes[row["path"]] = row.get("sha256_local", row.get("sha256"))
    for name, (relative, _) in SPECS.items():
        if name == "N32-XY":
            continue  # locally generated; no server-preservation record
        if relative not in hashes:
            raise ValueError(f"no preservation hash for {relative}")
    return hashes


def get_labels(h5: h5py.File, kind: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    group = h5["labels"]
    sep = group["strict_separable"][:].astype(bool)
    classes = {key: group[key][:].astype(bool) if key in group else np.zeros(len(sep), bool)
               for key in ("E2", "E3", "E4")}
    unknown = group["unknown"][:].astype(bool) if "unknown" in group else np.zeros(len(sep), bool)
    flags = sep.astype(int) + unknown.astype(int)
    for values in classes.values():
        flags += values.astype(int)
    if not np.all(flags == 1):
        raise ValueError(f"{kind}: labels are not an exclusive partition")
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
        data = h5["features/Joint_K16_inputs"]
        if data.shape[1:] != (16, 21):
            raise ValueError(f"unexpected adaptive token shape {data.shape}")
        for first in range(0, len(data), batch_size):
            yield data[first:first + batch_size]
        return
    raw = "features" if kind == "xy32" else "raw"
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
    return np.concatenate(blocks)


def previous_difference(archive: Path, name: str, ids: np.ndarray, p: np.ndarray) -> float | None:
    if name not in PREVIOUS:
        return None
    path = archive / PREVIOUS[name]
    with path.open(newline="", encoding="utf-8") as stream:
        old = {int(row["sample_id"]): float(row["ensemble_probability"])
               for row in csv.DictReader(stream)}
    if len(old) != len(ids) or set(old) != set(map(int, ids)):
        raise ValueError(f"{name}: saved score IDs do not match current HDF5")
    return float(np.max(np.abs(p - np.asarray([old[int(i)] for i in ids]))))


def evaluate(name: str, archive: Path, output: Path, models: list[torch.nn.Module],
             channels: list[int], hashes: dict[str, str], batch_size: int) -> dict:
    relative, kind = SPECS[name]
    path = archive / relative
    actual = sha256(path)
    expected = hashes.get(relative)
    if expected and actual != expected:
        raise ValueError(f"{name}: dataset SHA-256 differs from preserved source")
    with h5py.File(path, "r") as h5:
        if kind != "adaptive" and not bool(h5.attrs.get("generation_complete", False)):
            raise ValueError(f"{name}: dataset generation not complete")
        sep, classes, ids = get_labels(h5, kind)
        scores = score(models, h5, kind, channels, batch_size)
    if len(scores) != len(ids) or scores.shape[1] != 3:
        raise ValueError(f"{name}: prediction shape mismatch")
    p = scores.mean(axis=1).astype(np.float64)
    known = ~classes["unknown"]
    ent = classes["E2"] | classes["E3"] | classes["E4"]
    if known.any() and sep.any() and ent.any():
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
    differences = {key: (None if value is None else float(metrics[key] - value))
                   for key, value in PAPER[name].items() if key in metrics and metrics[key] is not None}
    old_diff = previous_difference(archive, name, ids, p)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{name}_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class", "ensemble_probability", *[f"probability_seed{s}" for s in SEEDS]])
        for i in range(len(p)):
            label = "Sep" if sep[i] else next((key for key in ("E2", "E3", "E4") if classes[key][i]), "Unknown")
            writer.writerow([int(ids[i]), label, float(p[i]), *[float(x) for x in scores[i]]])
    return {"benchmark": name, "model": "Dv1-only Joint-K16", "dataset": relative,
            "dataset_sha256": actual, "preservation_hash_match": expected is not None,
            "counts": counts, "metrics": metrics, "paper_metrics": PAPER[name],
            "difference_from_paper": differences,
            "max_abs_difference_vs_saved_per_row": old_diff,
            "unknown_excluded": True, "threshold": 0.5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase3_benchmarks")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    models, channels = load_final_ensemble(ROOT)
    hashes = expected_hashes(args.archive_root)
    reports = []
    for name in SPECS:
        result = evaluate(name, args.archive_root, args.output, models, channels, hashes, args.batch_size)
        reports.append(result)
        print(name, result["counts"], result["metrics"], flush=True)
    (args.output / "benchmark_reproduction.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

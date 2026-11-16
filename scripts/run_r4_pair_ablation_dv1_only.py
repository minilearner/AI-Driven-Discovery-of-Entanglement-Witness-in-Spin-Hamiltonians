"""Paired-seed Dv1-only rerun of the frozen r=4 pair-channel ablation.

This keeps the three previously chosen physical channel sets unchanged. It
trains all three arms from scratch on the same Dv1-only route and normalization,
then evaluates frozen checkpoints on the original, disjoint benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import audit_dv1_only_joint_selection_budget as dv1  # noqa: E402
import evaluate_corrected_pair_range_models as evaluator  # noqa: E402
import run_raw_t4_invariant_transformer as pipeline  # noqa: E402
from train_invariant_transformer import exact500_labels  # noqa: E402


EXPERIMENT = "r4_pair_ablation_dv1_only_v1"
OUT = ROOT / "analysis" / EXPERIMENT
MODELS = ROOT / "models" / EXPERIMENT
FROZEN = ROOT / "configs/r4_ablation"
DV1 = ROOT / "analysis/dv1_only_joint_selection_budget_audit_v1"
HIST = ROOT / "analysis/invariant_transformer_raw_t4_v1"
CACHE = OUT / "training_cache"
GENERAL_H5 = ROOT / "data/general_separable_ood_n32_v1/dataset.h5"
SEEDS = (20260829, 20260830, 20260831)
ARMS = (
    ("Dv1_Baseline_K16", 16, DV1 / "selection_manifests/joint_K16.json"),
    ("Dv1_DropR4_K14", 14, FROZEN / "joint_K14.json"),
    ("Dv1_ReplaceR4_T4_K16", 16, FROZEN / "joint_K16_higher_order.json"),
)
MASKS = [117, 118, 119, 120, 121]
EXPECTED_BASELINE = [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 44, 83, 91, 51, 3, 59]
EXPECTED_DROP = [x for x in EXPECTED_BASELINE if x not in (44, 51)]
EXPECTED_REPLACE = [*EXPECTED_DROP, 85, 112]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def model_dir(label: str, seed: int) -> Path:
    return MODELS / f"RoPETransformer_LocalInv123_RawT4_{label}_seed{seed}"


def source_manifest(label: str) -> Path:
    return next(path for name, _, path in ARMS if name == label)


def prepare() -> dict[str, Any]:
    if not GENERAL_H5.exists():
        raise FileNotFoundError(GENERAL_H5)
    geometry_source = DV1 / "dpi_e4_certificate_geometry.json"
    geometry_target = OUT / geometry_source.name
    if not geometry_source.exists():
        raise FileNotFoundError(geometry_source)
    geometry_target.parent.mkdir(parents=True, exist_ok=True)
    if not geometry_target.exists():
        shutil.copy2(geometry_source, geometry_target)
    elif digest(geometry_target) != digest(geometry_source):
        raise RuntimeError("DPI certificate geometry differs from the Dv1-only baseline")
    route = dv1.dv1_route()
    counts = {split: len(route["routes"][split][1]) for split in ("train", "validation", "test")}
    if counts != {"train": 35000, "validation": 7500, "test": 7500}:
        raise RuntimeError(f"unexpected Dv1-only split sizes: {counts}")
    if any(np.any(route["routes"][split][0] != 0) for split in counts):
        raise RuntimeError("non-Dv1 rows entered the training/validation/test route")
    norm_cache = DV1 / "cache"
    if not all((norm_cache / name).exists() for name in ("input_mean_full.npy", "input_std_full.npy")):
        raise RuntimeError("Dv1-only normalization is not prepared")
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_sources = {
        "input_mean_full.npy": norm_cache / "input_mean_full.npy",
        "input_std_full.npy": norm_cache / "input_std_full.npy",
    }
    cache_sources.update({
        f"{name}_Inv123RawT4_Robs4.npy": HIST / "cache" / f"{name}_Inv123RawT4_Robs4.npy"
        for name in pipeline.BASE
    })
    cache_sources.update({
        name: HIST / "cache" / name
        for name in (
            "exact500_Inv123RawT4_Robs4.npy",
            "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy",
            "N32_XY_Inv123RawT4_Robs4.npy",
            "N32_DPI_Inv123RawT4_Robs4.npy",
        )
    })
    for name, source in cache_sources.items():
        target = CACHE / name
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"prepared cache link differs from source: {name}")
        else:
            os.symlink(source, target)
    previous = [
        ROOT / "models/dv1_only_joint_budget_audit_v1"
        / f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}"
        for seed in SEEDS
    ]
    reference_reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in previous]
    if any(report["route_hash"] != route["route_hash"] for report in reference_reports):
        raise RuntimeError("Dv1 route differs from the audited canonical K16 route")
    if any(report["training_config"] != asdict(pipeline.TrainingConfig(seed=seed))
           for report, seed in zip(reference_reports, SEEDS)):
        raise RuntimeError("training configuration differs from canonical Dv1-only K16")
    manifests: dict[str, dict[str, Any]] = {}
    expected = (EXPECTED_BASELINE, EXPECTED_DROP, EXPECTED_REPLACE)
    for (label, budget, source), physical in zip(ARMS, expected):
        item = json.loads(source.read_text(encoding="utf-8"))
        if item["selected_physical_full_channels"] != physical:
            raise RuntimeError(f"frozen physical channels changed: {label}")
        if item["selected_full_token_channels"] != [*physical, *MASKS]:
            raise RuntimeError(f"frozen masks or channel order changed: {label}")
        if item["physical_channel_count"] != budget:
            raise RuntimeError(f"channel budget changed: {label}")
        manifests[label] = item
        target = OUT / "selection_manifests" / f"{label}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(source):
            raise RuntimeError(f"prepared manifest differs from frozen source: {label}")
        shutil.copy2(source, target)
    full_mean = np.load(CACHE / "input_mean_full.npy")
    full_std = np.load(CACHE / "input_std_full.npy")
    baseline_channels = manifests[ARMS[0][0]]["selected_full_token_channels"]
    for path in previous:
        if not np.array_equal(np.load(path / "input_mean.npy"), full_mean[baseline_channels]):
            raise RuntimeError("Dv1 normalization mean differs from canonical baseline")
        if not np.array_equal(np.load(path / "input_std.npy"), full_std[baseline_channels]):
            raise RuntimeError("Dv1 normalization scale differs from canonical baseline")
    audit = {
        "status": "prepared",
        "experiment": EXPERIMENT,
        "training_sources": ["D_v1"],
        "source_ids_in_all_splits": [0],
        "split_counts": counts,
        "route_hash": route["route_hash"],
        "normalization_source": "Dv1 training rows only, valid supports",
        "normalization_sha256": {name: digest(CACHE / name) for name in ("input_mean_full.npy", "input_std_full.npy")},
        "paired_seeds": list(SEEDS),
        "training_config_except_seed": {k: v for k, v in asdict(pipeline.TrainingConfig(seed=0)).items() if k != "seed"},
        "checkpoint_selection": "minimum canonical N16 validation BCE only",
        "frozen_selection": True,
        "removed_r4_full_channels": [44, 51],
        "replacement_t4_full_channels": [85, 112],
        "arms": {label: {"K": budget, "source_sha256": digest(source),
                         "selected_physical_full_channels": item["selected_physical_full_channels"],
                         "selected_full_token_channels": item["selected_full_token_channels"]}
                 for (label, budget, source), item in zip(ARMS, manifests.values())},
        "evaluation_data_used_for_selection": False,
        "exact5000_accessed": False,
    }
    save_json(OUT / "route_feature_audit.json", audit)
    return audit


def train(only_arm: Optional[str] = None) -> None:
    audit = prepare()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    route = dv1.dv1_route()
    arrays = pipeline.load_cached_inputs(CACHE)
    original = pipeline.model_job
    try:
        for label, budget, _ in ARMS:
            if only_arm is not None and label != only_arm:
                continue
            target = OUT / "selection_manifests" / f"{label}.json"
            pipeline.model_job = (
                lambda family, requested_budget, analysis_dir, *,
                _label=label, _budget=budget, _target=target:
                (_label, _target) if family == "joint" and requested_budget == _budget
                else (_ for _ in ()).throw(RuntimeError("unexpected training job"))
            )
            for seed in SEEDS:
                report = pipeline.train_model(
                    "joint", budget, seed, arrays=arrays, route_data=route,
                    cache_dir=CACHE, analysis_dir=OUT, models_dir=MODELS,
                    device=torch.device("cuda"), config=pipeline.TrainingConfig(seed=seed), force=False,
                )
                expected_channels = audit["arms"][label]["selected_full_token_channels"]
                if report["route_hash"] != audit["route_hash"] or report["selected_full_token_channels"] != expected_channels:
                    raise RuntimeError(f"training provenance mismatch: {label}/{seed}")
                print(json.dumps({"trained_and_checked": label, "seed": seed,
                                  "best_checkpoint_sha256": digest(model_dir(label, seed) / "best.pt")}), flush=True)
    finally:
        pipeline.model_job = original


def evaluate() -> dict[str, Any]:
    audit = prepare()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    cache_root = HIST / "cache"
    caches = {
        "exact": np.load(cache_root / "exact500_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "balanced": np.load(cache_root / "XY_factorized_balanced_N16_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "xy": np.load(cache_root / "N32_XY_Inv123RawT4_Robs4.npy", mmap_mode="r"),
        "dpi": np.load(cache_root / "N32_DPI_Inv123RawT4_Robs4.npy", mmap_mode="r"),
    }
    general = evaluator.build_general_separable_cache(
        GENERAL_H5, OUT / "cache/general_separable_R4_exact_from_proof.npy",
        r_obs=4, width=122, batch_size=256,
    )
    if general.shape != (2000, 32, 122):
        raise RuntimeError(f"general-separable feature shape changed: {general.shape}")
    with h5py.File(GENERAL_H5, "r") as h5:
        family = h5["meta/family"].asstr()[:]
        k_band = h5["meta/k_band"].asstr()[:]
        if not np.all(h5["labels/strict_separable"][:]) or np.any(h5["labels/binary_entangled"][:]):
            raise RuntimeError("constructively separable labels changed")
    labels = {
        "exact": exact500_labels(),
        "balanced": pipeline.n16_balanced_labels(),
        "xy": pipeline.n32_labels("xy"),
        "dpi": pipeline.n32_labels("dpi"),
    }
    results = []
    for label, budget, _ in ARMS:
        dirs = [model_dir(label, seed) for seed in SEEDS]
        reports = [json.loads((path / "report.json").read_text(encoding="utf-8")) for path in dirs]
        if any(report["route_hash"] != audit["route_hash"] for report in reports):
            raise RuntimeError(f"mixed-route checkpoint entered evaluation: {label}")
        results.append(evaluator.evaluate_one_family(
            label=label, cutoff=4, target_k=budget, model_dirs=dirs, reports=reports,
            cache=caches, general_cache=general, labels=labels,
            general_family=family, general_k_band=k_band,
            output_dir=OUT, device=device, batch_size=256,
        ))
    payload = {
        "status": "evaluated", "experiment": EXPERIMENT,
        "route_hash": audit["route_hash"], "training_sources": ["D_v1"],
        "seeds": list(SEEDS), "general_separable_sha256": digest(GENERAL_H5),
        "results": results, "selector_retrained": False,
        "exact5000_accessed": False,
    }
    save_json(OUT / "evaluation.json", payload)
    return payload


def summarize() -> dict[str, Any]:
    payload = json.loads((OUT / "evaluation.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "route_feature_audit.json").read_text(encoding="utf-8"))
    if payload["route_hash"] != audit["route_hash"] or len(payload["results"]) != 3:
        raise RuntimeError("evaluation and route audit disagree")
    summary = {"status": "complete", "experiment": EXPERIMENT,
               "training_sources": ["D_v1"], "route_hash": audit["route_hash"],
               "seeds": list(SEEDS), "arms": {}}
    for arm in payload["results"]:
        ensemble = arm["probability_average_ensemble"]["metrics"]
        per_seed = {}
        for row in arm["per_seed"]:
            m = row["metrics"]
            per_seed[str(row["seed"])] = {
                "exact500_AUROC": m["N16_exact500"]["AUROC"],
                "balanced_AUROC": m["N16_XY_balanced"]["AUROC"],
                "N32_XY_AUROC": m["N32_XY_matched"]["AUROC"],
                "GeneralSep_TNR": m["N32_general_separable"]["specificity"],
            }
        metrics = {
            "exact500_AUROC": ensemble["N16_exact500"]["AUROC"],
            "balanced_AUROC": ensemble["N16_XY_balanced"]["AUROC"],
            "N32_XY_AUROC": ensemble["N32_XY_matched"]["AUROC"],
            "GeneralSep_TNR": ensemble["N32_general_separable"]["specificity"],
            "GeneralSep_false_positives": ensemble["N32_general_separable"]["false_positives"],
        }
        summary["arms"][arm["label"]] = {
            "K": arm["K"], "channels": audit["arms"][arm["label"]]["selected_full_token_channels"],
            "per_seed": per_seed, "probability_average_ensemble": metrics,
            "seed_mean": {key: float(np.mean([row[key] for row in per_seed.values()]))
                          for key in next(iter(per_seed.values()))},
            "seed_sample_std": {key: float(np.std([row[key] for row in per_seed.values()], ddof=1))
                                for key in next(iter(per_seed.values()))},
            "checkpoints_unchanged_during_evaluation": arm["checkpoints_unchanged"],
        }
    baseline = summary["arms"][ARMS[0][0]]["per_seed"]
    for label in (ARMS[1][0], ARMS[2][0]):
        current = summary["arms"][label]["per_seed"]
        summary["arms"][label]["paired_seed_deltas_vs_baseline"] = {
            seed: {metric: current[seed][metric] - baseline[seed][metric] for metric in baseline[seed]}
            for seed in baseline
        }
    save_json(OUT / "summary.json", summary)
    print(json.dumps({"status": "complete", "ensemble": {
        name: row["probability_average_ensemble"] for name, row in summary["arms"].items()
    }}, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "train", "evaluate", "summarize", "all"), default="all")
    parser.add_argument("--arm", choices=[arm[0] for arm in ARMS])
    args = parser.parse_args()
    if any("exact5000" in item.lower() for item in sys.argv):
        raise RuntimeError("exact5000 access is prohibited in this rerun")
    if args.phase in ("prepare", "all"):
        print(json.dumps(prepare(), indent=2), flush=True)
    if args.phase in ("train", "all"):
        train(args.arm)
    if args.phase in ("evaluate", "all"):
        evaluate()
    if args.phase in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()

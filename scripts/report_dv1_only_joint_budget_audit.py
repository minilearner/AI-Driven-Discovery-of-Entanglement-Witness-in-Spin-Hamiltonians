"""Independently recompute the frozen-ranking Dv1-only budget scan metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/dv1_only_joint_selection_budget_audit_v1"
MODELS = ROOT / "models/dv1_only_joint_budget_audit_v1"
CANONICAL = ROOT / "models/joint_k16_training_distribution_ablation_v1"
BUDGETS = (4, 8, 12, 16, 20, 24, 28, 32)
SEEDS = (20260829, 20260830, 20260831)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_scores(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or [int(row["row"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"missing or reordered score rows: {path}")
    labels = {
        key: np.asarray([row[key].lower() == "true" for row in rows], bool)
        for key in ("separable", "entangled", "E2", "E_gt_2", "unknown")
    }
    labels["sample_id"] = np.asarray([int(row["sample_id"]) for row in rows], np.int64)
    if (labels["separable"].sum(), labels["E2"].sum(), labels["E_gt_2"].sum(), labels["unknown"].sum()) != (432, 313, 119, 48):
        raise RuntimeError(f"balanced benchmark label composition changed: {path}")
    if not np.array_equal(labels["entangled"], labels["E2"] | labels["E_gt_2"]):
        raise RuntimeError("certified class partition changed")
    return np.asarray([float(row["probability"]) for row in rows], np.float64), labels


def metrics(probability: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, float]:
    sep = labels["separable"]
    ent = labels["entangled"]
    strict = sep | ent
    return {
        "AUROC": float(roc_auc_score(ent[strict], probability[strict])),
        "E2_recall": float(np.mean(probability[labels["E2"]] >= 0.5)),
        "E_gt_2_recall": float(np.mean(probability[labels["E_gt_2"]] >= 0.5)),
        "Sep_FPR": float(np.mean(probability[sep] >= 0.5)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    full_mean = np.load(OUT / "cache" / "input_mean_full.npy")
    full_std = np.load(OUT / "cache" / "input_std_full.npy")
    if not np.array_equal(full_mean[-5:], np.zeros(5)) or not np.array_equal(full_std[-5:], np.ones(5)):
        raise RuntimeError("geometry masks must remain unstandardized")
    per_seed, ensembles, checkpoint_audit = [], [], []
    common_labels = None
    k16_average = None
    for k in BUDGETS:
        scores = []
        for seed in SEEDS:
            directory = MODELS / f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K{k}_seed{seed}"
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            if report["route_hash"] != audit["Dv1_only_route_hash"]:
                raise RuntimeError(f"non-Dv1 route at K={k} seed={seed}")
            if report["budget"] != k or report["training_config"]["seed"] != seed:
                raise RuntimeError("checkpoint config mismatch")
            selection = json.loads((OUT / "selection_manifests" / f"joint_K{k}.json").read_text(encoding="utf-8"))
            if report["selected_full_token_channels"] != selection["selected_full_token_channels"]:
                raise RuntimeError("channel selection mismatch")
            channels = np.asarray(selection["selected_full_token_channels"], dtype=np.int64)
            if not np.array_equal(np.load(directory / "input_mean.npy"), full_mean[channels]):
                raise RuntimeError("checkpoint input mean differs from Dv1-only preparation")
            if not np.array_equal(np.load(directory / "input_std.npy"), full_std[channels]):
                raise RuntimeError("checkpoint input scale differs from Dv1-only preparation")
            if report["parameter_count"] != 481921 + 128 * k:
                raise RuntimeError("model parameter-count mismatch")
            probability, labels = load_scores(directory / "scores_XY_factorized_balanced_N16.csv")
            if common_labels is None:
                common_labels = labels
            elif any(not np.array_equal(labels[key], common_labels[key]) for key in labels):
                raise RuntimeError("benchmark rows or labels differ between checkpoints")
            result = metrics(probability, labels)
            published = report["metrics"]["XY_factorized_balanced_N16"]
            for key in result:
                if abs(result[key] - published[key]) > 1e-10:
                    raise RuntimeError(f"metric recomputation mismatch for {key}")
            per_seed.append({"K": k, "seed": seed, "input_width": k + 5, **result})
            checkpoint_audit.append({"K": k, "seed": seed,
                "best_checkpoint_sha256": digest(directory / "best.pt"),
                "report_sha256": digest(directory / "report.json"),
                "best_epoch": report["best_epoch"],
                "best_validation_loss": report["best_validation_loss"],
                "Dv1_only_route_hash": report["route_hash"]})
            scores.append(probability)
        average = np.stack(scores).mean(axis=0)
        if k == 16:
            k16_average = average.copy()
        ensembles.append({"K": k, "input_width": k + 5, "seeds": len(SEEDS),
                          **metrics(average, common_labels)})
    canonical = []
    for seed in SEEDS:
        path = CANONICAL / f"RoPETransformer_JointK16_Dv1_only_seed{seed}" / "scores_N16_XY_balanced.csv"
        probability, labels = load_scores(path)
        if any(not np.array_equal(labels[key], common_labels[key]) for key in labels):
            raise RuntimeError("canonical K16 benchmark rows or labels differ")
        canonical.append(probability)
    canonical_average = np.stack(canonical).mean(axis=0)
    canonical_metrics = metrics(canonical_average, common_labels)
    result = {
        "status": "complete", "training_sources": ["D_v1"],
        "frozen_ranking_source": audit["frozen_ranking_source"],
        "frozen_ranking_sha256": audit["frozen_ranking_sha256"],
        "budget_k16_metrics": next(row for row in ensembles if row["K"] == 16),
        "established_canonical_joint_k16_metrics": canonical_metrics,
        "budget_k16_equals_established_baseline": all(
            abs(next(row for row in ensembles if row["K"] == 16)[key] - canonical_metrics[key]) <= 1e-10
            for key in canonical_metrics),
        "budget_k16_max_abs_probability_difference_from_established_baseline":
            float(np.max(np.abs(k16_average - canonical_average))),
        "budget_k16_same_score_array_as_established_baseline":
            bool(np.array_equal(k16_average, canonical_average)),
        "checkpoints": checkpoint_audit,
    }
    write_csv(OUT / "balanced_xy_per_seed_metrics.csv", per_seed)
    write_csv(OUT / "balanced_xy_ensemble_metrics.csv", ensembles)
    (OUT / "budget_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "K16_matches_canonical": result["budget_k16_equals_established_baseline"],
                      "ensemble_rows": len(ensembles)}), flush=True)


if __name__ == "__main__":
    main()

"""Small CPU smoke test of the Dv1-only data-to-training path.

This creates a new *smoke-only* checkpoint. It does not claim to reproduce the
three final training runs or their validation scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from entanglement_data.final_entnet import tokens_from_pauli_features  # noqa: E402
from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    InvariantTokenTransformer, InvariantTransformerConfig, NormalizedInvariantTransformer,
)
from run_joint_k16_training_distribution_ablation import fit_normalization  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(split: np.ndarray, y: np.ndarray, name: str, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        choices = np.flatnonzero((split == name) & (y == label))
        if len(choices) < per_class:
            raise ValueError(f"insufficient {name} label {label} rows")
        rows.extend(rng.choice(choices, per_class, replace=False))
    return np.sort(np.asarray(rows, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "reports/phase3_train_smoke")
    ap.add_argument("--train-per-class", type=int, default=64)
    ap.add_argument("--validation-per-class", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(4)
    np.random.seed(20261116)
    torch.manual_seed(20261116)
    args.output.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.dataset, "r") as h5:
        y = h5["labels/y"][:].astype(np.float32)
        split = h5["metadata/split"].asstr()[:]
        parent = h5["metadata/parent_state_id"].asstr()[:]
        train_ids = choose(split, y, "train", args.train_per_class, 11)
        val_ids = choose(split, y, "validation", args.validation_per_class, 12)
        if set(parent[train_ids]) & set(parent[val_ids]):
            raise ValueError("train-validation parent leakage")
        train_full = tokens_from_pauli_features(h5["features/X"][train_ids], N=16)
        val_full = tokens_from_pauli_features(h5["features/X"][val_ids], N=16)
    channels = json.loads((ROOT / "configs/joint_K16.json").read_text())["selected_full_token_channels"]
    route = (np.zeros(len(train_ids), np.int8), np.arange(len(train_ids)), y[train_ids])
    mean, std, norm_audit = fit_normalization([train_full], route, np.asarray(channels))
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=21)), mean, std
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(np.ascontiguousarray(train_full[:, :, channels]))
    y_train = torch.from_numpy(y[train_ids])
    model.train()
    losses = []
    for block in torch.randperm(len(train_ids)).split(32):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train[block]), y_train[block])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():
        p = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(val_full[:, :, channels])))).numpy()
    checkpoint = args.output / "smoke_only.pt"
    torch.save({"model_state_dict": model.state_dict(), "tag": "smoke_only_not_final"}, checkpoint)
    loaded = NormalizedInvariantTransformer(
        InvariantTokenTransformer(InvariantTransformerConfig(physical_channels=21)), mean, std
    )
    loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"], strict=True)
    loaded.eval()
    with torch.inference_mode():
        reloaded = torch.sigmoid(loaded(torch.from_numpy(np.ascontiguousarray(val_full[:, :, channels])))).numpy()
    if not np.allclose(p, reloaded, atol=1e-7):
        raise RuntimeError("smoke checkpoint reload changed predictions")
    checkpoint.unlink()  # Do not package a non-final model as a release artifact.
    report = {"status": "passed", "scope": "small Dv1-only training smoke, not final-model reproduction",
              "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
              "train_rows": len(train_ids), "validation_rows": len(val_ids), "input_shape": list(train_full.shape),
              "selected_channel_count": len(channels), "train_loss_mean": float(np.mean(losses)),
              "validation_AUROC": float(roc_auc_score(y[val_ids], p)),
              "checkpoint_reload_max_abs_diff": float(np.max(np.abs(p - reloaded))),
              "normalization_valid_anchor_counts": norm_audit["counts_by_selected_channel"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Separate Dv1+R2Support control ensemble used in appendix interventions.

This is a distinct model route from the final Dv1-only EntNet ensemble. Its
weights must never be substituted for those of the main-paper classifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from entanglement_data.ml.invariant_transformer import (
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
)

SEEDS = (20260829, 20260830, 20260831)


def load_control_ensemble(root: Path, *, device: str = "cpu") -> list[torch.nn.Module]:
    """Load the three appendix-control checkpoints with route checks."""
    root = Path(root)
    manifest = json.loads((root / "configs/joint_K16.json").read_text(encoding="utf-8"))
    channels = [int(value) for value in manifest["selected_full_token_channels"]]
    if len(channels) != 21 or channels[-5:] != list(range(117, 122)):
        raise ValueError("unexpected frozen K16 channel schema")
    models = []
    for seed in SEEDS:
        directory = root / "checkpoints/dv1_r2support_control" / f"seed{seed}"
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        audit = report["expanded_route_audit"]
        if audit["training_sources"] != ["D_v1", "R2Support"]:
            raise ValueError(f"seed {seed}: unexpected control training sources")
        if audit["excluded_sources"][:2] != ["D_pair", "D_H"]:
            raise ValueError(f"seed {seed}: auxiliary historical sources not excluded")
        if [int(value) for value in report["selected_full_token_channels"]] != channels:
            raise ValueError(f"seed {seed}: selected channels differ")
        config = InvariantTransformerConfig(**report["model_config"])
        mean = np.load(directory / "input_mean.npy")
        std = np.load(directory / "input_std.npy")
        model = NormalizedInvariantTransformer(InvariantTokenTransformer(config), mean, std)
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("metadata", {}).get("route_hash") not in (None, report["route_hash"]):
            raise ValueError(f"seed {seed}: route hash differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval().to(device)
        models.append(model)
    return models


def predict_control_ensemble(
    selected_tokens: np.ndarray,
    root: Path,
    *,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Return mean sigmoid score for [batch, sites, 21] control inputs."""
    values = np.asarray(selected_tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 21 or not np.isfinite(values).all():
        raise ValueError("expected finite selected tokens with shape [batch, sites, 21]")
    models = load_control_ensemble(root, device=device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            block = torch.from_numpy(np.ascontiguousarray(values[start:start + batch_size])).to(device)
            outputs.append(np.mean([
                torch.sigmoid(model(block)).cpu().numpy() for model in models
            ], axis=0))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

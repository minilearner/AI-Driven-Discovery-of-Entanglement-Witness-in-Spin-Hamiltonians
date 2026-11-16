"""Re-evaluate Appendix feature-block interventions on J1-J2 token data.

The Dv1+R2Support checkpoints used here are distinct from the final Dv1-only
classifier and from the earlier fixed-K16 control used for the thermal grid.
Mean replacement is applied only where the corresponding geometry mask is
valid. Such modified inputs need not describe physical quantum states.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.control_entnet import load_control_ensemble  # noqa: E402


def replaced_tokens(
    values: np.ndarray,
    means: np.ndarray,
    manifest: dict,
    *,
    replace_pair: bool,
) -> np.ndarray:
    """Apply the source intervention's mask-aware training-mean replacement."""
    channels = [int(value) for value in manifest["selected_full_token_channels"]]
    local_channel = {channel: index for index, channel in enumerate(channels)}
    result = np.asarray(values, dtype=np.float32).copy()
    for index, row in enumerate(manifest["selected_channels"]):
        family = row["family"]
        if (family == "pair") != replace_pair:
            continue
        if family == "pair":
            valid = values[:, :, local_channel[116 + int(row["pair_distance"])]] > 0.5
        elif family == "contiguous_three_body":
            valid = values[:, :, local_channel[121]] > 0.5
        elif family == "contiguous_four_body":
            valid = values[:, :, local_channel[119]] > 0.5
        else:
            raise ValueError(f"unexpected selected family {family}")
        result[:, :, index][valid] = float(means[index])
    return result


def _predict(model: torch.nn.Module, values: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            block = torch.from_numpy(np.ascontiguousarray(values[start:start + batch_size], dtype=np.float32))
            outputs.append(torch.sigmoid(model(block)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def intervention_scores(values: np.ndarray, *, batch_size: int = 256) -> dict[str, np.ndarray]:
    manifest = json.loads((ROOT / "configs/joint_K16.json").read_text(encoding="utf-8"))
    models = load_control_ensemble(ROOT)
    predictions = {key: [] for key in ("p_full", "p_no_high", "p_no_pair")}
    for seed, model in zip((20260829, 20260830, 20260831), models):
        means = np.load(ROOT / "checkpoints/dv1_r2support_control" / f"seed{seed}/input_mean.npy")
        modified_high = replaced_tokens(values, means, manifest, replace_pair=False)
        modified_pair = replaced_tokens(values, means, manifest, replace_pair=True)
        predictions["p_full"].append(_predict(model, values, batch_size))
        predictions["p_no_high"].append(_predict(model, modified_high, batch_size))
        predictions["p_no_pair"].append(_predict(model, modified_pair, batch_size))
    return {key: np.mean(np.stack(parts), axis=0) for key, parts in predictions.items()}


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1.0e-7, 1.0 - 1.0e-7)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as h5:
        values = h5["features/Joint_K16_inputs"][:]
        class_masks = {
            "separable": h5["labels/strict_separable"][:].astype(bool),
            "E2": h5["labels/E2"][:].astype(bool),
            "E_gt_2": (h5["labels/E3"][:] | h5["labels/E4"][:]).astype(bool),
            "unknown": h5["labels/unknown"][:].astype(bool),
        }
    membership = np.stack(list(class_masks.values())).sum(axis=0)
    if not np.all(membership == 1):
        raise ValueError("benchmark classes are not an exact disjoint cover")
    scores = intervention_scores(values, batch_size=args.batch_size)
    logits = {key.replace("p_", "z_"): _logit(value) for key, value in scores.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "row", "sample_class", *scores, *logits])
        writer.writeheader()
        for index in range(len(values)):
            name = next(key for key, mask in class_masks.items() if mask[index])
            writer.writerow({
                "model": "dv1_r2support_control", "row": index, "sample_class": name,
                **{key: float(value[index]) for key, value in scores.items()},
                **{key: float(value[index]) for key, value in logits.items()},
            })
    print(f"wrote {len(values)} rows to {args.output}")


if __name__ == "__main__":
    main()

"""Extend the frozen 117-channel consensus ranking to nested K budgets.

This script never reruns selection and never overwrites an existing manifest.
It preserves the five geometry masks and verifies the historical K16/20/24/32
manifests before creating any missing K4/8/12/28 manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selection_payload(k: int, channels: list[int], rows: list[dict]) -> dict:
    return {
        "status": "complete",
        "family": "joint",
        "budget": k,
        "physical_channel_count": k,
        "T4_channel_count": sum(channel >= 77 for channel in channels),
        "mask_channel_count": 5,
        "input_width": k + 5,
        "selected_physical_full_channels": channels,
        "selected_full_token_channels": [*channels, *range(117, 122)],
        "selected_channels": rows,
        "N32_used_for_selection": False,
        "exact500_used_for_selection": False,
        "exact5000_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=ROOT / "analysis/invariant_transformer_raw_t4_v1",
    )
    args = parser.parse_args()
    analysis = args.analysis_dir.resolve()
    manifest_dir = analysis / "selection_manifests"
    consensus = read_json(analysis / "joint_channel_selection_summary.json")
    ranking = consensus["consensus_ranking_best_to_worst_full_channels"]
    if len(ranking) != 117 or len(set(ranking)) != 117:
        raise RuntimeError("expected the frozen 117-channel permutation")
    k32 = read_json(manifest_dir / "joint_K32.json")
    if k32["selected_physical_full_channels"] != ranking[:32]:
        raise RuntimeError("historical K32 manifest differs from frozen ranking")
    ranked_rows = k32["selected_channels"]
    if [row["channel"] for row in ranked_rows] != ranking[:32]:
        raise RuntimeError("K32 channel metadata differs from frozen ranking")
    for k in (16, 20, 24, 32):
        prior = read_json(manifest_dir / f"joint_K{k}.json")
        if prior["selected_physical_full_channels"] != ranking[:k]:
            raise RuntimeError(f"historical K{k} manifest differs from frozen ranking")
    for k in (4, 8, 12, 28):
        path = manifest_dir / f"joint_K{k}.json"
        expected = selection_payload(k, ranking[:k], ranked_rows[:k])
        if path.exists():
            if read_json(path) != expected:
                raise RuntimeError(f"existing K{k} manifest differs; refusing overwrite")
            print(f"verified existing {path}")
        else:
            path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
            print(f"created {path}")


if __name__ == "__main__":
    main()

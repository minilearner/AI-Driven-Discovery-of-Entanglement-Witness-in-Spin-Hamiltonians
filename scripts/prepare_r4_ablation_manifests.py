"""Rebuild the two frozen r=4 ablation manifests from saved channel IDs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.ml.invariant_transformer import local_inv123_raw_t4_channel_manifest  # noqa: E402
from run_raw_t4_invariant_transformer import selection_payload  # noqa: E402


def main() -> None:
    summary = json.loads((ROOT / "paper_reproduction/data/r4_pair_ablation_dv1_only_summary.json").read_text(encoding="utf-8"))
    expected = {
        "Dv1_DropR4_K14": ("joint_K14.json", [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 83, 91, 3, 59]),
        "Dv1_ReplaceR4_T4_K16": ("joint_K16_higher_order.json", [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 83, 91, 3, 59, 85, 112]),
    }
    rows = local_inv123_raw_t4_channel_manifest(4)
    output = ROOT / "configs/r4_ablation"
    output.mkdir(parents=True, exist_ok=True)
    for arm, (name, physical) in expected.items():
        if summary["arms"][arm]["channels"] != [*physical, 117, 118, 119, 120, 121]:
            raise ValueError(f"saved {arm} channels differ")
        payload = selection_payload(
            family="joint", budget=len(physical), physical=physical,
            selected_rows=[rows[index] for index in physical],
        )
        path = output / name
        encoded = json.dumps(payload, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"refusing to overwrite changed manifest {path}")
        path.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()

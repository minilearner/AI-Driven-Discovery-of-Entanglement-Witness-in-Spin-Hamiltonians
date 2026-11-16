"""Lightweight inference example using frozen benchmark tokens."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.final_entnet import predict_final_ensemble  # noqa: E402


def main() -> None:
    fixture = np.load(ROOT / "tests" / "fixtures" / "xy_reference_9rows.npz")
    scores = predict_final_ensemble(fixture["full_tokens"], ROOT)
    for row, score in zip(fixture["rows"], scores):
        print(f"XY row {int(row):3d}: p(entangled) = {float(score):.8f}")


if __name__ == "__main__":
    main()

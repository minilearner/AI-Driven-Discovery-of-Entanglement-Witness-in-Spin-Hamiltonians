from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.xy.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the gated finite N=16 XY PBC benchmark")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "xy_n16_pbc_pilot.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "xy_n16_pbc_v1")
    args = parser.parse_args()
    report = run_benchmark(args.config.resolve(), args.output.resolve())
    print(report)


if __name__ == "__main__":
    main()

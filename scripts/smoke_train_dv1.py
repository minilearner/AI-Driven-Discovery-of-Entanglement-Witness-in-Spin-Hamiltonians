"""Run one epoch through the actual final Dv1-only training entry point.

The default input is a 24-row public XY fixture with genuine physical Pauli
correlations and constructive/NPT labels. This is an integration check, not a
retraining of the released model or a new scientific result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "tests/fixtures/physical_smoke_n16.h5")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/train_smoke_final_route")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"{output} already exists; choose a fresh --output directory")
    source = output / "source"
    dest = source / "data/n16_production_v1/dataset.h5"
    dest.parent.mkdir(parents=True)
    shutil.copyfile(args.dataset, dest)
    cache = output / "cache"
    subprocess.run([sys.executable, str(ROOT / "scripts/prepare_final_training_cache.py"),
                    "--dataset", str(dest), "--cache-dir", str(cache)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/train_final_joint_k16.py"),
                    "--seed", "20260829", "--source-root", str(source),
                    "--cache-dir", str(cache), "--analysis-dir", str(output / "analysis"),
                    "--models-dir", str(output / "models"), "--device", "cpu",
                    "--max-epochs", "1", "--patience", "1", "--train-only"], check=True)
    report_path = output / "models/RoPETransformer_JointK16_Dv1_only_seed20260829/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_train_only":
        raise RuntimeError("final-route smoke training did not complete")
    print(json.dumps({"status": "passed", "scope": "one-epoch final training route smoke",
                      "dataset": str(args.dataset), "training_rows": report["training_set_audit"]["splits"]["train"]["total_rows"],
                      "validation_rows": report["training_set_audit"]["splits"]["validation"]["total_rows"],
                      "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

"""Run the recorded final Dv1-only training route with K16 channels fixed.

This wrapper only selects the final route. See PROVENANCE.md: the exact
checkpoint training-time code commit is unknown and full retraining has not
yet been verified against the released weights.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from run_joint_k16_training_distribution_ablation import main as source_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one final Dv1-only Joint-K16 seed")
    parser.add_argument("--seed", type=int, required=True, choices=(20260829, 20260830, 20260831))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--general-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    command = ["train", "--route", "Dv1_only", "--seed", str(args.seed)]
    for name in ("source_root", "cache_dir", "analysis_dir", "models_dir", "general_cache",
                 "device", "max_epochs", "patience"):
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    if args.train_only:
        command.append("--train-only")
    sys.argv = [sys.argv[0], *command]
    source_main()


if __name__ == "__main__":
    main()

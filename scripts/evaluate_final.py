"""Evaluate frozen EntNet on a canonical benchmark HDF5 file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.final_entnet import (  # noqa: E402
    predict_final_ensemble,
    tokens_from_pauli_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as h5:
        raw = h5["features/X"]
        sep = h5["labels/certified_separable"][:].astype(bool)
        ent = h5["labels/certified_entangled"][:].astype(bool)
        e2 = h5["labels/E2"][:].astype(bool)
        higher = h5["labels/E_gt_2"][:].astype(bool)
        if len(raw) != len(sep) or np.any(sep & ent):
            raise ValueError("benchmark labels are inconsistent")
        token_batches = []
        for start in range(0, len(raw), args.batch_size):
            token_batches.append(tokens_from_pauli_features(raw[start:start + args.batch_size], N=args.N))
    p = predict_final_ensemble(np.concatenate(token_batches), ROOT, batch_size=args.batch_size)
    known = sep | ent
    report = {
        "dataset": str(args.dataset), "rows": int(len(p)),
        "known_rows": int(known.sum()), "unknown_rows": int((~known).sum()),
        "AUROC": float(roc_auc_score(ent[known], p[known])),
        "E2_recall": float(np.mean(p[e2] >= 0.5)) if e2.any() else None,
        "E_gt_2_recall": float(np.mean(p[higher] >= 0.5)) if higher.any() else None,
        "TNR": float(np.mean(p[sep] < 0.5)) if sep.any() else None,
        "threshold": 0.5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

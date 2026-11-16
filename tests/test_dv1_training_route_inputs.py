"""The final Dv1-only route must not require historical auxiliary datasets."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_joint_k16_training_distribution_ablation import ensure_inputs, load_metadata  # noqa: E402


def test_dv1_route_does_not_require_dpair_or_dh(tmp_path: Path) -> None:
    dataset = tmp_path / "data/n16_production_v1/dataset.h5"
    dataset.parent.mkdir(parents=True)
    with h5py.File(dataset, "w") as h5:
        h5["labels/y"] = np.array([0, 1], dtype=np.int8)
        for name, values in {
            "split": ["train", "validation"],
            "parent_state_id": ["parent-a", "parent-b"],
            "certificate_type": ["constructive_separable", "two_site_npt"],
            "generator_type": ["test", "test"],
        }.items():
            h5.create_dataset(f"metadata/{name}", data=np.array(values, dtype="S"))
        h5["metadata/npt_min_eigenvalue"] = np.array([0.0, -0.1])
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "D_v1_Inv123RawT4_Robs4.npy", np.zeros((2, 16, 122), np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    ensure_inputs(tmp_path, cache, manifest, ("D_v1",))
    metadata, audit = load_metadata(tmp_path, tmp_path / "audit", ("D_v1",))
    assert list(metadata) == ["D_v1"]
    assert audit["D_v1"]["parents_crossing_splits"] == 0

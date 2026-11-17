"""Package the six verified paper benchmarks for a public data release.

Maintainer command; source HDF5 files are read only. The resulting archive is
an output artifact and is never committed to the code repository.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
ASSET = "entnet-benchmarks-v1.tar.gz"
SPECS = (
    ("XXZ", "data/adaptive_gibbs_ood_n16_v1/xxz/dataset.h5", "xxz/dataset.h5",
     "N=16 XXZ adaptive Gibbs states", "Cross-Hamiltonian generalization",
     "scripts/generate_adaptive_gibbs_ood.py"),
    ("J1-J2", "data/adaptive_gibbs_ood_n16_v1/j1j2/dataset.h5", "j1j2/dataset.h5",
     "N=16 J1-J2 adaptive Gibbs states", "Cross-Hamiltonian generalization",
     "scripts/generate_adaptive_gibbs_ood.py"),
    ("Random-local-H", "data/adaptive_gibbs_ood_n16_v1/dh_family/dataset.h5",
     "random_local_h/dataset.h5", "N=16 random-local-H adaptive Gibbs states",
     "Cross-Hamiltonian generalization", "scripts/generate_adaptive_gibbs_ood.py"),
    ("N32-XY", "data/xy_n32_pbc_matched_exact500_v1/dataset.h5", "n32_xy/dataset.h5",
     "N=32 matched periodic XY Gibbs states", "System-size generalization",
     "scripts/generate_xy_n32_size_transfer.py"),
    ("N32-synthetic", "data/dpi_n32_general_v1/D_PI_N32.h5",
     "n32_synthetic/dataset.h5", "N=32 independently constructed synthetic states",
     "Pairwise-invisible and higher-order generalization",
     "scripts/generate_dpi_n32_general.py"),
    ("General-Sep", "data/general_separable_ood_n32_v1/dataset.h5",
     "general_sep/dataset.h5", "N=32 constructive separable mixtures",
     "False-positive control", "scripts/generate_general_separable_ood.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(path: Path) -> tuple[int, dict[str, int]]:
    with h5py.File(path, "r") as h5:
        if "labels/strict_separable" not in h5:
            raise ValueError(f"missing strict_separable label: {path}")
        labels = h5["labels"]
        n = len(labels["strict_separable"])
        counts = {}
        for name in ("strict_separable", "E2", "E3", "E4", "unknown"):
            counts[name] = int(labels[name][:].astype(bool).sum()) if name in labels else 0
        if sum(counts.values()) != n:
            raise ValueError(f"labels do not form an exclusive partition: {path}")
        return n, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root containing the six verified source HDF5 files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/benchmark_release")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = {row["benchmark"]: row for row in json.loads(
        (ROOT / "reports/phase3_benchmarks/benchmark_reproduction.json").read_text(encoding="utf-8"))}
    files = []
    for key, source, destination, description, experiment, generator in SPECS:
        path = args.source_root / source
        if not path.is_file():
            raise FileNotFoundError(path)
        n, labels = inspect(path)
        digest = sha256(path)
        report = reference[key]
        if digest != report["dataset_sha256"] or n != report["counts"]["total"]:
            raise ValueError(f"{key}: source differs from verified paper benchmark")
        for cls, target in (("strict_separable", "Sep"), ("E2", "E2"),
                            ("E3", "E3"), ("E4", "E4"), ("unknown", "Unknown")):
            if labels[cls] != report["counts"][target]:
                raise ValueError(f"{key}: {cls} count differs from paper benchmark")
        files.append({"id": key, "path": destination, "sha256": digest,
                      "size_bytes": path.stat().st_size, "samples": n, "labels": labels,
                      "description": description, "paper_experiment": experiment,
                      "generator": generator, "schema": {
                          "XXZ": "adaptive", "J1-J2": "adaptive", "Random-local-H": "adaptive",
                          "N32-XY": "xy32", "N32-synthetic": "synthetic32",
                          "General-Sep": "separable32"}[key],
                      "paper_metrics": report["paper_metrics"], "source": source})
    archive = args.output_dir / ASSET
    if archive.exists():
        raise FileExistsError(f"release asset already exists: {archive}")
    internal = json.dumps({"schema_version": 1, "files": files}, indent=2).encode("utf-8") + b"\n"
    with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            info = tarfile.TarInfo("benchmark_files.json")
            info.size = len(internal)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(internal))
            for item in files:
                path = args.source_root / item["source"]
                info = tarfile.TarInfo(item["path"])
                info.size = item["size_bytes"]
                info.mode = 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    manifest = {"schema_version": 1,
                "archive": {"asset_name": ASSET, "url": None,
                            "sha256": sha256(archive), "size_bytes": archive.stat().st_size},
                "files": files}
    output = args.output_dir / "benchmark_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(files)} verified benchmarks ({sum(x['samples'] for x in files)} rows)")
    print(f"Archive: {archive} ({archive.stat().st_size:,} bytes)")
    print(f"SHA-256: {manifest['archive']['sha256']}")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()

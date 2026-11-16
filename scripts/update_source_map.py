"""Refresh release-file SHA-256 values without requiring the private archive."""
from __future__ import annotations

import csv
import fnmatch
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "SOURCE_MAP.csv"
COPIED = {
    "scripts/generate_adaptive_gibbs_ood.py": "remote_work/run_adaptive_gibbs_ood_v1.py",
    "src/entanglement_data/thermal_purification.py":
        "remote_results/adaptive_gibbs_ood_n16_v1/src/entanglement_data/thermal_purification.py",
}
FIELDS = ("release_path", "release_sha256", "origin_type",
          "origin_path_from_private_root", "origin_sha256", "byte_identical")
GENERATED = (
    "paper_reproduction/figures/*.png",
    "paper_reproduction/figures/*.svg",
    "paper_reproduction/figures/*.tiff",
    "paper_reproduction/figures/*.alignment-layout.json",
    "reports/phase3_thermal/*_plot.npz",
    "reports/phase3_thermal/new_boundaries_plot.csv",
    "reports/phase3_thermal/new_fit_plot.json",
    "reports/phase3_thermal/fig3cd_detail.*",
    "reports/phase3_thermal/fig3cd_detail_alignment.json",
    "reports/phase3_thermal/xy_thermal_predictions_combined.alignment-*.json",
    "reports/phase3_thermal/xy_thermal_predictions_combined.alignment-*.svg",
    "reports/phase3_thermal/xy_thermal_predictions_combined.manifest.json",
    "reports/phase3_thermal/xy_thermal_predictions_combined.png",
    "reports/phase3_thermal/xy_thermal_predictions_combined.svg",
    "reports/phase3_thermal/xy_thermal_predictions_combined.tiff",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with MAP.open(newline="", encoding="utf-8") as stream:
        previous = {row["release_path"]: row for row in csv.DictReader(stream)}
    rows = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != MAP
                       and not any(part in {".git", ".venv", "__pycache__", ".pytest_cache"}
                                   for part in p.relative_to(ROOT).parts)
                       and p.suffix not in {".pyc", ".pyo"}):
        relative = path.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in GENERATED):
            continue
        digest = sha256(path)
        old = previous.get(relative)
        origin = old["origin_path_from_private_root"] if old else COPIED.get(relative, "")
        source_digest = old["origin_sha256"] if old else ""
        origin_type = (old["origin_type"] if old else
                       "private_archive" if origin else
                       "phase3_result" if relative.startswith("reports/phase3_") else
                       "release_integration")
        rows.append({"release_path": relative, "release_sha256": digest,
                     "origin_type": origin_type,
                     "origin_path_from_private_root": origin,
                     "origin_sha256": source_digest,
                     "byte_identical": str(bool(origin and digest == source_digest)).lower()})
    with MAP.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"source map refreshed: {len(rows)} mapped files")


if __name__ == "__main__":
    main()

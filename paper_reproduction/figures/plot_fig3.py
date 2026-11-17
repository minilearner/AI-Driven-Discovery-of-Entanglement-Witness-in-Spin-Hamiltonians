"""Regenerate the author-adopted Fig. 3 from final compact numerical data.

Panels (a,b) use Dv1-only trajectories and certificates; panels (c,d) use the
final Dv1+R2Support ensemble. No model retraining occurs in this script.
"""
from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import thermal_layout as template

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/phase3_thermal"
OUTPUT = ROOT / "outputs/figures/thermal"


def validate_dense_against_base() -> None:
    """Check direct predictions at exactly shared physical grid coordinates."""
    with np.load(SOURCE / "new_grid.npz") as base:
        base_h, base_t, base_p = base["h"], base["temperature"], base["p"]
    comparisons = {}
    for panel in ("lower", "upper"):
        with np.load(SOURCE / f"new_dense_{panel}.npz") as dense:
            h, temperature, probability = dense["h"], dense["temperature"], dense["p"]
        h_pairs = [(i, int(np.argmin(abs(base_h - value)))) for i, value in enumerate(h)]
        t_pairs = [(i, int(np.argmin(abs(base_t - value)))) for i, value in enumerate(temperature)]
        h_pairs = [(i, j) for i, j in h_pairs if abs(h[i] - base_h[j]) < 1e-12]
        t_pairs = [(i, j) for i, j in t_pairs if abs(temperature[i] - base_t[j]) < 1e-12]
        if not h_pairs or not t_pairs:
            raise ValueError(f"no common sampled points for {panel}")
        differences = abs(probability[np.ix_([i for i, _ in h_pairs], [i for i, _ in t_pairs])]
                          - base_p[np.ix_([j for _, j in h_pairs], [j for _, j in t_pairs])])
        maximum = float(differences.max())
        if maximum > 5e-5:
            raise ValueError(f"dense {panel} disagrees with base grid: {maximum}")
        comparisons[panel] = {"common_points": int(differences.size),
                              "max_abs_p_difference": maximum}
    (OUTPUT / "dense_grid_validation.json").write_text(json.dumps(comparisons, indent=2) + "\n")


def compatibility_grid(source: Path, destination: Path) -> None:
    with np.load(source) as data:
        h = np.asarray(data["h"], dtype=float)
        temperature = np.asarray(data["temperature"], dtype=float)
        probability = np.asarray(data["p"], dtype=float)
        seed = np.asarray(data["p_seed"], dtype=float)
    if probability.shape != (len(h), len(temperature)) or seed.shape != (*probability.shape, 3):
        raise ValueError(f"invalid grid shape: {source}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError(f"invalid probability grid: {source}")
    bounded = np.clip(probability, 1e-12, 1 - 1e-12)
    disagreement = np.any(seed >= 0.5, axis=2) & np.any(seed < 0.5, axis=2)
    np.savez_compressed(destination, h=h, temperature=temperature,
                        ensemble_logit=np.log(bounded / (1 - bounded)),
                        seed_sign_disagreement=disagreement)


def prepare_sources() -> None:
    if not (SOURCE / "dense_grid_report.json").is_file():
        raise FileNotFoundError(f"missing dense grid report in {SOURCE}")
    for name, destination in (("new_grid.npz", "new_grid_plot.npz"),
                              ("new_dense_lower.npz", "new_dense_lower_plot.npz"),
                              ("new_dense_upper.npz", "new_dense_upper_plot.npz")):
        compatibility_grid(SOURCE / name, OUTPUT / destination)
    with (SOURCE / "new_boundaries.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (OUTPUT / "new_boundaries_plot.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("h", "temperature", "log10_temperature",
                                                  "left_temperature", "right_temperature"))
        writer.writeheader()
        for row in rows:
            temperature = float(row["temperature"])
            half_width = float(row["bracket_log10_width"]) / 2
            writer.writerow({"h": row["h"], "temperature": temperature,
                             "log10_temperature": row["log10_temperature"],
                             "left_temperature": temperature * 10 ** -half_width,
                             "right_temperature": temperature * 10 ** half_width})
    fit = json.loads((SOURCE / "thermal_report.json").read_text())["low_boundary_fit"]
    compatible_fit = {"selection": {"h_range": [fit["h_min"], fit["h_max"]]},
                      "models": {"activated": {"C": fit["C_star"], "metrics": {
        "rmse_log10_temperature": fit["RMSE_log10T"],
        "r2_log10_temperature": fit["R2_log10T"]}}}}
    (OUTPUT / "new_fit_plot.json").write_text(json.dumps(compatible_fit, indent=2) + "\n")


def set_template_sources() -> None:
    template.OUTPUT = OUTPUT
    template.GRID_NPZ = OUTPUT / "new_grid_plot.npz"
    template.DENSE_LOWER_GRID_NPZ = OUTPUT / "new_dense_lower_plot.npz"
    template.DENSE_UPPER_GRID_NPZ = OUTPUT / "new_dense_upper_plot.npz"
    template.BOUNDARY_CSV = OUTPUT / "new_boundaries_plot.csv"
    template.FIT_REPORT = OUTPUT / "new_fit_plot.json"


def separate_cd() -> None:
    template.configure_style()
    figure = plt.figure(figsize=(6.844465, 3.20))
    outer = figure.add_gridspec(1, 2, width_ratios=[1.20, 1.0], left=0.080,
                                right=0.985, bottom=0.18, top=0.94, wspace=0.31)
    map_container = figure.add_subplot(outer[0, 0])
    map_container.set_axis_off()
    map_upper = map_container.inset_axes([0.0, 0.61, 0.88, 0.39])
    map_lower = map_container.inset_axes([0.0, 0.0, 0.88, 0.54])
    colorbar_axis = map_container.inset_axes([0.925, 0.0, 0.035, 1.0])
    template.panel_label(map_container, "c")
    fit_axis = figure.add_subplot(outer[0, 1])
    template.draw_reentry_panels(figure, (map_upper, map_lower), colorbar_axis, fit_axis)
    template.require_matplotlib_panel_alignment(
        figure, axes=[map_container, fit_axis],
        panel_ids={map_container: "c", fit_axis: "d"},
        exclude_axes=[map_upper, map_lower, colorbar_axis],
        json_out=str(OUTPUT / "fig3cd_detail_alignment.json"),
        require_panel_labels=True, strict=True, row_groups=[["c", "d"]],
    )
    figure.savefig(OUTPUT / "fig3cd_detail.pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(OUTPUT / "fig3cd_detail.png", dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def full_figure() -> None:
    template.make_figure()
    path = OUTPUT / "xy_thermal_predictions_combined.manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update({
        "status": "author_adopted_final_figure",
        "model_by_panel": {"a": "Baseline EntNet", "b": "model-independent certificates",
                           "c": "Distance-two-augmented EntNet", "d": "Distance-two-augmented EntNet"},
        "visible_lower_and_upper_windows_recomputed": True,
        "no_prediction_interpolation": True,
        "dense_grid_report": str((SOURCE / "dense_grid_report.json").relative_to(ROOT)).replace("\\", "/"),
        "fit_source": str((SOURCE / "thermal_report.json").relative_to(ROOT)).replace("\\", "/"),
        "boundary_errorbar_convention": "symmetric half of refined log10 bracket width",
    })
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    global SOURCE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace existing figure outputs")
    args = parser.parse_args()
    SOURCE, OUTPUT = args.source_dir.resolve(), args.output.resolve()
    if not args.force and (OUTPUT / "xy_thermal_predictions_combined.pdf").exists():
        parser.error(f"figure output exists in {OUTPUT}; choose --output elsewhere or --force")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    validate_dense_against_base()
    prepare_sources()
    set_template_sources()
    separate_cd()
    full_figure()


if __name__ == "__main__":
    main()

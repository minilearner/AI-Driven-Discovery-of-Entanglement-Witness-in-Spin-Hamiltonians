from __future__ import annotations

"""Shared four-panel XY thermal layout for the final Fig. 3 generator.

Run ``plot_fig3.py`` to prepare and plot the new Dv1+R2Support grids. This
module holds the paper's plotting geometry and is not an independent experiment.
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lambertw


def discover_root() -> Path:
    resolved = Path(__file__).resolve()
    for ancestor in (resolved.parent, *resolved.parents):
        # The generator may live either in the code repository or in the
        # adjacent report/entanglement_witness_prl_draft/figures directory.
        for candidate in (ancestor, ancestor / "纠缠Witness" / "code"):
            if (candidate / "reports/phase3_thermal").is_dir():
                return candidate
    raise RuntimeError("could not locate repository root")


ROOT = discover_root()
OUTPUT = Path(__file__).resolve().parent
TEMPERATURE_SOURCE = (
    ROOT / "reports/xy_temperature_sweep_dv1_v1/xy_temperature_sweep_source_data.csv"
)
GRID_NPZ = ROOT / "reports/phase3_thermal/new_grid_plot.npz"
DENSE_LOWER_GRID_NPZ = ROOT / "reports/phase3_thermal/new_dense_lower_plot.npz"
DENSE_UPPER_GRID_NPZ = ROOT / "reports/phase3_thermal/new_dense_upper_plot.npz"
BOUNDARY_CSV = ROOT / "reports/phase3_thermal/new_boundaries_plot.csv"
FIT_REPORT = ROOT / "reports/phase3_thermal/new_fit_plot.json"
ALIGNMENT_HELPER = Path(__file__).resolve().parent
sys.path.insert(0, str(ALIGNMENT_HELPER))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


TEMPERATURE_POINTS = (
    ("ising_ordered", r"$\gamma=1,\ h=0.5$"),
    ("ising_critical", r"$\gamma=1,\ h=1$"),
    ("ising_field", r"$\gamma=1,\ h=1.4$"),
    ("anisotropic_ordered", r"$\gamma=0.5,\ h=0.5$"),
    ("xx_gapless", r"$\gamma=0,\ h=0.5$"),
    ("xx_saturated", r"$\gamma=0,\ h=1.2$"),
)
TEMPERATURE_ROW_LABELS = tuple(label for _, label in TEMPERATURE_POINTS)
TEMPERATURE_COLORS = ("#0057B8", "#C61F3A", "#228B22", "#7B2CBF", "#B57A00", "#202020")
TEMPERATURE_MARKERS = ("o", "s", "^", "D", "v", "P")
CERTIFICATE_COLORS = {
    "E2": "#B2182B",
    "E3": "#EF8A62",
    "E4": "#FDB863",
    "unknown": "#4D8BC9",
    "infinite_T_separable": "#4D4D4D",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 7.8,
            "axes.labelsize": 8.3,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_label(axis: mpl.axes.Axes, label: str, *, x_offset: float = -17.0) -> None:
    axis.annotate(
        label,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_offset, 4.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
    )


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def load_temperature() -> dict[str, list[dict[str, object]]]:
    numeric = {
        "temperature",
        "ensemble_logit",
        "ensemble_probability",
        "pair_npt_margin_all",
        "triple_margin_all",
        "four_margin_all",
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with TEMPERATURE_SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            for key in numeric:
                parsed[key] = float(row[key])
            grouped[row["point_id"]].append(parsed)
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item["temperature"]))
    return grouped


def read_boundaries(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError("boundary table is empty")
    return rows


def activated_temperature(h_values: np.ndarray, constant: float) -> np.ndarray:
    distance = np.asarray(h_values, dtype=float) - 1.0
    if np.any(distance <= 0.0) or constant <= 0.0:
        raise ValueError("activated boundary requires h>1 and C>0")
    argument = 4.0 * distance / (constant * constant)
    result = 4.0 * distance / np.real(lambertw(argument))
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("activated boundary evaluation failed")
    return result


def draw_temperature_panels(
    score_axis: mpl.axes.Axes,
    hierarchy_axis: mpl.axes.Axes,
    grouped: dict[str, list[dict[str, object]]],
) -> tuple[list[mpl.lines.Line2D], list[mpl.lines.Line2D]]:
    infinity_x = 3000.0
    for index, (point_id, label) in enumerate(TEMPERATURE_POINTS):
        rows = grouped[point_id]
        finite = [row for row in rows if math.isfinite(float(row["temperature"]))]
        temperatures = np.asarray([float(row["temperature"]) for row in finite])
        probabilities = np.asarray([float(row["ensemble_probability"]) for row in finite])
        if np.any(temperatures <= 0.0):
            raise ValueError(f"logarithmic temperature axis requires T>0 for {point_id}")
        color = TEMPERATURE_COLORS[index]
        score_axis.plot(
            temperatures,
            probabilities,
            color=color,
            marker=TEMPERATURE_MARKERS[index],
            markersize=2.0,
            label=label,
        )
        infinity = next(row for row in rows if math.isinf(float(row["temperature"])))
        score_axis.scatter(
            infinity_x,
            float(infinity["ensemble_probability"]),
            color=color,
            marker="D",
            s=13,
            zorder=4,
        )
        for left, right in zip(finite[:-1], finite[1:]):
            hierarchy_axis.plot(
                [float(left["temperature"]), float(right["temperature"])],
                [index, index],
                color=CERTIFICATE_COLORS[str(left["class_name"])],
                linewidth=4.4,
                solid_capstyle="butt",
            )
        hierarchy_axis.scatter(
            temperatures,
            np.full_like(temperatures, index),
            color=[CERTIFICATE_COLORS[str(row["class_name"])] for row in finite],
            marker="s",
            s=6,
            edgecolors="none",
            zorder=3,
        )
        hierarchy_axis.scatter(
            infinity_x,
            index,
            color=CERTIFICATE_COLORS["infinite_T_separable"],
            marker="D",
            s=14,
            zorder=4,
        )

    ticks = [0.02, 0.1, 1, 10, 100, 1000, 3000]
    tick_labels = ["0.02", "0.1", "1", "10", "100", "1000", r"$\infty$"]
    for axis in (score_axis, hierarchy_axis):
        axis.set_xscale("log")
        axis.set_xlim(0.018, 4200.0)
        axis.set_xticks(ticks, tick_labels)
        axis.set_xlabel(r"Temperature $T/J$")
    score_axis.axhline(0.5, color="black", linewidth=0.7, linestyle="--")
    score_axis.set_ylim(-0.025, 1.025)
    score_axis.set_ylabel(r"Predicted probability $p=\sigma(z)$")
    score_axis.grid(axis="y", color="#DDDDDD", linewidth=0.45)
    hierarchy_axis.set_yticks(range(len(TEMPERATURE_POINTS)), TEMPERATURE_ROW_LABELS)
    hierarchy_axis.invert_yaxis()
    hierarchy_axis.grid(axis="x", color="#E9E9E9", linewidth=0.45)
    panel_label(score_axis, "a")
    panel_label(hierarchy_axis, "b")

    trajectory_handles = [
        mpl.lines.Line2D([], [], color=TEMPERATURE_COLORS[index],
                         marker=TEMPERATURE_MARKERS[index], markersize=3.0, label=label)
        for index, (_, label) in enumerate(TEMPERATURE_POINTS)
    ]
    certificate_handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="s" if name != "infinite_T_separable" else "D",
            linestyle="",
            color=CERTIFICATE_COLORS[name],
            label=(
                "Infinite-$T$ separable"
                if name == "infinite_T_separable"
                else ("Unknown" if name == "unknown" else rf"$\mathrm{{E}}_{{{name[1:]}}}$")
            ),
        )
        for name in ("E2", "E3", "E4", "unknown", "infinite_T_separable")
    ]
    return trajectory_handles, certificate_handles


def draw_reentry_panels(
    figure: plt.Figure,
    map_axes: tuple[mpl.axes.Axes, mpl.axes.Axes],
    colorbar_axis: mpl.axes.Axes,
    fit_axis: mpl.axes.Axes,
) -> dict[str, float]:
    grid = np.load(GRID_NPZ)
    h_values = np.asarray(grid["h"], dtype=float)
    temperatures = np.asarray(grid["temperature"], dtype=float)
    logits = np.asarray(grid["ensemble_logit"], dtype=float)
    probabilities = stable_sigmoid(logits)
    disagreement = np.asarray(grid["seed_sign_disagreement"], dtype=bool)
    if logits.shape != (h_values.size, temperatures.size):
        raise ValueError("logit grid shape does not match h and temperature axes")
    if np.any(temperatures <= 0.0):
        raise ValueError("temperatures must be positive")
    dense_grid = np.load(DENSE_LOWER_GRID_NPZ)
    dense_h = np.asarray(dense_grid["h"], dtype=float)
    dense_temperature = np.asarray(dense_grid["temperature"], dtype=float)
    dense_logits = np.asarray(dense_grid["ensemble_logit"], dtype=float)
    dense_disagreement = np.asarray(dense_grid["seed_sign_disagreement"], dtype=bool)
    if dense_logits.shape != (dense_h.size, dense_temperature.size):
        raise ValueError("dense lower grid shape does not match its coordinates")
    if dense_h.size < 100 or dense_temperature.size < 75:
        raise ValueError("dense lower grid is not sufficiently refined")
    if not (np.isclose(dense_h[0], 1.0) and np.isclose(dense_h[-1], 1.5)
            and np.isclose(dense_temperature[0], 0.01)
            and np.isclose(dense_temperature[-1], 10**-0.5)):
        raise ValueError("dense lower grid does not cover the specified plot window")
    dense_probability = stable_sigmoid(dense_logits)
    upper_grid = np.load(DENSE_UPPER_GRID_NPZ)
    upper_h = np.asarray(upper_grid["h"], dtype=float)
    upper_temperature = np.asarray(upper_grid["temperature"], dtype=float)
    upper_logits = np.asarray(upper_grid["ensemble_logit"], dtype=float)
    upper_disagreement = np.asarray(upper_grid["seed_sign_disagreement"], dtype=bool)
    if upper_logits.shape != (upper_h.size, upper_temperature.size):
        raise ValueError("dense upper grid shape does not match its coordinates")
    if upper_h.size < 90 or upper_temperature.size < 50:
        raise ValueError("dense upper grid is not sufficiently refined")
    if not (np.isclose(upper_h[0], 0.6) and np.isclose(upper_h[-1], 1.5)
            and np.isclose(upper_temperature[0], 10.0)
            and np.isclose(upper_temperature[-1], 100.0)):
        raise ValueError("dense upper grid does not cover the specified plot window")
    upper_probability = stable_sigmoid(upper_logits)

    boundaries = read_boundaries(BOUNDARY_CSV)
    fit_report = json.loads(FIT_REPORT.read_text(encoding="utf-8"))
    activated = fit_report["models"]["activated"]
    fit_h_min, fit_h_max = fit_report["selection"]["h_range"]
    lower = sorted(
        (row for row in boundaries
         if fit_h_min - 1e-9 <= row["h"] <= fit_h_max + 1e-9
         and row["temperature"] < 1.0),
        key=lambda row: row["h"],
    )
    constant = float(activated["C"])
    rmse_log10 = float(activated["metrics"]["rmse_log10_temperature"])
    r2_log10 = float(activated["metrics"]["r2_log10_temperature"])

    log_temperature = np.log10(temperatures)
    dense_log_temperature = np.log10(dense_temperature)
    upper_log_temperature = np.log10(upper_temperature)
    map_upper, map_lower = map_axes
    for axis in map_axes:
        mesh = axis.pcolormesh(
            h_values, log_temperature, probabilities.T,
            shading="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0,
            rasterized=True,
        )
        if axis is map_lower:
            # Only this saturated low-temperature sector is recomputed.  The
            # original grid remains untouched elsewhere; no interpolation is used.
            axis.pcolormesh(dense_h, dense_log_temperature, dense_probability.T,
                            shading="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0,
                            rasterized=True, zorder=2)
            axis.contour(dense_h, dense_log_temperature, dense_probability.T,
                         levels=[0.5], colors="black", linewidths=0.95,
                         zorder=4)
            disagreement_h = dense_h
            disagreement_t = dense_log_temperature
            disagreement_values = dense_disagreement
        else:
            axis.pcolormesh(upper_h, upper_log_temperature, upper_probability.T,
                            shading="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0,
                            rasterized=True, zorder=2)
            axis.contour(upper_h, upper_log_temperature, upper_probability.T,
                         levels=[0.5], colors="black", linewidths=0.95,
                         zorder=4)
            disagreement_h = upper_h
            disagreement_t = upper_log_temperature
            disagreement_values = upper_disagreement
        if np.any(disagreement_values):
            axis.contourf(disagreement_h, disagreement_t,
                          disagreement_values.T.astype(float),
                          levels=[0.5, 1.5], colors=["#737373"], alpha=0.22,
                          zorder=3)
            axis.contour(disagreement_h, disagreement_t,
                         disagreement_values.T.astype(float),
                         levels=[0.5], colors=["#4D4D4D"], linewidths=0.65,
                         linestyles="--", zorder=4)
        axis.scatter([row["h"] for row in boundaries],
                     [row["log10_temperature"] for row in boundaries],
                     s=5.5, color="black", linewidths=0, zorder=5)
        axis.axvline(1.0, color="white", lw=1.4, ls="--", zorder=6)
        axis.axvline(1.2, color="#FFD33D", lw=1.4, ls="--", zorder=6)
        axis.set_xlim(0.6, float(h_values.max()))
    map_upper.set_ylim(1.0, float(log_temperature.max()))
    map_lower.set_ylim(float(log_temperature.min()), -0.5)
    map_upper.set_xticklabels([])
    map_upper.tick_params(axis="x", bottom=False)
    map_upper.spines["bottom"].set_visible(False)
    map_lower.spines["top"].set_visible(False)
    map_upper.set_yticks([1.0, 1.5, 2.0])
    map_lower.set_yticks([-2.0, -1.5, -1.0, -0.5])
    map_lower.set_xlabel(r"$h/J$")
    map_lower.set_ylabel(r"$\log_{10}(T/J)$", labelpad=3)
    map_lower.yaxis.set_label_coords(-0.11, 0.94)
    # Small diagonal marks explicitly disclose the omitted -0.5 < logT < 1 region.
    for axis, y in ((map_upper, 0.0), (map_lower, 1.0)):
        axis.plot((-0.012, 0.012), (y - 0.02, y + 0.02),
                  transform=axis.transAxes, color="#333333", lw=0.75, clip_on=False)
        axis.plot((0.988, 1.012), (y - 0.02, y + 0.02),
                  transform=axis.transAxes, color="#333333", lw=0.75, clip_on=False)
    map_lower.text(
        1.47,
        -1.96,
        "Saturated state\n" + r"($T=0$)",
        color="white",
        ha="right",
        va="bottom",
        fontsize=7.4,
        fontweight="bold",
        linespacing=1.18,
    )
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.ax.set_title(r"$p$", fontsize=8.0, pad=2.0)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=7.2)
    colorbar.outline.set_linewidth(0.65)

    lower_h = np.asarray([row["h"] for row in lower], dtype=float)
    lower_t = np.asarray([row["temperature"] for row in lower], dtype=float)
    lower_left = np.asarray([row["left_temperature"] for row in lower], dtype=float)
    lower_right = np.asarray([row["right_temperature"] for row in lower], dtype=float)
    lower_log_t = np.log10(lower_t)
    y_error = np.vstack(
        [
            lower_log_t - np.log10(lower_left),
            np.log10(lower_right) - lower_log_t,
        ]
    )
    dense_h = np.linspace(float(lower_h.min()), float(lower_h.max()), 600)
    dense_t = activated_temperature(dense_h, constant)
    fit_axis.errorbar(
        lower_h,
        lower_log_t,
        yerr=y_error,
        fmt="o",
        ms=2.8,
        color="#111111",
        ecolor="#777777",
        elinewidth=0.65,
        capsize=1.3,
        zorder=4,
        label="Refined boundary",
    )
    fit_axis.plot(
        dense_h,
        np.log10(dense_t),
        color="#D55E00",
        lw=1.5,
        ls="--",
        label="Activated fit",
    )
    fit_axis.set_xlim(float(lower_h.min()) - 0.008, float(lower_h.max()) + 0.008)
    fit_axis.set_ylim(-1.79, -0.70)
    fit_axis.set_xlabel(r"$h/J$")
    fit_axis.set_ylabel(r"$\log_{10}(T_*/J)$")
    fit_axis.text(
        0.98, 0.045,
        r"$\sqrt{T_*}\exp[-2(h-1)/T_*]=C_*$",
        transform=fit_axis.transAxes, ha="right", va="bottom", fontsize=9.2,
    )
    fit_axis.legend(loc="upper left", frameon=False, handlelength=1.8)
    panel_label(fit_axis, "d")

    return {
        "C": constant,
        "r2_log10_temperature": r2_log10,
        "rmse_log10_temperature": rmse_log10,
        "selected_points": len(lower),
    }


def make_figure() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temperature = load_temperature()

    # Narrow the artwork by 5% so a 0.95-textwidth LaTeX placement keeps
    # essentially the original physical height and typography.
    figure = plt.figure(figsize=(6.844465, 5.82))
    outer = figure.add_gridspec(
        2,
        2,
        width_ratios=[1.20, 1.0],
        height_ratios=[0.83, 1.10],
        left=0.080,
        right=0.985,
        bottom=0.080,
        top=0.805,
        wspace=0.31,
        hspace=0.48,
    )
    score_axis = figure.add_subplot(outer[0, 0])
    hierarchy_axis = figure.add_subplot(outer[0, 1])
    map_container = figure.add_subplot(outer[1, 0])
    map_container.set_axis_off()
    map_upper = map_container.inset_axes([0.0, 0.61, 0.88, 0.39])
    map_lower = map_container.inset_axes([0.0, 0.0, 0.88, 0.54])
    colorbar_axis = map_container.inset_axes([0.925, 0.0, 0.035, 1.0])
    panel_label(map_container, "c")
    fit_axis = figure.add_subplot(outer[1, 1])

    trajectory_handles, certificate_handles = draw_temperature_panels(
        score_axis, hierarchy_axis, temperature
    )
    fit_metrics = draw_reentry_panels(figure, (map_upper, map_lower), colorbar_axis, fit_axis)

    score_axis.legend(handles=trajectory_handles, loc="lower center",
                      bbox_to_anchor=(0.50, 1.03), ncol=2, frameon=False,
                      handlelength=1.1, columnspacing=0.7, labelspacing=0.25,
                      fontsize=7.3)
    hierarchy_axis.legend(handles=certificate_handles, loc="lower center",
                          bbox_to_anchor=(0.50, 1.03), ncol=3, frameon=False,
                          handlelength=0.9, columnspacing=0.55, labelspacing=0.25,
                          fontsize=7.3)

    base = OUTPUT / "xy_thermal_predictions_combined"
    main_axes = [score_axis, hierarchy_axis, map_container, fit_axis]
    require_matplotlib_panel_alignment(
        figure,
        axes=main_axes,
        panel_ids={axis: panel for axis, panel in zip(main_axes, "abcd")},
        exclude_axes=[map_upper, map_lower, colorbar_axis],
        json_out=str(base.with_suffix(".alignment-layout.json")),
        overlay_svg=str(base.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        row_groups=[["a", "b"], ["c", "d"]],
    )
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        base.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    manifest = {
        "status": "complete",
        "core_conclusion": (
            "The frozen Joint-K16 response follows the decay of thermal correlations at high "
            "temperature and resolves a gap-controlled reentrant positive region above XX saturation."
        ),
        "backend": "python",
        "sources": [
            str(TEMPERATURE_SOURCE.relative_to(ROOT)).replace("\\", "/"),
            str(GRID_NPZ.relative_to(ROOT)).replace("\\", "/"),
            str(DENSE_LOWER_GRID_NPZ.relative_to(ROOT)).replace("\\", "/"),
            str(DENSE_UPPER_GRID_NPZ.relative_to(ROOT)).replace("\\", "/"),
            str(BOUNDARY_CSV.relative_to(ROOT)).replace("\\", "/"),
            str(FIT_REPORT.relative_to(ROOT)).replace("\\", "/"),
        ],
        "panel_roles": {
            "a": "one-dimensional thermal score trajectories",
            "b": "independent PPT/CCNR certificate hierarchy",
            "c": "two-dimensional XX thermal decision landscape",
            "d": "activated explanation of the lower entry boundary",
        },
        "fit": fit_metrics,
        "visible_lower_and_upper_windows_recomputed": True,
        "no_prediction_interpolation": True,
        "interpretation_boundary": (
            "The p=0.5 contour is a finite-size model decision boundary, not a rigorous "
            "thermodynamic entanglement transition."
        ),
    }
    base.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit("Run plot_fig3.py to prepare the final numerical sources first.")

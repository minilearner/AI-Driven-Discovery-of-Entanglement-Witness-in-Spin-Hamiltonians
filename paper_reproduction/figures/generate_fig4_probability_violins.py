from __future__ import annotations

"""Render Fig. 4 as probability violins for two controlled training routes.

The top row uses the frozen Dv1-only per-state scores already underlying the
manuscript.  The bottom row uses newly computed scores from checkpoints trained
only on Dv1 plus R2Support.  Full-input and train-mean-replaced predictions are
shown side by side without changing any sample label or model output.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from audit_panel_alignment import require_matplotlib_panel_alignment


CLASS_ORDER = ("separable", "E2", "E_gt_2", "unknown")
CLASS_STYLE = {
    "separable": {"label": "Separable", "short": "Sep", "color": "#238B78", "n": 30},
    "E2": {"label": r"$\mathrm{E}_{2}$", "short": r"$\mathrm{E}_{2}$", "color": "#B2182B", "n": 155},
    "E_gt_2": {"label": r"$\mathrm{E}_{>2}$", "short": r"$\mathrm{E}_{>2}$", "color": "#EF8A62", "n": 110},
    "unknown": {"label": "Unknown", "short": "Unk", "color": "#2166AC", "n": 168},
}
EXPECTED_COUNTS = {name: int(style["n"]) for name, style in CLASS_STYLE.items()}
MODEL_ORDER = ("dv1_only", "dv1_r2support_control")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    probability = np.empty_like(z)
    positive = z >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    probability[~positive] = exp_z / (1.0 + exp_z)
    return probability


def normalize_class(value: str) -> str:
    mapping = {
        "separable": "separable",
        "E2": "E2",
        "E3": "E_gt_2",
        "E4": "E_gt_2",
        "E_gt_2": "E_gt_2",
        "E>2": "E_gt_2",
        "unknown": "unknown",
    }
    if value not in mapping:
        raise ValueError(f"unexpected class label {value!r}")
    return mapping[value]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty source table: {path}")
    return rows


def load_sources(
    dv1_path: Path, control_path: Path
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, Any]]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = {
        model: {
            class_name: {key: [] for key in ("p_full", "p_no_high", "p_no_pair")}
            for class_name in CLASS_ORDER
        }
        for model in MODEL_ORDER
    }
    derived: list[dict[str, Any]] = []

    for row in read_csv(dv1_path):
        if row.get("model") != "dv1_only":
            continue
        class_name = normalize_class(row["sample_class"])
        probabilities = {
            "p_full": float(stable_sigmoid(np.asarray([float(row["z_full"])]))[0]),
            "p_no_high": float(stable_sigmoid(np.asarray([float(row["z_no_high"])]))[0]),
            "p_no_pair": float(stable_sigmoid(np.asarray([float(row["z_no_pair"])]))[0]),
        }
        for key, probability in probabilities.items():
            values["dv1_only"][class_name][key].append(probability)
        derived.append(
            {
                "model": "dv1_only",
                "row": int(row["row"]),
                "sample_class": class_name,
                **probabilities,
            }
        )

    for row in read_csv(control_path):
        if row.get("model") != "dv1_r2support_control":
            raise ValueError("control table contains a non-control model row")
        class_name = normalize_class(row["sample_class"])
        probabilities = {key: float(row[key]) for key in ("p_full", "p_no_high", "p_no_pair")}
        for key, probability in probabilities.items():
            values["dv1_r2support_control"][class_name][key].append(probability)
        derived.append(
            {
                "model": "dv1_r2support_control",
                "row": int(row["row"]),
                "sample_class": class_name,
                **probabilities,
            }
        )

    grouped: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model in MODEL_ORDER:
        grouped[model] = {}
        counts: dict[str, int] = {}
        for class_name in CLASS_ORDER:
            grouped[model][class_name] = {}
            for key, rows in values[model][class_name].items():
                array = np.asarray(rows, dtype=float)
                if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
                    raise ValueError(f"invalid probabilities for {model}/{class_name}/{key}")
                grouped[model][class_name][key] = array
            counts[class_name] = int(len(values[model][class_name]["p_full"]))
        if counts != EXPECTED_COUNTS:
            raise ValueError(f"sample counts changed for {model}: {counts}")
    if len(derived) != 2 * sum(EXPECTED_COUNTS.values()):
        raise AssertionError("consolidated source table does not contain both complete routes")
    return grouped, derived


def darken(color: str, factor: float = 0.72) -> tuple[float, float, float]:
    return tuple(factor * value for value in mpl.colors.to_rgb(color))


def lighten(color: str, weight: float = 0.77) -> tuple[float, float, float]:
    rgb = np.asarray(mpl.colors.to_rgb(color))
    return tuple((1.0 - weight) * rgb + weight * np.ones(3))


def draw_distribution(
    axis: mpl.axes.Axes,
    probability: np.ndarray,
    position: float,
    class_name: str,
    condition: str,
) -> None:
    color = CLASS_STYLE[class_name]["color"]
    width = 0.29
    if np.ptp(probability) > 1.0e-12:
        violin = axis.violinplot(
            [probability],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.18,
            points=240,
        )
        body = violin["bodies"][0]
        if condition == "full":
            body.set_facecolor(color)
            body.set_edgecolor(darken(color))
            body.set_alpha(0.78)
            body.set_linestyle("-")
        else:
            body.set_facecolor(lighten(color))
            body.set_edgecolor(color)
            body.set_alpha(0.92)
            body.set_linestyle((0, (2.5, 1.6)))
        body.set_linewidth(0.75)
    else:
        linestyle = "-" if condition == "full" else (0, (2.5, 1.6))
        axis.hlines(
            probability[0],
            position - width / 2,
            position + width / 2,
            color=darken(color) if condition == "full" else color,
            linewidth=3.4 if condition == "full" else 2.0,
            linestyle=linestyle,
        )

    q1, median, q3 = np.quantile(probability, [0.25, 0.5, 0.75])
    axis.vlines(
        position,
        q1,
        q3,
        color="#202020" if condition == "full" else color,
        linewidth=1.15,
        linestyle="-" if condition == "full" else (0, (2.0, 1.4)),
        zorder=5,
    )
    axis.scatter(
        [position],
        [median],
        s=9.5,
        facecolor=color if condition == "full" else "white",
        edgecolor="#202020" if condition == "full" else color,
        linewidth=0.6,
        zorder=6,
    )


def populate_panel(
    axis: mpl.axes.Axes,
    grouped: dict[str, dict[str, dict[str, np.ndarray]]],
    *,
    model: str,
    replaced_key: str,
    title: str,
    panel: str,
) -> None:
    x = np.arange(len(CLASS_ORDER), dtype=float)
    for index, class_name in enumerate(CLASS_ORDER):
        draw_distribution(axis, grouped[model][class_name]["p_full"], x[index] - 0.17, class_name, "full")
        draw_distribution(axis, grouped[model][class_name][replaced_key], x[index] + 0.17, class_name, "replaced")
    axis.set_xticks(x, [CLASS_STYLE[name]["short"] for name in CLASS_ORDER])
    axis.set_xlim(-0.55, len(CLASS_ORDER) - 0.45)
    axis.set_ylim(-0.025, 1.025)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.axhline(0.5, color="#555555", linewidth=0.75, linestyle=(0, (3.0, 2.0)), zorder=0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.78)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", length=0, pad=2)
    axis.tick_params(axis="x", labelsize=7.3)
    axis.set_title(title, pad=4.5)
    axis.annotate(
        panel,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(-17.0, 4.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def class_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=CLASS_STYLE[name]["color"],
            edgecolor=darken(CLASS_STYLE[name]["color"]),
            alpha=0.78,
            label=f'{CLASS_STYLE[name]["label"]} (n={CLASS_STYLE[name]["n"]})',
        )
        for name in CLASS_ORDER
    ]


def condition_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="-", color="#333333", markerfacecolor="#777777", markersize=4.2, label="Full input"),
        Line2D([], [], marker="o", linestyle=(0, (2.5, 1.6)), color="#777777", markerfacecolor="white", markersize=4.2, label="Block replaced"),
    ]


def write_source(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["model", "row", "sample_class", "p_full", "p_no_high", "p_no_pair"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    grouped: dict[str, dict[str, dict[str, np.ndarray]]], path: Path, provenance_path: Path
) -> None:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "figure": "Fig. 4 probability-violin redesign",
        "dataset": "expanded clean-Gibbs N=16 J1-J2",
        "decision_threshold": 0.5,
        "training_routes": {
            "dv1_only": ["D_v1"],
            "dv1_r2support_control": provenance["training_sources"],
        },
        "excluded_control_training_sources": provenance["excluded_training_sources"],
        "class_counts": EXPECTED_COUNTS,
        "panels": {},
    }
    for model in MODEL_ORDER:
        summary["panels"][model] = {}
        for class_name in CLASS_ORDER:
            summary["panels"][model][class_name] = {}
            for key, values in grouped[model][class_name].items():
                summary["panels"][model][class_name][key] = {
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "positive_count": int(np.sum(values >= 0.5)),
                }
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def draw(grouped: dict[str, dict[str, dict[str, np.ndarray]]], stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2047, 4.55), sharex=True, sharey=True)
    specs = [
        (axes[0, 0], "dv1_only", "p_no_high", r"Dv1 only: T3/T4 $\rightarrow$ training means", "a"),
        (axes[0, 1], "dv1_only", "p_no_pair", r"Dv1 only: Pair block $\rightarrow$ training means", "b"),
        (axes[1, 0], "dv1_r2support_control", "p_no_high", r"Dv1 + R2Support: T3/T4 $\rightarrow$ training means", "c"),
        (axes[1, 1], "dv1_r2support_control", "p_no_pair", r"Dv1 + R2Support: Pair block $\rightarrow$ training means", "d"),
    ]
    for axis, model, key, title, panel in specs:
        populate_panel(axis, grouped, model=model, replaced_key=key, title=title, panel=panel)
    axes[0, 0].set_ylabel(r"Predicted entanglement probability, $p$")
    axes[1, 0].set_ylabel(r"Predicted entanglement probability, $p$")
    fig.legend(
        handles=class_handles() + condition_handles(),
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=6,
        frameon=False,
        handlelength=1.35,
        columnspacing=1.05,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.075, top=0.89, hspace=0.34, wspace=0.17)
    panel_ids = {axis: panel for axis, _, _, _, panel in specs}
    require_matplotlib_panel_alignment(
        fig,
        axes=list(axes.ravel()),
        panel_ids=panel_ids,
        json_out=str(stem.with_suffix(".alignment-layout.json")),
        overlay_svg=str(stem.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
    )
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.025, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--dv1-source", type=Path, default=here / "joint_k16_dv1_r2support_interventions_source.csv")
    parser.add_argument("--control-source", type=Path, default=here / "dv1_r2support_control_j1j2_interventions.csv")
    parser.add_argument("--control-report", type=Path, default=here / "dv1_r2support_control_intervention_report.json")
    parser.add_argument("--output-dir", type=Path, default=here)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    grouped, derived = load_sources(args.dv1_source, args.control_source)
    write_source(derived, args.output_dir / "fig4_probability_violin_source_data.csv")
    write_summary(grouped, args.output_dir / "fig4_probability_violin_summary.json", args.control_report)
    stem = args.output_dir / "joint_k16_dv1_r2support_control_probability_violins"
    draw(grouped, stem)
    print(json.dumps({"status": "complete", "figure": str(stem.with_suffix('.pdf')), "rows": len(derived)}, indent=2))


if __name__ == "__main__":
    main()

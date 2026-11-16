"""Regenerate manuscript Figs. 2 and 3 from frozen selector and score tables.

The script changes presentation only. It does not refit models, rerank channels,
or alter the saved benchmark and intervention probabilities. Figure 3(b) reads
the separately audited Dv1-only rerun of the frozen r=4 channel ablation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from audit_panel_alignment import require_matplotlib_panel_alignment
from generate_fig4_probability_violins import (
    class_handles,
    condition_handles,
    load_sources,
    populate_panel,
)


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SOURCE = HERE / "combined_structural_lu_controls_source.csv"
SELECTION = DATA / "joint_channel_selection_summary.json"
SELECTED = DATA / "joint_K16_selection_manifest.json"
K_BUDGET_SOURCE = DATA / "joint_k_budget_balanced_metrics.csv"
R4_DV1_SUMMARY = DATA / "r4_pair_ablation_dv1_only_summary.json"
COLORS = {"T2": "#BF6A14", "C2": "#8C6335", "T3": "#3D7FA3", "T4": "#7A5AA5"}
MODELS = (
    ("raw", "Raw correlators", "#858585"),
    ("soft_lu", "Soft LU constraint", "#D55E00"),
    ("joint_k16", "Enforced LU invariance", "#2166AC"),
)
METRICS = (("AUROC", "AUROC"),
           ("E2_recall", r"Recall $\mathrm{E}_{2}$"),
           ("E_gt_2_recall", r"Recall $\mathrm{E}_{>2}$"))
METRIC_STYLE = (
    ("AUROC", "AUROC", "#A50021", "o"),
    ("E2_recall", r"Recall $\mathrm{E}_{2}$", "#D55E00", "s"),
    ("E_gt_2_recall", r"Recall $\mathrm{E}_{>2}$", "#2166AC", "^"),
    ("TNR", "TNR", "#5E6368", "D"),
)
PAIR_DISTANCE_COLORS = {1: "#C43532", 2: "#C49800", 3: "#2166AC", 4: "#666B70"}
OOD = (("XXZ", "XXZ"), ("J1-J2", r"$J_1$--$J_2$"),
       ("Random_local_H", "Random local $H$"))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        # Configured fallback for tooling; the active manuscript face is Times New Roman.
        "font.sans-serif": ["Arial", "Helvetica"],
        "mathtext.fontset": "stix",
        "font.size": 6.8,
        "axes.labelsize": 7.3,
        "axes.titlesize": 7.3,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax: mpl.axes.Axes, name: str) -> None:
    ax.annotate(name, (0, 1), xycoords="axes fraction", xytext=(-17, 3),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=8, fontweight="bold")


def source_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 57:
        raise ValueError(f"expected 57 current benchmark rows, found {len(rows)}")
    if any(row["model_or_variant"] in {"siamese", "Drop_r4", "Replace_by_T4"}
           for row in rows):
        raise ValueError("historical LU or r=4 ablation rows entered the current figure")
    return rows


def metric(rows: list[dict[str, str]], section: str, panel_id: str,
           benchmark: str, model: str, name: str) -> float:
    matches = [r for r in rows if (r["section"], r["panel"], r["benchmark"],
                                    r["model_or_variant"], r["metric"]) ==
               (section, panel_id, benchmark, model, name)]
    if len(matches) != 1:
        raise ValueError(f"missing or duplicate frozen value: {section}/{panel_id}/{benchmark}/{model}/{name}")
    return float(matches[0]["value"])


def selection_rows() -> tuple[list[dict], list[dict]]:
    summary = json.loads(SELECTION.read_text(encoding="utf-8"))
    manifest = json.loads(SELECTED.read_text(encoding="utf-8"))
    ids = summary["consensus_ranking_best_to_worst_full_channels"]
    if summary["candidate_count"] != 117 or ids[:16] != manifest["selected_physical_full_channels"]:
        raise ValueError("selector ranking and frozen K16 manifest disagree")
    lookup = {int(row["channel"]): row for row in summary["channels"]}
    ranked = []
    for rank, channel in enumerate(ids[:32], 1):
        row = dict(lookup[int(channel)])
        row["overall_rank"] = rank
        row["score"] = 118.0 - float(row["rank_mean"])
        family = row["family"]
        row["kind"] = ("T2" if row["raw_or_connected"] == "raw" else "C2") if family == "pair" else ("T3" if family == "contiguous_three_body" else "T4")
        ranked.append(row)
    pair_rows = [row for row in ranked if row["family"] == "pair"]
    if len(pair_rows) != 13:
        raise ValueError(f"expected 13 pair channels in top 32, found {len(pair_rows)}")
    return ranked[:16], pair_rows


def selection_panels(ax_a: mpl.axes.Axes, ax_b: mpl.axes.Axes) -> None:
    selected, pairs = selection_rows()
    x = np.arange(1, 17)
    ax_a.bar(x, [r["score"] for r in selected], width=0.72,
             color=[COLORS[r["kind"]] for r in selected], zorder=2)
    tensor_labels = {"T3": r"$T^{(3)}$", "T4": r"$T^{(4)}$"}
    ax_a.set_xticks(x, [str(row["overall_rank"]) for row in selected], fontsize=7.2)
    for position, row in zip(x, selected):
        label = (f"r={int(row['pair_distance'])}" if row["family"] == "pair"
                 else tensor_labels[row["kind"]])
        ax_a.annotate(label, (position, 0), xycoords=ax_a.get_xaxis_transform(),
                      xytext=(0, -10), textcoords="offset points", ha="center",
                      va="top", fontsize=7.2, annotation_clip=False)
    ax_a.set_xlim(0.45, 16.55)
    ax_a.set_ylim(90, 113)
    ax_a.set_yticks([90, 95, 100, 105, 110])
    ax_a.set_ylabel(r"Consensus score $S$")
    ax_a.set_title("Selected 16 LU-invariant channels", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color="#E4E4E4", linewidth=0.5, zorder=0)
    ax_a.tick_params(axis="x", length=0, pad=2)
    panel(ax_a, "a")

    # Distance is encoded by color; only the first-ranked channel at each
    # distance receives a star. Every data marker stays on the same baseline.
    ax_b.axhline(0.39, color="#CED3D7", linewidth=0.75, zorder=1)
    ax_b.axvline(16.0, ymin=0.0, ymax=0.73,
                 color="#71767A", linestyle="--", linewidth=0.8, zorder=1)
    first_ranks: dict[int, int] = {}
    for row in pairs:
        rank = int(row["overall_rank"])
        distance = int(row["pair_distance"])
        first = distance not in first_ranks
        if first:
            first_ranks[distance] = rank
        ax_b.scatter(rank, 0.39, s=66 if first else 27,
                     marker="*" if first else "o",
                     facecolor=PAIR_DISTANCE_COLORS[distance],
                     edgecolor="#30363A", linewidth=0.35, zorder=3)
        label_y = 0.64 if rank in (15, 23, 31) else 0.53
        ax_b.text(rank, label_y, str(rank), ha="center", va="bottom",
                  fontsize=5.7, color="#343A40")
    if first_ranks != {1: 2, 4: 11, 3: 17, 2: 28}:
        raise ValueError(f"unexpected first pair-distance ranks: {first_ranks}")
    ax_b.text(17.0, 0.12, "$K=16$", ha="left", va="center", fontsize=7.2)
    distance_handles = [Line2D([], [], linestyle="none", marker="o",
                               markerfacecolor=PAIR_DISTANCE_COLORS[distance],
                               markeredgecolor="#30363A", markeredgewidth=0.3,
                               markersize=4.3, label=f"$r={distance}$")
                        for distance in (1, 2, 3, 4)]
    distance_handles.append(Line2D([], [], linestyle="none", marker="*",
                                   markerfacecolor="#333333", markeredgecolor="#333333",
                                   markersize=6.2, label="First occurrence"))
    ax_b.legend(handles=distance_handles, loc="upper center",
                bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                fontsize=7.2, columnspacing=0.50, handletextpad=0.30,
                handlelength=0.65, borderaxespad=0.0)
    ax_b.set_xlim(0.4, 32.6)
    ax_b.set_ylim(0.05, 1.0)
    ax_b.set_xticks([1, 8, 16, 24, 32])
    ax_b.set_yticks([])
    ax_b.set_xlabel("")
    ax_b.set_title("Pair-channel ranking and performance vs. K", loc="left",
                   fontweight="bold")
    ax_b.tick_params(axis="x", length=2.5, pad=2)
    ax_b.tick_params(axis="x", labelbottom=False)
    panel(ax_b, "b")


def budget_panel(ax: mpl.axes.Axes) -> None:
    with K_BUDGET_SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    budgets = [int(row["K"]) for row in rows]
    if budgets != [4, 8, 12, 16, 20, 24, 28, 32]:
        raise ValueError("K-budget figure requires all eight audited, ordered budgets")
    for row in rows:
        if int(row["seeds"]) != 3 or int(row["input_width"]) != int(row["K"]) + 5:
            raise ValueError("K-budget provenance does not match the three-seed five-mask protocol")
    x = np.asarray(budgets)
    for key, label, color, marker in METRIC_STYLE:
        source_key = "Sep_FPR" if key == "TNR" else key
        y = np.asarray([float(row[source_key]) for row in rows])
        if not np.all(np.isfinite(y) & (y >= 0) & (y <= 1)):
            raise ValueError(f"invalid K-budget metric {key}")
        if key == "TNR":
            y = 1.0 - y
        ax.plot(x, y, marker=marker, markersize=3.3, linewidth=1.05,
                linestyle="--" if key == "TNR" else "-", color=color,
                label=label, zorder=3)
    ax.axvline(16, color="#909090", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xticks(budgets)
    # Shared with the pair-ranking panel: retain ranks 1--3 beside K=4.
    ax.set_xlim(0.4, 32.6)
    ax.set_ylim(-0.04, 1.055)
    ax.set_xlabel("# Distilled features $K$")
    ax.set_ylabel("AUROC / recall / TNR")
    # This lower axis and the pair-ranking axis form one composite panel b.
    # Their shared x axis directly locates each pair channel against K.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2,
              frameon=False, fontsize=7.2, labelspacing=0.2,
              columnspacing=0.7, handlelength=1.2, handletextpad=0.3)


def ood_panel(rows: list[dict[str, str]], ax: mpl.axes.Axes) -> None:
    # Use a larger-is-better scale for all four metrics. Small horizontal
    # offsets separate distinct metrics that are exactly one in a family.
    styles = (
        ("AUROC", "AUROC", "#2C5C85", "o"),
        ("E2_recall", r"Recall $\mathrm{E}_{2}$", "#B2182B", "s"),
        ("E_gt_2_recall", r"Recall $\mathrm{E}_{>2}$", "#EF8A62", "^"),
        ("FPR", r"TNR $=1-\mathrm{FPR}$", "#5E6368", "D"),
    )
    offsets = (-0.18, -0.06, 0.06, 0.18)
    for family_index, (key, _) in enumerate(OOD):
        for metric_index, (name, _, color, marker) in enumerate(styles):
            original = metric(rows, "ood", "d", key, "Dv1-only Joint-K16", name)
            if not 0.0 <= original <= 1.0:
                raise ValueError(f"invalid OOD metric {key}/{name}: {original}")
            result = 1.0 - original if name == "FPR" else original
            ax.scatter(family_index + offsets[metric_index], result,
                       marker=marker, s=27, facecolor=color,
                       edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks(np.arange(len(OOD)), [label for _, label in OOD])
    ax.tick_params(axis="x", labelsize=7.4, length=0, pad=4)
    ax.set_xlim(-0.48, len(OOD) - 0.52)
    ax.set_ylim(0.75, 1.035)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.axhline(1.0, color="#868C91", linestyle="--", linewidth=0.8,
               zorder=0)
    ax.set_ylabel("AUROC / recall / TNR")
    ax.set_title("Transfer to unseen Hamiltonians", loc="left", fontweight="bold")
    # Keep the lower legend band clear; tightly spaced 0.05 gridlines would
    # pass through the legend text at this final physical size.
    metric_handles = [Line2D([], [], linestyle="none", marker=marker,
                              markerfacecolor=color, markeredgecolor="white",
                              markeredgewidth=0.4, markersize=4.5, label=label)
                      for _, label, color, marker in styles]
    ax.legend(handles=metric_handles, loc="lower left",
              bbox_to_anchor=(0.02, 0.02), ncol=2, frameon=False,
              fontsize=7.2, columnspacing=0.45, handletextpad=0.15,
              handlelength=0.7, labelspacing=0.20)
    panel(ax, "c")


def representation_panel(rows: list[dict[str, str]], ax: mpl.axes.Axes,
                         source_panel: str, benchmark: str, title: str,
                         display_panel: str, ylabel: bool = False) -> None:
    x = np.arange(3)
    width = 0.25
    for index, (key, _, color) in enumerate(MODELS):
        values = [metric(rows, "representation", source_panel, benchmark, key, measure)
                  for measure, _ in METRICS]
        ax.bar(x + (index - 1) * width, values, width=width,
               color=color, zorder=2)
    ax.set_xticks(x, ["AUROC", r"Recall\n$\mathrm{E}_{2}$".replace("\\n", "\n"),
                      r"Recall\n$\mathrm{E}_{>2}$".replace("\\n", "\n")],
                  fontsize=7.3)
    ax.set_ylim(0, 1.04)
    ax.set_title(title, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel("Score / rate")
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.5, zorder=0)
    panel(ax, display_panel)


def tnr_panel(rows: list[dict[str, str]], ax: mpl.axes.Axes) -> None:
    """Collect all four separable controls on one larger-is-better scale."""
    tests = (("e", "N16_balanced", "XY 16"),
             ("f", "N32_XY", "XY 32"),
             ("g", "N32_DPI", "Synthetic"),
             ("h", "N32_GeneralSep", "Gen.-Sep"))
    for test_index, (source_panel, benchmark, _) in enumerate(tests):
        center = 3 - test_index
        for model_index, (key, _, color) in enumerate(MODELS):
            fpr = metric(rows, "representation", source_panel, benchmark, key, "FPR")
            if not 0.0 <= fpr <= 1.0:
                raise ValueError(f"invalid false-positive rate: {benchmark}/{key}")
            ax.scatter(1.0 - fpr, center + (model_index - 1) * 0.18,
                       marker=("o", "s", "^")[model_index], s=25,
                       facecolor=color, edgecolor="white", linewidth=0.3, zorder=3)
    ax.set_yticks([3, 2, 1, 0], [label for _, _, label in tests])
    ax.set_xlim(-0.035, 1.035)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(-0.40, 3.40)
    ax.set_title("Separable-state TNR", loc="left", fontweight="bold")
    ax.set_xlabel("TNR")
    ax.grid(axis="x", color="#E4E4E4", linewidth=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="y", labelsize=7.3, pad=1)
    panel(ax, "g")


def save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.025,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_fig2(rows: list[dict[str, str]]) -> None:
    fig = plt.figure(figsize=(7.2047, 5.25))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.22, 1.0], hspace=0.48,
                             left=0.075, right=0.985, top=0.985, bottom=0.065)
    upper = outer[0].subgridspec(1, 2, wspace=0.27)
    left = upper[0, 0].subgridspec(2, 1, height_ratios=[1.06, 0.94],
                                   hspace=0.43)
    # The two right-hand axes are a single panel, with one label and one
    # shared rank/K axis; close the gap without compressing the left column.
    right = upper[0, 1].subgridspec(2, 1, height_ratios=[1.06, 0.94],
                                    hspace=0.15)
    third = outer[1].subgridspec(1, 4, wspace=0.30)
    ax_a = fig.add_subplot(left[0, 0])
    ax_b_rank = fig.add_subplot(right[0, 0])
    ax_b_budget = fig.add_subplot(right[1, 0], sharex=ax_b_rank)
    ax_c = fig.add_subplot(left[1, 0])
    axes = [ax_a, ax_b_rank, ax_b_budget, ax_c]
    axes += [fig.add_subplot(third[0, i]) for i in range(4)]
    selection_panels(*axes[:2])
    budget_panel(axes[2])
    ood_panel(rows, axes[3])
    representation_panel(rows, axes[4], "e", "N16_balanced",
                         r"XY, $N=16$", "d", True)
    representation_panel(rows, axes[5], "f", "N32_XY",
                         r"XY, $N=32$", "e")
    representation_panel(rows, axes[6], "g", "N32_DPI",
                         r"Synthetic states, $N=32$", "f")
    tnr_panel(rows, axes[7])
    fig.legend(handles=[Patch(color=color, label=label) for _, label, color in MODELS],
               loc="center", bbox_to_anchor=(0.60, 0.340), ncol=3,
               frameon=False, columnspacing=1.5, handlelength=1.1)
    base = HERE / "combined_structural_lu_controls"
    if ax_b_rank.get_xlim() != ax_b_budget.get_xlim():
        raise AssertionError("composite panel b must share its rank/K limits")
    require_matplotlib_panel_alignment(
        fig, axes=axes,
        panel_ids={ax: label for ax, label in zip(axes,
                   ("a", "b", "b-budget", "c", "d", "e", "f", "g"))},
        json_out=str(base.with_suffix(".alignment-layout.json")),
        overlay_svg=str(base.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        # The lower half of composite panel b intentionally has no second
        # panel letter; validate its shared column and the other panel row.
        require_panel_labels=False,
        row_groups=[["d", "e", "f", "g"]],
        column_groups=[["a", "c"], ["b", "b-budget"]],
    )
    save(fig, base)


def make_fig3(rows: list[dict[str, str]]) -> None:
    grouped, _ = load_sources(
        DATA / "joint_k16_dv1_r2support_interventions_source.csv",
        DATA / "dv1_r2support_control_j1j2_interventions.csv",
    )
    fig = plt.figure(figsize=(5.7638, 4.30))
    # The two narrow bands separate the ablation metric key from the class and
    # condition key that applies only to the four intervention panels.
    outer = fig.add_gridspec(6, 2,
                             height_ratios=[0.88, 0.17, 0.25, 0.90, 0.18, 0.90],
                             hspace=0.29, wspace=0.25,
                             left=0.085, right=0.985, top=0.97, bottom=0.07)
    axes = [fig.add_subplot(outer[0, j]) for j in range(2)]
    axes += [fig.add_subplot(outer[i, j]) for i in (3, 5) for j in range(2)]
    x = np.arange(3)
    e4 = [metric(rows, "structural", "b", "N32_DPI", key, "E4_recall")
          for key in ("T1-T3", "+2_T4", "Joint-K16")]
    axes[0].scatter(x, e4, marker="^", s=42, color="#2C5C85", zorder=3)
    for position, value in zip(x, e4):
        offset = (8, 13) if position == 0 else (0, 10)
        axes[0].annotate(f"{value:.3f}", (position, value), xytext=offset,
                         textcoords="offset points",
                         ha="left" if position == 0 else "center", va="bottom",
                         fontsize=5.7, color="#2C5C85")
    axes[0].set_xticks(x, ["T1–T3", "+2 T4", "Joint-K16"])
    axes[0].set_ylim(0, 1.13)
    axes[0].set_ylabel(r"$\mathrm{E}_{4}$ recall")
    axes[0].set_title("Four-body contribution", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#E4E4E4", linewidth=0.5, zorder=0)
    panel(axes[0], "a")

    rerun = json.loads(R4_DV1_SUMMARY.read_text(encoding="utf-8"))
    if (rerun.get("status") != "complete" or
            rerun.get("training_sources") != ["D_v1"] or
            rerun.get("seeds") != [20260829, 20260830, 20260831]):
        raise ValueError("Fig. 3(b) requires the audited three-seed Dv1-only rerun")
    arm_names = ("Dv1_Baseline_K16", "Dv1_DropR4_K14", "Dv1_ReplaceR4_T4_K16")
    expected_channels = (
        [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 44, 83, 91, 51, 3, 59],
        [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 83, 91, 3, 59],
        [60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 83, 91, 3, 59, 85, 112],
    )
    ablation = [rerun["arms"][name] for name in arm_names]
    if any(row["channels"][:row["K"]] != expected for row, expected in
           zip(ablation, expected_channels)):
        raise ValueError("Fig. 3(b) frozen physical channels changed")
    auroc = [row["probability_average_ensemble"]["exact500_AUROC"] for row in ablation]
    tnr = [row["probability_average_ensemble"]["GeneralSep_TNR"] for row in ablation]
    if any(not 0.0 <= value <= 1.0 for value in auroc + tnr):
        raise ValueError("Fig. 3(b) metrics must be probabilities or AUROC values")
    if any(abs(row["probability_average_ensemble"]["GeneralSep_false_positives"] /
               2000 + value - 1.0) > 1e-12 for row, value in zip(ablation, tnr)):
        raise ValueError("Fig. 3(b) TNR disagrees with the 2,000 separable controls")
    axes[1].bar(x - 0.18, auroc, width=0.35, color="#A50021",
                label="Exact AUROC", zorder=2)
    axes[1].bar(x + 0.18, tnr, width=0.35, color="#D55E00",
                label="General-Sep TNR", zorder=2)
    axes[1].set_xticks(x, ["Baseline", "Drop $r=4$", "Replace by T₄"])
    axes[1].set_ylim(0, 1.04)
    axes[1].set_ylabel("AUROC / TNR")
    axes[1].set_title("$r=4$ pair ablation", loc="left", fontweight="bold")
    axes[1].grid(axis="y", color="#E4E4E4", linewidth=0.5, zorder=0)
    panel(axes[1], "b")

    specs = (
        (axes[2], "dv1_only", "p_no_high", r"Dv1: T3/T4 $\rightarrow$ mean", "c"),
        (axes[3], "dv1_only", "p_no_pair", r"Dv1: pair $\rightarrow$ mean", "d"),
        (axes[4], "dv1_r2support_control", "p_no_high", r"Dv1 + R2Support: T3/T4 $\rightarrow$ mean", "e"),
        (axes[5], "dv1_r2support_control", "p_no_pair", r"Dv1 + R2Support: pair $\rightarrow$ mean", "f"),
    )
    for ax, model, key, title, label in specs:
        populate_panel(ax, grouped, model=model, replaced_key=key, title=title, panel=label)
    axes[2].set_ylabel(r"Classifier output $p$")
    axes[4].set_ylabel(r"Classifier output $p$")
    metric_band = outer[1, 1].get_position(fig)
    fig.legend(handles=[Patch(color="#A50021", label="Exact AUROC"),
                        Patch(color="#D55E00", label="General-Sep TNR")],
               loc="center", bbox_to_anchor=(0.5 * (metric_band.x0 + metric_band.x1),
                                              0.5 * (metric_band.y0 + metric_band.y1)),
               ncol=2, frameon=False, handlelength=0.9, columnspacing=0.8,
               fontsize=6.0)
    legend_band = outer[2, :].get_position(fig)
    fig.legend(handles=class_handles() + condition_handles(), loc="center",
               bbox_to_anchor=(0.5 * (legend_band.x0 + legend_band.x1),
                               0.5 * (legend_band.y0 + legend_band.y1) + 0.020),
               ncol=3, frameon=False,
               handlelength=1.1, columnspacing=0.7, fontsize=7.3)
    base = HERE / "joint_k16_structural_interventions"
    require_matplotlib_panel_alignment(
        fig, axes=axes, panel_ids={ax: label for ax, label in zip(axes, "abcdef")},
        json_out=str(base.with_suffix(".alignment-layout.json")),
        overlay_svg=str(base.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        require_panel_labels=True, row_groups=[["a", "b"], ["c", "d"], ["e", "f"]],
        column_groups=[["c", "e"], ["d", "f"]],
    )
    save(fig, base)


if __name__ == "__main__":
    style()
    frozen = source_rows()
    make_fig2(frozen)
    make_fig3(frozen)

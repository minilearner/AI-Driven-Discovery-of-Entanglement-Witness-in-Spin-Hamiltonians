from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch


def discover_root() -> Path:
    resolved = Path(__file__).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "src/entanglement_data").is_dir():
            return candidate
    raise RuntimeError("could not locate repository root containing src/entanglement_data")


ROOT = discover_root()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.general_n32 import support_certificates  # noqa: E402
from entanglement_data.final_entnet import load_final_ensemble  # noqa: E402
from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    structured_inv123_raw_t4_local_tokens,
)
from entanglement_data.ood_hamiltonian_n16 import CachedMomentState  # noqa: E402
from entanglement_data.xy.free_fermion_exact import XYFreeFermionExactBackend  # noqa: E402
from entanglement_data.xy.size_transfer import (  # noqa: E402
    LocalRawSpec as TransferRawSpec,
    thermal_local_raw,
)


N = 16
R_OBS = 4
TOL_PT = 1.0e-9
TOL_CCNR = 1.0e-9
TRAINING_SEEDS = (20260829, 20260830, 20260831)
EXPECTED_CHANNELS = (60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 44, 83, 91, 51, 3, 59,
                     117, 118, 119, 120, 121)

POINTS = (
    {"id": "ising_ordered", "gamma": 1.0, "h": 0.5,
     "label": r"Ising ordered: $\gamma=1,\ h=0.5$"},
    {"id": "ising_critical", "gamma": 1.0, "h": 1.0,
     "label": r"Ising critical: $\gamma=1,\ h=1$"},
    {"id": "ising_field", "gamma": 1.0, "h": 1.4,
     "label": r"Ising field-polarized: $\gamma=1,\ h=1.4$"},
    {"id": "anisotropic_ordered", "gamma": 0.5, "h": 0.5,
     "label": r"Anisotropic ordered: $\gamma=0.5,\ h=0.5$"},
    {"id": "factorization_line", "gamma": 0.6, "h": 0.8,
     "label": r"Factorization line: $\gamma=0.6,\ h=0.8$"},
    {"id": "xx_gapless", "gamma": 0.0, "h": 0.5,
     "label": r"XX gapless: $\gamma=0,\ h=0.5$"},
    {"id": "xx_saturated", "gamma": 0.0, "h": 1.2,
     "label": r"XX saturated: $\gamma=0,\ h=1.2$"},
)

CLASS_ORDER = ("E2", "E3", "E4", "unknown", "infinite_T_separable")
CLASS_COLORS = {
    "E2": "#B2182B",
    "E3": "#EF8A62",
    "E4": "#FDB863",
    "unknown": "#4D8BC9",
    "infinite_T_separable": "#4D4D4D",
}
CURVE_COLORS = ("#1B4F72", "#B2182B", "#2E7D32", "#7B3294", "#E08214", "#008C95", "#666666")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def orbit_representatives(n: int, k: int) -> tuple[tuple[int, ...], ...]:
    """One support from every cyclic-translation orbit of k-site subsets."""
    representatives: dict[tuple[int, ...], tuple[int, ...]] = {}
    for support in itertools.combinations(range(n), k):
        images = [tuple(sorted(((site - shift) % n for site in support))) for shift in range(n)]
        key = min(images)
        representatives.setdefault(key, support)
    return tuple(sorted(representatives))


PAIR_REPS = orbit_representatives(N, 2)
TRIPLE_REPS = orbit_representatives(N, 3)
FOUR_REPS = orbit_representatives(N, 4)


def frozen_channels() -> np.ndarray:
    path = ROOT / "configs/joint_K16.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = tuple(int(value) for value in payload["selected_full_token_channels"])
    if channels != EXPECTED_CHANNELS:
        raise RuntimeError("frozen Joint K16 channel ordering changed")
    return np.asarray(channels, dtype=np.int64)


@torch.no_grad()
def model_logits(models: list[torch.nn.Module], values: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
    return np.stack([model(tensor).detach().cpu().numpy().astype(np.float64) for model in models])


def certificate_margin(result) -> float:
    return float(max(result.npt_margin, result.ccnr_margin))


def exact_record(backend: XYFreeFermionExactBackend, temperature: float, channels: np.ndarray) -> dict[str, Any]:
    beta = 0.0 if math.isinf(temperature) else 1.0 / temperature
    blocks = thermal_local_raw(backend, beta, TransferRawSpec(N, R_OBS))
    full = structured_inv123_raw_t4_local_tokens(
        blocks["T1"][None], blocks["T2"][None], blocks["K2"][None],
        blocks["T3"][None], blocks["K3"][None], blocks["T4"][None],
        two_mask=blocks["T2_mask"][None], three_mask=blocks["T3_mask"][None],
        four_mask=blocks["T4_mask"][None], R_obs=R_OBS, output_dtype=np.float32,
    )[0]
    state = CachedMomentState(N, lambda sites: backend.reduced_density_matrix(sites, beta))
    pair = support_certificates(state, PAIR_REPS, ccnr=False)
    triple = support_certificates(state, TRIPLE_REPS, ccnr=True)
    four = support_certificates(state, FOUR_REPS, ccnr=True)
    nearest = support_certificates(state, ((0, 1),), ccnr=False)
    contiguous3 = support_certificates(state, ((0, 1, 2),), ccnr=True)
    contiguous4 = support_certificates(state, ((0, 1, 2, 3),), ccnr=True)

    pair_margin = float(pair.npt_margin)
    triple_margin = certificate_margin(triple)
    four_margin = certificate_margin(four)
    if math.isinf(temperature):
        class_name = "infinite_T_separable"
    elif pair_margin > TOL_PT:
        class_name = "E2"
    elif triple_margin > max(TOL_PT, TOL_CCNR):
        class_name = "E3"
    elif four_margin > max(TOL_PT, TOL_CCNR):
        class_name = "E4"
    else:
        class_name = "unknown"
    return {
        "temperature": float(temperature),
        "beta": float(beta),
        "class_name": class_name,
        "pair_npt_margin_all": pair_margin,
        "pair_npt_margin_nearest": float(nearest.npt_margin),
        "triple_margin_all": triple_margin,
        "four_margin_all": four_margin,
        "triple_margin_contiguous": certificate_margin(contiguous3),
        "four_margin_contiguous": certificate_margin(contiguous4),
        "compact_tokens": full[:, channels],
        "max_abs_full_token": float(np.max(np.abs(full))),
    }


def attach_predictions(records: list[dict[str, Any]], models: list[torch.nn.Module], device: torch.device) -> None:
    values = np.stack([row["compact_tokens"] for row in records])
    logits = model_logits(models, values, device)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    ensemble = probabilities.mean(axis=0)
    ensemble_logit = np.log(np.clip(ensemble, 1e-12, 1.0 - 1e-12) / np.clip(1.0 - ensemble, 1e-12, 1.0))
    for index, row in enumerate(records):
        row["seed_logits"] = [float(value) for value in logits[:, index]]
        row["seed_probabilities"] = [float(value) for value in probabilities[:, index]]
        row["ensemble_probability"] = float(ensemble[index])
        row["ensemble_logit"] = float(ensemble_logit[index])


def needs_refinement(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if math.isinf(right["temperature"]):
        return False
    ratio = right["temperature"] / left["temperature"]
    if ratio < 1.08:
        return False
    class_change = left["class_name"] != right["class_name"]
    sign_change = left["ensemble_logit"] * right["ensemble_logit"] <= 0.0
    steep = abs(left["ensemble_logit"] - right["ensemble_logit"]) > 1.5
    probability_change = abs(left["ensemble_probability"] - right["ensemble_probability"]) > 0.08
    return class_change or sign_change or steep or probability_change


def crossing_temperature(records: list[dict[str, Any]]) -> float | None:
    finite = sorted((row for row in records if np.isfinite(row["temperature"])), key=lambda row: row["temperature"])
    for left, right in zip(finite[:-1], finite[1:]):
        z0, z1 = left["ensemble_logit"], right["ensemble_logit"]
        if z0 == 0.0:
            return float(left["temperature"])
        if z0 * z1 < 0.0:
            x0, x1 = math.log(left["temperature"]), math.log(right["temperature"])
            fraction = -z0 / (z1 - z0)
            return float(math.exp(x0 + fraction * (x1 - x0)))
    return None


def last_certified_temperature(records: list[dict[str, Any]], order: str) -> float | None:
    finite = [row["temperature"] for row in records if np.isfinite(row["temperature"])]
    if order == "E2":
        chosen = [row["temperature"] for row in records if row["pair_npt_margin_all"] > TOL_PT]
    elif order == "E3":
        chosen = [row["temperature"] for row in records if row["triple_margin_all"] > TOL_CCNR]
    else:
        chosen = [row["temperature"] for row in records if row["four_margin_all"] > TOL_CCNR]
    chosen = [value for value in chosen if np.isfinite(value)]
    return float(max(chosen)) if chosen else None


def path_summary(point: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    finite = sorted((row for row in records if np.isfinite(row["temperature"])), key=lambda row: row["temperature"])
    z = np.asarray([row["ensemble_logit"] for row in finite])
    differences = np.diff(z)
    infinite = next(row for row in records if math.isinf(row["temperature"]))
    return {
        "point": point,
        "sample_count": len(records),
        "low_temperature": finite[0]["temperature"],
        "low_temperature_logit": finite[0]["ensemble_logit"],
        "infinite_temperature_logit": infinite["ensemble_logit"],
        "infinite_temperature_probability": infinite["ensemble_probability"],
        "z_zero_crossing_temperature_log_interpolation": crossing_temperature(records),
        "last_certified_temperature": {
            name: last_certified_temperature(records, name) for name in ("E2", "E3", "E4")
        },
        "nonincreasing_step_fraction": float(np.mean(differences <= 1.0e-6)) if len(differences) else None,
        "upward_steps_over_0p1": int(np.sum(differences > 0.1)),
        "minimum_logit": float(np.min(z)),
        "maximum_logit": float(np.max(z)),
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, base: Path, *, dpi: int = 400) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def plot_overview(results: list[tuple[dict[str, Any], list[dict[str, Any]]]], output: Path) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.15, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.15, 1.0), hspace=0.25)
    ax = fig.add_subplot(grid[0])
    class_ax = fig.add_subplot(grid[1], sharex=ax)
    infinity_x = 2000.0
    for index, (point, records) in enumerate(results):
        finite = sorted((row for row in records if np.isfinite(row["temperature"])), key=lambda row: row["temperature"])
        x = np.asarray([row["temperature"] for row in finite])
        y = np.asarray([row["ensemble_logit"] for row in finite])
        color = CURVE_COLORS[index]
        ax.plot(x, y, color=color, lw=1.45, label=point["label"])
        ax.scatter(x, y, color=color, s=8, edgecolors="none", zorder=3)
        infinity = next(row for row in records if math.isinf(row["temperature"]))
        ax.scatter([infinity_x], [infinity["ensemble_logit"]], color=color, s=25, marker="D", zorder=4)
        for row in finite:
            class_ax.scatter(row["temperature"], index, s=18, marker="s",
                             color=CLASS_COLORS[row["class_name"]], edgecolors="none")
        class_ax.scatter(infinity_x, index, s=25, marker="D", color=CLASS_COLORS["infinite_T_separable"])
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.text(0.023, 0.45, r"decision threshold $z=0$", fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_ylabel(r"Frozen Dv1-only ensemble logit $z$")
    ax.set_title("(a) Temperature response of the frozen Joint-K16 classifier", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handlelength=1.8)
    class_ax.set_xscale("log")
    class_ax.set_xlabel(r"Temperature $T/J$  (diamond at $T=\infty$)")
    class_ax.set_yticks(range(len(POINTS)), [point["id"].replace("_", " ") for point in POINTS])
    class_ax.invert_yaxis()
    class_ax.set_title("(b) Independent PPT/CCNR hierarchy", loc="left", fontweight="bold")
    class_ax.grid(axis="x", color="#EEEEEE", lw=0.5)
    handles = [mpl.lines.Line2D([], [], marker="s", linestyle="", color=CLASS_COLORS[name], label=name.replace("_", " "))
               for name in CLASS_ORDER]
    class_ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=5, frameon=False)
    ax.set_xlim(0.018, 2500.0)
    ax.set_xticks([0.02, 0.1, 1, 10, 100, 1000, 2000], ["0.02", "0.1", "1", "10", "100", "1000", r"$\infty$"])
    fig.align_ylabels([ax, class_ax])
    save_figure(fig, output / "xy_temperature_sweep_overview")


def plot_individual(point: dict[str, Any], records: list[dict[str, Any]], output: Path) -> None:
    configure_matplotlib()
    finite = sorted((row for row in records if np.isfinite(row["temperature"])), key=lambda row: row["temperature"])
    x = np.asarray([row["temperature"] for row in finite])
    fig, axes = plt.subplots(2, 1, figsize=(5.1, 4.8), sharex=True, gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.12})
    ax, margin_ax = axes
    ax.plot(x, [row["ensemble_logit"] for row in finite], color="#1B4F72", lw=1.6, marker="o", ms=2.5)
    for seed_index, seed in enumerate(TRAINING_SEEDS):
        ax.plot(x, [row["seed_logits"][seed_index] for row in finite], color="#AAAAAA", lw=0.55, alpha=0.65,
                label="individual seeds" if seed_index == 0 else None)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel(r"Classifier logit $z$")
    ax.set_title(point["label"], loc="left")
    ax.grid(axis="y", color="#E4E4E4", lw=0.5)
    ax.legend(frameon=False, loc="best")
    margin_specs = (
        ("pair_npt_margin_all", "all-pair E2", "#B2182B", "-"),
        ("pair_npt_margin_nearest", "nearest-neighbor E2", "#B2182B", ":"),
        ("triple_margin_all", "all-subset E3", "#EF8A62", "-"),
        ("four_margin_all", "all-subset E4", "#FDB863", "-"),
        ("triple_margin_contiguous", "contiguous E3", "#4D8BC9", "--"),
        ("four_margin_contiguous", "contiguous E4", "#7B3294", "--"),
    )
    for key, label, color, style in margin_specs:
        y = np.maximum([row[key] for row in finite], 1.0e-14)
        margin_ax.plot(x, y, label=label, color=color, ls=style, lw=1.15)
    margin_ax.axhline(TOL_PT, color="black", lw=0.7, ls=":")
    margin_ax.set_xscale("log")
    margin_ax.set_yscale("log")
    margin_ax.set_xlabel(r"Temperature $T/J$")
    margin_ax.set_ylabel("Certificate margin")
    margin_ax.set_ylim(5.0e-14, max(1.0, float(max(row["pair_npt_margin_all"] for row in finite)) * 2.0))
    margin_ax.grid(color="#EAEAEA", lw=0.45, which="both")
    margin_ax.legend(frameon=False, ncol=2, loc="lower left", fontsize=6.7)
    fig.align_ylabels(axes)
    save_figure(fig, output / f"xy_temperature_{point['id']}")


def write_csv(path: Path, results: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> None:
    rows = []
    for point, records in results:
        for row in sorted(records, key=lambda item: item["temperature"]):
            flat = {key: value for key, value in row.items() if key != "compact_tokens"}
            flat.update({"point_id": point["id"], "gamma": point["gamma"], "h": point["h"]})
            for index, seed in enumerate(TRAINING_SEEDS):
                flat[f"logit_seed_{seed}"] = row["seed_logits"][index]
                flat[f"probability_seed_{seed}"] = row["seed_probabilities"][index]
            flat.pop("seed_logits")
            flat.pop("seed_probabilities")
            rows.append(flat)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/xy_temperature_sweep_dv1_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refinement-rounds", type=int, default=3)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    channels = frozen_channels()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    models, loaded_channels = load_final_ensemble(ROOT, device=str(device))
    if not np.array_equal(channels, loaded_channels):
        raise RuntimeError("thermal input channels differ from frozen checkpoint channels")
    model_audit = []
    for seed in TRAINING_SEEDS:
        model_dir = ROOT / "checkpoints/final_joint_k16" / f"seed{seed}"
        report = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
        model_audit.append({
            "seed": seed,
            "model_dir": str(model_dir),
            "training_route": report["training_route"],
            "selected_channels": report["selected_full_token_channels"],
        })

    base_temperatures = list(np.geomspace(0.02, 1000.0, 29)) + [math.inf]
    results: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for point_index, point in enumerate(POINTS):
        print(json.dumps({"stage": "path_start", "index": point_index, "point": point}), flush=True)
        backend = XYFreeFermionExactBackend(N, point["gamma"], point["h"], "PBC")
        cache: dict[float, dict[str, Any]] = {}

        def ensure(temperatures: list[float]) -> None:
            missing = [value for value in temperatures if value not in cache]
            for index, temperature in enumerate(missing):
                print(json.dumps({"stage": "state", "point": point["id"], "temperature": temperature,
                                  "item": index + 1, "missing": len(missing)}), flush=True)
                cache[temperature] = exact_record(backend, temperature, channels)
            attach_predictions(list(cache.values()), models, device)

        ensure(base_temperatures)
        for round_index in range(args.refinement_rounds):
            finite = sorted((row for row in cache.values() if np.isfinite(row["temperature"])),
                            key=lambda row: row["temperature"])
            additions = [math.sqrt(left["temperature"] * right["temperature"])
                         for left, right in zip(finite[:-1], finite[1:]) if needs_refinement(left, right)]
            additions = sorted({float(value) for value in additions if value not in cache})
            print(json.dumps({"stage": "refine", "point": point["id"], "round": round_index + 1,
                              "additions": len(additions)}), flush=True)
            if not additions:
                break
            ensure(additions)
        records = sorted(cache.values(), key=lambda row: row["temperature"])
        results.append((point, records))
        print(json.dumps({"stage": "path_complete", "point": point["id"],
                          "summary": path_summary(point, records)}), flush=True)

    write_csv(args.output / "xy_temperature_sweep_source_data.csv", results)
    summaries = [path_summary(point, records) for point, records in results]
    report = {
        "status": "complete",
        "scientific_question": (
            "Whether the frozen Dv1-only Joint-K16 logit changes continuously and crosses z=0 as exact finite-N "
            "XY Gibbs states approach the maximally mixed, fully separable infinite-temperature state."
        ),
        "N": N,
        "boundary_condition": "PBC",
        "hamiltonian_convention": "H=-sum_i[(1+gamma)/2 X_iX_{i+1}+(1-gamma)/2 Y_iY_{i+1}+h Z_i]",
        "certificate_definition": {
            "E2": "PPT over one representative of every cyclic orbit of all C(16,2) supports",
            "E3": "PPT or CCNR over every cyclic orbit of all C(16,3) supports, after E2 fails",
            "E4": "PPT or CCNR over every cyclic orbit of all C(16,4) supports, after E2 and E3 fail",
            "translation_orbit_justification": "The exact finite-size PBC Gibbs state is translation invariant.",
            "tol_pt": TOL_PT,
            "tol_ccnr": TOL_CCNR,
            "orbit_representative_counts": {"pair": len(PAIR_REPS), "triple": len(TRIPLE_REPS), "four": len(FOUR_REPS)},
            "additional_local_diagnostics": "nearest-neighbor E2 and contiguous E3/E4 margins are also recorded",
        },
        "temperature_schedule": {
            "base": "29 logarithmic points from 0.02 to 1000 plus exact beta=0",
            "adaptive_rule": "insert geometric midpoints at certificate-class changes, z sign changes, |Delta z|>1.5, or |Delta p|>0.08",
            "refinement_rounds": args.refinement_rounds,
        },
        "frozen_model": {"ensemble": "probability average of three seeds", "audit": model_audit},
        "points": summaries,
        "wall_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    save_json(args.output / "xy_temperature_sweep_report.json", report)
    plot_overview(results, args.output)
    for point, records in results:
        plot_individual(point, records, args.output)
    generated = sorted(str(path.name) for path in args.output.iterdir())
    save_json(args.output / "artifact_manifest.json", {"status": "complete", "files": generated})
    print(json.dumps({"stage": "complete", "output": str(args.output), "wall_seconds": report["wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

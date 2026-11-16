"""Recompute XY thermal response with the *new* Dv1+R2Support ensemble.

This program never reads old model probabilities as input. It reconstructs
thermal Pauli correlations from the exact XY backend for every grid point.
The original paper and figure files are left untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entanglement_data.control_entnet import load_control_ensemble  # noqa: E402
from entanglement_data.ml.invariant_transformer import structured_inv123_raw_t4_local_tokens  # noqa: E402
from entanglement_data.xy.free_fermion_exact import XYFreeFermionExactBackend  # noqa: E402
from entanglement_data.xy.size_transfer import LocalRawSpec, thermal_local_raw  # noqa: E402

SEEDS = (20260829, 20260830, 20260831)
N = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokens(backend: XYFreeFermionExactBackend, temperature: float, channels: list[int]) -> np.ndarray:
    raw = thermal_local_raw(backend, 1.0 / temperature, LocalRawSpec(N, 4))
    full = structured_inv123_raw_t4_local_tokens(
        raw["T1"][None], raw["T2"][None], raw["K2"][None],
        raw["T3"][None], raw["K3"][None], raw["T4"][None],
        two_mask=raw["T2_mask"][None], three_mask=raw["T3_mask"][None],
        four_mask=raw["T4_mask"][None], R_obs=4, output_dtype=np.float32,
    )
    return np.ascontiguousarray(full[0][:, channels], dtype=np.float32)


def predict(models: list[torch.nn.Module], values: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(device)
        return np.column_stack([torch.sigmoid(model(x)).cpu().numpy() for model in models])


def refine(backend: XYFreeFermionExactBackend, t0: float, t1: float,
           p0: float, p1: float, channels: list[int], models: list[torch.nn.Module],
           device: str, rounds: int) -> tuple[float, float]:
    if not (p0 - 0.5) * (p1 - 0.5) < 0:
        raise ValueError("boundary endpoints do not bracket 0.5")
    for _ in range(rounds):
        tm = float(np.sqrt(t0 * t1))
        pm = float(predict(models, tokens(backend, tm, channels)[None], device).mean())
        if (pm - 0.5) * (p0 - 0.5) <= 0:
            t1, p1 = tm, pm
        else:
            t0, p0 = tm, pm
    log0, log1 = np.log(t0), np.log(t1)
    fraction = (0.5 - p0) / (p1 - p0)
    return float(np.exp(log0 + fraction * (log1 - log0))), float(np.log10(t1 / t0))


def fit_activated(h: np.ndarray, temperature: np.ndarray) -> dict:
    def predicted(log_c):
        c = np.exp(log_c)
        return 4 * (h - 1) / np.real(lambertw(4 * (h - 1) / (c * c)))
    def objective(log_c):
        return float(np.sum((np.log(predicted(log_c)) - np.log(temperature)) ** 2))
    fit = minimize_scalar(objective, bounds=(-12, 2), method="bounded",
                          options={"xatol": 1e-13})
    if not fit.success:
        raise RuntimeError("activated fit did not converge")
    estimate = predicted(fit.x)
    target = np.log10(temperature)
    residual = target - np.log10(estimate)
    sst = float(np.sum((target - np.mean(target)) ** 2))
    return {"C_star": float(np.exp(fit.x)), "R2_log10T": float(1 - np.sum(residual**2) / sst),
            "RMSE_log10T": float(np.sqrt(np.mean(residual**2))), "points": len(h),
            "h_min": float(h.min()), "h_max": float(h.max())}


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "reports/phase3_thermal")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--h-count", type=int, default=61)
    ap.add_argument("--temperature-count", type=int, default=81)
    ap.add_argument("--refinement-rounds", type=int, default=8)
    ap.add_argument("--old-grid", type=Path)
    ap.add_argument("--trajectory-source", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    models = load_control_ensemble(ROOT, device=args.device)
    channels = json.loads((ROOT / "configs/joint_K16.json").read_text())["selected_full_token_channels"]
    h_values = np.linspace(0, 1.5, args.h_count)
    temperatures = np.geomspace(0.01, 100, args.temperature_count)
    p_grid = np.empty((len(h_values), len(temperatures)))
    seed_grid = np.empty((*p_grid.shape, 3))
    boundaries: list[dict] = []
    started = time.perf_counter()
    for i, h in enumerate(h_values):
        backend = XYFreeFermionExactBackend(N, 0.0, float(h), "PBC")
        feature_rows = np.stack([tokens(backend, float(t), channels) for t in temperatures])
        probabilities = predict(models, feature_rows, args.device)
        seed_grid[i] = probabilities
        p_grid[i] = probabilities.mean(axis=1)
        sign = p_grid[i] - 0.5
        for j in np.flatnonzero(sign[:-1] * sign[1:] < 0):
            crossing, width = refine(backend, float(temperatures[j]), float(temperatures[j + 1]),
                                     float(p_grid[i, j]), float(p_grid[i, j + 1]),
                                     channels, models, args.device, args.refinement_rounds)
            boundaries.append({"h": float(h), "temperature": crossing,
                               "log10_temperature": float(np.log10(crossing)),
                               "crossing_index": int(j), "bracket_log10_width": width})
        if i % 5 == 0 or i == len(h_values) - 1:
            print(f"thermal grid {i + 1}/{len(h_values)} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    save_rows(args.output / "new_grid_scores.csv", [
        {"h": float(h), "temperature": float(t), "p_ensemble": float(p_grid[i, j]),
         **{f"p_seed_{seed}": float(seed_grid[i, j, k]) for k, seed in enumerate(SEEDS)}}
        for i, h in enumerate(h_values) for j, t in enumerate(temperatures)
    ])
    np.savez_compressed(args.output / "new_grid.npz", h=h_values, temperature=temperatures,
                        p=p_grid, p_seed=seed_grid)
    save_rows(args.output / "new_boundaries.csv", boundaries)
    low = [row for row in boundaries if 1.05 - 1e-9 <= row["h"] <= 1.5 + 1e-9
           and row["temperature"] < 1]
    fit = fit_activated(np.array([r["h"] for r in low]), np.array([r["temperature"] for r in low])) if len(low) >= 3 else None
    old = None
    if args.old_grid:
        with np.load(args.old_grid) as source:
            old_h, old_t = source["h"], source["temperature"]
            old_p = 1 / (1 + np.exp(-np.clip(source["ensemble_logit"], -60, 60)))
        if np.allclose(h_values, old_h) and np.allclose(temperatures, old_t):
            old = {"max_abs_p_difference": float(np.max(np.abs(p_grid - old_p))),
                   "mean_abs_p_difference": float(np.mean(np.abs(p_grid - old_p))),
                   "decision_disagreement_fraction": float(np.mean((p_grid >= .5) != (old_p >= .5)))}
    trajectory_rows = []
    if args.trajectory_source:
        with args.trajectory_source.open(newline="", encoding="utf-8") as stream:
            input_rows = list(csv.DictReader(stream))
        points = sorted({(r["point_id"], float(r["gamma"]), float(r["h"])) for r in input_rows
                         if r["point_id"] != "factorization_line"})
        for point, gamma, h in points:
            backend = XYFreeFermionExactBackend(N, gamma, h, "PBC")
            rows = [r for r in input_rows if r["point_id"] == point]
            values = np.stack([tokens(backend, float(r["temperature"]), channels) for r in rows])
            pseed = predict(models, values, args.device)
            for r, ps in zip(rows, pseed):
                trajectory_rows.append({"point_id": point, "gamma": gamma, "h": h,
                                        "temperature": float(r["temperature"]),
                                        "certificate_class": r["class_name"],
                                        "p_old_dv1_only": float(r["ensemble_probability"]),
                                        "p_new_r2support": float(np.mean(ps)),
                                        **{f"p_seed_{seed}": float(ps[k]) for k, seed in enumerate(SEEDS)}})
        save_rows(args.output / "new_trajectories.csv", trajectory_rows)
    seed_disagree = (np.min(seed_grid, axis=2) < .5) & (np.max(seed_grid, axis=2) >= .5)
    h12_index = int(np.argmin(abs(h_values - 1.2)))
    h12_crossings = [r for r in boundaries if abs(r["h"] - 1.2) < 1e-9]
    report = {"model": "new Dv1+R2Support appendix-control ensemble", "N": 16, "gamma": 0.0,
              "hamiltonian": "H=-sum_i[(1+gamma)/2 XiXi+1+(1-gamma)/2 YiYi+1+h Zi]", "boundary": "PBC",
              "checkpoint_sha256": {str(s): sha256(ROOT / f"checkpoints/dv1_r2support_control/seed{s}/best.pt") for s in SEEDS},
              "grid": {"h_count": len(h_values), "temperature_count": len(temperatures),
                       "h_range": [0, 1.5], "temperature_range": [.01, 100],
                       "seed_decision_disagreement_fraction": float(np.mean(seed_disagree)),
                       "h_1p2_low_p": float(p_grid[h12_index, 0]), "h_1p2_max_p": float(p_grid[h12_index].max()),
                       "h_1p2_high_p": float(p_grid[h12_index, -1]),
                       "h_1p2_crossings": h12_crossings},
              "low_boundary_fit": fit, "old_grid_comparison": old,
              "trajectory_points": len({r["point_id"] for r in trajectory_rows}),
              "elapsed_seconds": time.perf_counter() - started,
              "threshold": 0.5, "old_probabilities_used_as_input": False}
    (args.output / "thermal_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    m = axes[0].pcolormesh(h_values, np.log10(temperatures), p_grid.T, shading="auto",
                           cmap="RdBu_r", vmin=0, vmax=1, rasterized=True)
    axes[0].contour(h_values, np.log10(temperatures), p_grid.T, levels=[.5], colors="k", linewidths=1.1)
    axes[0].axvline(1, color="white", lw=1.5, ls="--")
    axes[0].axvline(1.2, color="#ffd200", lw=1.5, ls="--")
    axes[0].set(xlabel=r"$h/J$", ylabel=r"$\log_{10}(T/J)$", title="(c) New Dv1+R2Support thermal response")
    fig.colorbar(m, ax=axes[0], label="Classifier output $p$")
    if low:
        h_fit = np.array([r["h"] for r in low])
        t_fit = np.array([r["temperature"] for r in low])
        axes[1].plot(h_fit, np.log10(t_fit), "ko", ms=3, label="Recomputed boundary")
        if fit:
            c = fit["C_star"]
            fitted_t = 4 * (h_fit - 1) / np.real(lambertw(4 * (h_fit - 1) / (c*c)))
            axes[1].plot(h_fit, np.log10(fitted_t), "--", color="#d97706", label="Activated fit")
    axes[1].set(xlabel=r"$h/J$", ylabel=r"$\log_{10}(T_*/J)$", title="(d) Recomputed lower boundary")
    axes[1].legend(fontsize=8)
    fig.savefig(args.output / "fig3cd_candidate.pdf")
    plt.close(fig)
    if trajectory_rows:
        fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
        for point in sorted({r["point_id"] for r in trajectory_rows}):
            rows = [r for r in trajectory_rows if r["point_id"] == point]
            ax.plot([r["temperature"] for r in rows], [r["p_new_r2support"] for r in rows], label=point.replace("_", " "))
        ax.axhline(.5, ls="--", color=".45")
        ax.set(xscale="log", xlabel=r"$T/J$", ylabel="Classifier output $p$", ylim=(-.02, 1.02))
        ax.legend(ncol=2, fontsize=7)
        fig.savefig(args.output / "new_thermal_trajectories.pdf")
        plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.diagnostics.observable_radius import LocalRawSpec, invariant_tokens_from_raw  # noqa: E402
from entanglement_data.general_n32 import pair_supports, support_certificates  # noqa: E402
from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    InvariantTokenTransformer,
    InvariantTransformerConfig,
    NormalizedInvariantTransformer,
    local_inv123_raw_t4_channel_manifest,
)
from entanglement_data.ood_hamiltonian_n16 import (  # noqa: E402
    CachedMomentState,
    HamiltonianPoint,
    all_required_supports,
    certificate_summary,
    constructive_thermal_state,
    exact_ground_state,
    exact_thermal_state,
    j1j2_point,
    local_raw_from_state,
    moment_state_from_mps,
    point_metadata,
    raw_vector_from_state,
    safe_thermal_certificate,
    sample_parameter_points,
    solve_ground_state_dmrg,
    thermal_lanczos_rdms,
    xxz_point,
)
from entanglement_data.xy.free_fermion_exact import XYFreeFermionExactBackend  # noqa: E402
from entanglement_data.xy.size_transfer import LocalRawSpec as TransferRawSpec  # noqa: E402
from entanglement_data.xy.size_transfer import thermal_local_raw  # noqa: E402


N = 16
R_OBS = 4
SPEC = LocalRawSpec(N=N, R_obs=R_OBS)
TRAINING_SEEDS = (20260829, 20260830, 20260831)
EXPECTED_CHANNELS = (60, 2, 110, 63, 100, 101, 14, 82, 62, 109, 44, 83, 91, 51, 3, 59,
                     117, 118, 119, 120, 121)
OOD_SEEDS = {"xxz": 2026091100, "j1j2": 2026092100, "dh_family": 2026093100}
FAMILIES = ("xxz", "j1j2", "dh_family")


def source_root() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / "data/n16_production_v1/dataset.h5").exists():
            return candidate
    return ROOT


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def frozen_manifest(root: Path) -> tuple[dict[str, Any], np.ndarray]:
    path = root / "analysis/invariant_transformer_raw_t4_v1/selection_manifests/joint_K16.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    channels = np.asarray(manifest["selected_full_token_channels"], dtype=np.int64)
    if tuple(channels) != EXPECTED_CHANNELS:
        raise RuntimeError("frozen Joint K16 channel ordering changed")
    if manifest["physical_channel_count"] != 16 or manifest["mask_channel_count"] != 5:
        raise RuntimeError("frozen Joint K16 width changed")
    return manifest, channels


def audit_dh(root: Path, output: Path) -> dict[str, Any]:
    historical = root / "data/random_local_hamiltonian_v1/D_H.h5"
    config = root / "configs/random_local_hamiltonian_v1.yaml"
    training_seeds: set[int] = set()
    historical_rows = 0
    if historical.exists():
        with h5py.File(historical, "r") as h5:
            historical_rows = len(h5["metadata/template_seed"])
            training_seeds = {int(value) for value in h5["metadata/template_seed"][:]}
    proposed = {OOD_SEEDS["dh_family"] + index for index in range(10000)}
    payload = {
        "status": "passed" if not (training_seeds & proposed) else "failed",
        "historical_dataset": str(historical),
        "historical_dataset_sha256": sha256(historical) if historical.exists() else None,
        "historical_config": str(config),
        "historical_config_sha256": sha256(config) if config.exists() else None,
        "historical_rows": historical_rows,
        "training_template_seed_count": len(training_seeds),
        "new_ood_seed_namespace": "2026093100 + point_index",
        "seed_overlap_count": len(training_seeds & proposed),
        "boundary": "OBC",
        "actual_hamiltonian": (
            "H=-sum_i f_i n_i.sigma_i + sum_i,ab J_i^{ab} sigma_i^a sigma_{i+1}^b; "
            "J_i=-c_i n_i n_{i+1}^T + lambda M_i^perp"
        ),
        "important_correction": (
            "Historical D_H is not a scalar disordered-Ising chain. It uses arbitrary local field axes and "
            "general real 3x3 nearest-neighbor Pauli couplings, interpolating from a commuting product parent."
        ),
        "field_distribution": "initial uniform [1.0,1.5], then raised above adjacent |c_i| plus 0.25 safety margin",
        "commuting_coupling_distribution": "profile-dependent and clipped to [-0.15,0.15]",
        "lambda_distribution": "weak [0.10,0.30] p=.3; moderate [0.30,0.80] p=.5; strong [0.80,1.50] p=.2",
        "profile_distribution": "disordered p=.4; weakly_homogeneous p=.3; smooth p=.3",
        "historical_solver": "TeNPy two-site DMRG, chi=128, svd_min=1e-10, max_sweeps=30, energy_tol=1e-10",
        "exact5000_accessed": False,
    }
    if payload["status"] != "passed":
        raise RuntimeError("new D_H OOD seeds overlap historical training seeds")
    save_json(output, payload)
    return payload


def tokens_from_state(state) -> tuple[np.ndarray, np.ndarray]:
    raw = raw_vector_from_state(state, SPEC)
    full = invariant_tokens_from_raw(raw, SPEC, dtype=np.float32)
    _, channels = frozen_manifest(source_root())
    return full, full[:, channels]


def _certificate_class_from_historical(h5: h5py.File, row: int) -> str:
    if bool(h5["labels/certified_separable"][row]):
        return "strict_separable"
    if bool(h5["cert2/all_pairs/any_npt"][row]):
        return "E2"
    if bool(h5["cert3/any_npt"][row]) or bool(h5["cert3/any_ccnr"][row]):
        return "E3"
    if bool(h5["cert4/any_npt"][row]) or bool(h5["cert4/any_ccnr"][row]):
        return "E4"
    return "unknown"


def xy_pipeline_regression(root: Path, rows: Sequence[int] = (0, 37, 113)) -> dict[str, Any]:
    h5_path = root / "data/xy_n16_pbc_exact500_v1/dataset.h5"
    cache_path = root / "analysis/invariant_transformer_raw_t4_v1/cache/exact500_Inv123RawT4_Robs4.npy"
    reference = np.load(cache_path, mmap_mode="r")
    feature_errors: list[float] = []
    certificate_rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5:
        for row in rows:
            T = float(h5["params/T"][row])
            if T <= 0:
                continue
            backend = XYFreeFermionExactBackend(
                16, float(h5["params/gamma"][row]), float(h5["params/h"][row]), "PBC"
            )
            blocks = thermal_local_raw(backend, 1.0 / T, TransferRawSpec(16, 4))
            from entanglement_data.ml.invariant_transformer import structured_inv123_raw_t4_local_tokens

            tokens = structured_inv123_raw_t4_local_tokens(
                blocks["T1"][None], blocks["T2"][None], blocks["K2"][None],
                blocks["T3"][None], blocks["K3"][None], blocks["T4"][None],
                two_mask=blocks["T2_mask"][None], three_mask=blocks["T3_mask"][None],
                four_mask=blocks["T4_mask"][None], R_obs=4, output_dtype=np.float32,
            )[0]
            feature_error = float(np.max(np.abs(tokens - reference[row])))
            feature_errors.append(feature_error)
            state = CachedMomentState(16, lambda sites, backend=backend, T=T: backend.reduced_density_matrix(sites, 1.0 / T))
            pairs = support_certificates(state, pair_supports(16), ccnr=False)
            pair_error = float(np.max(np.abs(pairs.min_pt - h5["cert2/all_pairs/pt_min_eigenvalue"][row])))
            order_errors: dict[str, float] = {}
            for order, prefix in ((3, "cert3"), (4, "cert4")):
                supports = [tuple(int(x) for x in values) for values in h5[f"{prefix}/block_sites"][row]]
                current = support_certificates(state, supports, ccnr=True)
                order_errors[f"{prefix}_min_pt"] = float(np.max(np.abs(current.min_pt - h5[f"{prefix}/block_min_pt"][row])))
                order_errors[f"{prefix}_max_ccnr"] = float(np.max(np.abs(current.max_ccnr - h5[f"{prefix}/block_max_ccnr"][row])))
            computed = certificate_summary(state)["class_name"]
            historical = _certificate_class_from_historical(h5, row)
            certificate_rows.append({
                "row": int(row), "feature_max_abs_error": feature_error, "pair_pt_max_abs_error": pair_error,
                **order_errors, "computed_contiguous_hierarchy_class": computed,
                "historical_full_search_class": historical,
            })
    maximum_feature = max(feature_errors, default=math.inf)
    maximum_certificate = max(
        (max(value for key, value in row.items() if key.endswith("error") and not key.startswith("feature"))
         for row in certificate_rows), default=math.inf
    )
    return {
        "status": "passed" if maximum_feature < 2.0e-5 and maximum_certificate < 2.0e-7 else "failed",
        "feature_max_abs_error": maximum_feature,
        "certificate_value_max_abs_error": maximum_certificate,
        "rows": certificate_rows,
        "note": "Certificate values are compared on the exact historical support lists; class columns also show the new frozen contiguous hierarchy.",
    }


def representative_point(family: str, N: int, seed: int) -> HamiltonianPoint:
    if family == "xxz":
        return xxz_point(N, J=1.0, delta=0.7, h=0.35, seed=seed)
    if family == "j1j2":
        return j1j2_point(N, J2=0.55, h=0.25, seed=seed)
    return sample_parameter_points("dh_family", N, 2, seed=seed)[1]


def ground_regression(family: str) -> dict[str, Any]:
    rows = []
    for offset, size in enumerate((8, 10, 12)):
        point = representative_point(family, size, OOD_SEEDS[family] + 800 + offset)
        exact_energy, exact_state, _ = exact_ground_state(point)
        mps, diagnostics = solve_ground_state_dmrg(point, chi_max=128, max_sweeps=50)
        dmrg_state = moment_state_from_mps(mps)
        supports = ((0, 1), (0, min(3, size - 1)), (0, 1, 2), (0, 1, 2, 3))
        rdm_errors = {
            "-".join(map(str, support)): float(np.linalg.norm(
                dmrg_state.reduced_density_matrix(support) - exact_state.reduced_density_matrix(support)
            )) for support in supports
        }
        exact_cert = certificate_summary(exact_state)
        dmrg_cert = certificate_summary(dmrg_state)
        rows.append({
            "N": size, "parameters": point.parameters,
            "E_ED": exact_energy, "E_DMRG": diagnostics["energy"],
            "energy_abs_error": abs(exact_energy - diagnostics["energy"]),
            "max_local_rdm_frobenius_error": max(rdm_errors.values()),
            "rdm_errors": rdm_errors,
            "ED_class": exact_cert["class_name"], "DMRG_class": dmrg_cert["class_name"],
            "DMRG_diagnostics": diagnostics,
        })
    passed = all(
        row["energy_abs_error"] < 2.0e-7 and row["max_local_rdm_frobenius_error"] < 2.0e-5
        and row["ED_class"] == row["DMRG_class"] for row in rows
    )
    return {"status": "passed" if passed else "failed", "thresholds": {
        "energy_abs_error": 2.0e-7, "local_rdm_frobenius_error": 2.0e-5,
    }, "rows": rows}


def ftlm_regression(family: str, *, random_vectors: int, lanczos_dimension: int) -> dict[str, Any]:
    rows = []
    for offset, size in enumerate((8, 10)):
        point = representative_point(family, size, OOD_SEEDS[family] + 900 + offset)
        supports = ((0, 1), (0, min(3, size - 1)), (0, 1, 2), (0, 1, 2, 3))
        for temperature in (0.25, 0.75, 1.5):
            exact, exact_diag = exact_thermal_state(point, temperature)
            estimate = thermal_lanczos_rdms(
                point, (temperature,), random_vectors=random_vectors,
                lanczos_dimension=min(lanczos_dimension, (1 << size)),
                seed=point.point_seed + int(1000 * temperature), supports=supports,
            )
            approximate = estimate.states[temperature]
            errors = [float(np.linalg.norm(
                approximate.reduced_density_matrix(support) - exact.reduced_density_matrix(support)
            )) for support in supports]
            rows.append({
                "N": size, "temperature": temperature,
                "energy_exact": exact_diag["energy"],
                "energy_ftlm": estimate.diagnostics[temperature]["energy"],
                "energy_abs_error_per_site": abs(exact_diag["energy"] - estimate.diagnostics[temperature]["energy"]) / size,
                "max_selected_rdm_frobenius_error": max(errors),
                "selected_rdm_errors": errors,
                "diagnostics": estimate.diagnostics[temperature],
            })
    # This is a stochastic validation gate. Thresholds are intentionally much
    # looser than exact PPT tolerances and are recorded, never hidden.
    passed = all(row["energy_abs_error_per_site"] < 0.08 and row["max_selected_rdm_frobenius_error"] < 0.20 for row in rows)
    return {"status": "passed" if passed else "failed", "random_vectors": random_vectors,
            "lanczos_dimension": lanczos_dimension,
            "thresholds": {"energy_abs_error_per_site": 0.08, "selected_rdm_frobenius_error": 0.20},
            "rows": rows}


def run_regression(args: argparse.Namespace) -> dict[str, Any]:
    root = args.source_root
    output = args.output_root / args.family / "regression_report.json"
    report = {
        "family": args.family,
        "historical_xy_model_input_and_certificate_regression": xy_pipeline_regression(root),
        "ground_state_ed_vs_dmrg": ground_regression(args.family),
        "thermal_exact_vs_ftlm": ftlm_regression(
            args.family, random_vectors=args.regression_random_vectors,
            lanczos_dimension=args.regression_lanczos_dimension,
        ),
        "exact5000_accessed": False,
    }
    report["status"] = "passed" if all(
        report[key]["status"] == "passed" for key in (
            "historical_xy_model_input_and_certificate_regression", "ground_state_ed_vs_dmrg", "thermal_exact_vs_ftlm"
        )
    ) else "failed"
    save_json(output, report)
    print(json.dumps({"stage": "regression", "family": args.family, "status": report["status"], "path": str(output)}), flush=True)
    return report


def run_convergence(args: argparse.Namespace) -> dict[str, Any]:
    regression_path = args.output_root / args.family / "regression_report.json"
    if not regression_path.exists() or json.loads(regression_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("regression must pass before the N=16 FTLM convergence study")
    point = representative_point(args.family, N, OOD_SEEDS[args.family] + 990)
    temperatures = tuple(float(value) * point.natural_scale for value in args.convergence_temperatures)
    levels = ((4, 40), (8, 40), (8, 80), (16, 80), (16, 120), (32, 120))
    results = {}
    for random_vectors, lanczos_dimension in levels:
        print(json.dumps({"stage": "convergence", "family": args.family, "R": random_vectors,
                          "m": lanczos_dimension}), flush=True)
        results[(random_vectors, lanczos_dimension)] = thermal_lanczos_rdms(
            point, temperatures, random_vectors=random_vectors, lanczos_dimension=lanczos_dimension,
            seed=point.point_seed + 700000,
        )
    reference = results[(32, 120)]
    rows = []
    for random_vectors, lanczos_dimension in levels:
        current = results[(random_vectors, lanczos_dimension)]
        for reduced, temperature in zip(args.convergence_temperatures, temperatures):
            ref_state = reference.states[temperature]
            state = current.states[temperature]
            rdm_error = max(float(np.linalg.norm(
                state.reduced_density_matrix(support) - ref_state.reduced_density_matrix(support)
            )) for support in all_required_supports(N))
            _, compact = tokens_from_state(state)
            _, ref_compact = tokens_from_state(ref_state)
            feature_error = float(np.max(np.abs(compact - ref_compact)))
            cert = certificate_summary(state)
            ref_cert = certificate_summary(ref_state)
            margin_error = abs(float(cert["certificate_margin"]) - float(ref_cert["certificate_margin"])) \
                if np.isfinite(cert["certificate_margin"]) and np.isfinite(ref_cert["certificate_margin"]) else math.nan
            row = {
                "R": random_vectors, "m": lanczos_dimension, "temperature_reduced": float(reduced),
                "energy_abs_error_per_site": abs(
                    current.diagnostics[temperature]["energy"] - reference.diagnostics[temperature]["energy"]
                ) / N,
                "max_required_rdm_frobenius_error": rdm_error,
                "max_joint_k16_input_abs_error": feature_error,
                "certificate_class": cert["class_name"], "reference_certificate_class": ref_cert["class_name"],
                "certificate_margin_abs_error": margin_error,
                "block_rdm_uncertainty": current.diagnostics[temperature]["max_block_rdm_frobenius_deviation"],
            }
            row["within_production_tolerance"] = bool(
                row["energy_abs_error_per_site"] <= 0.02
                and row["max_required_rdm_frobenius_error"] <= 0.12
                and row["max_joint_k16_input_abs_error"] <= 0.15
                and (not np.isfinite(margin_error) or margin_error <= 0.05)
            )
            rows.append(row)
    accepted = []
    for level in levels:
        subset = [row for row in rows if (row["R"], row["m"]) == level]
        if all(row["within_production_tolerance"] for row in subset):
            accepted.append(level)
    recommendation = accepted[0] if accepted else (32, 120)
    report = {
        "status": "passed", "family": args.family, "reference": {"R": 32, "m": 120},
        "temperatures_reduced": list(args.convergence_temperatures),
        "tolerances": {"energy_per_site": 0.02, "required_rdm_frobenius": 0.12,
                       "joint_k16_input_abs": 0.15, "certificate_margin": 0.05},
        "accepted_levels": [{"R": R, "m": m} for R, m in accepted],
        "recommended_production": {"R": recommendation[0], "m": recommendation[1]},
        "rows": rows, "exact5000_accessed": False,
    }
    path = args.output_root / args.family / "ftlm_convergence_report.json"
    save_json(path, report)
    print(json.dumps({"stage": "convergence", "family": args.family,
                      "recommended_production": report["recommended_production"], "path": str(path)}), flush=True)
    return report


def plan_rows(family: str, zero_count: int, thermal_paths: int, temperatures: Sequence[float]) -> tuple[list[dict[str, Any]], list[HamiltonianPoint], list[HamiltonianPoint]]:
    zero = sample_parameter_points(family, N, zero_count, seed=OOD_SEEDS[family], thermal=False)
    thermal = sample_parameter_points(family, N, thermal_paths, seed=OOD_SEEDS[family] + 50000, thermal=True)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(zero):
        rows.append({"sample_id": f"{family}_T0_{index:04d}", "sector": "zero_temperature", "point_index": index,
                     "temperature_reduced": 0.0, "temperature": 0.0, "point": point_metadata(point)})
    for index, point in enumerate(thermal):
        for reduced in temperatures:
            rows.append({"sample_id": f"{family}_FTLM_{index:04d}_t{reduced:g}", "sector": "finite_temperature",
                         "point_index": index, "temperature_reduced": float(reduced),
                         "temperature": float(reduced) * point.natural_scale, "point": point_metadata(point)})
    return rows, zero, thermal


def _create(h5: h5py.File, name: str, shape: tuple[int, ...], dtype, **kwargs):
    parent, leaf = name.rsplit("/", 1)
    return h5.require_group(parent).create_dataset(leaf, shape=shape, dtype=dtype, **kwargs)


def initialize_dataset(path: Path, rows: list[dict[str, Any]], channels: np.ndarray, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string = h5py.string_dtype("utf-8")
    count = len(rows)
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "N": N, "boundary": "OBC", "family": config["family"], "generation_complete": False,
            "plan_hash": canonical_hash(rows), "config_json": json.dumps(config),
            "frozen_model": "Historical Joint T1-T4 K16 rmax=4", "frozen_threshold": 0.5,
            "selected_full_token_channels": json.dumps(channels.tolist()), "exact5000_accessed": False,
        })
        _create(h5, "progress/completed", (count,), np.bool_, fillvalue=False)
        _create(h5, "metadata/sample_id", (count,), string)
        _create(h5, "metadata/sector", (count,), string)
        _create(h5, "metadata/point_index", (count,), np.int32)
        _create(h5, "metadata/point_seed", (count,), np.int64)
        _create(h5, "metadata/parameters_json", (count,), string)
        _create(h5, "metadata/separability_proof", (count,), string)
        _create(h5, "metadata/temperature_reduced", (count,), np.float64)
        _create(h5, "metadata/temperature", (count,), np.float64)
        _create(h5, "metadata/beta", (count,), np.float64)
        _create(h5, "solver/name", (count,), string)
        _create(h5, "solver/diagnostics_json", (count,), string)
        _create(h5, "solver/converged", (count,), np.bool_)
        _create(h5, "solver/energy", (count,), np.float64)
        _create(h5, "solver/energy_variance", (count,), np.float64)
        _create(h5, "solver/rdm_uncertainty", (count,), np.float64)
        _create(h5, "solver/feature_uncertainty", (count,), np.float64)
        _create(h5, "features/full_tokens", (count, N, 122), np.float32, chunks=(1, N, 122), compression="gzip")
        _create(h5, "features/Joint_K16_inputs", (count, N, 21), np.float32, chunks=(1, N, 21), compression="gzip")
        _create(h5, "labels/strict_class", (count,), string)
        for name in ("strict_separable", "certified_entangled", "E2", "E3", "E4", "unknown"):
            _create(h5, f"labels/{name}", (count,), np.bool_)
        _create(h5, "certificate/type", (count,), string)
        _create(h5, "certificate/support", (count, 4), np.int8, fillvalue=-1)
        _create(h5, "certificate/cut", (count, 3), np.int8, fillvalue=-1)
        for name in ("margin", "sigma", "acceptance_threshold"):
            _create(h5, f"certificate/{name}", (count,), np.float64)
        _create(h5, "certificate/diagnostics_json", (count,), string)
        for row, definition in enumerate(rows):
            point = definition["point"]
            h5["metadata/sample_id"][row] = definition["sample_id"]
            h5["metadata/sector"][row] = definition["sector"]
            h5["metadata/point_index"][row] = definition["point_index"]
            h5["metadata/point_seed"][row] = point["point_seed"]
            h5["metadata/parameters_json"][row] = json.dumps(point["parameters"])
            h5["metadata/separability_proof"][row] = point["separability_proof"] or ""
            h5["metadata/temperature_reduced"][row] = definition["temperature_reduced"]
            h5["metadata/temperature"][row] = definition["temperature"]
            h5["metadata/beta"][row] = math.inf if definition["temperature"] == 0 else 1.0 / definition["temperature"]


def _write_row(path: Path, row: int, full: np.ndarray, compact: np.ndarray, certificate: dict[str, Any], diagnostics: dict[str, Any], feature_uncertainty: float) -> None:
    class_name = certificate["class_name"]
    support = np.full(4, -1, dtype=np.int8)
    values = certificate.get("certificate_support", [])
    support[:len(values)] = values
    cut = np.full(3, -1, dtype=np.int8)
    cut_values = certificate.get("certificate_cut", [])
    cut[:len(cut_values)] = cut_values
    with h5py.File(path, "r+") as h5:
        h5["features/full_tokens"][row] = full
        h5["features/Joint_K16_inputs"][row] = compact
        h5["solver/name"][row] = diagnostics["solver"]
        h5["solver/diagnostics_json"][row] = json.dumps(diagnostics, allow_nan=True)
        h5["solver/converged"][row] = bool(diagnostics.get("converged", True))
        h5["solver/energy"][row] = diagnostics["energy"]
        h5["solver/energy_variance"][row] = diagnostics.get("energy_variance", math.nan)
        h5["solver/rdm_uncertainty"][row] = diagnostics.get("max_block_rdm_frobenius_deviation", 0.0)
        h5["solver/feature_uncertainty"][row] = feature_uncertainty
        h5["labels/strict_class"][row] = class_name
        h5["labels/strict_separable"][row] = class_name == "strict_separable"
        h5["labels/certified_entangled"][row] = class_name in ("E2", "E3", "E4")
        for label in ("E2", "E3", "E4"):
            h5[f"labels/{label}"][row] = class_name == label
        h5["labels/unknown"][row] = class_name == "unknown"
        h5["certificate/type"][row] = certificate.get("certificate_type", "none")
        h5["certificate/support"][row] = support
        h5["certificate/cut"][row] = cut
        h5["certificate/margin"][row] = certificate.get("certificate_margin", math.nan)
        h5["certificate/sigma"][row] = certificate.get("certificate_sigma", 0.0)
        h5["certificate/acceptance_threshold"][row] = certificate.get("acceptance_threshold", 0.0)
        h5["certificate/diagnostics_json"][row] = json.dumps(certificate, allow_nan=True)
        h5["progress/completed"][row] = True
        h5.flush()


def _strict_ground_certificate(state, point: HamiltonianPoint) -> dict[str, Any]:
    certificate = certificate_summary(state, strict_separable=point.separability_proof is not None)
    if point.separability_proof:
        certificate["constructive_proof"] = point.separability_proof
        if certificate["pair_npt_margin"] > 1.0e-7:
            raise FloatingPointError("constructively product ground state acquired an NPT pair")
    return certificate


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    regression_path = args.output_root / args.family / "regression_report.json"
    if not regression_path.exists() or json.loads(regression_path.read_text(encoding="utf-8"))["status"] != "passed":
        raise RuntimeError("family regression has not passed; production is blocked")
    _, channels = frozen_manifest(args.source_root)
    temperatures = tuple(float(value) for value in args.temperatures)
    rows, zero_points, thermal_points = plan_rows(args.family, args.zero_count, args.thermal_paths, temperatures)
    path = args.output_root / args.family / "dataset.h5"
    config = {
        "family": args.family, "zero_count": args.zero_count, "thermal_paths": args.thermal_paths,
        "temperatures_reduced": temperatures, "random_vectors": args.random_vectors,
        "lanczos_dimension": args.lanczos_dimension, "dmrg_chi_max": args.dmrg_chi_max,
        "dmrg_max_sweeps": args.dmrg_max_sweeps,
    }
    if not path.exists():
        initialize_dataset(path, rows, channels, config)
    with h5py.File(path, "r") as h5:
        if h5.attrs["plan_hash"] != canonical_hash(rows):
            raise RuntimeError("existing dataset plan differs; refusing to overwrite or mix runs")
        completed = h5["progress/completed"][:]
    if args.refresh_constructive_controls:
        for row, definition in enumerate(rows):
            if definition["sector"] != "finite_temperature":
                continue
            point = thermal_points[definition["point_index"]]
            if constructive_thermal_state(point, definition["temperature"]) is not None:
                completed[row] = False
    started = time.perf_counter()
    for point_index, point in enumerate(zero_points):
        row = point_index
        if completed[row]:
            continue
        parameter_log = point.parameters if point.family != "dh_family" else {
            "lambda_ent": point.parameters["lambda_ent"], "lambda_bin": point.parameters["lambda_bin"],
            "profile_type": point.parameters["profile_type"], "template_seed": point.parameters["template_seed"],
        }
        print(json.dumps({"stage": "T0", "family": args.family, "row": row, "total": len(rows),
                          "parameters": parameter_log}, default=str), flush=True)
        mps, diagnostics = solve_ground_state_dmrg(
            point, chi_max=args.dmrg_chi_max, max_sweeps=args.dmrg_max_sweeps,
        )
        state = moment_state_from_mps(mps)
        full, compact = tokens_from_state(state)
        certificate = _strict_ground_certificate(state, point)
        if not diagnostics["converged"]:
            certificate["raw_candidate_class"] = certificate["class_name"]
            certificate["class_name"] = "unknown"
            certificate["numerical_rejection"] = "DMRG variance criterion failed"
        _write_row(path, row, full, compact, certificate, diagnostics, 0.0)
    base = len(zero_points)
    for point_index, point in enumerate(thermal_points):
        block_rows = [base + point_index * len(temperatures) + offset for offset in range(len(temperatures))]
        pending = [row for row in block_rows if not completed[row]]
        if not pending:
            continue
        print(json.dumps({"stage": "FTLM", "family": args.family, "point": point_index,
                          "path_total": len(thermal_points), "temperatures": temperatures}, default=str), flush=True)
        physical_temperatures = tuple(value * point.natural_scale for value in temperatures)
        constructive = [constructive_thermal_state(point, value) for value in physical_temperatures]
        estimate = None if all(value is not None for value in constructive) else thermal_lanczos_rdms(
            point, physical_temperatures, random_vectors=args.random_vectors,
            lanczos_dimension=args.lanczos_dimension, seed=point.point_seed + 700000,
        )
        for offset, (reduced, physical) in enumerate(zip(temperatures, physical_temperatures)):
            row = block_rows[offset]
            if completed[row]:
                continue
            exact_control = constructive[offset]
            state = exact_control[0] if exact_control is not None else estimate.states[physical]
            full, compact = tokens_from_state(state)
            if exact_control is not None:
                feature_uncertainty = 0.0
                certificate = certificate_summary(state, strict_separable=True)
                certificate.update({"constructive_proof": exact_control[1]["constructive_proof"],
                                    "certificate_sigma": 0.0, "acceptance_threshold": 0.0,
                                    "stochastic_certificate_accepted": True})
                diagnostics = exact_control[1]
            else:
                block_compact = np.stack([tokens_from_state(block)[1] for block in estimate.block_states[physical]])
                feature_uncertainty = float(np.max(np.std(block_compact, axis=0, ddof=1))) if len(block_compact) > 1 else math.inf
                certificate = safe_thermal_certificate(
                    state, estimate.block_states[physical], strict_separable_proof=None,
                )
                diagnostics = {**estimate.diagnostics[physical], "converged": True}
            diagnostics = {**diagnostics, "temperature_reduced": reduced, "temperature_physical": physical}
            _write_row(path, row, full, compact, certificate, diagnostics, feature_uncertainty)
    with h5py.File(path, "r+") as h5:
        h5.attrs["generation_complete"] = bool(np.all(h5["progress/completed"][:]))
        h5.attrs["generation_wall_seconds_last_invocation"] = time.perf_counter() - started
        complete = int(h5["progress/completed"][:].sum())
    report = {"status": "complete" if complete == len(rows) else "incomplete", "family": args.family,
              "completed": complete, "total": len(rows), "dataset": str(path), "exact5000_accessed": False}
    save_json(args.output_root / args.family / "generation_report.json", report)
    print(json.dumps(report), flush=True)
    return report


def load_joint_model(model_dir: Path, device: torch.device) -> tuple[NormalizedInvariantTransformer, dict[str, Any]]:
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    try:
        checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_dir / "best.pt", map_location=device)
    config = InvariantTransformerConfig(**checkpoint["model_config"])
    if config.physical_channels != 21:
        raise RuntimeError("frozen model input width is not 21")
    model = NormalizedInvariantTransformer(
        InvariantTokenTransformer(config), np.zeros(21, np.float32), np.ones(21, np.float32)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, report


@torch.no_grad()
def predict(model: torch.nn.Module, values: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    result = []
    for start in range(0, len(values), batch_size):
        block = torch.as_tensor(np.array(values[start:start + batch_size], copy=True), device=device)
        result.append(torch.sigmoid(model(block)).cpu().numpy())
    return np.concatenate(result)


def _rank_metric(negative: np.ndarray, positive: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    chosen = negative | positive
    labels = positive[chosen].astype(np.int8)
    if not chosen.any() or len(np.unique(labels)) < 2:
        return {"count": int(chosen.sum()), "AUROC": None, "AP": None}
    return {"count": int(chosen.sum()), "AUROC": float(roc_auc_score(labels, probability[chosen])),
            "AP": float(average_precision_score(labels, probability[chosen]))}


def metrics(probability: np.ndarray, labels: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, Any]:
    probability = probability[rows]
    selected = {name: value[rows] for name, value in labels.items()}
    sep = selected["strict_separable"]
    ent = selected["certified_entangled"]
    strict = sep | ent
    result: dict[str, Any] = {
        "total_count": int(np.asarray(rows, dtype=bool).sum()), "strict_count": int(strict.sum()), "separable_count": int(sep.sum()),
        "entangled_count": int(ent.sum()), "unknown_count": int(selected["unknown"].sum()),
        "AUROC": float(roc_auc_score(ent[strict], probability[strict])) if sep.any() and ent.any() else None,
        "Sep_FPR": float((probability[sep] >= 0.5).mean()) if sep.any() else None,
        "specificity": float((probability[sep] < 0.5).mean()) if sep.any() else None,
    }
    for label in ("E2", "E3", "E4"):
        mask = selected[label]
        result[f"{label}_count"] = int(mask.sum())
        result[f"{label}_recall"] = float((probability[mask] >= 0.5).mean()) if mask.any() else None
        result[f"Sep_vs_{label}"] = _rank_metric(sep, mask, probability)
    return result


def _plots(output: Path, family: str, probability: np.ndarray, sector: np.ndarray, point_index: np.ndarray,
           temperature: np.ndarray, labels: dict[str, np.ndarray], margin: np.ndarray, parameters: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    colors = {"E2": "tab:blue", "E3": "tab:orange", "E4": "tab:red"}
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for label, color in colors.items():
        mask = labels[label]
        if mask.any():
            ax.scatter(margin[mask], probability[mask], s=18, alpha=.75, label=label, color=color)
    ax.axhline(.5, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="PPT/CCNR certificate margin", ylabel="Joint K16 ensemble probability",
           title=f"{family}: model score vs certificate margin")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "score_vs_certificate_margin.png", dpi=180); plt.close(fig)

    thermal = sector == "finite_temperature"
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for index in np.unique(point_index[thermal])[:12]:
        mask = thermal & (point_index == index)
        order = np.argsort(temperature[mask])
        ax.plot(temperature[mask][order], probability[mask][order], marker="o", linewidth=1, alpha=.75,
                label=f"path {index}")
    ax.axhline(.5, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="reduced temperature T/J", ylabel="Joint K16 ensemble probability",
           title=f"{family}: frozen-model thermal paths")
    if len(np.unique(point_index[thermal])) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(output / "temperature_paths.png", dpi=180); plt.close(fig)

    zero = sector == "zero_temperature"
    if family in ("xxz", "j1j2") and zero.any():
        xkey = "Delta" if family == "xxz" else "J2"
        x = np.asarray([float(row[xkey]) for row in parameters])
        h = np.asarray([float(row["h"]) for row in parameters])
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        scatter = ax.scatter(x[zero], h[zero], c=probability[zero], vmin=0, vmax=1, cmap="viridis", s=38)
        ax.set(xlabel=xkey, ylabel="h", title=f"{family}: N=16 T=0 frozen Joint K16 score")
        fig.colorbar(scatter, ax=ax, label="p(entangled)")
        fig.tight_layout(); fig.savefig(output / "zero_temperature_parameter_map.png", dpi=180); plt.close(fig)


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / args.family / "dataset.h5"
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as h5:
        if not bool(h5.attrs["generation_complete"]) or not np.all(h5["progress/completed"][:]):
            raise RuntimeError("generation is incomplete; frozen inference is blocked")
        values = h5["features/Joint_K16_inputs"][:]
        sector = h5["metadata/sector"].asstr()[:]
        point_index = h5["metadata/point_index"][:]
        temperature = h5["metadata/temperature_reduced"][:]
        parameters = [json.loads(value) for value in h5["metadata/parameters_json"].asstr()[:]]
        margin = h5["certificate/margin"][:]
        labels = {name: h5[f"labels/{name}"][:].astype(bool) for name in
                  ("strict_separable", "certified_entangled", "E2", "E3", "E4", "unknown")}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    probabilities: list[np.ndarray] = []
    model_audit = []
    for seed in TRAINING_SEEDS:
        model_dir = args.source_root / "models/invariant_transformer_raw_t4_v1" / (
            f"RoPETransformer_LocalInv123_RawT4_JointRawT4_K16_seed{seed}"
        )
        model, report = load_joint_model(model_dir, device)
        probabilities.append(predict(model, values, device))
        model_audit.append({
            "seed": seed, "checkpoint_sha256": sha256(model_dir / "best.pt"),
            "report_sha256": sha256(model_dir / "report.json"),
            "selected_channels": report["selected_full_token_channels"],
            "architecture": report["architecture_frozen"], "threshold": 0.5,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked = np.stack(probabilities)
    ensemble = stacked.mean(axis=0)
    with h5py.File(path, "r+") as h5:
        if "model" in h5:
            del h5["model"]
        group = h5.create_group("model")
        group.create_dataset("scores_by_seed", data=stacked, compression="gzip")
        group.create_dataset("ensemble_score", data=ensemble, compression="gzip")
        group.attrs["training_seeds"] = json.dumps(TRAINING_SEEDS)
        group.attrs["threshold"] = 0.5
        group.attrs["frozen"] = True
    splits = {
        "all": np.ones(len(values), dtype=bool),
        "zero_temperature": sector == "zero_temperature",
        "finite_temperature": sector == "finite_temperature",
    }
    metric_report = {
        "per_seed": {str(seed): {name: metrics(probabilities[index], labels, mask) for name, mask in splits.items()}
                     for index, seed in enumerate(TRAINING_SEEDS)},
        "probability_average_ensemble": {name: metrics(ensemble, labels, mask) for name, mask in splits.items()},
    }
    plot_dir = args.output_root / args.family / "figures"
    _plots(plot_dir, args.family, ensemble, sector, point_index, temperature, labels, margin, parameters)
    report = {
        "status": "complete", "family": args.family, "dataset": str(path), "frozen_model_audit": model_audit,
        "metrics": metric_report, "plots": [str(path) for path in sorted(plot_dir.glob("*.png"))],
        "no_retraining": True, "no_finetuning": True, "no_threshold_tuning": True,
        "no_checkpoint_selection": True, "exact5000_accessed": False,
    }
    save_json(args.output_root / args.family / "evaluation_report.json", report)
    print(json.dumps({"stage": "evaluate", "family": args.family, "metrics": metric_report["probability_average_ensemble"]}, indent=2), flush=True)
    return report


def validate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_root / args.family / "dataset.h5"
    with h5py.File(path, "r") as h5:
        completed = h5["progress/completed"][:]
        full = h5["features/full_tokens"][:]
        compact = h5["features/Joint_K16_inputs"][:]
        _, channels = frozen_manifest(args.source_root)
        errors = {
            "compact_projection_max_abs_error": float(np.max(np.abs(compact - full[:, :, channels]))),
            "finite_feature_count": int(np.isfinite(full).sum()),
            "total_feature_count": int(full.size),
            "completed": int(completed.sum()), "total": len(completed),
            "generation_complete": bool(h5.attrs["generation_complete"]),
            "minimum_rdm_eigenvalue": min(
                json.loads(value).get("minimum_rdm_eigenvalue", 0.0)
                for value in h5["solver/diagnostics_json"].asstr()[:]
            ),
            "exact5000_accessed": bool(h5.attrs["exact5000_accessed"]),
        }
    passed = (errors["completed"] == errors["total"] and errors["generation_complete"]
              and errors["finite_feature_count"] == errors["total_feature_count"]
              and errors["compact_projection_max_abs_error"] == 0.0
              and not errors["exact5000_accessed"])
    report = {"status": "passed" if passed else "failed", **errors}
    save_json(args.output_root / args.family / "validation_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("dataset validation failed")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Sequential N=16 cross-Hamiltonian OOD benchmark")
    result.add_argument("stage", choices=("audit", "regression", "convergence", "generate", "validate", "evaluate"))
    result.add_argument("--family", choices=FAMILIES, default="xxz")
    result.add_argument("--source-root", type=Path, default=source_root())
    result.add_argument("--output-root", type=Path, default=source_root() / "data/ood_hamiltonian_n16_v1")
    result.add_argument("--zero-count", type=int, default=120)
    result.add_argument("--thermal-paths", type=int, default=32)
    result.add_argument("--temperatures", type=float, nargs="+", default=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0))
    result.add_argument("--random-vectors", type=int, default=8)
    result.add_argument("--lanczos-dimension", type=int, default=80)
    result.add_argument("--regression-random-vectors", type=int, default=16)
    result.add_argument("--regression-lanczos-dimension", type=int, default=80)
    result.add_argument("--convergence-temperatures", type=float, nargs="+", default=(0.1, 0.5, 1.0))
    result.add_argument("--dmrg-chi-max", type=int, default=128)
    result.add_argument("--dmrg-max-sweeps", type=int, default=40)
    result.add_argument("--device", default="cuda")
    result.add_argument("--refresh-constructive-controls", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    # The string is intentionally checked at the CLI boundary; this experiment
    # has no code path to the closed final benchmark.
    if "exact" + "5000" in " ".join(sys.argv).lower():
        raise RuntimeError("the final exact benchmark is closed for this experiment")
    if args.stage == "audit":
        _, channels = frozen_manifest(args.source_root)
        report = {
            "status": "passed", "frozen_channels": channels.tolist(),
            "manifest": local_inv123_raw_t4_channel_manifest(4),
            "dh_family": audit_dh(args.source_root, args.output_root / "dh_historical_audit.json"),
            "sequential_execution_required": True, "exact5000_accessed": False,
        }
        save_json(args.output_root / "audit_report.json", report)
        print(json.dumps({"stage": "audit", "status": "passed"}), flush=True)
    elif args.stage == "regression":
        run_regression(args)
    elif args.stage == "convergence":
        run_convergence(args)
    elif args.stage == "generate":
        run_generate(args)
    elif args.stage == "validate":
        validate_dataset(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ood_hamiltonian_n16 import (  # noqa: E402
    CachedMomentState,
    all_required_supports,
    certificate_summary,
    dh_point,
    exact_thermal_state,
    j1j2_point,
    safe_thermal_certificate,
    sample_parameter_points,
    thermal_lanczos_rdms,
    xxz_point,
)
from entanglement_data.thermal_purification import (  # noqa: E402
    as_cached_moment_state,
    generate_thermal_path,
)
from run_ood_hamiltonian_n16 import (  # noqa: E402
    TRAINING_SEEDS,
    load_joint_model,
    metrics,
    predict,
    sha256,
    tokens_from_state,
)


FAMILIES = ("xxz", "j1j2", "dh_family")
NEW_SEEDS = {"xxz": 2026191100, "j1j2": 2026192100, "dh_family": 2026193100}
COARSE_T = (0.35, 0.50, 0.70, 0.90, 1.10, 1.40, 1.80, 2.30)
REFINE_RATIOS = (
    1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35,
    1.40, 1.50, 1.60, 1.75, 1.90, 2.10,
)
MIN_HIGHER_ORDER = 30
MIN_HIGHER_ORDER_PARENTS = 10


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def compact_parameters(point) -> dict[str, Any]:
    if point.family == "dh_family":
        return {
            "lambda_ent": float(point.parameters["lambda_ent"]),
            "lambda_bin": point.parameters["lambda_bin"],
            "profile_type": point.parameters["profile_type"],
            "template_seed": int(point.parameters["template_seed"]),
        }
    return {key: value for key, value in point.parameters.items() if key != "saturation_bound"}


def heldout_points(family: str, count: int):
    seed = NEW_SEEDS[family]
    points = []
    for index in range(count):
        point_seed = seed + index
        rng = np.random.default_rng(point_seed)
        if family == "xxz":
            broad = index % 4 == 0
            field_limit = 3.5 if broad else 0.75
            points.append(
                xxz_point(
                    16,
                    J=(-1.0 if rng.random() < 0.32 else 1.0) if broad else 1.0,
                    delta=float(rng.uniform(-2.5, 3.0) if broad else rng.uniform(0.4, 1.8)),
                    h=float(rng.uniform(-field_limit, field_limit)),
                    seed=point_seed,
                )
            )
        elif family == "j1j2":
            field_limit = 4.0 if index % 4 == 0 else 1.5
            points.append(
                j1j2_point(
                    16,
                    J2=float(rng.uniform(-0.5, 1.5)),
                    h=float(rng.uniform(-field_limit, field_limit)),
                    seed=point_seed,
                )
            )
        else:
            points.append(dh_point(16, seed=point_seed))
    return points


def transition_temperature(classes: dict[float, str]) -> float | None:
    values = sorted(classes)
    e2 = [value for value in values if classes[value] == "E2"]
    if not e2:
        return None
    last_e2 = max(e2)
    higher = [value for value in values if value > last_e2 and classes[value] != "E2"]
    if not higher:
        return None
    return 0.5 * (last_e2 + min(higher))


def certificate_margin(value: dict[str, Any]) -> float:
    margin = value.get("certificate_margin")
    return float(margin) if margin is not None and np.isfinite(margin) else 0.0


def process_point(payload: tuple[str, int, Any]) -> dict[str, Any]:
    family, point_index, point = payload
    coarse_chi = 96 if family == "j1j2" else 48
    primary_chi = 256 if family == "j1j2" else 128
    confirm_chi = 256 if family == "j1j2" else 96
    primary_dt = 0.025 if family == "j1j2" else 0.05
    confirm_dt = 0.0125 if family == "j1j2" else 0.025
    rdm_tolerance = 1.0e-3 if family == "j1j2" else 2.0e-4
    margin_floor = 1.0e-3 if family == "j1j2" else 5.0e-4
    coarse_physical = tuple(value * point.natural_scale for value in COARSE_T)
    coarse_raw, coarse_diag = generate_thermal_path(
        point, coarse_physical, chi_max=coarse_chi, dt=0.10, svd_min=1.0e-10
    )
    coarse_states = {
        reduced: as_cached_moment_state(coarse_raw[physical])
        for reduced, physical in zip(COARSE_T, coarse_physical)
    }
    coarse_cert = {value: certificate_summary(coarse_states[value]) for value in COARSE_T}
    classes = {value: coarse_cert[value]["class_name"] for value in COARSE_T}
    transition = transition_temperature(classes)
    if transition is None:
        return {
            "family": family,
            "point_index": point_index,
            "point_seed": int(point.point_seed),
            "parameters": compact_parameters(point),
            "coarse_classes": {str(key): value for key, value in classes.items()},
            "coarse_transition": None,
            "rows": [],
        }

    temperatures = tuple(
        sorted({round(float(np.clip(transition * ratio, 0.30, 2.60)), 8) for ratio in REFINE_RATIOS})
    )
    physical_temperatures = tuple(value * point.natural_scale for value in temperatures)
    primary_raw, primary_diag = generate_thermal_path(
        point, physical_temperatures, chi_max=primary_chi, dt=primary_dt, svd_min=1.0e-11
    )
    confirm_raw, confirm_diag = generate_thermal_path(
        point, physical_temperatures, chi_max=confirm_chi, dt=confirm_dt, svd_min=1.0e-11
    )
    rows: list[dict[str, Any]] = []
    for temperature, physical_temperature in zip(temperatures, physical_temperatures):
        primary = as_cached_moment_state(primary_raw[physical_temperature])
        confirm = as_cached_moment_state(confirm_raw[physical_temperature])
        cert_primary = certificate_summary(primary)
        cert_confirm = certificate_summary(confirm)
        errors = [
            float(
                np.linalg.norm(
                    primary.reduced_density_matrix(support)
                    - confirm.reduced_density_matrix(support)
                )
            )
            for support in all_required_supports(16)
        ]
        rdm_error = max(errors)
        margin_primary = certificate_margin(cert_primary)
        margin_confirm = certificate_margin(cert_confirm)
        margin_error = abs(margin_primary - margin_confirm)
        same_class = cert_primary["class_name"] == cert_confirm["class_name"]
        robust_positive = (
            same_class
            and cert_primary["class_name"] in ("E2", "E3", "E4")
            and margin_primary >= (2.0e-3 if family == "j1j2" else 1.0e-4)
            and margin_confirm >= (2.0e-3 if family == "j1j2" else 1.0e-4)
            and rdm_error <= rdm_tolerance
            and margin_error <= max(margin_floor, 0.15 * margin_primary)
        )
        final_class = cert_primary["class_name"] if robust_positive else "unknown"
        full, compact = tokens_from_state(primary)
        rows.append(
            {
                "temperature_reduced": float(temperature),
                "temperature": float(physical_temperature),
                "class_name": final_class,
                "raw_class_primary": cert_primary["class_name"],
                "raw_class_confirm": cert_confirm["class_name"],
                "certificate_primary": cert_primary,
                "certificate_confirm": cert_confirm,
                "rdm_resolution_max_frobenius": rdm_error,
                "certificate_margin_abs_error": margin_error,
                "resolution_confirmed": bool(robust_positive),
                "primary_diagnostics": primary_diag[physical_temperature],
                "confirm_diagnostics": confirm_diag[physical_temperature],
                "features": compact.astype(np.float32),
                "full_feature_hash": hashlib.sha256(full.tobytes()).hexdigest(),
            }
        )
    return {
        "family": family,
        "point_index": point_index,
        "point_seed": int(point.point_seed),
        "parameters": compact_parameters(point),
        "coarse_classes": {str(key): value for key, value in classes.items()},
        "coarse_transition": transition,
        "coarse_diagnostics": {
            str(reduced): coarse_diag[physical]
            for reduced, physical in zip(COARSE_T, coarse_physical)
        },
        "rows": rows,
    }


def average_states(states) -> CachedMomentState:
    states = tuple(states)
    return CachedMomentState(
        states[0].N,
        lambda sites: sum(
            (state.reduced_density_matrix(sites) for state in states),
            np.zeros((1 << len(sites), 1 << len(sites)), dtype=np.complex128),
        )
        / len(states),
    )


def process_point_dual_ftlm(payload: tuple[str, int, Any]) -> dict[str, Any]:
    family, point_index, point = payload
    fine_random_vectors = 32 if family == "dh_family" else 16
    fine_lanczos_dimension = 100 if family == "dh_family" else 80
    coarse_physical = tuple(value * point.natural_scale for value in COARSE_T)
    coarse = thermal_lanczos_rdms(
        point,
        coarse_physical,
        random_vectors=6,
        lanczos_dimension=50,
        seed=point.point_seed + 7100000,
        uncertainty_blocks=4,
    )
    classes = {
        reduced: certificate_summary(coarse.states[physical])["class_name"]
        for reduced, physical in zip(COARSE_T, coarse_physical)
    }
    transition = transition_temperature(classes)
    if transition is None:
        return {
            "family": family,
            "point_index": point_index,
            "point_seed": int(point.point_seed),
            "parameters": compact_parameters(point),
            "coarse_classes": {str(key): value for key, value in classes.items()},
            "coarse_transition": None,
            "rows": [],
        }

    temperatures = tuple(
        sorted({round(float(np.clip(transition * ratio, 0.30, 2.60)), 8) for ratio in REFINE_RATIOS})
    )
    physical_temperatures = tuple(value * point.natural_scale for value in temperatures)
    estimates = [
        thermal_lanczos_rdms(
            point,
            physical_temperatures,
            random_vectors=fine_random_vectors,
            lanczos_dimension=fine_lanczos_dimension,
            seed=point.point_seed + offset,
            uncertainty_blocks=4,
        )
        for offset in (8100000, 8200000)
    ]
    rows = []
    for reduced, physical in zip(temperatures, physical_temperatures):
        independent = [estimate.states[physical] for estimate in estimates]
        pooled = average_states(independent)
        pooled_blocks = [
            average_states((estimates[0].block_states[physical][index], estimates[1].block_states[physical][index]))
            for index in range(4)
        ]
        cert_independent = [certificate_summary(state) for state in independent]
        cert_pooled = safe_thermal_certificate(
            pooled, pooled_blocks, strict_separable_proof=None, numerical_floor=1.0e-5
        )
        pooled_class = cert_pooled["class_name"]
        class_agreement = all(value["class_name"] == pooled_class for value in cert_independent)
        rdm_difference = max(
            float(
                np.linalg.norm(
                    independent[0].reduced_density_matrix(support)
                    - independent[1].reduced_density_matrix(support)
                )
            )
            for support in all_required_supports(16)
        )
        margin_difference = abs(
            certificate_margin(cert_independent[0]) - certificate_margin(cert_independent[1])
        )
        robust_positive = (
            pooled_class in ("E2", "E3", "E4")
            and class_agreement
            and bool(cert_pooled.get("stochastic_certificate_accepted", False))
        )
        final_class = pooled_class if robust_positive else "unknown"
        full, compact = tokens_from_state(pooled)
        rows.append(
            {
                "temperature_reduced": float(reduced),
                "temperature": float(physical),
                "class_name": final_class,
                "raw_class_primary": cert_independent[0]["class_name"],
                "raw_class_confirm": cert_independent[1]["class_name"],
                "certificate_primary": cert_pooled,
                "certificate_confirm": cert_independent[1],
                "rdm_resolution_max_frobenius": rdm_difference,
                "certificate_margin_abs_error": margin_difference,
                "resolution_confirmed": bool(robust_positive),
                "primary_diagnostics": estimates[0].diagnostics[physical],
                "confirm_diagnostics": estimates[1].diagnostics[physical],
                "features": compact.astype(np.float32),
                "full_feature_hash": hashlib.sha256(full.tobytes()).hexdigest(),
            }
        )
    return {
        "family": family,
        "point_index": point_index,
        "point_seed": int(point.point_seed),
        "parameters": compact_parameters(point),
        "coarse_classes": {str(key): value for key, value in classes.items()},
        "coarse_transition": transition,
        "coarse_diagnostics": {
            str(reduced): coarse.diagnostics[physical]
            for reduced, physical in zip(COARSE_T, coarse_physical)
        },
        "rows": rows,
    }


def regression_point(family: str):
    if family == "xxz":
        return xxz_point(8, J=1.0, delta=0.7, h=0.35, seed=NEW_SEEDS[family] - 10)
    if family == "j1j2":
        return j1j2_point(8, J2=0.55, h=0.25, seed=NEW_SEEDS[family] - 10)
    return dh_point(8, seed=NEW_SEEDS[family] - 10)


def run_regression(output_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"families": {}}
    for family in FAMILIES:
        point = regression_point(family)
        temperatures = (0.70 * point.natural_scale, 1.20 * point.natural_scale)
        regression_chi = 256 if family == "j1j2" else 96
        regression_dt = 0.025 if family == "j1j2" else 0.05
        raw, diagnostics = generate_thermal_path(
            point, temperatures, chi_max=regression_chi, dt=regression_dt,
            svd_min=1.0e-13 if family == "j1j2" else 1.0e-11,
        )
        rows = []
        for temperature in temperatures:
            current = as_cached_moment_state(raw[temperature])
            exact, _ = exact_thermal_state(point, temperature)
            errors = [
                float(
                    np.linalg.norm(
                        current.reduced_density_matrix(support)
                        - exact.reduced_density_matrix(support)
                    )
                )
                for support in all_required_supports(8)
            ]
            rows.append(
                {
                    "temperature": temperature,
                    "max_required_rdm_frobenius_error": max(errors),
                    "purification_class": certificate_summary(current)["class_name"],
                    "exact_class": certificate_summary(exact)["class_name"],
                    "diagnostics": diagnostics[temperature],
                }
            )
        regression_tolerance = 1.2e-3 if family == "j1j2" else 2.0e-5
        passed = all(
            row["max_required_rdm_frobenius_error"] <= regression_tolerance
            and row["purification_class"] == row["exact_class"]
            for row in rows
        )
        report["families"][family] = {"status": "passed" if passed else "failed", "rows": rows}
    report["status"] = (
        "passed" if all(value["status"] == "passed" for value in report["families"].values()) else "failed"
    )
    save_json(output_root / "regression_report.json", report)
    if report["status"] != "passed":
        raise RuntimeError("purification regression failed")
    return report


def old_rows(family: str, base_root: Path) -> list[dict[str, Any]]:
    path = base_root / "data/ood_hamiltonian_n16_v1" / family / "dataset.h5"
    result: list[dict[str, Any]] = []
    with h5py.File(path, "r") as h5:
        classes = h5["labels/strict_class"].asstr()[:]
        selected = np.flatnonzero(np.isin(classes, ("strict_separable", "E2")))
        for index in selected:
            result.append(
                {
                    "source": "historical_clean_gibbs_control",
                    "source_row": int(index),
                    "parent_id": f"historical_{int(h5['metadata/point_index'][index])}",
                    "point_seed": int(h5["metadata/point_seed"][index]),
                    "parameters": json.loads(h5["metadata/parameters_json"].asstr()[index]),
                    "temperature_reduced": float(h5["metadata/temperature_reduced"][index]),
                    "temperature": float(h5["metadata/temperature"][index]),
                    "class_name": str(classes[index]),
                    "resolution_confirmed": True,
                    "features": h5["features/Joint_K16_inputs"][index].astype(np.float32),
                    "certificate": json.loads(h5["certificate/diagnostics_json"].asstr()[index]),
                }
            )
    return result


def write_dataset(path: Path, family: str, rows: list[dict[str, Any]], audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            {
                "family": family,
                "N": 16,
                "boundary": "OBC",
                "state_type": "clean Hamiltonian Gibbs states only",
                "noise_added": False,
                "selection_uses_model_scores": False,
                "threshold_E3": MIN_HIGHER_ORDER,
                "threshold_E4": MIN_HIGHER_ORDER,
                "generation_audit_hash": canonical_hash(audit),
            }
        )
        h5.create_dataset("features/Joint_K16_inputs", data=np.stack([row["features"] for row in rows]), compression="gzip")
        h5.create_dataset("metadata/source", data=np.asarray([row["source"] for row in rows], dtype=object), dtype=string)
        h5.create_dataset("metadata/parent_id", data=np.asarray([row["parent_id"] for row in rows], dtype=object), dtype=string)
        h5.create_dataset("metadata/point_seed", data=np.asarray([row["point_seed"] for row in rows], dtype=np.int64))
        h5.create_dataset("metadata/parameters_json", data=np.asarray([json.dumps(row["parameters"]) for row in rows], dtype=object), dtype=string)
        h5.create_dataset("metadata/temperature_reduced", data=np.asarray([row["temperature_reduced"] for row in rows], dtype=np.float64))
        h5.create_dataset("metadata/temperature", data=np.asarray([row["temperature"] for row in rows], dtype=np.float64))
        h5.create_dataset("labels/strict_class", data=np.asarray([row["class_name"] for row in rows], dtype=object), dtype=string)
        for label in ("strict_separable", "E2", "E3", "E4", "unknown"):
            h5.create_dataset(f"labels/{label}", data=np.asarray([row["class_name"] == label for row in rows], dtype=np.bool_))
        h5.create_dataset("metadata/resolution_confirmed", data=np.asarray([row["resolution_confirmed"] for row in rows], dtype=np.bool_))
        h5.create_dataset("certificate/json", data=np.asarray([json.dumps(row["certificate"], allow_nan=True) for row in rows], dtype=object), dtype=string)


def run_generate(family: str, output_root: Path, paths: int, workers: int,
                 base_root: Path) -> dict[str, Any]:
    regression = output_root / "regression_report.json"
    if not regression.exists() or json.loads(regression.read_text())["status"] != "passed":
        raise RuntimeError("run and pass regression before generation")
    points = heldout_points(family, paths)
    payloads = [(family, index, point) for index, point in enumerate(points)]
    results: list[dict[str, Any]] = []
    cache_root = output_root / family / "parent_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    pending = []
    for payload in payloads:
        _, index, point = payload
        cache_key = canonical_hash(
            {
                "family": family,
                "point_seed": point.point_seed,
                "parameters": compact_parameters(point),
                "coarse_t": COARSE_T,
                "refine_ratios": REFINE_RATIOS,
                "version": 9,
            }
        )[:12]
        cache_path = cache_root / f"parent_{index:04d}_{cache_key}.pt"
        if cache_path.exists():
            results.append(torch.load(cache_path, map_location="cpu"))
        else:
            pending.append((payload, cache_path))
    if results:
        print(json.dumps({"family": family, "resumed_paths": len(results), "pending_paths": len(pending)}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        worker = process_point_dual_ftlm if family == "j1j2" else process_point
        futures = {pool.submit(worker, payload): cache_path for payload, cache_path in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            cache_path = futures[future]
            temporary = cache_path.with_suffix(".tmp")
            torch.save(result, temporary)
            temporary.replace(cache_path)
            results.append(result)
            counts: dict[str, int] = {}
            for item in results:
                for row in item["rows"]:
                    counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
            print(json.dumps({"family": family, "completed_new_paths": completed, "available_paths": len(results), "total_paths": paths, "counts": counts}), flush=True)
    results.sort(key=lambda value: value["point_index"])
    rows = old_rows(family, base_root)
    for result in results:
        for row in result["rows"]:
            cert = row["certificate_primary"]
            rows.append(
                {
                    "source": "adaptive_clean_gibbs_dual_ftlm" if family == "j1j2" else "adaptive_clean_gibbs_purification",
                    "source_row": -1,
                    "parent_id": f"adaptive_{result['point_seed']}",
                    "point_seed": result["point_seed"],
                    "parameters": result["parameters"],
                    "temperature_reduced": row["temperature_reduced"],
                    "temperature": row["temperature"],
                    "class_name": row["class_name"],
                    "resolution_confirmed": row["resolution_confirmed"],
                    "features": row.pop("features"),
                    "certificate": {
                        **cert,
                        "confirm_class": row["raw_class_confirm"],
                        "rdm_resolution_max_frobenius": row["rdm_resolution_max_frobenius"],
                        "certificate_margin_abs_error": row["certificate_margin_abs_error"],
                    },
                }
            )
    classes = np.asarray([row["class_name"] for row in rows])
    counts = {label: int(np.sum(classes == label)) for label in ("strict_separable", "E2", "E3", "E4", "unknown")}
    parents = {
        label: len({row["parent_id"] for row in rows if row["class_name"] == label})
        for label in ("E3", "E4")
    }
    threshold_met = counts["E3"] >= MIN_HIGHER_ORDER and counts["E4"] >= MIN_HIGHER_ORDER and parents["E3"] >= MIN_HIGHER_ORDER_PARENTS and parents["E4"] >= MIN_HIGHER_ORDER_PARENTS
    audit = {
        "family": family,
        "requested_paths": paths,
        "new_seed_namespace": NEW_SEEDS[family],
        "parameter_sampling": "Every fourth XXZ/J1-J2 parent uses the broad field range; the other XXZ parents use J=1, Delta in [0.4,1.8], |h|<=0.75, and the other J1-J2 parents use |h|<=1.5. Random-local-H uses its preregistered held-out generator.",
        "coarse_temperatures_reduced": COARSE_T,
        "refine_ratios": REFINE_RATIOS,
        "label_solver": "two independent canonical-typicality Lanczos estimates pooled to R=32" if family == "j1j2" else "two-resolution deterministic tensor-network thermal purification",
        "primary_resolution": {"random_vectors": 16, "lanczos_dimension": 80} if family == "j1j2" else {"chi_max": 128, "imaginary_time_step_max": 0.05},
        "confirmation_resolution": {"random_vectors": 16, "lanczos_dimension": 80} if family == "j1j2" else {"chi_max": 96, "imaginary_time_step_max": 0.025},
        "acceptance": ({
            "independent_hierarchy_class_agreement": True,
            "pooled_acceptance_threshold": "max(1e-5, 5*four-block sigma)",
            "pooled_random_vectors": 32,
        } if family == "j1j2" else {
            "independent_hierarchy_class_agreement": True,
            "max_rdm_frobenius_difference": 2.0e-4,
            "minimum_positive_margin": 1.0e-4,
            "max_margin_difference": "max(5e-4, 0.15*primary_margin)",
        }),
        "counts": counts,
        "higher_order_parent_counts": parents,
        "threshold_met": threshold_met,
        "noise_added": False,
        "model_scores_used_for_generation": False,
        "results": [
            {
                **{key: value for key, value in result.items() if key != "rows"},
                "rows": [
                    {key: value for key, value in row.items() if key != "features"}
                    for row in result["rows"]
                ],
            }
            for result in results
        ],
    }
    family_root = output_root / family
    save_json(family_root / "generation_audit.json", audit)
    write_dataset(family_root / "dataset.h5", family, rows, audit)
    save_json(family_root / "summary.json", {key: value for key, value in audit.items() if key != "results"})
    return audit


def evaluate_route(path: Path, model_template: str, values: np.ndarray, labels: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    probabilities = []
    model_audit = []
    for seed in TRAINING_SEEDS:
        model_dir = ROOT / model_template.format(seed=seed)
        model, report = load_joint_model(model_dir, device)
        probabilities.append(predict(model, values, device))
        model_audit.append({"seed": seed, "checkpoint_sha256": sha256(model_dir / "best.pt"), "report_sha256": sha256(model_dir / "report.json"), "threshold": 0.5})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    stacked = np.stack(probabilities)
    ensemble = stacked.mean(axis=0)
    mask = np.ones(len(values), dtype=bool)
    return {
        "model_audit": model_audit,
        "per_seed": {str(seed): metrics(stacked[index], labels, mask) for index, seed in enumerate(TRAINING_SEEDS)},
        "probability_average_ensemble": metrics(ensemble, labels, mask),
        "scores_sha256": hashlib.sha256(stacked.tobytes()).hexdigest(),
    }


def run_evaluate(output_root: Path, device_name: str) -> dict[str, Any]:
    summaries = {family: json.loads((output_root / family / "summary.json").read_text()) for family in FAMILIES}
    if not all(summary["threshold_met"] for summary in summaries.values()):
        raise RuntimeError("evaluation gate closed: every family must meet E3/E4 count and parent thresholds")
    routes = {
        "Dv1_only_JointK16": "models/joint_k16_training_distribution_ablation_v1/RoPETransformer_JointK16_Dv1_only_seed{seed}",
        "Dv1_plus_R2Support_JointK16": "models/k24_r2support_domain_generalization_v1/Fixed_K16_R2Support_seed{seed}",
    }
    device = torch.device(device_name)
    report: dict[str, Any] = {"families": {}, "threshold_gate": summaries, "no_retraining": True, "no_threshold_tuning": True, "no_model_score_selection": True}
    for family in FAMILIES:
        with h5py.File(output_root / family / "dataset.h5", "r") as h5:
            values = h5["features/Joint_K16_inputs"][:]
            labels = {name: h5[f"labels/{name}"][:].astype(bool) for name in ("strict_separable", "E2", "E3", "E4", "unknown")}
            labels["certified_entangled"] = labels["E2"] | labels["E3"] | labels["E4"]
        report["families"][family] = {
            route: evaluate_route(output_root, template, values, labels, device)
            for route, template in routes.items()
        }
    save_json(output_root / "evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("regression", "generate"))
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--paths", type=int, default=40)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/adaptive_gibbs_ood_n16_v1")
    parser.add_argument("--base-root", type=Path, default=ROOT,
                        help="root containing the prerequisite 312-row OOD HDF5 files")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.stage == "regression":
        print(json.dumps(run_regression(args.output_root), indent=2), flush=True)
    elif args.stage == "generate":
        if args.family is None:
            raise ValueError("--family is required for generate")
        print(json.dumps({key: value for key, value in run_generate(args.family, args.output_root, args.paths, args.workers, args.base_root).items() if key != "results"}, indent=2), flush=True)
    else:
        print(json.dumps(run_evaluate(args.output_root, args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()

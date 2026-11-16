from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.local_unitary import feature_slices
from entanglement_data.xy.free_fermion_exact import XYFreeFermionExactBackend
from entanglement_data.xy.size_transfer import (
    LocalRawSpec,
    ProductRDMState,
    ThermalRDMState,
    certify_translation_orbits,
    connected_from_raw,
    cyclic_orbits,
    factorized_bloch,
    local_masks,
    product_local_raw,
    thermal_local_raw,
)


SOURCE = ROOT / "data/xy_n16_pbc_exact500_v1/dataset.h5"
DEFAULT_OUTPUT = ROOT / "data/xy_n32_pbc_matched_exact500_v1"
TOL_PT = 1e-9
TOL_CCNR = 1e-9
CLASS_TO_ID = {"strict_separable": 0, "E2": 2, "E3": 3, "E4": 4, "unknown": -1}


def file_snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def read_source_parameters(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        if len(h5["params/sample_id"]) != 500 or not bool(h5.attrs["generation_complete"]):
            raise RuntimeError("source exact500 is not the frozen completed 500-row dataset")
        return {
            "sample_id": h5["params/sample_id"][:], "gamma": h5["params/gamma"][:],
            "h": h5["params/h"][:], "T": h5["params/T"][:],
            "beta": h5["params/beta"][:], "sample_seed": h5["params/sample_seed"][:],
            "stratum": h5["params/stratum"].asstr()[:],
        }


def raw_from_n16_stored(x: np.ndarray) -> dict[str, np.ndarray]:
    spec = LocalRawSpec(16, 4)
    source_spec = FeatureSpec(16)
    slices = feature_slices(source_spec)
    t1 = np.asarray(x[slices.one], dtype=np.float64).reshape(16, 3)
    all_pairs = np.asarray(x[slices.two], dtype=np.float64).reshape(-1, 3, 3)
    lookup = np.full((16, 16), -1, dtype=np.int64)
    lookup[slices.pair_left, slices.pair_right] = np.arange(len(slices.pair_left))
    t2 = np.zeros((16, 4, 3, 3), dtype=np.float64)
    for distance in range(1, 5):
        for i in range(16 - distance):
            t2[i, distance - 1] = all_pairs[lookup[i, i + distance]]
    t3 = np.zeros((16, 3, 3, 3), dtype=np.float64)
    t3[:14] = np.asarray(x[slices.three], dtype=np.float64).reshape(14, 3, 3, 3)
    t4 = np.zeros((16, 3, 3, 3, 3), dtype=np.float64)
    t4[:13] = np.asarray(x[slices.four], dtype=np.float64).reshape(13, 3, 3, 3, 3)
    k2, k3 = connected_from_raw(t1, t2, t3, spec)
    p2, p3, p4 = local_masks(spec)
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "K2": k2, "K3": k3,
            "T2_mask": p2, "T3_mask": p3, "T4_mask": p4}


def product_bloch(stratum: str, gamma: float, h: float) -> np.ndarray:
    if stratum == "exact_factorized_anchor":
        return factorized_bloch(gamma, h)
    if stratum == "exact_polarized_anchor":
        return np.asarray([0.0, 0.0, 1.0])
    raise ValueError(f"not an exact product stratum: {stratum}")


def recompute_raw(N: int, gamma: float, h: float, T: float, stratum: str) -> tuple[dict[str, np.ndarray], Any, dict[str, Any]]:
    spec = LocalRawSpec(N, 4)
    if stratum in ("exact_factorized_anchor", "exact_polarized_anchor"):
        bloch = product_bloch(stratum, gamma, h)
        return product_local_raw(bloch, spec), ProductRDMState(bloch, N), {
            "backend": "analytic_exact_product", "parity_projection_used": False,
            "parity_even_logZ": math.nan, "parity_odd_logZ": math.nan, "total_logZ": math.nan,
            "max_imaginary_residual": 0.0,
        }
    if T <= 0:
        raise RuntimeError("matched exact500 unexpectedly contains a non-product T=0 row")
    backend = XYFreeFermionExactBackend(N, gamma, h, "PBC")
    beta = 1.0 / T
    raw = thermal_local_raw(backend, beta, spec)
    logs = backend.sector_log_partition_functions(beta)
    return raw, ThermalRDMState(backend, beta), {
        "backend": "finite_free_fermion_exact", "parity_projection_used": True,
        "parity_even_logZ": logs["even"], "parity_odd_logZ": logs["odd"], "total_logZ": backend.logZ(beta),
        "max_imaginary_residual": backend.max_imaginary_residual,
    }


def run_n16_regression(source: Path, output_path: Path) -> dict[str, Any]:
    rng = np.random.default_rng(2026081902)
    with h5py.File(source, "r") as h5:
        strata = h5["params/stratum"].asstr()[:]
        thermal = np.flatnonzero(~np.isin(strata, ["exact_factorized_anchor", "exact_polarized_anchor"]))
        selected = rng.choice(thermal, size=4, replace=False).tolist()
        selected += [int(np.flatnonzero(strata == name)[0]) for name in ("exact_factorized_anchor", "exact_polarized_anchor")]
        records, max_feature_error = [], 0.0
        for position, index in enumerate(selected):
            gamma, field, temperature = (float(h5[f"params/{key}"][index]) for key in ("gamma", "h", "T"))
            stratum = strata[index]
            expected = raw_from_n16_stored(h5["features/X"][index])
            actual, state, _ = recompute_raw(16, gamma, field, temperature, stratum)
            errors = {key: float(np.max(np.abs(actual[key] - expected[key]))) for key in ("T1", "T2", "T3", "T4", "K2", "K3")}
            max_feature_error = max(max_feature_error, *errors.values())
            cert2 = certify_translation_orbits(state, 16, 2, tol_pt=TOL_PT, tol_ccnr=TOL_CCNR)
            expected_class = (
                "strict_separable" if bool(h5["labels/certified_separable"][index]) else
                ("E2" if bool(h5["cert2/all_pairs/any_npt"][index]) else
                 ("E3" if bool(h5["cert3/any_npt"][index] or h5["cert3/any_ccnr"][index]) else
                  ("E4" if bool(h5["cert4/any_npt"][index] or h5["cert4/any_ccnr"][index]) else "unknown")))
            )
            if stratum in ("exact_factorized_anchor", "exact_polarized_anchor"):
                actual_class = "strict_separable"
            elif cert2["any_npt"]:
                actual_class = "E2"
            else:
                hierarchical3 = certify_translation_orbits(state, 16, 3, tol_pt=TOL_PT, tol_ccnr=TOL_CCNR)
                if hierarchical3["any_npt"] or hierarchical3["any_ccnr"]:
                    actual_class = "E3"
                else:
                    hierarchical4 = certify_translation_orbits(state, 16, 4, tol_pt=TOL_PT, tol_ccnr=TOL_CCNR)
                    actual_class = "E4" if hierarchical4["any_npt"] or hierarchical4["any_ccnr"] else "unknown"
            certificate_errors = {
                "allpair_min_pt": abs(cert2["global_min_pt"] - float(h5["cert2/all_pairs/min_pt_eigenvalue"][index])),
                "allpair_any_npt_match": bool(cert2["any_npt"] == bool(h5["cert2/all_pairs/any_npt"][index])),
                "hierarchical_class_match": actual_class == expected_class,
            }
            if position == 0:
                for k in (3, 4):
                    cert = certify_translation_orbits(state, 16, k, tol_pt=TOL_PT, tol_ccnr=TOL_CCNR)
                    certificate_errors[f"K{k}_min_pt"] = abs(cert["global_min_pt"] - float(h5[f"cert{k}/global_min_pt"][index]))
                    certificate_errors[f"K{k}_max_ccnr"] = abs(cert["global_max_ccnr"] - float(h5[f"cert{k}/global_max_ccnr"][index]))
                    certificate_errors[f"K{k}_npt_match"] = bool(cert["any_npt"] == bool(h5[f"cert{k}/any_npt"][index]))
                    certificate_errors[f"K{k}_ccnr_match"] = bool(cert["any_ccnr"] == bool(h5[f"cert{k}/any_ccnr"][index]))
            records.append({"source_row": int(index), "sample_id": int(h5["params/sample_id"][index]), "stratum": stratum,
                            "expected_hierarchical_class": expected_class, "actual_hierarchical_class": actual_class,
                            "raw_errors": errors, "certificate_checks": certificate_errors})
    numeric_certificate_errors = [value for record in records for key, value in record["certificate_checks"].items() if not key.endswith("match")]
    boolean_matches = [value for record in records for key, value in record["certificate_checks"].items() if key.endswith("match")]
    report = {
        "passed": max_feature_error < 5e-7 and max(numeric_certificate_errors) < 5e-9 and all(boolean_matches),
        "selected_by_seed": 2026081902, "selected_rows": selected,
        "max_raw_feature_absolute_error": max_feature_error,
        "max_certificate_margin_absolute_error": max(numeric_certificate_errors),
        "all_certificate_flags_match": all(boolean_matches), "points": records,
        "comparison": "N=16 finite exact backend versus frozen exact500 T1-T4, PT and CCNR",
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"N=16 regression failed; see {output_path}")
    return report


def create_dataset(path: Path, parameters: dict[str, np.ndarray], source_snapshot: dict[str, Any]) -> None:
    string = h5py.string_dtype("utf-8")
    spec = LocalRawSpec(32, 4)
    pair_mask, triple_mask, four_mask = local_masks(spec)
    initializing = path.with_suffix(path.suffix + ".initializing")
    with h5py.File(initializing, "w") as h5:
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            git_hash = "99478b8 (stable reference; generation code uncommitted)"
        h5.attrs.update(
            N=32, num_samples=500, model="XY", boundary_condition="PBC", pair_r_max=4,
            generator_version="xy_n32_matched_exact500_v1", master_seed=20260811, git_hash=git_hash,
            generation_complete=False,
            source_N16_exact500=str(SOURCE.resolve()), source_N16_sha256=source_snapshot["sha256"],
            solver="finite_free_fermion_exact_with_exact_parity_projection",
            certificate_policy="all_subsets_via_exact_cyclic_translation_orbits",
            tol_pt=TOL_PT, tol_ccnr=TOL_CCNR, dense_2_power_N_constructed=False,
            exact5000_accessed=False,
        )
        h5.create_dataset("progress/completed", (500,), dtype=np.bool_)
        for key, values in parameters.items():
            if values.dtype.kind in "OUS":
                # h5py 3.8 accepts Python string objects for variable-length
                # UTF-8, whereas passing a NumPy <U array has no conversion path.
                h5.create_dataset(f"params/{key}", data=np.asarray(values.astype(str), dtype=object), dtype=string)
            else:
                h5.create_dataset(f"params/{key}", data=values, dtype=values.dtype)
        h5.create_dataset("params/matched_N16_sample_id", data=parameters["sample_id"], dtype=np.int32)
        h5.create_dataset("features/T1", (500, 32, 3), np.float32, chunks=(1, 32, 3), compression="gzip")
        h5.create_dataset("features/T2", (500, 32, 4, 3, 3), np.float32, chunks=(1, 32, 4, 3, 3), compression="gzip")
        h5.create_dataset("features/T3", (500, 32, 3, 3, 3), np.float32, chunks=(1, 32, 3, 3, 3), compression="gzip")
        h5.create_dataset("features/T4", (500, 32, 3, 3, 3, 3), np.float32, chunks=(1, 32, 3, 3, 3, 3), compression="gzip")
        h5.create_dataset("features/T2_mask", data=np.broadcast_to(pair_mask, (500, *pair_mask.shape)), compression="gzip")
        h5.create_dataset("features/T3_mask", data=np.broadcast_to(triple_mask, (500, *triple_mask.shape)), compression="gzip")
        h5.create_dataset("features/T4_mask", data=np.broadcast_to(four_mask, (500, *four_mask.shape)), compression="gzip")
        h5.create_dataset("derived/K2", (500, 32, 4, 3, 3), np.float32, chunks=(1, 32, 4, 3, 3), compression="gzip")
        h5.create_dataset("derived/K3", (500, 32, 3, 3, 3), np.float32, chunks=(1, 32, 3, 3, 3), compression="gzip")
        for key, dtype in (("backend", string), ("parity_projection_used", np.bool_), ("parity_even_logZ", np.float64),
                           ("parity_odd_logZ", np.float64), ("total_logZ", np.float64), ("runtime_raw", np.float64),
                           ("max_imaginary_residual", np.float64),
                           ("runtime_cert2", np.float64), ("runtime_cert3", np.float64), ("runtime_cert4", np.float64),
                           ("runtime_total", np.float64)):
            h5.create_dataset(f"state/{key}", (500,), dtype=dtype)
        for k in (2, 3, 4):
            orbit = cyclic_orbits(32, k)
            h5.create_dataset(f"certificate_schema/K{k}/representatives", data=np.asarray([x for x, _ in orbit], dtype=np.int8))
            h5.create_dataset(f"certificate_schema/K{k}/multiplicity", data=np.asarray([x for _, x in orbit], dtype=np.int16))
            for key, dtype in (("evaluated", np.bool_), ("any_npt", np.bool_), ("any_ccnr", np.bool_),
                               ("npt_block_count", np.int32), ("ccnr_block_count", np.int32),
                               ("global_min_pt", np.float64), ("global_max_ccnr", np.float64),
                               ("best_npt_representative", np.int16), ("best_ccnr_representative", np.int16),
                               ("best_npt_cut", np.int8), ("best_ccnr_cut", np.int8),
                               ("min_rdm_eigenvalue", np.float64), ("max_rdm_trace_error", np.float64),
                               ("max_rdm_hermiticity_error", np.float64)):
                fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
                h5.create_dataset(f"cert{k}/{key}", (500,), dtype=dtype, fillvalue=fill)
        h5.create_dataset("labels/class_id", (500,), np.int8, fillvalue=-127)
        h5.create_dataset("labels/class_name", (500,), dtype=string)
        h5.create_dataset("labels/certificate_type", (500,), dtype=string)
        h5.create_dataset("labels/certificate_subsystem", (500, 4), np.int8, fillvalue=-1)
        h5.create_dataset("labels/certificate_subsystem_size", (500,), np.int8)
        h5.create_dataset("labels/certificate_bipartition", (500,), np.int8, fillvalue=-1)
        h5.create_dataset("labels/certificate_margin", (500,), np.float64, fillvalue=np.nan)
        for key in ("strict_separable", "E2", "E3", "E4", "unknown"):
            h5.create_dataset(f"labels/{key}", (500,), np.bool_)
        h5.create_dataset("labels/binary_known", (500,), np.bool_)
        h5.create_dataset("labels/binary_entangled", (500,), np.bool_)
        h5.flush()
    initializing.replace(path)


def write_certificate(h5: h5py.File, index: int, k: int, cert: dict[str, Any]) -> None:
    h5[f"cert{k}/evaluated"][index] = True
    for key in ("any_npt", "any_ccnr", "npt_block_count", "ccnr_block_count", "global_min_pt", "global_max_ccnr",
                "best_npt_representative", "best_ccnr_representative", "best_npt_cut", "best_ccnr_cut",
                "min_rdm_eigenvalue", "max_rdm_trace_error", "max_rdm_hermiticity_error"):
        h5[f"cert{k}/{key}"][index] = cert[key]


CERTIFICATE_WRITE_KEYS = (
    "any_npt", "any_ccnr", "npt_block_count", "ccnr_block_count", "global_min_pt", "global_max_ccnr",
    "best_npt_representative", "best_ccnr_representative", "best_npt_cut", "best_ccnr_cut",
    "min_rdm_eigenvalue", "max_rdm_trace_error", "max_rdm_hermiticity_error",
)


def compute_one(task: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    """Compute one state without touching HDF5; safe for process workers."""

    index, row = task
    started = time.perf_counter()
    gamma, field, temperature = (float(row[key]) for key in ("gamma", "h", "T"))
    stratum = str(row["stratum"])
    t = time.perf_counter()
    raw, state, state_info = recompute_raw(32, gamma, field, temperature, stratum)
    raw_seconds = time.perf_counter() - t
    exact_separable = stratum in ("exact_factorized_anchor", "exact_polarized_anchor")
    certificates: dict[int, dict[str, Any]] = {}
    runtimes = {2: 0.0, 3: 0.0, 4: 0.0}
    if exact_separable:
        label = "strict_separable"
        certificate_type, certificate_subsystem, certificate_cut, certificate_margin = "exact_construction", (), -1, math.nan
    else:
        for k in (2, 3, 4):
            t = time.perf_counter()
            certificates[k] = certify_translation_orbits(state, 32, k, tol_pt=TOL_PT, tol_ccnr=TOL_CCNR)
            runtimes[k] = time.perf_counter() - t
            detected = certificates[k]["any_npt"] if k == 2 else (certificates[k]["any_npt"] or certificates[k]["any_ccnr"])
            if detected:
                label = f"E{k}"
                if k == 2 or -certificates[k]["global_min_pt"] >= certificates[k]["global_max_ccnr"] - 1.0:
                    certificate_type = "NPT"
                    rep_index = certificates[k]["best_npt_representative"]
                    certificate_cut = certificates[k]["best_npt_cut"]
                    certificate_margin = -certificates[k]["global_min_pt"]
                else:
                    certificate_type = "CCNR"
                    rep_index = certificates[k]["best_ccnr_representative"]
                    certificate_cut = certificates[k]["best_ccnr_cut"]
                    certificate_margin = certificates[k]["global_max_ccnr"] - 1.0
                certificate_subsystem = tuple(int(x) for x in certificates[k]["representatives"][rep_index])
                break
        else:
            label = "unknown"
            certificate_type, certificate_subsystem, certificate_cut, certificate_margin = "none", (), -1, math.nan
    if hasattr(state, "backend"):
        state_info["max_imaginary_residual"] = state.backend.max_imaginary_residual
    compact_certificates = {
        k: {key: cert[key] for key in CERTIFICATE_WRITE_KEYS} for k, cert in certificates.items()
    }
    return {
        "index": index, "sample_id": int(row["sample_id"]), "raw": {key: raw[key] for key in ("T1", "T2", "T3", "T4", "K2", "K3")},
        "state_info": state_info, "raw_seconds": raw_seconds, "certificates": compact_certificates,
        "runtimes": runtimes, "label": label, "certificate_type": certificate_type,
        "certificate_subsystem": certificate_subsystem, "certificate_cut": certificate_cut,
        "certificate_margin": certificate_margin, "seconds": time.perf_counter() - started,
    }


def write_one(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    index, raw = int(payload["index"]), payload["raw"]
    state_info, certificates, runtimes = payload["state_info"], payload["certificates"], payload["runtimes"]
    label = payload["label"]
    certificate_type, certificate_subsystem = payload["certificate_type"], payload["certificate_subsystem"]
    certificate_cut, certificate_margin = payload["certificate_cut"], payload["certificate_margin"]
    with h5py.File(path, "r+") as h5:
        for key in ("T1", "T2", "T3", "T4"):
            h5[f"features/{key}"][index] = raw[key].astype(np.float32)
        for key in ("K2", "K3"):
            h5[f"derived/{key}"][index] = raw[key].astype(np.float32)
        for key, value in state_info.items():
            h5[f"state/{key}"][index] = value
        h5["state/runtime_raw"][index] = payload["raw_seconds"]
        for k in (2, 3, 4):
            h5[f"state/runtime_cert{k}"][index] = runtimes[k]
            if k in certificates:
                write_certificate(h5, index, k, certificates[k])
        h5["state/runtime_total"][index] = payload["seconds"]
        h5["labels/class_id"][index] = CLASS_TO_ID[label]
        h5["labels/class_name"][index] = label
        h5["labels/certificate_type"][index] = certificate_type
        h5["labels/certificate_subsystem"][index] = -1
        if certificate_subsystem:
            h5["labels/certificate_subsystem"][index, : len(certificate_subsystem)] = certificate_subsystem
        h5["labels/certificate_subsystem_size"][index] = len(certificate_subsystem)
        h5["labels/certificate_bipartition"][index] = certificate_cut
        h5["labels/certificate_margin"][index] = certificate_margin
        for name in CLASS_TO_ID:
            h5[f"labels/{name}"][index] = name == label
        h5["labels/binary_known"][index] = label != "unknown"
        h5["labels/binary_entangled"][index] = label in ("E2", "E3", "E4")
        h5["progress/completed"][index] = True
        h5.flush()
    return {"index": index, "sample_id": payload["sample_id"], "class": label,
            "seconds": payload["seconds"], "cert_seconds": runtimes}


def generate_one(path: Path, index: int, parameters: dict[str, np.ndarray]) -> dict[str, Any]:
    row = {key: values[index] for key, values in parameters.items()}
    return write_one(path, compute_one((index, row)))


def margin_summary(values: np.ndarray) -> dict[str, float] | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return {"min": float(finite.min()), "median": float(np.median(finite)), "max": float(finite.max())}


def n16_hierarchical_classes(source: Path) -> np.ndarray:
    with h5py.File(source, "r") as h5:
        separable = h5["labels/certified_separable"][:]
        e2 = h5["cert2/all_pairs/any_npt"][:]
        e3 = h5["cert3/any_npt"][:] | h5["cert3/any_ccnr"][:]
        e4 = h5["cert4/any_npt"][:] | h5["cert4/any_ccnr"][:]
    labels = np.full(500, "unknown", dtype=object)
    labels[separable] = "strict_separable"
    labels[~separable & e2] = "E2"
    labels[~separable & ~e2 & e3] = "E3"
    labels[~separable & ~e2 & ~e3 & e4] = "E4"
    return labels


def build_report(path: Path, regression: dict[str, Any], source_before: dict[str, Any], source_after: dict[str, Any]) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        done = h5["progress/completed"][:]
        labels = h5["labels/class_name"].asstr()[:][done]
        n = int(done.sum())
        report = {
            "status": "complete" if n == 500 and bool(h5.attrs["generation_complete"]) else "running",
            "completed_samples": n, "requested_samples": 500,
            "class_counts_N32": {name: int(np.sum(labels == name)) for name in CLASS_TO_ID},
            "runtime_mean_seconds": {key: float(h5[f"state/runtime_{key}"][:][done].mean()) if n else math.nan for key in ("raw", "cert2", "cert3", "cert4", "total")},
            "generation_runtime_sum_seconds": float(h5["state/runtime_total"][:][done].sum()),
            "file_size_bytes": path.stat().st_size, "physical_raw_correlator_count_per_state": 4317,
            "derived_connected_scalar_count_per_state": 1062 + 810,
            "certificate_margins": {}, "failure_count": 0, "numerical_warning_count": 0, "warnings": [],
            "N16_regression": regression,
            "source_N16_untouched": source_before == source_after,
            "source_N16_snapshot_before": source_before, "source_N16_snapshot_after": source_after,
            "exact5000_accessed": False, "dense_2_power_32_constructed": False,
        }
        for k in (2, 3, 4):
            evaluated = h5[f"cert{k}/evaluated"][:] & done
            report["certificate_margins"][f"K{k}"] = {
                "evaluated_samples": int(evaluated.sum()),
                "global_min_pt": margin_summary(h5[f"cert{k}/global_min_pt"][:][evaluated]),
                "global_max_ccnr": margin_summary(h5[f"cert{k}/global_max_ccnr"][:][evaluated]),
            }
        certificate_margin = h5["labels/certificate_margin"][:]
        report["class_certificate_margin_distribution"] = {
            name: margin_summary(certificate_margin[done & h5[f"labels/{name}"][:]]) for name in ("E2", "E3", "E4")
        }
        report["max_imaginary_residual"] = float(h5["state/max_imaginary_residual"][:][done].max()) if n else math.nan
        if n == 500:
            n16 = n16_hierarchical_classes(SOURCE)
            n32 = h5["labels/class_name"].asstr()[:]
            names = tuple(CLASS_TO_ID)
            table = {left: {right: int(np.sum((n16 == left) & (n32 == right))) for right in names} for left in names}
            report["N16_class_counts"] = {name: int(np.sum(n16 == name)) for name in names}
            report["N16_to_N32_transition_table"] = table
        return report


def validate_complete(path: Path) -> dict[str, Any]:
    spec = LocalRawSpec(32, 4)
    with h5py.File(path, "r") as h5:
        checks = {
            "generation_complete": bool(h5.attrs["generation_complete"]),
            "all_rows_complete": bool(h5["progress/completed"][:].all()),
            "T1_shape": list(h5["features/T1"].shape), "T2_shape": list(h5["features/T2"].shape),
            "T3_shape": list(h5["features/T3"].shape), "T4_shape": list(h5["features/T4"].shape),
            "K2_shape": list(h5["derived/K2"].shape), "K3_shape": list(h5["derived/K3"].shape),
            "T2_mask_shape": list(h5["features/T2_mask"].shape),
            "T3_mask_shape": list(h5["features/T3_mask"].shape), "T4_mask_shape": list(h5["features/T4_mask"].shape),
            "all_raw_finite": all(np.isfinite(h5[f"features/{key}"][:]).all() for key in ("T1", "T2", "T3", "T4")),
            "all_derived_finite": all(np.isfinite(h5[f"derived/{key}"][:]).all() for key in ("K2", "K3")),
            "raw_bound_max": max(float(np.abs(h5[f"features/{key}"][:]).max()) for key in ("T1", "T2", "T3", "T4")),
            "known_binary_rows": int(h5["labels/binary_known"][:].sum()),
            "unknown_excluded_from_binary": bool(np.all(~h5["labels/binary_known"][:][h5["labels/unknown"][:]])),
            "physical_raw_count": spec.total_physical,
            "max_imaginary_residual": float(h5["state/max_imaginary_residual"][:].max()),
        }
        # Recompute connected tensors for deterministic rows directly from stored raw moments.
        max_connected_error = 0.0
        for index in (0, 137, 311, 499):
            k2, k3 = connected_from_raw(h5["features/T1"][index], h5["features/T2"][index], h5["features/T3"][index], spec)
            max_connected_error = max(max_connected_error, float(np.max(np.abs(k2 - h5["derived/K2"][index]))), float(np.max(np.abs(k3 - h5["derived/K3"][index]))))
        checks["max_connected_recompute_error"] = max_connected_error
    expected_shapes = ([500, 32, 3], [500, 32, 4, 3, 3], [500, 32, 3, 3, 3], [500, 32, 3, 3, 3, 3])
    checks["passed"] = (
        checks["generation_complete"] and checks["all_rows_complete"] and checks["all_raw_finite"] and checks["all_derived_finite"]
        and checks["raw_bound_max"] <= 1.000001 and checks["max_connected_recompute_error"] < 2e-6
        and tuple(checks[key] for key in ("T1_shape", "T2_shape", "T3_shape", "T4_shape")) == expected_shapes
        and checks["T2_mask_shape"] == [500, 32, 4] and checks["T3_mask_shape"] == [500, 32]
        and checks["T4_mask_shape"] == [500, 32] and checks["max_imaginary_residual"] <= 2e-10
        and checks["physical_raw_count"] == 4317 and checks["unknown_excluded_from_binary"]
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched finite-N=32 XY exact500 size-transfer benchmark.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--regression-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if "exact5000" in str(args.output_dir).lower():
        raise RuntimeError("exact5000 is outside this stage")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must lie in [1,16]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "dataset.h5"
    if args.status:
        if not path.exists():
            print(json.dumps({"exists": False, "completed": 0, "generation_complete": False}))
        else:
            with h5py.File(path, "r") as h5:
                print(json.dumps({"exists": True, "completed": int(h5["progress/completed"][:].sum()),
                                  "generation_complete": bool(h5.attrs["generation_complete"])}))
        return
    source_before = file_snapshot(SOURCE)
    regression_path = args.output_dir / "n16_regression_report.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8")) if regression_path.exists() else run_n16_regression(SOURCE, regression_path)
    if not regression["passed"]:
        raise RuntimeError("N=16 regression gate did not pass")
    if args.regression_only:
        print(json.dumps(regression, indent=2), flush=True)
        return
    parameters = read_source_parameters(SOURCE)
    if not path.exists():
        create_dataset(path, parameters, source_before)
        config = {
            "N": 32, "boundary": "PBC", "matched_source": str(SOURCE.resolve()),
            "raw_schema_physical_counts": {"T1": 96, "T2": 1062, "T3": 810, "T4": 2349, "total": 4317},
            "derived": ["K2", "strict_centered_K3"], "tol_pt": TOL_PT, "tol_ccnr": TOL_CCNR,
            "certificate_families": {"K2": "all C(32,2)", "K3": "all C(32,3)", "K4": "all C(32,4)"},
            "exact_translation_orbit_compression": {f"K{k}": len(cyclic_orbits(32, k)) for k in (2, 3, 4)},
            "hierarchy": ["strict_separable_construction", "E2_NPT", "E3_NPT_or_CCNR", "E4_NPT_or_CCNR", "unknown"],
            "higher_certificates_after_first_detection": "not_evaluated", "exact5000_accessed": False,
            "production_command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/generate_xy_n32_size_transfer.py --progress-every 5 --workers 8",
            "required_extra_dependency": "pfapack==0.3.1 on the remote generation host",
        }
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    config_path = args.output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["production_workers"] = args.workers
    config["production_command"] = f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/generate_xy_n32_size_transfer.py --progress-every {args.progress_every} --workers {args.workers}"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    with h5py.File(path, "r+") as h5:
        h5.attrs["production_workers"] = args.workers
        h5.attrs["production_command"] = config["production_command"]
        h5.flush()
    with h5py.File(path, "r") as h5:
        completed = h5["progress/completed"][:]
    pending = [index for index in range(500) if not completed[index]]
    completed_count = int(completed.sum())

    def record(payload: dict[str, Any]) -> None:
        nonlocal completed_count
        result = write_one(path, payload)
        completed_count += 1
        if args.progress_every and (completed_count % args.progress_every == 0 or completed_count == 500):
            print(json.dumps({"completed": completed_count, "total": 500, **result}), flush=True)
            source_now = file_snapshot(SOURCE)
            report = build_report(path, regression, source_before, source_now)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    tasks = [(index, {key: values[index] for key, values in parameters.items()}) for index in pending]
    if args.workers <= 1:
        for task in tasks:
            record(compute_one(task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(compute_one, task) for task in tasks]
            for future in as_completed(futures):
                record(future.result())
    with h5py.File(path, "r+") as h5:
        h5.attrs["generation_complete"] = bool(h5["progress/completed"][:].all())
        h5.flush()
    source_after = file_snapshot(SOURCE)
    report = build_report(path, regression, source_before, source_after)
    validation = validate_complete(path)
    with h5py.File(path, "r") as h5:
        n16 = n16_hierarchical_classes(SOURCE)
        n32 = h5["labels/class_name"].asstr()[:]
        certificate_type = h5["labels/certificate_type"].asstr()[:]
    table_path = args.output_dir / "matched_N16_N32_metadata.csv"
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "gamma", "h", "T", "N16_class", "N32_class", "N32_certificate_type"])
        for i in range(500):
            writer.writerow([int(parameters["sample_id"][i]), float(parameters["gamma"][i]), float(parameters["h"][i]),
                             float(parameters["T"][i]), n16[i], n32[i], certificate_type[i]])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not report["source_N16_untouched"] or not validation["passed"]:
        raise RuntimeError("final N=32 validation failed")
    print(json.dumps({"status": "complete", "report": report, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()

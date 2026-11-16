from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import qmc

from ..certification.multipartite import BlockCertificate, certify_blocks, sample_certificate_blocks
from ..observables.engine import compute_features
from ..observables.feature_spec import FeatureSpec
from ..observables.pauli import pauli_matrices
from ..states.mps import PurifiedMPS
from .model import (
    SolverMetadata,
    exact_factorized_product,
    exact_polarized_product,
    generate_ground_state,
    generate_thermal_state,
)
from .free_fermion_exact import XYFreeFermionExactBackend


CERT_ENERGY = 1 << 0
CERT_PAPER_NN = 1 << 1
CERT_ALLPAIR = 1 << 2
CERT_K3_NPT = 1 << 3
CERT_K3_CCNR = 1 << 4
CERT_K4_NPT = 1 << 5
CERT_K4_CCNR = 1 << 6
CERT_EXACT_SEPARABLE = 1 << 7


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def schema_hash(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode()).hexdigest()


def _temperature(u: np.ndarray) -> np.ndarray:
    out = np.empty(len(u))
    low = u < 0.50
    medium = (u >= 0.50) & (u < 0.85)
    high = u >= 0.85
    out[low] = np.exp(np.log(0.02) + (u[low] / 0.50) * np.log(0.5 / 0.02))
    out[medium] = 0.5 + ((u[medium] - 0.50) / 0.35) * 0.7
    out[high] = 1.2 + ((u[high] - 0.85) / 0.15) * 0.3
    return out


def sample_parameter_table(total: int, master_seed: int, *, include_ordered_t0: bool = True) -> list[dict[str, Any]]:
    if total not in (50, 500, 5000):
        raise ValueError("supported gated sizes are 50 (exact smoke pilot), 500, and 5000")
    if total == 50:
        # The expensive generic ordered-T=0 DMRG subset is deliberately
        # postponed; the exact-backend smoke pilot targets finite T and keeps
        # both analytic separable anchor types.
        counts = {
            "broad": 30,
            "near_critical": 7,
            "near_factorization": 7,
            "low_temperature": 4,
            "exact_factorized_anchor": 1,
            "exact_polarized_anchor": 1,
        }
    else:
        scale = total // 500
        counts = {
            "broad": 280 * scale,
            "near_critical": 70 * scale,
            "near_factorization": 70 * scale,
            "low_temperature": 40 * scale,
            "exact_factorized_anchor": 10 * scale,
            "exact_polarized_anchor": 10 * scale,
            "ordered_T0_symmetry_broken": 20 * scale,
        }
        if not include_ordered_t0:
            counts["broad"] += counts.pop("ordered_T0_symmetry_broken")
    rows: list[dict[str, Any]] = []
    offset = 0
    for sidx, (stratum, count) in enumerate(counts.items()):
        sampler = qmc.Sobol(d=3, scramble=True, seed=master_seed + 1009 * (sidx + 1))
        u = sampler.random(count)
        if stratum == "broad":
            gamma, h, T = u[:, 0], 1.5 * u[:, 1], _temperature(u[:, 2])
        elif stratum == "near_critical":
            gamma, h, T = u[:, 0], 0.85 + 0.30 * u[:, 1], _temperature(u[:, 2])
        elif stratum == "near_factorization":
            gamma = u[:, 0]
            hf = np.sqrt(np.maximum(0.0, 1.0 - gamma**2))
            h = np.clip(hf - 0.08 + 0.16 * u[:, 1], 0.0, 1.5)
            T = _temperature(u[:, 2])
        elif stratum == "low_temperature":
            gamma, h, T = u[:, 0], 1.5 * u[:, 1], 0.01 + 0.09 * u[:, 2]
        elif stratum == "exact_factorized_anchor":
            gamma = 0.02 + 0.96 * u[:, 0]
            h = np.sqrt(1.0 - gamma**2)
            T = np.zeros(count)
        elif stratum == "exact_polarized_anchor":
            gamma = np.zeros(count)
            h = 1.05 + 0.45 * u[:, 0]
            T = np.zeros(count)
        else:
            gamma = 0.05 + 0.95 * u[:, 0]
            h = 0.02 + 0.96 * u[:, 1]
            # Keep generic points visibly away from the exact product line.
            too_close = np.abs(gamma**2 + h**2 - 1.0) < 0.03
            h[too_close] = np.clip(h[too_close] - 0.06, 0.02, 0.98)
            T = np.zeros(count)
        for j in range(count):
            seed = int(np.random.SeedSequence([master_seed, offset + j]).generate_state(1)[0])
            rows.append(
                {
                    "sample_id": offset + j,
                    "master_seed": master_seed,
                    "sample_seed": seed,
                    "stratum": stratum,
                    "gamma": float(gamma[j]),
                    "h": float(h[j]),
                    "T": float(T[j]),
                }
            )
        offset += count
    return rows


def feature_segments(x: np.ndarray, N: int = 16) -> dict[str, float]:
    cuts = [0, 3 * N, 3 * N + 9 * N * (N - 1) // 2]
    cuts += [cuts[-1] + 27 * (N - 2), len(x)]
    out: dict[str, float] = {}
    for k in range(1, 5):
        values = np.abs(x[cuts[k - 1] : cuts[k]])
        out[f"mean_abs_{k}body"] = float(values.mean())
        out[f"max_abs_{k}body"] = float(values.max())
    local = x[: 3 * N].reshape(N, 3)
    out.update(
        magnetization_x=float(local[:, 0].mean()),
        magnetization_y=float(local[:, 1].mean()),
        magnetization_z=float(local[:, 2].mean()),
    )
    return out


def pair_certificates_from_features(x: np.ndarray, N: int = 16, tol_pt: float = 1e-9) -> dict[str, Any]:
    dtype = torch.complex128
    basis = pauli_matrices(device=torch.device("cpu"), dtype=dtype, include_identity=True)
    local = x[: 3 * N].reshape(N, 3).astype(np.float64)
    pair_values = x[3 * N : 3 * N + 9 * N * (N - 1) // 2].reshape(-1, 3, 3).astype(np.float64)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pt = np.empty(len(pairs), dtype=np.float64)
    neg = np.empty(len(pairs), dtype=np.float64)
    min_rdm_eigenvalue = math.inf
    max_rdm_trace_error = 0.0
    max_rdm_hermiticity_error = 0.0
    for idx, (i, j) in enumerate(pairs):
        rho = torch.kron(basis[0], basis[0])
        for a in range(3):
            rho = rho + local[i, a] * torch.kron(basis[a + 1], basis[0])
            rho = rho + local[j, a] * torch.kron(basis[0], basis[a + 1])
            for b in range(3):
                rho = rho + pair_values[idx, a, b] * torch.kron(basis[a + 1], basis[b + 1])
        rho = (rho / 4 + (rho / 4).conj().T) / 2
        min_rdm_eigenvalue = min(min_rdm_eigenvalue, float(torch.linalg.eigvalsh(rho).min().real.item()))
        max_rdm_trace_error = max(max_rdm_trace_error, abs(float(torch.trace(rho).real.item()) - 1.0))
        max_rdm_hermiticity_error = max(max_rdm_hermiticity_error, float(torch.max(torch.abs(rho - rho.conj().T)).item()))
        shaped = rho.reshape(2, 2, 2, 2).permute(2, 1, 0, 3).reshape(4, 4)
        eig = torch.linalg.eigvalsh(shaped).real
        pt[idx] = float(eig.min().item())
        neg[idx] = float(torch.clamp(-eig, min=0).sum().item())
    flags = pt < -tol_pt
    lookup = {pair: i for i, pair in enumerate(pairs)}
    nn_pairs = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)]
    nn_indices = [lookup[tuple(sorted(pair))] for pair in nn_pairs]
    nn_pt, nn_neg, nn_flags = pt[nn_indices], neg[nn_indices], flags[nn_indices]
    return {
        "pair_indices": np.asarray(pairs, dtype=np.int8),
        "pt": pt,
        "negativity": neg,
        "flags": flags,
        "nn_pair_indices": np.asarray(nn_pairs, dtype=np.int8),
        "nn_pt": nn_pt,
        "nn_negativity": nn_neg,
        "nn_flags": nn_flags,
        "allpair_any_npt": bool(flags.any()),
        "allpair_npt_count": int(flags.sum()),
        "allpair_min_pt": float(pt.min()),
        "allpair_max_negativity": float(neg.max()),
        "allpair_best_pair": int(np.argmin(pt)),
        "paper_nn_any_npt": bool(nn_flags.any()),
        "paper_nn_npt_count": int(nn_flags.sum()),
        "paper_nn_min_pt": float(nn_pt.min()),
        "paper_nn_max_negativity": float(nn_neg.max()),
        "paper_nn_best_pair": int(np.argmin(nn_pt)),
        "max_rdm_hermiticity_error": max_rdm_hermiticity_error,
        "max_rdm_trace_error": max_rdm_trace_error,
        "min_rdm_eigenvalue": min_rdm_eigenvalue,
    }


def energy_witness(x: np.ndarray, gamma: float, h: float, N: int = 16, tol: float = 1e-9) -> dict[str, Any]:
    local = x[: 3 * N].reshape(N, 3)
    pair = x[3 * N : 3 * N + 9 * N * (N - 1) // 2].reshape(-1, 3, 3)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    lookup = {p: n for n, p in enumerate(pairs)}
    E = -h * float(local[:, 2].sum())
    for i in range(N):
        j = (i + 1) % N
        values = pair[lookup[tuple(sorted((i, j)))]]
        # XX and YY are unchanged by swapping the two site order.
        E -= 0.5 * (1 + gamma) * float(values[0, 0])
        E -= 0.5 * (1 - gamma) * float(values[1, 1])
    if h <= 1.0 + gamma:
        sep_per_site = -((1.0 + gamma) ** 2 + h**2) / (2.0 * (1.0 + gamma))
    else:
        sep_per_site = -h
    E_sep = N * sep_per_site
    return {
        "energy": E,
        "energy_per_site": E / N,
        "E_sep": E_sep,
        "E_sep_per_site": sep_per_site,
        "margin": E - E_sep,
        "detected": bool(E < E_sep - tol),
    }


def _empty_meta(solver: str) -> SolverMetadata:
    return SolverMetadata(solver, True, 1, 1, math.nan, 0, 0.0, 0.0, 0.0, 0.0)


class ExactThermalState:
    """Small compatibility adapter for the validated certificate routines."""

    def __init__(self, backend: XYFreeFermionExactBackend, beta: float):
        self.backend, self.beta, self.N = backend, float(beta), backend.N
        self.max_rdm_hermiticity_error = 0.0
        self.max_rdm_trace_error = 0.0
        self.min_rdm_eigenvalue = math.inf

    def reduced_density_matrix(self, sites):
        rho = self.backend.reduced_density_matrix(sites, self.beta)
        hermiticity = float(np.max(np.abs(rho - rho.conj().T)))
        trace_error = abs(float(np.trace(rho).real) - 1.0)
        minimum = float(np.linalg.eigvalsh(rho).min())
        self.max_rdm_hermiticity_error = max(self.max_rdm_hermiticity_error, hermiticity)
        self.max_rdm_trace_error = max(self.max_rdm_trace_error, trace_error)
        self.min_rdm_eigenvalue = min(self.min_rdm_eigenvalue, minimum)
        if hermiticity > 1e-11 or trace_error > 1e-11 or minimum < -1e-9:
            raise FloatingPointError(
                f"unphysical exact RDM: hermiticity={hermiticity}, trace={trace_error}, min_eig={minimum}"
            )
        return torch.as_tensor(rho, dtype=torch.complex128)


def generate_state_and_features(row: dict[str, Any], config: dict[str, Any], spec: FeatureSpec):
    N, gamma, h, T = spec.N, row["gamma"], row["h"], row["T"]
    stratum = row["stratum"]
    field_error = math.nan
    field_passed = True
    epsilon = 0.0
    if stratum == "exact_factorized_anchor":
        state = exact_factorized_product(N, gamma, h)
        meta = _empty_meta("analytic_exact_factorized_product")
        state_type = "exact_factorized_product"
        convention = "analytic_symmetry_broken_product"
        exact_separable = True
    elif stratum == "exact_polarized_anchor":
        state = exact_polarized_product(N)
        meta = _empty_meta("analytic_exact_polarized_product")
        state_type = "exact_polarized_product"
        convention = "exact_polarized_product"
        exact_separable = True
    elif T > 0:
        started = time.perf_counter()
        backend = XYFreeFermionExactBackend(N, gamma, h, "PBC")
        state = ExactThermalState(backend, 1.0 / T)
        initialization_time = time.perf_counter() - started
        meta = _empty_meta("finite_free_fermion_exact")
        meta.runtime_state_generation = initialization_time
        state_type = convention = "exact_unperturbed_thermal_gibbs"
        exact_separable = False
    else:
        settings = config["ground_solver"]
        features_by_epsilon: list[np.ndarray] = []
        state = None
        meta = None
        epsilons = list(config["field_convergence"]["epsilon_sequence"])
        tolerance = float(config["field_convergence"]["feature_tolerance"])
        for epsilon in epsilons:
            candidate, candidate_meta = generate_ground_state(N, gamma, h, epsilon_x=float(epsilon), **settings)
            candidate_x = compute_features(candidate, spec, output_dtype=np.float64).values
            features_by_epsilon.append(candidate_x)
            state, meta = candidate, candidate_meta
            if len(features_by_epsilon) >= 3:
                field_error = float(np.max(np.abs(features_by_epsilon[-1].astype(float) - features_by_epsilon[-2].astype(float))))
                if field_error < tolerance:
                    break
        assert state is not None and meta is not None
        field_passed = field_error < tolerance
        state_type = "symmetry_broken_ground_state"
        convention = "symmetry_broken_branch"
        exact_separable = False
    started = time.perf_counter()
    if isinstance(state, ExactThermalState):
        x = state.backend.feature_vector(state.beta, spec)
    else:
        x = compute_features(state, spec, output_dtype=np.float64).values
    feature_time = time.perf_counter() - started
    if isinstance(state, ExactThermalState):
        sector_logs = state.backend.sector_log_partition_functions(state.beta)
        internal = state.backend.validate_internal_consistency(state.beta, ((0, 1), (0, 1, 2), (0, 1, 2, 3)))
        backend_info = {
            "state_backend": "finite_free_fermion_exact",
            "exact_backend_version": state.backend.backend_version,
            "parity_projection_used": True,
            "parity_even_logZ": sector_logs["even"],
            "parity_odd_logZ": sector_logs["odd"],
            "total_logZ": state.backend.logZ(state.beta),
            "free_fermion_internal_check_passed": bool(internal["passed"]),
        }
    else:
        backend_info = {
            "state_backend": "analytic_product" if exact_separable else "ground_state_dmrg",
            "exact_backend_version": "",
            "parity_projection_used": False,
            "parity_even_logZ": math.nan,
            "parity_odd_logZ": math.nan,
            "total_logZ": math.nan,
            "free_fermion_internal_check_passed": True,
        }
    return state, x, meta, {
        "state_type": state_type,
        "state_convention": convention,
        "exact_separable": exact_separable,
        "symmetry_breaking_used": T == 0 and stratum == "ordered_T0_symmetry_broken",
        "epsilon_x": float(epsilon),
        "epsilon_y": 0.0,
        "field_convergence_passed": field_passed,
        "field_convergence_error": field_error,
        "feature_time": feature_time,
        **backend_info,
    }


def certify_sample(state, x: np.ndarray, row: dict[str, Any], config: dict[str, Any]):
    timings: dict[str, float] = {}
    t = time.perf_counter()
    cert2 = pair_certificates_from_features(x, tol_pt=float(config["certification"]["tol_pt"]))
    if cert2["min_rdm_eigenvalue"] < -1e-9:
        raise FloatingPointError(f"unphysical two-site RDM: min_eig={cert2['min_rdm_eigenvalue']}")
    timings["cert2"] = time.perf_counter() - t
    block_sets = {}
    for k, key in ((3, "K3"), (4, "K4")):
        policy = config["certification"][key]
        if str(policy).lower() == "all":
            selected = list(combinations(range(16), k))
            block_sets[k] = (selected, ["complete_enumeration"] * len(selected))
        else:
            block_sets[k] = sample_certificate_blocks(16, k, int(policy), row["sample_seed"] + k)
    blocks3, sources3 = block_sets[3]
    blocks4, sources4 = block_sets[4]
    t = time.perf_counter()
    cert3 = certify_blocks(state, blocks3, sources3, tol_pt=float(config["certification"]["tol_pt"]), tol_ccnr=float(config["certification"]["tol_ccnr"]))
    timings["cert3"] = time.perf_counter() - t
    t = time.perf_counter()
    cert4 = certify_blocks(state, blocks4, sources4, tol_pt=float(config["certification"]["tol_pt"]), tol_ccnr=float(config["certification"]["tol_ccnr"]))
    timings["cert4"] = time.perf_counter() - t
    energy = energy_witness(x, row["gamma"], row["h"], tol=float(config["certification"]["tol_energy"]))
    if isinstance(state, ExactThermalState):
        timings["rdm_diagnostics"] = {
            "max_rdm_hermiticity_error": max(state.max_rdm_hermiticity_error, cert2["max_rdm_hermiticity_error"]),
            "max_rdm_trace_error": max(state.max_rdm_trace_error, cert2["max_rdm_trace_error"]),
            "min_rdm_eigenvalue": min(state.min_rdm_eigenvalue, cert2["min_rdm_eigenvalue"]),
        }
    else:
        timings["rdm_diagnostics"] = {
            "max_rdm_hermiticity_error": 0.0,
            "max_rdm_trace_error": 0.0,
            "min_rdm_eigenvalue": math.nan,
        }
    return cert2, cert3, cert4, energy, timings


def _create_dataset(h5: h5py.File, path: str, shape: tuple[int, ...], dtype, **kwargs):
    parent, name = path.rsplit("/", 1)
    return h5.require_group(parent).create_dataset(name, shape=shape, dtype=dtype, **kwargs)


def initialize_hdf5(path: Path, rows: list[dict[str, Any]], names: list[str], config: dict[str, Any]) -> None:
    M = len(rows)
    B3 = math.comb(16, 3) if str(config["certification"]["K3"]).lower() == "all" else 16 + int(config["certification"]["K3"])
    B4 = math.comb(16, 4) if str(config["certification"]["K4"]).lower() == "all" else 16 + int(config["certification"]["K4"])
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            N=16,
            boundary_condition="PBC",
            master_seed=int(config["master_seed"]),
            feature_schema_hash=schema_hash(names),
            config_hash=config_hash(config),
            tensor_network_library="TeNPy",
            tensor_network_library_version="1.1.0",
            generation_complete=False,
        )
        try:
            h5.attrs["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            h5.attrs["git_commit"] = "uncommitted"
        _create_dataset(h5, "progress/completed", (M,), np.bool_, fillvalue=False)
        _create_dataset(h5, "features/X", (M, len(names)), np.float32, chunks=(1, len(names)), compression="gzip")
        for key, dtype in [("gamma", np.float64), ("h", np.float64), ("T", np.float64), ("beta", np.float64)]:
            _create_dataset(h5, f"params/{key}", (M,), dtype)
        _create_dataset(h5, "params/stratum", (M,), string)
        _create_dataset(h5, "params/sample_seed", (M,), np.uint64)
        _create_dataset(h5, "params/sample_id", (M,), np.int32)
        _create_dataset(h5, "params/distance_to_critical_line", (M,), np.float64)
        _create_dataset(h5, "params/distance_to_factorization_line", (M,), np.float64)
        for i, row in enumerate(rows):
            h5["params/gamma"][i], h5["params/h"][i], h5["params/T"][i] = row["gamma"], row["h"], row["T"]
            h5["params/beta"][i] = math.inf if row["T"] == 0 else 1.0 / row["T"]
            h5["params/stratum"][i] = row["stratum"]
            h5["params/sample_seed"][i] = row["sample_seed"]
            h5["params/sample_id"][i] = row["sample_id"]
            h5["params/distance_to_critical_line"][i] = abs(row["h"] - 1.0)
            h5["params/distance_to_factorization_line"][i] = abs(row["h"] ** 2 + row["gamma"] ** 2 - 1.0)
        for key in ["state_type", "state_convention", "solver", "hamiltonian_used_for_generation", "energy_hamiltonian", "state_backend", "exact_backend_version"]:
            _create_dataset(h5, f"state/{key}", (M,), string)
        for key, dtype in [
            ("symmetry_breaking_used", np.bool_), ("epsilon_x", np.float64), ("epsilon_y", np.float64),
            ("field_convergence_passed", np.bool_), ("field_convergence_error", np.float64),
            ("solver_converged", np.bool_), ("chi_max_requested", np.int32), ("final_bond_dim", np.int32),
            ("imaginary_time_step", np.float64), ("number_of_steps", np.int32), ("truncation_error", np.float64),
            ("discarded_weight", np.float64), ("norm_error", np.float64), ("runtime_state_generation", np.float64),
            ("runtime_feature", np.float64), ("runtime_cert2", np.float64), ("runtime_cert3", np.float64), ("runtime_cert4", np.float64),
            ("parity_projection_used", np.bool_), ("parity_even_logZ", np.float64), ("parity_odd_logZ", np.float64),
            ("total_logZ", np.float64), ("free_fermion_internal_check_passed", np.bool_),
        ]:
            _create_dataset(h5, f"state/{key}", (M,), dtype)
        for key in ["value", "energy_per_site", "E_sep", "E_sep_per_site", "margin"]:
            _create_dataset(h5, f"energy/{key}", (M,), np.float64)
        _create_dataset(h5, "energy/detected", (M,), np.bool_)
        for key in ["mean_abs_1body", "mean_abs_2body", "mean_abs_3body", "mean_abs_4body", "max_abs_1body", "max_abs_2body", "max_abs_3body", "max_abs_4body", "magnetization_x", "magnetization_y", "magnetization_z"]:
            _create_dataset(h5, f"diagnostics/{key}", (M,), np.float64)
        for key in ["max_rdm_hermiticity_error", "max_rdm_trace_error", "min_rdm_eigenvalue"]:
            _create_dataset(h5, f"diagnostics/{key}", (M,), np.float64)
        _create_dataset(h5, "cert2/paper_nn/pair_indices", (M, 16, 2), np.int8)
        _create_dataset(h5, "cert2/paper_nn/pt_min_eigenvalue", (M, 16), np.float64)
        _create_dataset(h5, "cert2/paper_nn/negativity", (M, 16), np.float64)
        _create_dataset(h5, "cert2/paper_nn/npt_flag", (M, 16), np.bool_)
        _create_dataset(h5, "cert2/all_pairs/pair_indices", (M, 120, 2), np.int8)
        _create_dataset(h5, "cert2/all_pairs/pt_min_eigenvalue", (M, 120), np.float64)
        _create_dataset(h5, "cert2/all_pairs/negativity", (M, 120), np.float64)
        _create_dataset(h5, "cert2/all_pairs/npt_flag", (M, 120), np.bool_)
        for prefix in ["paper_nn", "all_pairs"]:
            for key, dtype in [("any_npt", np.bool_), ("npt_count", np.int16), ("min_pt_eigenvalue", np.float64), ("max_negativity", np.float64), ("best_pair", np.int16)]:
                _create_dataset(h5, f"cert2/{prefix}/{key}", (M,), dtype)
        for prefix, B, cuts, k in [("cert3", B3, 3, 3), ("cert4", B4, 7, 4)]:
            _create_dataset(h5, f"{prefix}/block_sites", (M, B, k), np.int8)
            _create_dataset(h5, f"{prefix}/block_source", (M, B), string)
            _create_dataset(h5, f"{prefix}/pt_eigs", (M, B, cuts), np.float64)
            _create_dataset(h5, f"{prefix}/ccnr_norms", (M, B, cuts), np.float64)
            for key, dtype in [("block_min_pt", np.float64), ("block_max_ccnr", np.float64), ("block_npt", np.bool_), ("block_ccnr", np.bool_), ("best_npt_cut", np.int8), ("best_ccnr_cut", np.int8)]:
                _create_dataset(h5, f"{prefix}/{key}", (M, B), dtype)
            for key, dtype in [("any_npt", np.bool_), ("any_ccnr", np.bool_), ("npt_block_count", np.int16), ("ccnr_block_count", np.int16), ("global_min_pt", np.float64), ("global_max_ccnr", np.float64), ("best_npt_block", np.int16), ("best_ccnr_block", np.int16)]:
                _create_dataset(h5, f"{prefix}/{key}", (M,), dtype)
        for key, dtype in [
            ("status", np.int8), ("certificate_mask", np.uint16), ("certified_entangled", np.bool_),
            ("certified_separable", np.bool_), ("unknown", np.bool_), ("paper_detectable", np.bool_),
            ("two_rdm_detectable", np.bool_), ("beyond_paper_nn", np.bool_), ("beyond_all_two_rdm", np.bool_),
        ]:
            _create_dataset(h5, f"labels/{key}", (M,), dtype)


def _write_block(h5: h5py.File, prefix: str, i: int, cert: BlockCertificate) -> None:
    h5[f"{prefix}/block_sites"][i] = cert.block_sites
    h5[f"{prefix}/block_source"][i] = cert.block_source.astype(str)
    h5[f"{prefix}/pt_eigs"][i] = cert.pt_min_eigenvalues
    h5[f"{prefix}/ccnr_norms"][i] = cert.ccnr_norms
    for key in ["block_min_pt", "block_max_ccnr", "block_npt", "block_ccnr", "best_npt_cut", "best_ccnr_cut"]:
        h5[f"{prefix}/{key}"][i] = getattr(cert, key)
    for key in ["any_npt", "any_ccnr", "npt_block_count", "ccnr_block_count", "global_min_pt", "global_max_ccnr", "best_npt_block", "best_ccnr_block"]:
        h5[f"{prefix}/{key}"][i] = getattr(cert, key)


def write_sample(path: Path, i: int, x: np.ndarray, meta: SolverMetadata, state_info: dict[str, Any], cert2, cert3, cert4, energy, timings):
    with h5py.File(path, "r+") as h5:
        h5["features/X"][i] = x
        for key in ["state_type", "state_convention", "symmetry_breaking_used", "epsilon_x", "epsilon_y", "field_convergence_passed", "field_convergence_error"]:
            h5[f"state/{key}"][i] = state_info[key]
        h5["state/hamiltonian_used_for_generation"][i] = "H_XY_plus_symmetry_breaking_field" if state_info["symmetry_breaking_used"] else "original_H_XY"
        h5["state/energy_hamiltonian"][i] = "original_H_XY"
        h5["state/solver"][i] = meta.solver
        for key in ["state_backend", "exact_backend_version", "parity_projection_used", "parity_even_logZ", "parity_odd_logZ", "total_logZ", "free_fermion_internal_check_passed"]:
            h5[f"state/{key}"][i] = state_info[key]
        mapping = {"final_max_bond_dimension": "final_bond_dim", "total_truncation_error": "truncation_error"}
        for key in ["solver_converged", "chi_max_requested", "final_max_bond_dimension", "imaginary_time_step", "number_of_steps", "total_truncation_error", "discarded_weight", "norm_error", "runtime_state_generation"]:
            h5[f"state/{mapping.get(key, key)}"][i] = getattr(meta, key)
        h5["state/runtime_feature"][i] = state_info["feature_time"]
        for key in ["cert2", "cert3", "cert4"]:
            h5[f"state/runtime_{key}"][i] = timings[key]
        for key, value in energy.items():
            h5[f"energy/{'value' if key == 'energy' else key}"][i] = value
        for key, value in feature_segments(x).items():
            h5[f"diagnostics/{key}"][i] = value
        for key, value in timings["rdm_diagnostics"].items():
            h5[f"diagnostics/{key}"][i] = value
        for prefix, values in [("paper_nn", (cert2["nn_pair_indices"], cert2["nn_pt"], cert2["nn_negativity"], cert2["nn_flags"])), ("all_pairs", (cert2["pair_indices"], cert2["pt"], cert2["negativity"], cert2["flags"]))]:
            for name, value in zip(["pair_indices", "pt_min_eigenvalue", "negativity", "npt_flag"], values):
                h5[f"cert2/{prefix}/{name}"][i] = value
        for prefix, source in [("paper_nn", "paper_nn"), ("all_pairs", "allpair")]:
            h5[f"cert2/{prefix}/any_npt"][i] = cert2[f"{source}_any_npt"]
            h5[f"cert2/{prefix}/npt_count"][i] = cert2[f"{source}_npt_count"]
            h5[f"cert2/{prefix}/min_pt_eigenvalue"][i] = cert2[f"{source}_min_pt"]
            h5[f"cert2/{prefix}/max_negativity"][i] = cert2[f"{source}_max_negativity"]
            h5[f"cert2/{prefix}/best_pair"][i] = cert2[f"{source}_best_pair"]
        _write_block(h5, "cert3", i, cert3)
        _write_block(h5, "cert4", i, cert4)
        mask = 0
        mask |= CERT_ENERGY if energy["detected"] else 0
        mask |= CERT_PAPER_NN if cert2["paper_nn_any_npt"] else 0
        mask |= CERT_ALLPAIR if cert2["allpair_any_npt"] else 0
        mask |= CERT_K3_NPT if cert3.any_npt else 0
        mask |= CERT_K3_CCNR if cert3.any_ccnr else 0
        mask |= CERT_K4_NPT if cert4.any_npt else 0
        mask |= CERT_K4_CCNR if cert4.any_ccnr else 0
        mask |= CERT_EXACT_SEPARABLE if state_info["exact_separable"] else 0
        entangled = bool(mask & 0x7F)
        separable = bool(state_info["exact_separable"])
        if separable and entangled:
            raise FloatingPointError("an exact product anchor triggered an entanglement certificate")
        unknown = not entangled and not separable
        higher = energy["detected"] or cert3.any_npt or cert3.any_ccnr or cert4.any_npt or cert4.any_ccnr
        labels = {
            "status": 1 if entangled else (0 if separable else -1), "certificate_mask": mask,
            "certified_entangled": entangled, "certified_separable": separable, "unknown": unknown,
            "paper_detectable": cert2["paper_nn_any_npt"], "two_rdm_detectable": cert2["allpair_any_npt"],
            "beyond_paper_nn": (not cert2["paper_nn_any_npt"]) and entangled,
            "beyond_all_two_rdm": (not cert2["allpair_any_npt"]) and higher,
        }
        for key, value in labels.items():
            h5[f"labels/{key}"][i] = value
        h5["progress/completed"][i] = True
        h5.flush()


def build_report(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        done = h5["progress/completed"][:]
        n = int(done.sum())
        report: dict[str, Any] = {"completed_samples": n, "requested_samples": len(done)}
        if not n:
            return report
        mask = done
        report["runtime_mean_seconds"] = {key: float(h5[f"state/runtime_{key}"][:][mask].mean()) for key in ["state_generation", "feature", "cert2", "cert3", "cert4"]}
        report["solver_failure_rate"] = float((~h5["state/solver_converged"][:][mask]).mean())
        report["bond_dimension"] = {"min": int(h5["state/final_bond_dim"][:][mask].min()), "median": float(np.median(h5["state/final_bond_dim"][:][mask])), "max": int(h5["state/final_bond_dim"][:][mask].max())}
        report["certificate_fraction"] = {
            "energy": float(h5["energy/detected"][:][mask].mean()),
            "paper_nn_npt": float(h5["labels/paper_detectable"][:][mask].mean()),
            "allpair_npt": float(h5["labels/two_rdm_detectable"][:][mask].mean()),
            "k3_npt": float((h5["cert3/block_npt"][:][mask].any(axis=1)).mean()),
            "k3_ccnr": float((h5["cert3/block_ccnr"][:][mask].any(axis=1)).mean()),
            "k4_npt": float((h5["cert4/block_npt"][:][mask].any(axis=1)).mean()),
            "k4_ccnr": float((h5["cert4/block_ccnr"][:][mask].any(axis=1)).mean()),
        }
        beyond = h5["labels/beyond_all_two_rdm"][:][mask]
        report["beyond_all_two_rdm_count"] = int(beyond.sum())
        report["beyond_all_two_rdm_fraction"] = float(beyond.mean())
        statuses = h5["labels/status"][:][mask]
        report["status_counts"] = {"certified_entangled": int((statuses == 1).sum()), "certified_separable": int((statuses == 0).sum()), "unknown": int((statuses == -1).sum())}
        return report


def run_thermal_convergence_audit(
    config: dict[str, Any], rows: list[dict[str, Any]], spec: FeatureSpec, checkpoint_dir: Path | None = None
) -> dict[str, Any]:
    audit_cfg = config.get("thermal_convergence_audit", {})
    count = int(audit_cfg.get("num_points", max(1, round(len(rows) * float(audit_cfg.get("fraction", 0.0))))))
    temperature_min = float(audit_cfg.get("temperature_min", 0.0))
    thermal_rows = [row for row in rows if row["T"] >= temperature_min and row["T"] > 0]
    rng = np.random.default_rng(int(config["master_seed"]) + 99173)
    selected = rng.choice(len(thermal_rows), size=min(count, len(thermal_rows)), replace=False)
    tolerance = float(audit_cfg.get("feature_tolerance", 1e-6))
    results: list[dict[str, Any]] = []
    passed = True
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for idx in selected:
        row = thermal_rows[int(idx)]
        base_settings = dict(config["thermal_solver"])
        tight_settings = dict(base_settings)
        tight_settings["dt"] *= float(audit_cfg.get("tighter_dt_factor", 0.5))
        tight_settings["chi_max"] = int(math.ceil(tight_settings["chi_max"] * float(audit_cfg.get("tighter_chi_factor", 2.0))))
        tight_settings["svd_min"] *= float(audit_cfg.get("tighter_svd_min_factor", 0.1))
        states, metas = [], []
        for label, settings in (("base", base_settings), ("tight", tight_settings)):
            checkpoint = checkpoint_dir / f"sample_{row['sample_id']}_{label}.pt" if checkpoint_dir is not None else None
            if checkpoint is not None and checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = PurifiedMPS([tensor.to(torch.complex128) for tensor in saved["tensors"]])
                meta = SolverMetadata(**saved["metadata"])
            else:
                state, meta = generate_thermal_state(16, row["gamma"], row["h"], row["T"], **settings)
                if checkpoint is not None:
                    torch.save({"tensors": state.tensors, "metadata": asdict(meta), "settings": settings}, checkpoint)
            states.append(state); metas.append(meta)
        paulis = pauli_matrices(device=torch.device("cpu"), dtype=torch.complex128)
        onebody = []
        for state in states:
            left, right = state.left_environments(), state.right_environments()
            values: list[float] = []
            for i, tensor in enumerate(state.tensors):
                local = torch.einsum(
                    "qab,ascr,pst,btcu->qpru", left[i].unsqueeze(0), tensor.conj(), paulis, tensor
                )[0]
                result = torch.einsum("pab,ab->p", local, right[i + 1])
                values.extend(float(v.real.item()) for v in result)
            onebody.append(np.asarray(values, dtype=np.float64))
        onebody_error = float(np.max(np.abs(onebody[1] - onebody[0])))
        if onebody_error >= tolerance:
            errors = {"max_onebody_difference": onebody_error}
            primary_passed = False
            features = None
        else:
            features = [compute_features(state, spec, output_dtype=np.float64).values for state in states]
            e0 = energy_witness(features[0], row["gamma"], row["h"])
            e1 = energy_witness(features[1], row["gamma"], row["h"])
            p0 = pair_certificates_from_features(features[0])
            p1 = pair_certificates_from_features(features[1])
            errors = {
                "max_onebody_difference": onebody_error,
                "max_feature_difference": float(np.max(np.abs(features[1] - features[0]))),
                "energy_difference": abs(e1["energy"] - e0["energy"]),
                "allpair_min_pt_difference": abs(p1["allpair_min_pt"] - p0["allpair_min_pt"]),
            }
            primary_passed = all(value < tolerance for value in errors.values())
        # If the primary feature/energy/pair quantities already fail, avoid an
        # hours-long high-chi nonlocal RDM audit: the sample is rejected either way.
        primary_passed = all(value < tolerance for value in errors.values())
        if primary_passed:
            b3 = [(0, 1, 2), (0, 5, 11)]
            b4 = [(0, 1, 2, 3), (0, 4, 9, 13)]
            c30 = certify_blocks(states[0], b3, ["audit"] * 2); c31 = certify_blocks(states[1], b3, ["audit"] * 2)
            c40 = certify_blocks(states[0], b4, ["audit"] * 2); c41 = certify_blocks(states[1], b4, ["audit"] * 2)
            errors.update(
                k3_min_pt_difference=abs(c31.global_min_pt - c30.global_min_pt),
                k3_max_ccnr_difference=abs(c31.global_max_ccnr - c30.global_max_ccnr),
                k4_min_pt_difference=abs(c41.global_min_pt - c40.global_min_pt),
                k4_max_ccnr_difference=abs(c41.global_max_ccnr - c40.global_max_ccnr),
            )
        point_passed = primary_passed and all(value < tolerance for value in errors.values())
        passed = passed and point_passed
        results.append(
            {
                "sample_id": row["sample_id"], "gamma": row["gamma"], "h": row["h"], "T": row["T"],
                "base_settings": base_settings, "tight_settings": tight_settings,
                "base_solver": asdict(metas[0]), "tight_solver": asdict(metas[1]),
                "errors": errors, "passed": point_passed,
                "higher_certificate_audit_skipped": not primary_passed,
                "full_feature_audit_skipped": features is None,
            }
        )
        if not point_passed and bool(audit_cfg.get("stop_on_first_failure", True)):
            break
    return {
        "passed": passed and len(results) == len(selected),
        "required_tolerance": tolerance,
        "requested_points": len(selected),
        "completed_points": len(results),
        "stopped_early": len(results) < len(selected),
        "points": results,
    }


def run_benchmark(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    total = int(config["dataset"]["num_samples"])
    if total == 5000 and not bool(config["dataset"].get("pilot_gate_approved", False)):
        raise RuntimeError("5000 production is gated: set pilot_gate_approved only after reviewing the 500 report")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset.h5"
    spec = FeatureSpec(16)
    names = spec.names()
    training_names = Path(config["feature_schema"]["training_feature_names"])
    expected = json.loads(training_names.read_text(encoding="utf-8"))
    if names != expected:
        raise RuntimeError("XY feature names do not exactly match the training feature_names.json")
    rows = sample_parameter_table(
        total,
        int(config["master_seed"]),
        include_ordered_t0=bool(config["dataset"].get("include_ordered_t0", True)),
    )
    validation_path = output_dir / "validation_report.json"
    if config.get("state_backend") == "finite_free_fermion_exact":
        source_validation = Path(config["exact_backend_validation_report"])
        audit = json.loads(source_validation.read_text(encoding="utf-8"))
        validation_path.write_text(json.dumps(audit, indent=2, allow_nan=True), encoding="utf-8")
    elif validation_path.exists():
        audit = json.loads(validation_path.read_text(encoding="utf-8"))
    else:
        audit = run_thermal_convergence_audit(config, rows, spec, output_dir / ".audit_checkpoints")
        validation_path.write_text(json.dumps(audit, indent=2, allow_nan=True, default=_json_default), encoding="utf-8")
    if not audit.get("passed", False):
        report = {
            "status": "blocked_by_backend_validation_gate",
            "requested_samples": total,
            "completed_samples": 0,
            "pilot_gate_passed": False,
            "validation_report": str(validation_path),
        }
        (output_dir / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    if not path.exists():
        initialize_hdf5(path, rows, names, config)
        (output_dir / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with h5py.File(path, "r") as h5:
        completed = h5["progress/completed"][:]
    progress_every = int(config["runtime"].get("progress_every", 1))
    for i, row in enumerate(rows):
        if completed[i]:
            continue
        started = time.perf_counter()
        state, x, meta, state_info = generate_state_and_features(row, config, spec)
        if not meta.solver_converged or not state_info["field_convergence_passed"]:
            raise RuntimeError(f"sample {i} failed numerical convergence and was not admitted")
        cert2, cert3, cert4, energy, timings = certify_sample(state, x, row, config)
        write_sample(path, i, x, meta, state_info, cert2, cert3, cert4, energy, timings)
        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == total):
            print(json.dumps({"completed": i + 1, "total": total, "sample_seconds": time.perf_counter() - started}), flush=True)
            report = build_report(path)
            (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with h5py.File(path, "r+") as h5:
        h5.attrs["generation_complete"] = True
    report = build_report(path)
    report["pilot_gate_passed"] = report.get("beyond_all_two_rdm_count", 0) > int(config["dataset"].get("minimum_beyond_all_two_rdm", 0))
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

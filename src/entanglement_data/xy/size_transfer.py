from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Any, Protocol, Sequence

import numpy as np

from .free_fermion_exact import AXES, XYFreeFermionExactBackend


class RDMState(Protocol):
    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray: ...


@dataclass(frozen=True)
class LocalRawSpec:
    N: int
    R_obs: int = 4

    @property
    def one_physical(self) -> int:
        return 3 * self.N

    @property
    def two_physical(self) -> int:
        return 9 * sum(self.N - d for d in range(1, self.R_obs + 1))

    @property
    def three_physical(self) -> int:
        return 27 * (self.N - 2)

    @property
    def four_physical(self) -> int:
        return 81 * (self.N - 3)

    @property
    def total_physical(self) -> int:
        return self.one_physical + self.two_physical + self.three_physical + self.four_physical


def local_masks(spec: LocalRawSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = np.zeros((spec.N, spec.R_obs), dtype=bool)
    for distance in range(1, spec.R_obs + 1):
        pair[: spec.N - distance, distance - 1] = True
    triple = np.arange(spec.N) < spec.N - 2
    four = np.arange(spec.N) < spec.N - 3
    return pair, triple, four


def connected_from_raw(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray, spec: LocalRawSpec) -> tuple[np.ndarray, np.ndarray]:
    """Derive K2 and the strict third cumulant K3, preserving padded shapes."""

    if t1.shape != (spec.N, 3) or t2.shape != (spec.N, spec.R_obs, 3, 3) or t3.shape != (spec.N, 3, 3, 3):
        raise ValueError("raw local tensor shape mismatch")
    k2 = np.zeros_like(t2)
    pair_mask, triple_mask, _ = local_masks(spec)
    for i, d0 in zip(*np.nonzero(pair_mask)):
        d = d0 + 1
        k2[i, d0] = t2[i, d0] - np.einsum("a,b->ab", t1[i], t1[i + d])
    k3 = np.zeros_like(t3)
    for i in np.flatnonzero(triple_mask):
        mi, mj, mk = t1[i], t1[i + 1], t1[i + 2]
        tij, tjk, tik = t2[i, 0], t2[i + 1, 0], t2[i, 1]
        k3[i] = (
            t3[i] - np.einsum("a,bc->abc", mi, tjk)
            - np.einsum("b,ac->abc", mj, tik)
            - np.einsum("c,ab->abc", mk, tij)
            + 2.0 * np.einsum("a,b,c->abc", mi, mj, mk)
        )
    return k2, k3


def _moment(backend: XYFreeFermionExactBackend, beta: float, sites: Sequence[int]) -> np.ndarray:
    k = len(sites)
    values = [backend.pauli_string_expectation(sites, axes, beta) for axes in product(AXES, repeat=k)]
    return np.asarray(values, dtype=np.float64).reshape((3,) * k)


def thermal_local_raw(backend: XYFreeFermionExactBackend, beta: float, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    """Exact finite-PBC local moments, exploiting only exact translation symmetry."""

    if backend.N != spec.N:
        raise ValueError("backend and feature size differ")
    pair_mask, triple_mask, four_mask = local_masks(spec)
    t1 = np.repeat(_moment(backend, beta, (0,))[None], spec.N, axis=0)
    t2 = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for distance in range(1, spec.R_obs + 1):
        base = _moment(backend, beta, (0, distance))
        t2[: spec.N - distance, distance - 1] = base
    t3 = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    t3[triple_mask] = _moment(backend, beta, (0, 1, 2))
    t4 = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    t4[four_mask] = _moment(backend, beta, (0, 1, 2, 3))
    k2, k3 = connected_from_raw(t1, t2, t3, spec)
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "K2": k2, "K3": k3,
            "T2_mask": pair_mask, "T3_mask": triple_mask, "T4_mask": four_mask}


def product_local_raw(bloch: np.ndarray, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    bloch = np.asarray(bloch, dtype=np.float64)
    if bloch.shape == (3,):
        bloch = np.repeat(bloch[None], spec.N, axis=0)
    if bloch.shape != (spec.N, 3):
        raise ValueError("bloch must have shape [3] or [N,3]")
    pair_mask, triple_mask, four_mask = local_masks(spec)
    t1 = bloch.copy()
    t2 = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for i, d0 in zip(*np.nonzero(pair_mask)):
        t2[i, d0] = np.einsum("a,b->ab", bloch[i], bloch[i + d0 + 1])
    t3 = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    for i in np.flatnonzero(triple_mask):
        t3[i] = np.einsum("a,b,c->abc", bloch[i], bloch[i + 1], bloch[i + 2])
    t4 = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    for i in np.flatnonzero(four_mask):
        t4[i] = np.einsum("a,b,c,d->abcd", bloch[i], bloch[i + 1], bloch[i + 2], bloch[i + 3])
    k2, k3 = connected_from_raw(t1, t2, t3, spec)
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "K2": k2, "K3": k3,
            "T2_mask": pair_mask, "T3_mask": triple_mask, "T4_mask": four_mask}


def factorized_bloch(gamma: float, h: float) -> np.ndarray:
    if abs(gamma * gamma + h * h - 1.0) > 1e-9:
        raise ValueError("factorized anchor is off gamma^2+h^2=1")
    phi = 0.5 * math.acos(float(np.clip(h / (1.0 + gamma), -1.0, 1.0)))
    return np.asarray([math.sin(2 * phi), 0.0, math.cos(2 * phi)], dtype=np.float64)


def cyclic_canonical(block: Sequence[int], N: int) -> tuple[int, ...]:
    values = tuple(sorted(int(x) for x in block))
    return min(tuple(sorted((site - shift) % N for site in values)) for shift in values)


@lru_cache(maxsize=16)
def cyclic_orbits(N: int, k: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    groups: dict[tuple[int, ...], int] = {}
    for block in combinations(range(N), k):
        key = cyclic_canonical(block, N)
        groups[key] = groups.get(key, 0) + 1
    return tuple(sorted(groups.items()))


def _partial_transpose_numpy(rho: np.ndarray, k: int, cut: tuple[int, ...]) -> np.ndarray:
    axes = list(range(2 * k))
    for site in cut:
        axes[site], axes[k + site] = axes[k + site], axes[site]
    return rho.reshape((2,) * (2 * k)).transpose(axes).reshape(2**k, 2**k)


def _ccnr_numpy(rho: np.ndarray, k: int, cut: tuple[int, ...]) -> float:
    right = tuple(i for i in range(k) if i not in cut)
    axes = list(cut) + [k + i for i in cut] + list(right) + [k + i for i in right]
    matrix = rho.reshape((2,) * (2 * k)).transpose(axes).reshape(4 ** len(cut), 4 ** len(right))
    return float(np.linalg.svd(matrix, compute_uv=False).sum())


def bipartitions(k: int) -> tuple[tuple[int, ...], ...]:
    cuts = []
    full = (1 << k) - 1
    for mask in range(1, full):
        complement = full ^ mask
        if mask < complement:
            cuts.append(tuple(i for i in range(k) if mask & (1 << i)))
    return tuple(cuts)


def certify_translation_orbits(
    state: RDMState, N: int, k: int, *, tol_pt: float = 1e-9, tol_ccnr: float = 1e-9
) -> dict[str, Any]:
    """Certify every k-subset exactly via PBC translation-orbit representatives."""

    orbit_data = cyclic_orbits(N, k)
    cuts = bipartitions(k)
    pt = np.empty((len(orbit_data), len(cuts)), dtype=np.float64)
    ccnr = np.empty_like(pt)
    min_rdm_eigenvalue = math.inf
    max_trace_error = 0.0
    max_hermiticity_error = 0.0
    for index, (block, _) in enumerate(orbit_data):
        rho = np.asarray(state.reduced_density_matrix(block), dtype=np.complex128)
        max_hermiticity_error = max(max_hermiticity_error, float(np.max(np.abs(rho - rho.conj().T))))
        max_trace_error = max(max_trace_error, abs(float(np.trace(rho).real) - 1.0))
        min_rdm_eigenvalue = min(min_rdm_eigenvalue, float(np.linalg.eigvalsh(rho).min()))
        for cut_index, cut in enumerate(cuts):
            pt[index, cut_index] = float(np.linalg.eigvalsh(_partial_transpose_numpy(rho, k, cut)).min())
            ccnr[index, cut_index] = _ccnr_numpy(rho, k, cut)
    block_min, block_max = pt.min(axis=1), ccnr.max(axis=1)
    npt, realigned = block_min < -tol_pt, block_max > 1.0 + tol_ccnr
    multiplicity = np.asarray([value for _, value in orbit_data], dtype=np.int32)
    best_pt, best_ccnr = int(np.argmin(block_min)), int(np.argmax(block_max))
    return {
        "representatives": np.asarray([block for block, _ in orbit_data], dtype=np.int8),
        "multiplicity": multiplicity, "pt_eigs": pt, "ccnr_norms": ccnr,
        "block_min_pt": block_min, "block_max_ccnr": block_max,
        "block_npt": npt, "block_ccnr": realigned,
        "any_npt": bool(npt.any()), "any_ccnr": bool(realigned.any()),
        "npt_block_count": int(multiplicity[npt].sum()),
        "ccnr_block_count": int(multiplicity[realigned].sum()),
        "global_min_pt": float(block_min[best_pt]), "global_max_ccnr": float(block_max[best_ccnr]),
        "best_npt_representative": best_pt, "best_ccnr_representative": best_ccnr,
        "best_npt_cut": int(np.argmin(pt[best_pt])), "best_ccnr_cut": int(np.argmax(ccnr[best_ccnr])),
        "min_rdm_eigenvalue": min_rdm_eigenvalue,
        "max_rdm_trace_error": max_trace_error, "max_rdm_hermiticity_error": max_hermiticity_error,
        "full_subset_count": math.comb(N, k), "representative_count": len(orbit_data),
    }


class ThermalRDMState:
    def __init__(self, backend: XYFreeFermionExactBackend, beta: float):
        self.backend, self.beta = backend, float(beta)

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        return self.backend.reduced_density_matrix(sites, self.beta)


class ProductRDMState:
    def __init__(self, bloch: np.ndarray, N: int):
        self.bloch = np.asarray(bloch, dtype=np.float64)
        if self.bloch.shape == (3,):
            self.bloch = np.repeat(self.bloch[None], N, axis=0)

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        paulis = (
            np.eye(2, dtype=np.complex128), np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128), np.array([[1, 0], [0, -1]], dtype=np.complex128),
        )
        rho = np.asarray([[1.0 + 0j]])
        for site in sites:
            local = (paulis[0] + sum(self.bloch[int(site), a] * paulis[a + 1] for a in range(3))) / 2
            rho = np.kron(rho, local)
        return rho

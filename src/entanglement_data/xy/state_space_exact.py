from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


PAULI_LABELS = ("X", "Y", "Z")


@dataclass(frozen=True)
class SectorGroundState:
    parity: int
    energies: np.ndarray
    vector: np.ndarray


@dataclass(frozen=True)
class ExactGroundStateResult:
    gamma: float
    h: float
    even: SectorGroundState
    odd: SectorGroundState
    selected_parity: int
    selected_vector: np.ndarray
    ground_energy: float
    global_gap: float
    parity_splitting: float
    degeneracy_count: int
    near_degenerate: bool
    convention: str


class XYParityExactDiagonalizer:
    """Sparse finite-N PBC XY diagonalizer in exact spin-parity sectors.

    It implements the same Pauli Hamiltonian convention as ``xy_tenpy_model``::

        H = -sum_i [(1+gamma)/2 X_i X_{i+1}
                    +(1-gamma)/2 Y_i Y_{i+1}] - h sum_i Z_i.

    The computational basis uses ``Z|0>=|0>``.  Even/odd mean eigenvalue
    +1/-1 of the physical spin parity ``prod_i Z_i``.
    """

    def __init__(self, N: int = 16, *, eig_tol: float = 2e-12, degeneracy_tol: float = 1e-9):
        if N < 2 or N > 20:
            raise ValueError("exact state-space diagonalization requires 2 <= N <= 20")
        self.N = int(N)
        self.dimension = 1 << self.N
        self.eig_tol = float(eig_tol)
        self.degeneracy_tol = float(degeneracy_tol)
        states = np.arange(self.dimension, dtype=np.uint32)
        # ``int.bit_count`` is unavailable on the frozen remote Python 3.8
        # runtime.  This equivalent path keeps the exact basis convention
        # unchanged while remaining compatible with that environment.
        popcount = np.fromiter((bin(int(value)).count("1") for value in states), dtype=np.int16)
        self._states = states
        self._popcount = popcount
        self._sector_cache = {
            parity: self._build_sector(parity, states[(popcount & 1) == (0 if parity == 1 else 1)])
            for parity in (1, -1)
        }

    def _build_sector(self, parity: int, basis: np.ndarray) -> dict[str, np.ndarray | csr_matrix]:
        index = np.full(self.dimension, -1, dtype=np.int32)
        index[basis] = np.arange(len(basis), dtype=np.int32)
        rows_exchange: list[np.ndarray] = []
        cols_exchange: list[np.ndarray] = []
        rows_pair: list[np.ndarray] = []
        cols_pair: list[np.ndarray] = []
        columns = np.arange(len(basis), dtype=np.int32)
        for site in range(self.N):
            right = (site + 1) % self.N
            flip = np.uint32((1 << site) | (1 << right))
            partners = index[basis ^ flip]
            same = ((basis >> site) & 1) == ((basis >> right) & 1)
            rows_pair.append(partners[same])
            cols_pair.append(columns[same])
            rows_exchange.append(partners[~same])
            cols_exchange.append(columns[~same])

        def matrix(rows: list[np.ndarray], cols: list[np.ndarray]) -> csr_matrix:
            row = np.concatenate(rows)
            col = np.concatenate(cols)
            return csr_matrix((-np.ones(len(row)), (row, col)), shape=(len(basis), len(basis)))

        popcount = self._popcount[basis]
        field_diagonal = -(self.N - 2.0 * popcount.astype(np.float64))
        return {
            "parity": np.asarray(parity),
            "basis": basis,
            "exchange": matrix(rows_exchange, cols_exchange),
            "pair": matrix(rows_pair, cols_pair),
            "field": field_diagonal,
        }

    def sector_hamiltonian(self, parity: int, gamma: float, h: float) -> csr_matrix:
        data = self._sector_cache[int(parity)]
        return data["exchange"] + float(gamma) * data["pair"] + diags(float(h) * data["field"])

    def sector_coordinates(self, parity: int, full_vector: np.ndarray) -> np.ndarray:
        basis = np.asarray(self._sector_cache[int(parity)]["basis"], dtype=np.int64)
        return np.asarray(full_vector, dtype=np.complex128)[basis].real

    def solve_sector(
        self,
        parity: int,
        gamma: float,
        h: float,
        *,
        levels: int = 3,
        v0: np.ndarray | None = None,
    ) -> SectorGroundState:
        matrix = self.sector_hamiltonian(parity, gamma, h)
        if v0 is None:
            rng = np.random.default_rng(1600 + (0 if parity == 1 else 1))
            v0 = rng.normal(size=matrix.shape[0])
        values, vectors = eigsh(
            matrix,
            k=int(levels),
            which="SA",
            v0=np.asarray(v0, dtype=np.float64),
            tol=self.eig_tol,
            maxiter=5000,
            ncv=max(20, 2 * int(levels) + 2),
        )
        order = np.argsort(values)
        values = np.asarray(values[order], dtype=np.float64)
        sector_vector = np.asarray(vectors[:, order[0]], dtype=np.complex128)
        full = np.zeros(self.dimension, dtype=np.complex128)
        full[np.asarray(self._sector_cache[int(parity)]["basis"], dtype=np.int64)] = sector_vector
        full /= np.linalg.norm(full)
        return SectorGroundState(int(parity), values, full)

    def solve(
        self,
        gamma: float,
        h: float,
        *,
        levels: int = 3,
        v0_even: np.ndarray | None = None,
        v0_odd: np.ndarray | None = None,
    ) -> ExactGroundStateResult:
        even = self.solve_sector(1, gamma, h, levels=levels, v0=v0_even)
        odd = self.solve_sector(-1, gamma, h, levels=levels, v0=v0_odd)
        splitting = float(abs(even.energies[0] - odd.energies[0]))
        if even.energies[0] <= odd.energies[0] + self.degeneracy_tol:
            selected = even
        else:
            selected = odd
        candidates = np.sort(np.concatenate((even.energies, odd.energies)))
        ground = float(candidates[0])
        above = candidates[candidates > ground + self.degeneracy_tol]
        gap = float(above[0] - ground) if len(above) else 0.0
        degeneracy = int(np.sum(np.abs(candidates - ground) <= self.degeneracy_tol))
        return ExactGroundStateResult(
            gamma=float(gamma),
            h=float(h),
            even=even,
            odd=odd,
            selected_parity=selected.parity,
            selected_vector=selected.vector,
            ground_energy=ground,
            global_gap=gap,
            parity_splitting=splitting,
            degeneracy_count=degeneracy,
            near_degenerate=splitting <= self.degeneracy_tol,
            convention="lowest physical-parity sector; even selected within degeneracy tolerance",
        )


def product_state_vector(local: Sequence[complex], N: int) -> np.ndarray:
    vector = np.asarray(local, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    result = np.asarray([1.0 + 0.0j])
    for _ in range(int(N)):
        result = np.kron(result, vector)
    return result


def matched_two_sector_branch(result: ExactGroundStateResult, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Project a reference branch onto the two lowest parity-sector states.

    Away from an exact degeneracy this is a symmetry-broken two-state proxy,
    not an eigenstate.  The returned energy variance is exactly the variance
    caused by mixing the two sector energies.
    """

    reference = np.asarray(reference, dtype=np.complex128)
    even_overlap = np.vdot(result.even.vector, reference)
    odd_overlap = np.vdot(result.odd.vector, reference)
    vector = even_overlap * result.even.vector + odd_overlap * result.odd.vector
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise FloatingPointError("reference has no overlap with the two-sector ground subspace")
    vector /= norm
    pe = abs(np.vdot(result.even.vector, vector)) ** 2
    po = abs(np.vdot(result.odd.vector, vector)) ** 2
    energies = np.asarray([result.even.energies[0], result.odd.energies[0]])
    mean = pe * energies[0] + po * energies[1]
    variance = pe * (energies[0] - mean) ** 2 + po * (energies[1] - mean) ** 2
    return vector, float(variance)


@dataclass(frozen=True)
class DensePureState:
    vector: np.ndarray
    N: int

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.complex128)
        if vector.shape != (1 << int(self.N),):
            raise ValueError("state-vector dimension does not match N")
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"state vector is not normalized: {norm}")
        object.__setattr__(self, "vector", vector / norm)

    @cached_property
    def basis(self) -> np.ndarray:
        return np.arange(1 << self.N, dtype=np.uint32)

    def pauli_expectation(self, sites: Sequence[int], labels: Sequence[str]) -> float:
        if len(sites) != len(labels) or len(set(sites)) != len(sites):
            raise ValueError("sites and labels must be equal-length distinct sequences")
        flipped = self.basis.copy()
        phase = np.ones(len(flipped), dtype=np.complex128)
        for site, label in zip(sites, labels):
            bit = (self.basis >> int(site)) & 1
            upper = str(label).upper()
            if upper == "X":
                flipped ^= np.uint32(1 << int(site))
            elif upper == "Y":
                phase *= 1j * (1.0 - 2.0 * bit)
                flipped ^= np.uint32(1 << int(site))
            elif upper == "Z":
                phase *= 1.0 - 2.0 * bit
            elif upper != "I":
                raise ValueError(f"unknown Pauli label {label!r}")
        value = np.vdot(self.vector[flipped], phase * self.vector)
        if abs(value.imag) > 2e-9:
            raise FloatingPointError(f"Pauli expectation has imaginary residual {value.imag}")
        return float(value.real)

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        sites = tuple(int(site) for site in sites)
        if not sites or len(set(sites)) != len(sites):
            raise ValueError("sites must be non-empty and distinct")
        axes = tuple(self.N - 1 - site for site in sites)
        remainder = tuple(axis for axis in range(self.N) if axis not in axes)
        matrix = self.vector.reshape((2,) * self.N).transpose(axes + remainder).reshape(1 << len(sites), -1)
        rho = matrix @ matrix.conj().T
        return (rho + rho.conj().T) / 2.0


def schmidt_metrics(vector: np.ndarray, N: int, cut: int | None = None, *, tolerance: float = 1e-12) -> dict[str, object]:
    cut = int(N // 2 if cut is None else cut)
    singular = np.linalg.svd(np.asarray(vector).reshape(1 << cut, 1 << (N - cut)), compute_uv=False)
    spectrum = np.maximum(singular * singular, 0.0)
    spectrum /= spectrum.sum()
    positive = spectrum[spectrum > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    renyi2 = -float(np.log(np.sum(spectrum * spectrum)))
    return {
        "spectrum": spectrum,
        "largest": float(spectrum[0]),
        "second": float(spectrum[1]) if len(spectrum) > 1 else 0.0,
        "rank_above_tolerance": int(np.sum(spectrum > tolerance)),
        "von_neumann_entropy": entropy,
        "renyi2_entropy": renyi2,
    }


def translation_invariant_compact(state: DensePureState, R_obs: int = 4) -> np.ndarray:
    """Build the exact frozen OBC-style compact T1/T2/T3 input."""

    N = state.N
    one = np.asarray([state.pauli_expectation((0,), (axis,)) for axis in PAULI_LABELS])
    pair_blocks = {
        distance: np.asarray([
            state.pauli_expectation((0, distance), axes)
            for axes in product(PAULI_LABELS, repeat=2)
        ]).reshape(3, 3)
        for distance in range(1, R_obs + 1)
    }
    triple = np.asarray([
        state.pauli_expectation((0, 1, 2), axes)
        for axes in product(PAULI_LABELS, repeat=3)
    ]).reshape(3, 3, 3)
    return np.concatenate((
        np.tile(one, N),
        *(np.tile(pair_blocks[distance].reshape(-1), N - distance) for distance in range(1, R_obs + 1)),
        np.tile(triple.reshape(-1), N - 2),
    )).astype(np.float64)

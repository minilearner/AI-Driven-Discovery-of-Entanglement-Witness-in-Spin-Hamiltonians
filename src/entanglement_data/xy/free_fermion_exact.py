from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence

import numpy as np
from pfapack import pfaffian

from ..observables.feature_spec import FeatureSpec


PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}
AXES = ("X", "Y", "Z")


@lru_cache(maxsize=4)
def _pauli_basis(k: int) -> tuple[tuple[tuple[str, ...], ...], np.ndarray]:
    strings = []
    operators = []
    for string in product(("I", "X", "Y", "Z"), repeat=k):
        operator = np.asarray([[1.0 + 0.0j]])
        for label in string:
            operator = np.kron(operator, PAULI[label])
        strings.append(string)
        operators.append(operator)
    return tuple(strings), np.stack(operators)


def _pf(matrix: np.ndarray) -> complex:
    if matrix.shape == (0, 0):
        return 1.0 + 0.0j
    # Parlett-Reid is substantially faster than Householder for the many
    # small (at most 32x32) Pfaffians in the N=16 feature workload.
    return complex(pfaffian.pfaffian(np.asarray(matrix, dtype=np.complex128), method="P"))


def _stable_log_2cosh(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(x, -x)


def _projected_mode_thermodynamics(
    energies: np.ndarray, beta: float, parity_sign: int
) -> tuple[float, float]:
    """Stable parity projection for independent positive quasiparticle modes.

    ``parity_sign`` is target physical parity times the quasiparticle-vacuum
    parity.  Zero selects an unprojected Gaussian trace (used for OBC).
    """
    energies = np.asarray(energies, dtype=np.float64)
    x = 0.5 * beta * energies
    logZ0 = float(np.sum(_stable_log_2cosh(x)))
    energy0 = -float(np.sum(0.5 * energies * np.tanh(x)))
    if parity_sign == 0:
        return logZ0, energy0
    if beta == 0 or np.any(energies == 0):
        return logZ0 - math.log(2.0), energy0
    y = np.exp(-2.0 * x)
    logp = float(np.sum(np.log1p(-y) - np.log1p(y)))
    dlogp = float(np.sum(2.0 * energies * np.exp(-2.0 * x) / (-np.expm1(-4.0 * x))))
    if parity_sign > 0:
        logfactor = math.log1p(math.exp(logp)) - math.log(2.0)
        correction = math.exp(logp) / (1.0 + math.exp(logp)) * dlogp
    else:
        one_minus_p = -math.expm1(logp)
        logfactor = math.log(one_minus_p) - math.log(2.0)
        correction = -math.exp(logp) / one_minus_p * dlogp
    return logZ0 + logfactor, energy0 - correction


def _simplify_majorana_word(word: Sequence[int], coefficient: complex = 1.0) -> tuple[complex, tuple[int, ...]]:
    """Canonicalize a Majorana word using a_i a_j=-a_j a_i and a_i^2=1."""
    ordered: list[int] = []
    coeff = complex(coefficient)
    for index in word:
        greater = sum(existing > index for existing in ordered)
        if greater % 2:
            coeff = -coeff
        if index in ordered:
            ordered.remove(index)
        else:
            ordered.append(index)
            ordered.sort()
    return coeff, tuple(ordered)


def pauli_to_majorana(sites: Sequence[int], paulis: Sequence[str]) -> tuple[complex, tuple[int, ...]]:
    if len(sites) != len(paulis):
        raise ValueError("sites and paulis must have equal length")
    if len(set(sites)) != len(sites):
        raise ValueError("sites must be distinct")
    word: list[int] = []
    coefficient = 1.0 + 0.0j
    for site, label in sorted(zip((int(i) for i in sites), (str(p).upper() for p in paulis))):
        if site < 0:
            raise ValueError("site index must be non-negative")
        if label == "I":
            continue
        if label in ("X", "Y"):
            for left in range(site):
                coefficient *= -1j
                word.extend((2 * left, 2 * left + 1))
            word.append(2 * site + (label == "Y"))
        elif label == "Z":
            coefficient *= -1j
            word.extend((2 * site, 2 * site + 1))
        else:
            raise ValueError(f"unknown Pauli label {label!r}")
    return _simplify_majorana_word(word, coefficient)


@dataclass(frozen=True)
class SectorData:
    name: str
    target_parity: int
    boundary_sign: int
    A: np.ndarray
    energies: np.ndarray


class XYFreeFermionExactBackend:
    """Exact finite-even-N XY Gibbs backend with PBC parity projection.

    Majoranas follow ``a[2j]=(prod_{l<j} Z_l) X_j`` and
    ``a[2j+1]=(prod_{l<j} Z_l) Y_j``. The quadratic Hamiltonian is
    ``H=(i/4) a^T A a``. The physical even sector uses antiperiodic
    fermions and the physical odd sector periodic fermions.
    """

    backend_version = "1.0.0"
    parity_sector_method = "majorana_gaussian_trace_with_exact_parity_projection"

    def __init__(self, N: int, gamma: float, h: float, boundary_condition: str = "PBC"):
        if N < 2 or N % 2:
            raise ValueError("N must be a positive even integer")
        boundary = boundary_condition.upper()
        if boundary not in ("PBC", "OBC"):
            raise ValueError("boundary_condition must be PBC or OBC")
        self.N = int(N)
        self.gamma = float(gamma)
        self.h = float(h)
        self.boundary_condition = boundary
        self._cache: dict[tuple[float, tuple[tuple[int, str], ...]], float] = {}
        self._thermal_cache: dict[float, tuple[tuple[SectorData, np.ndarray, float, complex], ...]] = {}
        self.max_imaginary_residual = 0.0
        if boundary == "PBC":
            self.sectors = (
                self._build_sector("even", target_parity=+1, boundary_sign=-1),
                self._build_sector("odd", target_parity=-1, boundary_sign=+1),
            )
        else:
            self.sectors = (self._build_sector("full", target_parity=0, boundary_sign=0),)

    def _majorana_matrix(self, boundary_sign: int) -> np.ndarray:
        N, g, h = self.N, self.gamma, self.h
        jx, jy = (1.0 + g) / 2.0, (1.0 - g) / 2.0
        A = np.zeros((2 * N, 2 * N), dtype=np.float64)

        def add(i: int, j: int, value: float) -> None:
            if i > j:
                i, j, value = j, i, -value
            A[i, j] += value
            A[j, i] -= value

        for site in range(N):
            add(2 * site, 2 * site + 1, 2.0 * h)
        for site in range(N - 1):
            add(2 * site + 1, 2 * (site + 1), 2.0 * jx)
            add(2 * site, 2 * (site + 1) + 1, -2.0 * jy)
        if boundary_sign:
            add(2 * (N - 1) + 1, 0, 2.0 * jx * boundary_sign)
            add(2 * (N - 1), 1, -2.0 * jy * boundary_sign)
        return A

    def _build_sector(self, name: str, target_parity: int, boundary_sign: int) -> SectorData:
        A = self._majorana_matrix(boundary_sign)
        eigenvalues = np.linalg.eigvalsh(1j * A).real
        energies = np.sort(eigenvalues)[self.N :]
        energies[np.abs(energies) < 1e-13] = 0.0
        return SectorData(name, target_parity, boundary_sign, A, energies)

    @staticmethod
    def _covariance(sector: SectorData, beta: float) -> np.ndarray:
        values, vectors = np.linalg.eigh(1j * sector.A)
        gamma = (vectors * np.tanh(0.5 * beta * values)) @ vectors.conj().T
        gamma = (gamma - gamma.T) / 2.0
        np.fill_diagonal(gamma, 0.0)
        return gamma

    def _sector_trace_data(self, sector: SectorData, beta: float):
        gamma = self._covariance(sector, beta)
        x = 0.5 * beta * sector.energies
        logZ0 = float(np.sum(_stable_log_2cosh(x)))
        parity_mean = ((-1j) ** self.N) * _pf(gamma)
        parity_mean = complex(float(np.real_if_close(parity_mean, tol=1000).real), 0.0)
        return gamma, logZ0, parity_mean

    def _all_sector_data(self, beta: float):
        if beta < 0 or not np.isfinite(beta):
            raise ValueError("beta must be finite and non-negative")
        key = float(beta)
        if key not in self._thermal_cache:
            self._thermal_cache[key] = tuple(
                (sector, *self._sector_trace_data(sector, key)) for sector in self.sectors
            )
        return self._thermal_cache[key]

    def log_partition_function(self, beta: float) -> float:
        if self.boundary_condition == "OBC":
            return _projected_mode_thermodynamics(self.sectors[0].energies, beta, 0)[0]
        sector_logs = self.sector_log_partition_functions(beta)
        log_even, log_odd = sector_logs["even"], sector_logs["odd"]
        return float(np.logaddexp(log_even, log_odd))

    # Short aliases form the public production interface requested by the
    # dataset pipeline; the descriptive names above remain useful in reports.
    def logZ(self, beta: float) -> float:
        return self.log_partition_function(beta)

    def sector_log_partition_functions(self, beta: float) -> dict[str, float]:
        if self.boundary_condition == "OBC":
            return {"full": _projected_mode_thermodynamics(self.sectors[0].energies, beta, 0)[0]}
        out = {}
        for sector in self.sectors:
            orientation = float(np.real_if_close(((-1j) ** self.N) * _pf(1j * sector.A), tol=1000).real)
            ground_parity = 1 if orientation >= 0 else -1
            projection_sign = sector.target_parity * ground_parity
            out[sector.name] = _projected_mode_thermodynamics(sector.energies, beta, projection_sign)[0]
        return out

    def thermal_energy(self, beta: float) -> float:
        """Return -d log(Z)/d beta from exact parity-projected spectra."""
        if self.boundary_condition == "OBC":
            return _projected_mode_thermodynamics(self.sectors[0].energies, beta, 0)[1]
        logs = self.sector_log_partition_functions(beta)
        values = []
        for sector in self.sectors:
            orientation = float(np.real_if_close(((-1j) ** self.N) * _pf(1j * sector.A), tol=1000).real)
            projection_sign = sector.target_parity * (1 if orientation >= 0 else -1)
            values.append(_projected_mode_thermodynamics(sector.energies, beta, projection_sign)[1])
        offset = max(logs.values())
        weights = np.asarray([math.exp(logs[sector.name] - offset) for sector in self.sectors])
        return float(weights @ np.asarray(values) / weights.sum())

    def energy(self, beta: float) -> float:
        return self.thermal_energy(beta)

    def _majorana_expectation(self, beta: float, coefficient: complex, indices: tuple[int, ...]) -> complex:
        if len(indices) % 2:
            return 0.0 + 0.0j
        if self.boundary_condition == "OBC":
            gamma = self._all_sector_data(beta)[0][1]
            return coefficient * _pf(gamma[np.ix_(indices, indices)])
        full = tuple(range(2 * self.N))
        terms = []
        for sector, gamma, logZ0, parity_mean in self._all_sector_data(beta):
            u0 = coefficient * _pf(gamma[np.ix_(indices, indices)])
            p_coeff, complement = _simplify_majorana_word(full + indices, ((-1j) ** self.N) * coefficient)
            uP = p_coeff * _pf(gamma[np.ix_(complement, complement)])
            terms.append((sector, logZ0, parity_mean, u0, uP))
        offset = max(item[1] for item in terms)
        numerator = 0.0 + 0.0j
        denominator = 0.0
        for sector, logZ0, parity_mean, u0, uP in terms:
            weight = math.exp(logZ0 - offset)
            numerator += weight * (u0 + sector.target_parity * uP)
            denominator += weight * (1.0 + sector.target_parity * parity_mean.real)
        return numerator / denominator

    def pauli_string_expectation(self, sites: Sequence[int], paulis: Sequence[str], beta: float) -> float:
        canonical = tuple(sorted((int(i), str(p).upper()) for i, p in zip(sites, paulis) if str(p).upper() != "I"))
        if any(i < 0 or i >= self.N for i, _ in canonical):
            raise ValueError("site index out of range")
        if len({i for i, _ in canonical}) != len(canonical):
            raise ValueError("sites must be distinct")
        if self.boundary_condition == "PBC" and canonical:
            canonical = min(
                tuple(sorted(((site - shift) % self.N, label) for site, label in canonical))
                for shift, _ in canonical
            )
        key = (float(beta), canonical)
        if key in self._cache:
            return self._cache[key]
        if sum(label in ("X", "Y") for _, label in canonical) % 2:
            value = 0.0
        else:
            coefficient, indices = pauli_to_majorana([i for i, _ in canonical], [p for _, p in canonical])
            raw = self._majorana_expectation(beta, coefficient, indices)
            self.max_imaginary_residual = max(self.max_imaginary_residual, abs(float(raw.imag)))
            if abs(raw.imag) > 2e-10:
                raise FloatingPointError(f"Pauli expectation has imaginary residue {raw.imag}")
            value = float(raw.real)
        self._cache[key] = value
        return value

    def pauli_expectation(self, sites: Sequence[int], paulis: Sequence[str], beta: float) -> float:
        return self.pauli_string_expectation(sites, paulis, beta)

    def reduced_density_matrix(self, sites: Sequence[int], beta: float) -> np.ndarray:
        sites = tuple(int(i) for i in sites)
        if not sites or len(set(sites)) != len(sites):
            raise ValueError("sites must be a non-empty distinct sequence")
        k = len(sites)
        strings, operators = _pauli_basis(k)
        coefficients = np.empty(len(strings), dtype=np.float64)
        for index, string in enumerate(strings):
            active = [(site, label) for site, label in zip(sites, string) if label != "I"]
            coefficients[index] = 1.0 if not active else self.pauli_string_expectation(
                [x[0] for x in active], [x[1] for x in active], beta
            )
        rho = np.tensordot(coefficients, operators, axes=(0, 0)) / (2**k)
        return (rho + rho.conj().T) / 2.0

    def feature_vector(self, beta: float, feature_schema: FeatureSpec) -> np.ndarray:
        if feature_schema.N != self.N:
            raise ValueError("feature schema N does not match backend")
        return np.asarray([
            self.pauli_string_expectation(term.sites, [AXES[a] for a in term.axes], beta)
            for term in feature_schema.terms()
        ], dtype=np.float64)

    def validate_internal_consistency(self, beta: float, blocks: Sequence[Sequence[int]] = ((0, 1),)) -> dict[str, float | bool]:
        hermiticity, trace_error, min_eigenvalue = 0.0, 0.0, math.inf
        for sites in blocks:
            rho = self.reduced_density_matrix(sites, beta)
            hermiticity = max(hermiticity, float(np.max(np.abs(rho - rho.conj().T))))
            trace_error = max(trace_error, abs(float(np.trace(rho).real) - 1.0))
            min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(rho).min()))
        return {
            "passed": hermiticity < 1e-11 and trace_error < 1e-11 and min_eigenvalue > -1e-10,
            "max_rdm_hermiticity_error": hermiticity,
            "max_rdm_trace_error": trace_error,
            "min_rdm_eigenvalue": min_eigenvalue,
        }


def appendix_a_thermodynamics(N: int, gamma: float, h: float, beta: float) -> tuple[float, float, float]:
    """Independent Appendix-A Eqs. (A1)-(A12): (logZ_even, logZ_odd, energy)."""
    if N % 2:
        raise ValueError("Appendix A formula requires even N")
    m = np.arange(1, N // 2 + 1)
    p = np.pi * (2 * m - 1) / N
    ep = 2.0 * np.sqrt((np.cos(p) - h) ** 2 + (gamma * np.sin(p)) ** 2)
    n = np.arange(1, N // 2)
    q = 2.0 * np.pi * n / N
    eq = 2.0 * np.sqrt((np.cos(q) - h) ** 2 + (gamma * np.sin(q)) ** 2)

    # Each non-endpoint momentum occurs in a +/- pair.  In the periodic
    # sector the Bogoliubov vacuum parity is sign(h^2-1); the physical odd
    # projection therefore keeps the vacuum for |h|<1 and removes it for
    # |h|>1.  This mode-product form is algebraically equivalent to A5-A12
    # but avoids subtraction of exponentially large C^(+) and C^(-) terms.
    logZe, energy_even = _projected_mode_thermodynamics(np.repeat(ep, 2), beta, +1)
    periodic_energies = np.concatenate((np.repeat(eq, 2), [2.0 * abs(h - 1.0), 2.0 * abs(h + 1.0)]))
    odd_projection_sign = +1 if abs(h) < 1.0 else -1
    logZo, energy_odd = _projected_mode_thermodynamics(periodic_energies, beta, odd_projection_sign)
    zoff = max(logZe, logZo)
    we, wo = math.exp(logZe - zoff), math.exp(logZo - zoff)
    return logZe, logZo, (we * energy_even + wo * energy_odd) / (we + wo)

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.linalg import eigh

from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.xy.free_fermion_exact import (
    PAULI,
    XYFreeFermionExactBackend,
    appendix_a_thermodynamics,
)


def _operator(N: int, sites: tuple[int, ...], labels: tuple[str, ...]) -> np.ndarray:
    lookup = dict(zip(sites, labels))
    out = np.asarray([[1.0 + 0.0j]])
    for site in range(N):
        out = np.kron(out, PAULI[lookup.get(site, "I")])
    return out


def _dense_gibbs(
    N: int, gamma: float, h: float, beta: float, boundary_condition: str = "PBC"
) -> tuple[float, float, np.ndarray]:
    H = np.zeros((2**N, 2**N), dtype=np.complex128)
    jx, jy = (1 + gamma) / 2, (1 - gamma) / 2
    bond_sites = range(N) if boundary_condition == "PBC" else range(N - 1)
    for site in bond_sites:
        neighbor = site + 1 if boundary_condition == "OBC" else (site + 1) % N
        H -= jx * _operator(N, (site, neighbor), ("X", "X"))
        H -= jy * _operator(N, (site, neighbor), ("Y", "Y"))
    for site in range(N):
        H -= h * _operator(N, (site,), ("Z",))
    eigenvalues, eigenvectors = eigh(H)
    exponents = -beta * eigenvalues
    offset = float(exponents.max())
    weights = np.exp(exponents - offset)
    partition = float(weights.sum())
    rho = (eigenvectors * weights) @ eigenvectors.conj().T / partition
    logZ = offset + np.log(partition)
    energy = float(eigenvalues @ weights / partition)
    return float(logZ), energy, rho


@pytest.mark.parametrize("N,boundary_condition", product((4, 6, 8), ("PBC", "OBC")))
def test_exact_backend_matches_dense_ed_random_pauli_strings(N: int, boundary_condition: str) -> None:
    rng = np.random.default_rng(4100 + N)
    for _ in range(2):
        gamma = float(rng.uniform(0.05, 0.95))
        h = float(rng.uniform(-1.25, 1.25))
        beta = float(rng.uniform(0.1, 4.0))
        reference_logZ, reference_energy, rho = _dense_gibbs(N, gamma, h, beta, boundary_condition)
        backend = XYFreeFermionExactBackend(N, gamma, h, boundary_condition)
        assert backend.logZ(beta) == pytest.approx(reference_logZ, abs=2e-11)
        assert backend.energy(beta) == pytest.approx(reference_energy, abs=2e-11)
        for order in (1, 2, 3, 4):
            for _ in range(3):
                sites = tuple(sorted(rng.choice(N, size=order, replace=False).tolist()))
                labels = tuple(rng.choice(("X", "Y", "Z"), size=order).tolist())
                reference = np.trace(rho @ _operator(N, sites, labels))
                assert abs(reference.imag) < 2e-11
                assert backend.pauli_expectation(sites, labels, beta) == pytest.approx(reference.real, abs=3e-11)


def test_appendix_a_thermodynamics_matches_majorana_backend() -> None:
    rng = np.random.default_rng(501)
    for _ in range(12):
        gamma = float(rng.uniform(0.01, 0.99))
        h = float(rng.uniform(-1.4, 1.4))
        # This comparison uses the direct double-precision Pfaffian parity
        # factor as an independent reference, so avoid the beta->infinity
        # cancellation that the Appendix-A log formula is designed to fix.
        beta = float(10 ** rng.uniform(-2.0, 0.7))
        backend = XYFreeFermionExactBackend(16, gamma, h)
        log_even, log_odd, energy = appendix_a_thermodynamics(16, gamma, h, beta)
        gaussian_logs = {}
        for sector, _, logZ0, parity_mean in backend._all_sector_data(beta):
            gaussian_logs[sector.name] = logZ0 + np.log((1 + sector.target_parity * parity_mean.real) / 2)
        assert log_even == pytest.approx(gaussian_logs["even"], abs=2e-9)
        assert log_odd == pytest.approx(gaussian_logs["odd"], abs=2e-9)
        step = 1e-5 * max(1.0, beta)
        lp = backend.logZ(beta + step)
        lm = backend.logZ(beta - step)
        assert energy == pytest.approx(-(lp - lm) / (2 * step), abs=2e-7)


def test_rdm_pauli_expansion_and_feature_schema() -> None:
    backend = XYFreeFermionExactBackend(4, 0.37, 0.21)
    beta = 1.7
    rho = backend.reduced_density_matrix((0, 2, 3), beta)
    assert np.max(np.abs(rho - rho.conj().T)) < 1e-12
    assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.eigvalsh(rho).min() > -1e-11
    for labels in product(("X", "Y", "Z"), repeat=3):
        reference = np.trace(rho @ _operator(3, (0, 1, 2), labels)).real
        assert backend.pauli_expectation((0, 2, 3), labels, beta) == pytest.approx(reference, abs=2e-11)
    features = backend.feature_vector(beta, FeatureSpec(4))
    assert features.shape == (FeatureSpec(4).num_features,)
    assert np.isfinite(features).all()


@pytest.mark.parametrize("boundary_condition", ("PBC", "OBC"))
@pytest.mark.parametrize("sites", ((0, 4), (0, 2, 5), (0, 1, 3, 5)))
def test_exact_rdm_matches_dense_partial_trace(boundary_condition: str, sites: tuple[int, ...]) -> None:
    N, gamma, h, beta = 6, 0.417, -0.283, 1.31
    _, _, rho = _dense_gibbs(N, gamma, h, beta, boundary_condition)
    rest = tuple(i for i in range(N) if i not in sites)
    axes = sites + rest + tuple(N + i for i in sites) + tuple(N + i for i in rest)
    shaped = rho.reshape((2,) * (2 * N)).transpose(axes)
    expected = np.einsum("abcb->ac", shaped.reshape(2 ** len(sites), 2 ** len(rest), 2 ** len(sites), 2 ** len(rest)))
    actual = XYFreeFermionExactBackend(N, gamma, h, boundary_condition).reduced_density_matrix(sites, beta)
    assert np.max(np.abs(actual - expected)) < 3e-11


def test_low_temperature_thermodynamics_stays_stable() -> None:
    for gamma, h in ((0.4, 0.3), (0.2, 0.95), (0.8, 1.2), (0.01, 1.0)):
        logZ, energy, _ = _dense_gibbs(8, gamma, h, 100.0)
        backend = XYFreeFermionExactBackend(8, gamma, h)
        assert backend.logZ(100.0) == pytest.approx(logZ, abs=2e-10)
        assert backend.energy(100.0) == pytest.approx(energy, abs=2e-11)

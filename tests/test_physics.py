from __future__ import annotations

from itertools import product

import numpy as np
import torch

from entanglement_data.certification.ppt import (
    apply_channel_to_two_site_rdms,
    certify_two_site_npt,
    certify_two_site_rdms,
    mps_two_site_density_matrices,
    partial_transpose_two_qubit,
    two_site_density_matrix,
)
from entanglement_data.observables.engine import compute_features
from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.pauli import pauli_matrices
from entanglement_data.states.mps import (
    mps_from_statevector,
    purified_mps_to_dense_density,
    random_mps,
)
from entanglement_data.states.noise import apply_local_noise, channel_kraus, haar_local_unitaries
from entanglement_data.states.separable import SeparableMixture


DEVICE = torch.device("cpu")
DTYPE = torch.complex128


def dense_expectation(rho: torch.Tensor, N: int, sites: tuple[int, ...], ops: tuple[torch.Tensor, ...]) -> torch.Tensor:
    op_map = dict(zip(sites, ops))
    full = torch.ones((1, 1), dtype=DTYPE)
    identity = torch.eye(2, dtype=DTYPE)
    for i in range(N):
        full = torch.kron(full, op_map.get(i, identity))
    return torch.trace(rho @ full)


def dense_separable(state: SeparableMixture) -> torch.Tensor:
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    identity = torch.eye(2, dtype=DTYPE)
    rho = torch.zeros((2**state.N, 2**state.N), dtype=DTYPE)
    for k in range(state.weights.numel()):
        component = torch.ones((1, 1), dtype=DTYPE)
        for i in range(state.N):
            local = identity.clone()
            for a in range(3):
                local += state.bloch[k, i, a] * paulis[a]
            component = torch.kron(component, local / 2)
        rho += state.weights[k] * component
    return rho


def test_random_mps_correlators_equal_dense() -> None:
    generator = torch.Generator().manual_seed(7)
    state = random_mps(5, 4, generator=generator, device=DEVICE)
    rho = purified_mps_to_dense_density(state)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    cases = [((0,), (2,)), ((1, 4), (0, 1)), ((0, 2, 3), (1, 2, 0)), ((0, 1, 2, 4), (0, 1, 2, 0))]
    for sites, axes in cases:
        ops = tuple(paulis[a] for a in axes)
        actual = state.expectation(sites, ops)
        expected = dense_expectation(rho, 5, sites, ops)
        assert torch.abs(actual - expected) < 1e-10


def test_noisy_purification_correlators_equal_dense() -> None:
    generator = torch.Generator().manual_seed(11)
    pure = random_mps(4, 3, generator=generator, device=DEVICE)
    state = apply_local_noise(pure, "amplitude_damping", 0.23)
    rho = purified_mps_to_dense_density(state)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    for sites, axes in [((0,), (0,)), ((0, 3), (2, 1)), ((1, 2, 3), (0, 1, 2))]:
        ops = tuple(paulis[a] for a in axes)
        assert torch.abs(state.expectation(sites, ops) - dense_expectation(rho, 4, sites, ops)) < 1e-10
    expected_purity = torch.trace(rho @ rho).real.item()
    assert state.purity() == pytest_approx(expected_purity, 1e-10)
    assert pure.purity() == pytest_approx(1.0, 1e-10)


def test_separable_formula_and_ppt() -> None:
    generator = torch.Generator().manual_seed(19)
    state = SeparableMixture.random(4, 4, generator=generator, device=DEVICE)
    rho = dense_separable(state)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    for sites, axes in [((2,), (1,)), ((0, 3), (2, 0)), ((0, 1, 2), (0, 1, 2))]:
        expected = dense_expectation(rho, 4, sites, tuple(paulis[a] for a in axes))
        actual = state.pauli_expectation(sites, axes)
        assert abs(float(actual) - float(expected.real)) < 1e-10
    assert not certify_two_site_npt(state, max_distance=3).certified


def test_classical_ghz_mixture_has_long_range_correlation_but_is_separable() -> None:
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    bloch = torch.zeros((2, 6, 3), dtype=torch.float64)
    bloch[0, :, 2] = 1.0
    bloch[1, :, 2] = -1.0
    state = SeparableMixture(weights, bloch)
    assert abs(float(state.pauli_expectation((0,), (2,)))) < 1e-12
    assert float(state.pauli_expectation((0, 5), (2, 2))) == pytest_approx(1.0, 1e-12)
    assert not certify_two_site_npt(state, max_distance=5).certified


def test_bell_pair_is_npt() -> None:
    vector = torch.zeros(4, dtype=DTYPE)
    vector[0] = vector[3] = 1 / np.sqrt(2)
    state = mps_from_statevector(vector, 2)
    certificate = certify_two_site_npt(state)
    assert certificate.certified
    assert certificate.min_eigenvalue == pytest_approx(-0.5, 1e-12)


def pytest_approx(value: float, tolerance: float):
    # Avoid importing pytest into library-facing numerical examples.
    import pytest

    return pytest.approx(value, abs=tolerance)


def test_ghz_known_correlations() -> None:
    N = 5
    vector = torch.zeros(2**N, dtype=DTYPE)
    vector[0] = vector[-1] = 1 / np.sqrt(2)
    state = mps_from_statevector(vector, N)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    assert state.expectation(tuple(range(N)), tuple(paulis[0] for _ in range(N))).real == pytest_approx(1.0, 1e-12)
    assert state.expectation((0, 4), (paulis[2], paulis[2])).real == pytest_approx(1.0, 1e-12)


def test_local_unitary_preserves_schmidt_spectrum_and_npt() -> None:
    generator = torch.Generator().manual_seed(23)
    state = random_mps(5, 3, generator=generator, device=DEVICE)
    before = state.schmidt_values(2)
    cert_before = certify_two_site_npt(state, max_distance=2)
    unitaries, _ = haar_local_unitaries(5, generator=generator, device=DEVICE)
    transformed = state.apply_local_unitaries(unitaries)
    after = transformed.schmidt_values(2)
    cert_after = certify_two_site_npt(transformed, max_distance=2)
    assert torch.allclose(before, after, atol=1e-10, rtol=1e-10)
    assert cert_before.certified == cert_after.certified
    assert cert_before.min_eigenvalue == pytest_approx(cert_after.min_eigenvalue, 1e-10)


def test_schmidt_values_equal_dense_svd() -> None:
    generator = torch.Generator().manual_seed(27)
    state = random_mps(6, 4, generator=generator, device=DEVICE)
    rho = purified_mps_to_dense_density(state)
    eigenvalues = torch.linalg.eigvalsh(rho)
    vector = torch.linalg.eigh(rho).eigenvectors[:, eigenvalues.argmax()]
    expected = torch.linalg.svdvals(vector.reshape(2**3, 2**3))
    expected = expected[expected > 1e-12]
    actual = state.schmidt_values(3)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_cached_npt_matches_independent_pair_rdms() -> None:
    generator = torch.Generator().manual_seed(28)
    state = apply_local_noise(random_mps(5, 3, generator=generator, device=DEVICE), "dephasing", 0.17)
    certificate = certify_two_site_npt(state, max_distance=2, tolerance=1.0)
    independent = []
    for distance in (1, 2):
        for i in range(state.N - distance):
            rho = two_site_density_matrix(state, i, i + distance)
            independent.append(float(torch.linalg.eigvalsh(partial_transpose_two_qubit(rho)).min()))
    assert not certificate.certified
    assert certificate.min_eigenvalue == pytest_approx(min(independent), 1e-10)


def test_rdm_channel_certificate_matches_noisy_purification() -> None:
    generator = torch.Generator().manual_seed(281)
    base = random_mps(5, 4, generator=generator, device=DEVICE)
    base_rdms = mps_two_site_density_matrices(base, max_distance=2)
    for noise_type in ("dephasing", "depolarizing", "amplitude_damping"):
        strength = 0.31
        kraus = channel_kraus(noise_type, strength, device=DEVICE, dtype=DTYPE)
        shortcut = certify_two_site_rdms(apply_channel_to_two_site_rdms(base_rdms, kraus))
        full = certify_two_site_npt(apply_local_noise(base, noise_type, strength), max_distance=2)
        assert shortcut.certified == full.certified
        assert shortcut.min_eigenvalue == pytest_approx(full.min_eigenvalue, 1e-10)


def test_gauge_transformation_preserves_observables() -> None:
    generator = torch.Generator().manual_seed(29)
    state = random_mps(5, 3, generator=generator, device=DEVICE)
    bond = 2
    dim = state.bond_dimensions[bond]
    gauge = torch.eye(dim, dtype=DTYPE) + 0.05 * torch.randn(dim, dim, generator=generator, dtype=torch.float64)
    gauged = state.apply_gauge(bond, gauge)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    for sites, axes in [((0,), (1,)), ((1, 4), (0, 2)), ((0, 1, 2, 3), (2, 1, 0, 2))]:
        ops = tuple(paulis[a] for a in axes)
        assert torch.abs(state.expectation(sites, ops) - gauged.expectation(sites, ops)) < 1e-10


def test_partial_transpose_is_involution() -> None:
    generator = torch.Generator().manual_seed(31)
    a = torch.randn(4, 4, generator=generator, dtype=torch.float64).to(DTYPE)
    rho = a @ a.H
    rho /= torch.trace(rho)
    assert torch.allclose(partial_transpose_two_qubit(partial_transpose_two_qubit(rho)), rho)


def test_feature_engine_count_order_and_dense_values() -> None:
    generator = torch.Generator().manual_seed(37)
    state = random_mps(4, 3, generator=generator, device=DEVICE)
    spec = FeatureSpec(4)
    result = compute_features(state, spec, connected=True)
    rho = purified_mps_to_dense_density(state)
    paulis = pauli_matrices(device=DEVICE, dtype=DTYPE)
    assert result.values.shape == (spec.num_features,)
    assert result.names == spec.names()
    for index, term in enumerate(spec.terms()):
        expected = dense_expectation(rho, 4, term.sites, tuple(paulis[a] for a in term.axes))
        assert abs(float(result.values[index]) - float(expected.real)) < 2e-6
    assert result.connected_values is not None
    assert result.connected_values.size == 9 * 6 + 27 * 2

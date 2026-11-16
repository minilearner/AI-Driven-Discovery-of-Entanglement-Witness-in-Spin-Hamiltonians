from __future__ import annotations

import numpy as np

from entanglement_data.certification.dpi_channels import (
    amplitude_damping_channel, analyze_channel_windows, coefficients_to_rdm, dephasing_channel, depolarizing_channel,
    pauli_channel, random_unit_vectors, rdm_pauli_coefficients, sample_entanglement_breaking_pauli_endpoint,
    pair_min_pt_transformed, prepare_rdms, separable_mixture_features, transform_coefficients,
)
from entanglement_data.certification.pairwise_invisible import rdms_from_legacy


def random_rho(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); a = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
    rho = a @ a.conj().T; return rho / np.trace(rho)


def test_pauli_coefficients_round_trip() -> None:
    rho = random_rho(4, 1)
    np.testing.assert_allclose(coefficients_to_rdm(rdm_pauli_coefficients(rho)), rho, atol=2e-15)


def test_channel_endpoints_are_product_or_classicalizing() -> None:
    rng = np.random.default_rng(2); axes = random_unit_vectors(rng, 4); rho = random_rho(4, 3); coeff = rdm_pauli_coefficients(rho)
    dep = coefficients_to_rdm(transform_coefficients(coeff, tuple(range(4)), depolarizing_channel(1.0, 4)))
    np.testing.assert_allclose(dep, np.eye(16) / 16, atol=2e-15)
    damp = coefficients_to_rdm(transform_coefficients(coeff, tuple(range(4)), amplitude_damping_channel(1.0, axes, np.ones(4))))
    assert np.isclose(np.trace(damp @ damp).real, 1.0, atol=2e-14)
    dephased = coefficients_to_rdm(transform_coefficients(coeff, tuple(range(4)), dephasing_channel(1.0, axes)))
    assert np.linalg.eigvalsh(dephased).min() > -2e-14


def test_sampled_pauli_endpoints_satisfy_entanglement_breaking_octrahedron() -> None:
    rng = np.random.default_rng(4); _, lambdas = sample_entanglement_breaking_pauli_endpoint(rng, 16)
    assert np.all(np.abs(lambdas).sum(axis=1) <= 0.95 + 1e-12)
    rotations = np.broadcast_to(np.eye(3), (16, 3, 3)).copy()
    channel = pauli_channel(1.0, rotations, lambdas)
    np.testing.assert_allclose(np.linalg.svd(channel.matrices, compute_uv=False).sum(axis=1), np.abs(lambdas).sum(axis=1), atol=1e-12)
    rotations = np.stack([np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(16)])
    rotations[np.linalg.det(rotations) < 0, :, 0] *= -1
    rotated = pauli_channel(1.0, rotations, lambdas)
    np.testing.assert_allclose(np.linalg.svd(rotated.matrices, compute_uv=False), np.sort(np.abs(lambdas), axis=1)[:, ::-1], atol=1e-12)


def test_separable_mixture_features_reconstruct_positive_rdms() -> None:
    rng = np.random.default_rng(5); bloch = rng.normal(size=(6, 16, 3)); bloch *= rng.uniform(0, .9, size=(6, 16, 1)) / np.linalg.norm(bloch, axis=2, keepdims=True)
    weights = rng.dirichlet(np.ones(6)); raw, skip = separable_mixture_features(bloch, weights)
    pairs, triples, fours = rdms_from_legacy(raw, skip)
    for _, rho in pairs[:2] + triples[:2] + fours[:2]:
        assert np.linalg.eigvalsh(rho).min() > -2e-13


def test_nonunital_window_stays_inside_connected_pair_ppt_safe_region() -> None:
    # A four-qubit GHZ parent has rank-deficient pair marginals.  Amplitude
    # damping first moves those pairs into the PPT interior, then returns them
    # to the PPT boundary at the pure-product endpoint gamma=1.
    psi = np.zeros(16, dtype=np.complex128); psi[0] = psi[-1] = 1 / np.sqrt(2)
    rho4 = np.outer(psi, psi.conj())
    rho3 = np.diag([.5, 0, 0, 0, 0, 0, 0, .5]).astype(np.complex128)
    rho2 = np.diag([.5, 0, 0, .5]).astype(np.complex128)
    pairs = prepare_rdms([((i, j), rho2) for i in range(4) for j in range(i + 1, 4)])
    triples = prepare_rdms([((0, 1, 2), rho3), ((1, 2, 3), rho3)])
    fours = prepare_rdms([((0, 1, 2, 3), rho4)])
    axes = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), (4, 3)).copy()
    path = lambda q: amplitude_damping_channel(q, axes, np.ones(4))
    result = analyze_channel_windows(pairs, triples, fours, path)
    assert result.e4_window is not None
    for q in np.linspace(*result.e4_window, 7):
        assert pair_min_pt_transformed(pairs, path(float(q))) >= 1e-6

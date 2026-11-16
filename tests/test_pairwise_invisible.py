from __future__ import annotations

import numpy as np
import torch

from entanglement_data.certification.pairwise_invisible import analyze_linear_noise_windows, analyze_parent_windows, disappearance_threshold, order_margins, partial_transpose, product_marginal_features, product_marginal_rdm, rdms_from_legacy, reconstruct_block_rdm
from entanglement_data.observables.engine import compute_features
from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.lu_invariants import decode_correlators
from entanglement_data.observables.skip3 import Skip3Spec, compute_skip3_features, decode_skip3
from entanglement_data.states.mps import mps_from_statevector


def ghz(k: int):
    vector = torch.zeros(2**k, dtype=torch.complex128)
    vector[0] = vector[-1] = 2**-0.5
    return mps_from_statevector(vector, k)


def test_rdm_reconstruction_matches_direct_dense_state() -> None:
    N = 5
    rng = np.random.default_rng(17)
    vector = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    state = mps_from_statevector(torch.tensor(vector, dtype=torch.complex128), N)
    raw = compute_features(state, FeatureSpec(N), output_dtype=np.float64).values
    decoded = decode_correlators(raw, FeatureSpec(N))
    skip3 = decode_skip3(compute_skip3_features(state, Skip3Spec(N), output_dtype=np.float64), Skip3Spec(N))[None, :]
    for sites in ((0, 4), (1, 2, 3), (0, 1, 2, 3)):
        expected = state.reduced_density_matrix(sites).detach().cpu().numpy()
        actual = reconstruct_block_rdm(decoded, sites, skip3=skip3 if len(sites) == 4 else None)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def test_three_qubit_ghz_is_pairwise_invisible_E3_parent() -> None:
    raw = compute_features(ghz(3), FeatureSpec(3), output_dtype=np.float64).values
    skip3 = compute_skip3_features(ghz(3), Skip3Spec(3), output_dtype=np.float64)
    pairs, triples, _ = rdms_from_legacy(raw, skip3, FeatureSpec(3))
    result = analyze_parent_windows(pairs, triples, triples)
    assert result.parent_order2.min_pt_eigenvalue >= -1e-12
    assert result.parent_order3.min_pt_eigenvalue < -0.49
    assert result.e3_window is not None


def test_four_qubit_ghz_is_E4_not_E3() -> None:
    state = ghz(4)
    raw = compute_features(state, FeatureSpec(4), output_dtype=np.float64).values
    skip3 = compute_skip3_features(state, Skip3Spec(4), output_dtype=np.float64)
    pairs, triples, fours = rdms_from_legacy(raw, skip3, FeatureSpec(4))
    result = analyze_parent_windows(pairs, triples, fours)
    assert result.parent_order2.min_pt_eigenvalue >= -1e-12
    assert result.parent_order3.min_pt_eigenvalue >= -1e-6
    assert result.parent_order3.max_ccnr_excess <= 1e-6
    assert result.parent_order4.min_pt_eigenvalue < -0.49
    assert result.e3_window is None
    assert result.e4_window is not None


def test_product_marginal_endpoint_matches_moment_and_rdm_construction() -> None:
    rng = np.random.default_rng(91)
    raw = rng.normal(size=2559)
    raw[:48] *= 0.05
    product_raw, product_skip = product_marginal_features(raw, rng.normal(size=702))
    pairs, triples, fours = rdms_from_legacy(product_raw, product_skip)
    one = decode_correlators(raw, FeatureSpec(16)).one[0]
    for support, rho in pairs[:3] + triples[:2] + fours[:2]:
        np.testing.assert_allclose(rho, product_marginal_rdm(one, support), atol=1e-12)


def test_generic_product_endpoint_finds_ghz_window() -> None:
    state = ghz(4)
    raw = compute_features(state, FeatureSpec(4), output_dtype=np.float64).values
    skip = compute_skip3_features(state, Skip3Spec(4), output_dtype=np.float64)
    pairs, triples, fours = rdms_from_legacy(raw, skip, FeatureSpec(4))
    one = decode_correlators(raw, FeatureSpec(4)).one[0]
    endpoints = lambda blocks: [(support, product_marginal_rdm(one, support)) for support, _ in blocks]
    result = analyze_linear_noise_windows(pairs, triples, fours, endpoints(pairs), endpoints(triples), endpoints(fours))
    assert result.e4_window is not None
    assert result.p4 is not None and result.p4 > result.p2


def test_partial_transpose_is_hermitian_and_trace_preserving() -> None:
    rho = ghz(4).reduced_density_matrix((0, 1, 2, 3)).detach().cpu().numpy()
    transformed = partial_transpose(rho, (0, 2))
    assert np.allclose(transformed, transformed.conj().T)
    assert np.isclose(np.trace(transformed), 1.0)


def test_fast_disappearance_threshold_matches_certificate_boundary() -> None:
    state = ghz(4)
    raw = compute_features(state, FeatureSpec(4), output_dtype=np.float64).values
    skip3 = compute_skip3_features(state, Skip3Spec(4), output_dtype=np.float64)
    _, _, fours = rdms_from_legacy(raw, skip3, FeatureSpec(4))
    threshold = disappearance_threshold(fours, 1e-8)
    assert threshold is not None
    assert order_margins(fours, threshold - 1e-8).min_pt_eigenvalue < -1e-8
    after = order_margins(fours, threshold + 1e-8)
    assert after.min_pt_eigenvalue >= -1e-8 - 1e-12
    assert after.max_ccnr_excess <= 1e-8 + 1e-10

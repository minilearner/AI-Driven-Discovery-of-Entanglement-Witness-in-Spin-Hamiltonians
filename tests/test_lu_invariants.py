from __future__ import annotations

from itertools import product

import numpy as np
import torch

from entanglement_data.observables.engine import compute_features
from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.local_unitary import random_local_rotations, transform_features
from entanglement_data.observables.lu_invariants import (
    LUInvariantSpec,
    compute_lu_invariant_features,
    connected_third_cumulant,
    decode_correlators,
    higher_order_four_body_residual,
    raw_t4_invariant_blocks,
)
from entanglement_data.observables.pauli import pauli_matrices
from entanglement_data.states.mps import mps_from_statevector


def _random_state(N: int, seed: int):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**N) + 1j * rng.normal(size=2**N)
    return mps_from_statevector(torch.tensor(vector, dtype=torch.complex128), N)


def test_feature_names_and_expected_dimension() -> None:
    spec = LUInvariantSpec(16)
    assert spec.inv12_stop == 1696
    assert spec.inv123_stop == 1976
    assert spec.num_features == 3016
    assert len(spec.names) == 3016
    assert any("higher_order_four_body_residual_R4_gt2" in name for name in spec.names)
    assert not any("centered" in name.lower() or "fourth_cumulant" in name.lower() for name in spec.names)


def test_all_invariants_are_local_su2_invariant_float64() -> None:
    N = 6
    spec = FeatureSpec(N)
    raw = compute_features(_random_state(N, 101), spec, output_dtype=np.float64).values
    rotations, _ = random_local_rotations(np.random.default_rng(202), N)
    rotated = transform_features(raw, rotations, spec, output_dtype=np.float64)
    phi = compute_lu_invariant_features(raw, spec, output_dtype=np.float64)
    phi_rotated = compute_lu_invariant_features(rotated, spec, output_dtype=np.float64)
    assert np.max(np.abs(phi - phi_rotated)) < 1e-8


def test_float32_invariance_is_near_pipeline_precision() -> None:
    N = 5
    spec = FeatureSpec(N)
    raw = compute_features(_random_state(N, 303), spec).values
    rotations, _ = random_local_rotations(np.random.default_rng(404), N)
    rotated = transform_features(raw, rotations, spec)
    phi = compute_lu_invariant_features(raw, spec)
    phi_rotated = compute_lu_invariant_features(rotated, spec)
    assert np.max(np.abs(phi - phi_rotated)) < 2e-6


def test_r4_gt2_is_zero_for_product_state() -> None:
    N = 4
    rng = np.random.default_rng(505)
    local = []
    for _ in range(N):
        vector = rng.normal(size=2) + 1j * rng.normal(size=2)
        local.append(vector / np.linalg.norm(vector))
    vector = local[0]
    for item in local[1:]:
        vector = np.kron(vector, item)
    spec = FeatureSpec(N)
    raw = compute_features(mps_from_statevector(torch.tensor(vector, dtype=torch.complex128), N), spec, output_dtype=np.float64).values
    residual = higher_order_four_body_residual(decode_correlators(raw, spec))
    assert np.max(np.abs(residual)) < 1e-10


def test_r4_gt2_transforms_covariantly() -> None:
    N = 5
    spec = FeatureSpec(N)
    raw = compute_features(_random_state(N, 606), spec, output_dtype=np.float64).values
    rotations, _ = random_local_rotations(np.random.default_rng(707), N)
    rotated = transform_features(raw, rotations, spec, output_dtype=np.float64)
    residual = higher_order_four_body_residual(decode_correlators(raw, spec))[0]
    residual_rotated = higher_order_four_body_residual(decode_correlators(rotated, spec))[0]
    starts = np.arange(N - 3)
    expected = np.einsum(
        "nam,nbp,ncq,ndr,nmpqr->nabcd",
        rotations[starts], rotations[starts + 1], rotations[starts + 2], rotations[starts + 3], residual,
        optimize=True,
    )
    assert np.max(np.abs(expected - residual_rotated)) < 1e-9


def test_r4_formula_matches_direct_dense_moments() -> None:
    N = 4
    state = _random_state(N, 808)
    spec = FeatureSpec(N)
    raw = compute_features(state, spec, output_dtype=np.float64).values
    residual = higher_order_four_body_residual(decode_correlators(raw, spec))[0, 0]
    paulis = pauli_matrices(device=torch.device("cpu"), dtype=torch.complex128)
    m = np.empty((4, 3), dtype=float)
    t2 = np.empty((4, 4, 3, 3), dtype=float)
    for i in range(4):
        for a in range(3):
            m[i, a] = state.expectation((i,), (paulis[a],)).real.item()
    for i in range(4):
        for j in range(i + 1, 4):
            for a, b in product(range(3), repeat=2):
                t2[i, j, a, b] = state.expectation((i, j), (paulis[a], paulis[b])).real.item()
                t2[j, i, b, a] = t2[i, j, a, b]
    direct_t4 = np.empty((3, 3, 3, 3), dtype=float)
    for axes in product(range(3), repeat=4):
        direct_t4[axes] = state.expectation(tuple(range(4)), tuple(paulis[a] for a in axes)).real.item()
    connected = {(i, j): t2[i, j] - np.outer(m[i], m[j]) for i in range(4) for j in range(i + 1, 4)}
    direct = direct_t4.copy()
    direct -= np.einsum("a,b,c,d->abcd", *m)
    direct -= np.einsum("ab,c,d->abcd", connected[0, 1], m[2], m[3])
    direct -= np.einsum("ac,b,d->abcd", connected[0, 2], m[1], m[3])
    direct -= np.einsum("ad,b,c->abcd", connected[0, 3], m[1], m[2])
    direct -= np.einsum("bc,a,d->abcd", connected[1, 2], m[0], m[3])
    direct -= np.einsum("bd,a,c->abcd", connected[1, 3], m[0], m[2])
    direct -= np.einsum("cd,a,b->abcd", connected[2, 3], m[0], m[1])
    direct -= np.einsum("ab,cd->abcd", connected[0, 1], connected[2, 3])
    direct -= np.einsum("ac,bd->abcd", connected[0, 2], connected[1, 3])
    direct -= np.einsum("ad,bc->abcd", connected[0, 3], connected[1, 2])
    assert np.max(np.abs(residual - direct)) < 1e-10


def test_one_and_two_body_components_match_direct_linear_algebra() -> None:
    N = 4
    spec = FeatureSpec(N)
    raw = compute_features(_random_state(N, 909), spec, output_dtype=np.float64).values
    decoded = decode_correlators(raw, spec)
    phi = compute_lu_invariant_features(raw, spec, output_dtype=np.float64)
    np.testing.assert_allclose(phi[:N], np.linalg.norm(decoded.one[0], axis=-1), atol=1e-12)
    tensor = decoded.two[0, 0]
    left, right = decoded.one[0, 0], decoded.one[0, 1]
    expected = np.concatenate(
        (
            np.linalg.svd(tensor, compute_uv=False),
            [np.linalg.det(tensor)],
            [left @ tensor @ right],
            [np.linalg.norm(tensor @ right)],
            [np.linalg.norm(tensor.T @ left)],
        )
    )
    np.testing.assert_allclose(phi[N : N + 7], expected, atol=1e-12)


def test_k3_matches_direct_dense_cumulant() -> None:
    N = 4
    state = _random_state(N, 1001)
    spec = FeatureSpec(N)
    raw = compute_features(state, spec, output_dtype=np.float64).values
    decoded = decode_correlators(raw, spec)
    k3 = connected_third_cumulant(decoded)[0, 0]
    paulis = pauli_matrices(device=torch.device("cpu"), dtype=torch.complex128)
    m = np.asarray(
        [[state.expectation((i,), (paulis[a],)).real.item() for a in range(3)] for i in range(3)]
    )
    t2 = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        t2[i, j] = np.asarray(
            [[state.expectation((i, j), (paulis[a], paulis[b])).real.item() for b in range(3)] for a in range(3)]
        )
    t3 = np.asarray(
        [
            state.expectation((0, 1, 2), tuple(paulis[a] for a in axes)).real.item()
            for axes in product(range(3), repeat=3)
        ]
    ).reshape(3, 3, 3)
    direct = (
        t3
        - np.einsum("a,bc->abc", m[0], t2[1, 2])
        - np.einsum("b,ac->abc", m[1], t2[0, 2])
        - np.einsum("c,ab->abc", m[2], t2[0, 1])
        + 2 * np.einsum("a,b,c->abc", *m)
    )
    np.testing.assert_allclose(k3, direct, atol=1e-10)


def test_raw_t4_candidate_invariants_are_lu_invariant_and_not_duplicated() -> None:
    rng = np.random.default_rng(1201)
    tensor = rng.normal(size=(64, 3, 3, 3, 3))
    rotations = []
    for _ in range(4):
        q, _ = np.linalg.qr(rng.normal(size=(64, 3, 3)))
        determinant = np.linalg.det(q)
        q[:, :, -1] *= np.where(determinant < 0.0, -1.0, 1.0)[:, None]
        rotations.append(q)
    rotated = np.einsum(
        "nae,nbf,ncg,ndh,nefgh->nabcd", *rotations, tensor, optimize=True,
    )
    original_invariants = raw_t4_invariant_blocks(tensor, output_dtype=np.float64)
    rotated_invariants = raw_t4_invariant_blocks(rotated, output_dtype=np.float64)
    np.testing.assert_allclose(original_invariants, rotated_invariants, atol=2e-12, rtol=2e-12)
    assert original_invariants.shape == (64, 40)
    for left in range(40):
        for right in range(left + 1, 40):
            assert not np.array_equal(original_invariants[:, left], original_invariants[:, right])

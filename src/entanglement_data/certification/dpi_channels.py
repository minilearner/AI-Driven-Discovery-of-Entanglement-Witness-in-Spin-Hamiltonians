from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable

import numpy as np

from .pairwise_invisible import PAULI, OrderMargins, ParentWindow, ccnr_norm, inequivalent_cuts, partial_transpose
from ..observables.feature_spec import FeatureSpec
from ..observables.skip3 import Skip3Spec


@dataclass(frozen=True)
class AffineQubitChannel:
    """Bloch affine map r' = M r + c for every site."""

    matrices: np.ndarray  # [N,3,3]
    shifts: np.ndarray  # [N,3]

    def __post_init__(self) -> None:
        if self.matrices.ndim != 3 or self.matrices.shape[1:] != (3, 3):
            raise ValueError("matrices must have shape [N,3,3]")
        if self.shifts.shape != self.matrices.shape[:2]:
            raise ValueError("shifts must have shape [N,3]")


@dataclass(frozen=True)
class PreparedRDM:
    support: tuple[int, ...]
    coefficients: np.ndarray


def rdm_pauli_coefficients(rho: np.ndarray) -> np.ndarray:
    k = len(rho).bit_length() - 1
    values = np.empty((4,) * k, dtype=np.float64)
    for axes in product(range(4), repeat=k):
        operator = PAULI[axes[0]]
        for axis in axes[1:]:
            operator = np.kron(operator, PAULI[axis])
        values[axes] = float(np.trace(rho @ operator).real)
    return values


def prepare_rdms(rdms: list[tuple[tuple[int, ...], np.ndarray]]) -> list[PreparedRDM]:
    return [PreparedRDM(support, rdm_pauli_coefficients(rho)) for support, rho in rdms]


def transform_coefficients(coefficients: np.ndarray, support: tuple[int, ...], channel: AffineQubitChannel) -> np.ndarray:
    local_maps = []
    for site in support:
        linear = np.zeros((4, 4), dtype=np.float64)
        linear[0, 0] = 1.0
        linear[1:, 0] = channel.shifts[site]
        linear[1:, 1:] = channel.matrices[site]
        local_maps.append(linear)
    out = coefficients
    for mode, linear in enumerate(local_maps):
        out = np.tensordot(linear, out, axes=(1, mode))
        out = np.moveaxis(out, 0, mode)
    return out


def coefficients_to_rdm(coefficients: np.ndarray) -> np.ndarray:
    k = coefficients.ndim
    rho = np.tensordot(coefficients.reshape(-1), _pauli_basis(k), axes=(0, 0)) / 2**k
    return (rho + rho.conj().T) / 2


@lru_cache(maxsize=4)
def _pauli_basis(k: int) -> np.ndarray:
    operators = []
    for axes in product(range(4), repeat=k):
        operator = PAULI[axes[0]]
        for axis in axes[1:]: operator = np.kron(operator, PAULI[axis])
        operators.append(operator)
    return np.stack(operators)


def separable_mixture_features(bloch: np.ndarray, weights: np.ndarray, spec: FeatureSpec | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Exact legacy and skip3 moments of a guaranteed separable mixture."""
    spec = spec or FeatureSpec(bloch.shape[1]); weights = np.asarray(weights, dtype=np.float64)
    if bloch.ndim != 3 or bloch.shape[1:] != (spec.N, 3) or weights.shape != (len(bloch),):
        raise ValueError("separable mixture shape mismatch")
    if np.max(np.linalg.norm(bloch, axis=2)) > 1 + 1e-12 or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("unphysical separable mixture")
    legacy = np.empty(spec.num_features, dtype=np.float64)
    for index, term in enumerate(spec.terms()):
        legacy[index] = np.sum(weights * np.prod([bloch[:, site, axis] for site, axis in zip(term.sites, term.axes)], axis=0))
    skip_spec = Skip3Spec(spec.N); skip = np.empty(skip_spec.num_features, dtype=np.float64)
    for index, term in enumerate(skip_spec.terms):
        skip[index] = np.sum(weights * np.prod([bloch[:, site, axis] for site, axis in zip(term.sites, term.axes)], axis=0))
    return legacy, skip


def transform_prepared(prepared: list[PreparedRDM], channel: AffineQubitChannel) -> list[tuple[tuple[int, ...], np.ndarray]]:
    return [(item.support, coefficients_to_rdm(transform_coefficients(item.coefficients, item.support, channel))) for item in prepared]


def random_unit_vectors(rng: np.random.Generator, N: int) -> np.ndarray:
    axes = rng.normal(size=(N, 3)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes


def dephasing_channel(parameter: float, axes: np.ndarray) -> AffineQubitChannel:
    matrices = (1.0 - parameter) * np.eye(3)[None] + parameter * np.einsum("ia,ib->iab", axes, axes)
    return AffineQubitChannel(matrices, np.zeros_like(axes))


def amplitude_damping_channel(parameter: float, axes: np.ndarray, exponents: np.ndarray) -> AffineQubitChannel:
    gamma = 1.0 - np.power(1.0 - parameter, exponents)
    matrices = np.empty((len(axes), 3, 3), dtype=np.float64)
    for i, (axis, value) in enumerate(zip(axes, gamma)):
        parallel = np.outer(axis, axis)
        matrices[i] = np.sqrt(1.0 - value) * (np.eye(3) - parallel) + (1.0 - value) * parallel
    return AffineQubitChannel(matrices, gamma[:, None] * axes)


def depolarizing_channel(parameter: float, N: int) -> AffineQubitChannel:
    return AffineQubitChannel(np.broadcast_to((1.0 - parameter) * np.eye(3), (N, 3, 3)).copy(), np.zeros((N, 3)))


def sample_entanglement_breaking_pauli_endpoint(rng: np.random.Generator, N: int) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.empty((N, 4), dtype=np.float64)
    lambdas = np.empty((N, 3), dtype=np.float64)
    for site in range(N):
        total = rng.uniform(0.45, 0.90)
        lam = total * rng.dirichlet(np.full(3, 2.0))
        probabilities[site] = np.asarray([
            1 + lam[0] + lam[1] + lam[2], 1 + lam[0] - lam[1] - lam[2],
            1 - lam[0] + lam[1] - lam[2], 1 - lam[0] - lam[1] + lam[2],
        ]) / 4
        lambdas[site] = lam
    return probabilities, lambdas


def pauli_channel(parameter: float, rotations: np.ndarray, endpoint_lambdas: np.ndarray) -> AffineQubitChannel:
    current = (1.0 - parameter) + parameter * endpoint_lambdas
    matrices = np.einsum("iab,ib,icb->iac", rotations, current, rotations, optimize=True)
    return AffineQubitChannel(matrices, np.zeros((len(rotations), 3)))


def order_margins_transformed(prepared: list[PreparedRDM], channel: AffineQubitChannel) -> OrderMargins:
    transformed = transform_prepared(prepared, channel)
    min_pt, max_ccnr = np.inf, -np.inf
    npt_support = ccnr_support = transformed[0][0]; npt_cut = ccnr_cut = (0,)
    for support, rho in transformed:
        for cut in inequivalent_cuts(len(support)):
            eigenvalue = float(np.linalg.eigvalsh(partial_transpose(rho, cut))[0])
            realignment = ccnr_norm(rho, cut) - 1.0
            if eigenvalue < min_pt: min_pt, npt_support, npt_cut = eigenvalue, support, cut
            if realignment > max_ccnr: max_ccnr, ccnr_support, ccnr_cut = realignment, support, cut
    return OrderMargins(min_pt, max_ccnr, npt_support, npt_cut, ccnr_support, ccnr_cut)


def pair_min_pt_transformed(prepared: list[PreparedRDM], channel: AffineQubitChannel) -> float:
    return min(float(np.linalg.eigvalsh(partial_transpose(rho, (0,)))[0]) for _, rho in transform_prepared(prepared, channel))


def analyze_channel_windows(
    pairs: list[PreparedRDM], triples: list[PreparedRDM], fours: list[PreparedRDM],
    channel_at: Callable[[float], AffineQubitChannel], *, tolerance: float = 1e-6, iterations: int = 24,
) -> ParentWindow:
    parent2 = order_margins_transformed(pairs, channel_at(0.0)); parent3 = order_margins_transformed(triples, channel_at(0.0)); parent4 = order_margins_transformed(fours, channel_at(0.0))
    grid = np.linspace(0.0, 1.0, 65)
    pair_values = np.asarray([pair_min_pt_transformed(pairs, channel_at(float(value))) for value in grid])
    safe = np.flatnonzero(pair_values >= tolerance)
    if not len(safe):
        return ParentWindow(1.0, None, None, np.nan, None, None, parent2, parent3, parent4)
    first = int(safe[0]); low, high = float(grid[max(0, first - 1)]), float(grid[first])
    for _ in range(iterations):
        middle = (low + high) / 2
        if pair_min_pt_transformed(pairs, channel_at(middle)) >= tolerance: high = middle
        else: low = middle
    p2 = high

    # Non-unital channels need not remain a finite distance inside the PPT cone.
    # In particular, amplitude damping ends at a pure product state, where the
    # smallest pair partial-transpose eigenvalue is exactly zero.  Restrict all
    # accepted windows to the connected PPT-safe component that starts at p2.
    unsafe_after = next(
        (index for index in range(first + 1, len(grid)) if pair_values[index] < tolerance),
        None,
    )
    if unsafe_after is None:
        pair_safe_upper = 1.0
    else:
        safe_low = float(grid[unsafe_after - 1])
        unsafe_high = float(grid[unsafe_after])
        for _ in range(iterations):
            middle = (safe_low + unsafe_high) / 2
            if pair_min_pt_transformed(pairs, channel_at(middle)) >= tolerance:
                safe_low = middle
            else:
                unsafe_high = middle
        pair_safe_upper = safe_low

    def disappears(blocks: list[PreparedRDM]) -> float | None:
        initial = order_margins_transformed(blocks, channel_at(0.0))
        if initial.min_pt_eigenvalue >= -tolerance and initial.max_ccnr_excess <= tolerance: return None
        values = []
        for value in np.linspace(p2, 1.0, 65):
            margins = order_margins_transformed(blocks, channel_at(float(value)))
            values.append((float(value), margins.min_pt_eigenvalue < -tolerance or margins.max_ccnr_excess > tolerance))
        transition = next((index for index, (_, active) in enumerate(values) if not active), None)
        if transition is None: return None
        if transition == 0: return p2
        lo, hi = values[transition - 1][0], values[transition][0]
        for _ in range(iterations):
            middle = (lo + hi) / 2; margins = order_margins_transformed(blocks, channel_at(middle))
            if margins.min_pt_eigenvalue < -tolerance or margins.max_ccnr_excess > tolerance: lo = middle
            else: hi = middle
        return hi

    p3, p4 = disappears(triples), disappears(fours)
    def window(lower: float, certificate_upper: float | None):
        if certificate_upper is None:
            return None
        upper = min(certificate_upper, pair_safe_upper)
        if upper <= lower:
            return None
        width = upper - lower; pad = min(max(1e-5, .02 * width), .2 * width)
        return (lower + pad, upper - pad) if width > 2 * pad else None
    e3 = window(p2, p3); e4 = window(max(p2, p3 or p2), p4)
    return ParentWindow(p2, p3, p4, pair_min_pt_transformed(pairs, channel_at(min(1.0, p2 + 1e-8))), e3, e4, parent2, parent3, parent4)


def features_from_transformed_rdms(
    one_body: np.ndarray, pairs: list[PreparedRDM], triples: list[PreparedRDM], fours: list[PreparedRDM], channel: AffineQubitChannel,
) -> tuple[np.ndarray, np.ndarray]:
    spec, skip_spec = FeatureSpec(len(one_body)), Skip3Spec(len(one_body))
    transformed_one = channel.shifts + np.einsum("iab,ib->ia", channel.matrices, one_body)
    p = {support: transform_coefficients(item.coefficients, support, channel) for item in pairs for support in (item.support,)}
    t = {support: transform_coefficients(item.coefficients, support, channel) for item in triples for support in (item.support,)}
    f = {support: transform_coefficients(item.coefficients, support, channel) for item in fours for support in (item.support,)}
    values = np.empty(spec.num_features, dtype=np.float32)
    for index, term in enumerate(spec.terms()):
        if len(term.sites) == 1: values[index] = transformed_one[term.sites[0], term.axes[0]]
        else:
            source = p if len(term.sites) == 2 else t if len(term.sites) == 3 else f
            values[index] = source[term.sites][tuple(axis + 1 for axis in term.axes)]
    skip = np.empty(skip_spec.num_features, dtype=np.float32)
    for index, term in enumerate(skip_spec.terms):
        block = f[(term.sites[0], term.sites[0] + 1, term.sites[0] + 2, term.sites[0] + 3)]
        axes = [0, 0, 0, 0]
        for site, axis in zip(term.sites, term.axes): axes[site - term.sites[0]] = axis + 1
        skip[index] = block[tuple(axes)]
    return values, skip

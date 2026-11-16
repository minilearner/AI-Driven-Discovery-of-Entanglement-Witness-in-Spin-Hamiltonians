from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from ..observables.feature_spec import FeatureSpec
from ..observables.lu_invariants import CorrelatorTensors, decode_correlators
from ..observables.skip3 import Skip3Spec, decode_skip3


PAULI = np.asarray(
    [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=np.complex128,
)


def inequivalent_cuts(k: int) -> list[tuple[int, ...]]:
    if k < 2:
        raise ValueError("k must be at least two")
    full = (1 << k) - 1
    return [tuple(i for i in range(k) if mask & (1 << i)) for mask in range(1, full) if mask < (full ^ mask)]


def partial_transpose(rho: np.ndarray, cut: tuple[int, ...]) -> np.ndarray:
    k = int(round(np.log2(rho.shape[0])))
    if rho.shape != (2**k, 2**k):
        raise ValueError("rho must be a square qubit density matrix")
    axes = list(range(2 * k))
    for site in cut:
        axes[site], axes[k + site] = axes[k + site], axes[site]
    return rho.reshape((2,) * (2 * k)).transpose(axes).reshape(rho.shape)


def ccnr_norm(rho: np.ndarray, cut: tuple[int, ...]) -> float:
    return float(np.linalg.svd(_realignment_matrix(rho, cut), compute_uv=False).sum())


def _realignment_matrix(rho: np.ndarray, cut: tuple[int, ...]) -> np.ndarray:
    k = int(round(np.log2(rho.shape[0])))
    right = tuple(i for i in range(k) if i not in cut)
    axes = tuple(cut) + tuple(k + i for i in cut) + right + tuple(k + i for i in right)
    return rho.reshape((2,) * (2 * k)).transpose(axes).reshape(4 ** len(cut), 4 ** len(right))


def white_noise_rdm(rho: np.ndarray, p: float) -> np.ndarray:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0,1]")
    return (1.0 - p) * rho + p * np.eye(len(rho), dtype=np.complex128) / len(rho)


def linear_noise_rdm(rho: np.ndarray, endpoint: np.ndarray, p: float) -> np.ndarray:
    if rho.shape != endpoint.shape or not 0.0 <= p <= 1.0:
        raise ValueError("rho/endpoint shape mismatch or p outside [0,1]")
    mixed = (1.0 - p) * rho + p * endpoint
    return (mixed + mixed.conj().T) / 2


def product_marginal_rdm(one_body: np.ndarray, sites: tuple[int, ...], shrink: float = 1.0) -> np.ndarray:
    """Product of a parent's one-site marginals on ``sites``."""
    rho = np.asarray([[1.0 + 0j]])
    for site in sites:
        local = (PAULI[0] + sum(shrink * float(one_body[site, a]) * PAULI[a + 1] for a in range(3))) / 2
        rho = np.kron(rho, local)
    return (rho + rho.conj().T) / 2


def product_marginal_features(
    features: np.ndarray, skip3_values: np.ndarray, spec: FeatureSpec | None = None, *, shrink: float = 1.0
):
    """Exact legacy/skip3 moments of the product of the input one-site marginals."""
    spec = spec or FeatureSpec(16)
    decoded = decode_correlators(features, spec)
    one = decoded.one[0]
    legacy = np.empty(spec.num_features, dtype=np.float64)
    for index, term in enumerate(spec.terms()):
        legacy[index] = np.prod([shrink * one[site, axis] for site, axis in zip(term.sites, term.axes)])
    auxiliary_spec = Skip3Spec(spec.N)
    auxiliary = np.empty(auxiliary_spec.num_features, dtype=np.float64)
    for index, term in enumerate(auxiliary_spec.terms):
        auxiliary[index] = np.prod([shrink * one[site, axis] for site, axis in zip(term.sites, term.axes)])
    return legacy, auxiliary


def _pair_lookup(decoded: CorrelatorTensors) -> np.ndarray:
    N = decoded.one.shape[1]
    lookup = np.full((N, N), -1, dtype=np.int64)
    lookup[decoded.pair_left, decoded.pair_right] = np.arange(len(decoded.pair_left))
    lookup[decoded.pair_right, decoded.pair_left] = np.arange(len(decoded.pair_left))
    return lookup


def reconstruct_block_rdm(
    decoded: CorrelatorTensors,
    sites: tuple[int, ...],
    row: int = 0,
    skip3: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct a 2/3/4-site RDM from the frozen legacy moment schema."""
    if len(sites) not in (2, 3, 4) or tuple(sorted(sites)) != sites:
        raise ValueError("sites must be an increasing support of size 2, 3, or 4")
    lookup = _pair_lookup(decoded)
    k = len(sites)
    rho = np.zeros((2**k, 2**k), dtype=np.complex128)
    for axes4 in product(range(4), repeat=k):
        support = tuple(local for local, axis in enumerate(axes4) if axis)
        global_support = tuple(sites[local] for local in support)
        axes0 = tuple(axes4[local] - 1 for local in support)
        order = len(support)
        if order == 0:
            value = 1.0
        elif order == 1:
            value = decoded.one[row, global_support[0], axes0[0]]
        elif order == 2:
            value = decoded.two[row, lookup[global_support[0], global_support[1]]][axes0]
        elif order == 3:
            if global_support == tuple(range(global_support[0], global_support[0] + 3)):
                value = decoded.three[row, global_support[0]][axes0]
            else:
                if skip3 is None or k != 4:
                    raise ValueError("four-site RDM reconstruction requires the exact skip3 auxiliary block")
                start = sites[0]
                branch = {(sites[0], sites[1], sites[3]): 0, (sites[0], sites[2], sites[3]): 1}.get(global_support)
                if branch is None:
                    raise ValueError(f"unsupported non-contiguous triple {global_support}")
                value = skip3[row, start, branch][axes0]
        else:
            if global_support != tuple(range(global_support[0], global_support[0] + 4)):
                raise ValueError("legacy schema stores only contiguous four-site moments")
            value = decoded.four[row, global_support[0]][axes0]
        operator = PAULI[axes4[0]]
        for axis in axes4[1:]:
            operator = np.kron(operator, PAULI[axis])
        rho += float(value) * operator
    rho /= 2**k
    return (rho + rho.conj().T) / 2


@dataclass(frozen=True)
class OrderMargins:
    min_pt_eigenvalue: float
    max_ccnr_excess: float
    best_npt_support: tuple[int, ...]
    best_npt_cut: tuple[int, ...]
    best_ccnr_support: tuple[int, ...]
    best_ccnr_cut: tuple[int, ...]

    @property
    def npt_margin(self) -> float:
        return -self.min_pt_eigenvalue


@dataclass(frozen=True)
class ParentWindow:
    p2: float
    p3: float | None
    p4: float | None
    pair_margin_at_lower: float
    e3_window: tuple[float, float] | None
    e4_window: tuple[float, float] | None
    parent_order2: OrderMargins
    parent_order3: OrderMargins
    parent_order4: OrderMargins


def order_margins(rdms: list[tuple[tuple[int, ...], np.ndarray]], p: float) -> OrderMargins:
    if not rdms:
        raise ValueError("at least one RDM is required")
    min_pt, max_ccnr = np.inf, -np.inf
    npt_support = ccnr_support = rdms[0][0]
    npt_cut = ccnr_cut = (0,)
    for support, parent_rho in rdms:
        rho = white_noise_rdm(parent_rho, p)
        for cut in inequivalent_cuts(len(support)):
            eigenvalue = float(np.linalg.eigvalsh(partial_transpose(rho, cut))[0])
            realignment = ccnr_norm(rho, cut) - 1.0
            if eigenvalue < min_pt:
                min_pt, npt_support, npt_cut = eigenvalue, support, cut
            if realignment > max_ccnr:
                max_ccnr, ccnr_support, ccnr_cut = realignment, support, cut
    return OrderMargins(min_pt, max_ccnr, npt_support, npt_cut, ccnr_support, ccnr_cut)


def linear_order_margins(
    rdms: list[tuple[tuple[int, ...], np.ndarray]],
    endpoints: list[tuple[tuple[int, ...], np.ndarray]],
    p: float,
) -> OrderMargins:
    if [support for support, _ in rdms] != [support for support, _ in endpoints]:
        raise ValueError("parent and endpoint RDM supports differ")
    mixed = [(support, linear_noise_rdm(rho, endpoint, p)) for (support, rho), (_, endpoint) in zip(rdms, endpoints)]
    return order_margins(mixed, 0.0)


def linear_pair_min_pt(rdms, endpoints, p: float) -> float:
    if [support for support, _ in rdms] != [support for support, _ in endpoints]:
        raise ValueError("parent and endpoint RDM supports differ")
    return min(
        float(np.linalg.eigvalsh(partial_transpose(linear_noise_rdm(rho, endpoint, p), (0,)))[0])
        for (_, rho), (_, endpoint) in zip(rdms, endpoints)
    )


def linear_min_pt(rdms, endpoints, p: float) -> float:
    parent, endpoint, _ = _linear_certificate_arrays(rdms, endpoints)
    return float(np.linalg.eigvalsh((1.0 - p) * parent + p * endpoint)[..., 0].min())


def linear_max_ccnr(rdms, endpoints, p: float) -> float:
    _, _, groups = _linear_certificate_arrays(rdms, endpoints)
    return max(float(np.linalg.svd((1.0 - p) * parent + p * endpoint, compute_uv=False).sum(axis=-1).max() - 1.0) for parent, endpoint in groups)


def _linear_certificate_arrays(rdms, endpoints):
    parent_pt, endpoint_pt = [], []
    realignment: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for (_, rho), (_, endpoint) in zip(rdms, endpoints):
        k = len(rho).bit_length() - 1
        for cut in inequivalent_cuts(k):
            parent_pt.append(partial_transpose(rho, cut))
            endpoint_pt.append(partial_transpose(endpoint, cut))
            a, b = _realignment_matrix(rho, cut), _realignment_matrix(endpoint, cut)
            if a.shape[0] > a.shape[1]:
                a, b = a.T, b.T
            bucket = realignment.setdefault(a.shape, ([], []))
            bucket[0].append(a); bucket[1].append(b)
    groups = [(np.stack(parent), np.stack(endpoint)) for parent, endpoint in realignment.values()]
    return np.stack(parent_pt), np.stack(endpoint_pt), groups


def linear_disappearance_threshold(rdms, endpoints, tolerance: float, iterations: int = 28) -> float | None:
    parent_pt, endpoint_pt, realignment = _linear_certificate_arrays(rdms, endpoints)

    def min_pt(p: float) -> float:
        return float(np.linalg.eigvalsh((1.0 - p) * parent_pt + p * endpoint_pt)[..., 0].min())

    def max_ccnr(p: float) -> float:
        return max(float(np.linalg.svd((1.0 - p) * parent + p * endpoint, compute_uv=False).sum(axis=-1).max() - 1.0) for parent, endpoint in realignment)

    parent_npt = min_pt(0.0)
    parent_ccnr = max_ccnr(0.0)
    if parent_npt >= -tolerance and parent_ccnr <= tolerance:
        return None
    if min_pt(1.0) < -tolerance or max_ccnr(1.0) > tolerance:
        return None
    low, high = 0.0, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if min_pt(middle) < -tolerance:
            low = middle
        else:
            high = middle
    if max_ccnr(high) <= tolerance:
        return high
    low, high = high, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if max_ccnr(middle) > tolerance:
            low = middle
        else:
            high = middle
    return high


def analyze_linear_noise_windows(
    pair_rdms,
    triple_rdms,
    four_rdms,
    pair_endpoints,
    triple_endpoints,
    four_endpoints,
    *,
    ppt_tolerance: float = 1e-6,
    certificate_tolerance: float = 1e-6,
) -> ParentWindow:
    parent2 = linear_order_margins(pair_rdms, pair_endpoints, 0.0)
    parent3 = linear_order_margins(triple_rdms, triple_endpoints, 0.0)
    parent4 = linear_order_margins(four_rdms, four_endpoints, 0.0)
    low, high = 0.0, 1.0
    for _ in range(28):
        middle = (low + high) / 2
        if linear_pair_min_pt(pair_rdms, pair_endpoints, middle) >= ppt_tolerance:
            high = middle
        else:
            low = middle
    p2 = high
    p3 = linear_disappearance_threshold(triple_rdms, triple_endpoints, certificate_tolerance)
    p4 = linear_disappearance_threshold(four_rdms, four_endpoints, certificate_tolerance)

    def safe_window(lower, upper):
        if upper is None or upper <= lower:
            return None
        width = upper - lower
        pad = min(max(1e-5, 0.02 * width), 0.2 * width)
        return (lower + pad, upper - pad) if width > 2 * pad else None

    e3 = safe_window(p2, p3)
    e4 = safe_window(max(p2, p3 or p2), p4)
    margin = linear_pair_min_pt(pair_rdms, pair_endpoints, min(1.0, p2 + 1e-8))
    return ParentWindow(p2, p3, p4, margin, e3, e4, parent2, parent3, parent4)


def certified(margins: OrderMargins, tolerance: float) -> bool:
    return margins.min_pt_eigenvalue < -tolerance or margins.max_ccnr_excess > tolerance


def npt_disappearance_threshold(rdms: list[tuple[tuple[int, ...], np.ndarray]], tolerance: float) -> float | None:
    """Exact white-noise threshold for the last selected NPT certificate.

    Partial transpose commutes with the identity admixture, so every eigenvalue
    follows (1-p)*lambda + p/d and needs no numerical search.
    """
    thresholds: list[float] = []
    for support, rho in rdms:
        inverse_dimension = 1.0 / len(rho)
        for cut in inequivalent_cuts(len(support)):
            eigenvalue = float(np.linalg.eigvalsh(partial_transpose(rho, cut))[0])
            if eigenvalue < -tolerance:
                thresholds.append((-tolerance - eigenvalue) / (inverse_dimension - eigenvalue))
    return float(np.clip(max(thresholds), 0.0, 1.0)) if thresholds else None


def disappearance_threshold(rdms: list[tuple[tuple[int, ...], np.ndarray]], tolerance: float, iterations: int = 32) -> float | None:
    parent = order_margins(rdms, 0.0)
    if not certified(parent, tolerance):
        return None
    npt_threshold = npt_disappearance_threshold(rdms, tolerance) or 0.0
    # In the common case NPT is the last surviving certificate. One CCNR check
    # at that exact boundary avoids thousands of redundant small SVDs.
    if order_margins(rdms, npt_threshold).max_ccnr_excess <= tolerance:
        return npt_threshold
    low, high = npt_threshold, 1.0
    for _ in range(iterations):
        middle = (low + high) / 2
        if order_margins(rdms, middle).max_ccnr_excess > tolerance:
            low = middle
        else:
            high = middle
    return high


def analyze_parent_windows(
    pair_rdms: list[tuple[tuple[int, ...], np.ndarray]],
    triple_rdms: list[tuple[tuple[int, ...], np.ndarray]],
    four_rdms: list[tuple[tuple[int, ...], np.ndarray]],
    *,
    ppt_tolerance: float = 1e-6,
    certificate_tolerance: float = 1e-6,
) -> ParentWindow:
    parent2 = order_margins(pair_rdms, 0.0)
    parent3 = order_margins(triple_rdms, 0.0)
    parent4 = order_margins(four_rdms, 0.0)
    pair_thresholds = []
    for _, rho in pair_rdms:
        value = float(np.linalg.eigvalsh(partial_transpose(rho, (0,)))[0])
        pair_thresholds.append(0.0 if value >= ppt_tolerance else (ppt_tolerance - value) / (0.25 - value))
    p2 = float(np.clip(max(pair_thresholds), 0.0, 1.0))
    p3 = disappearance_threshold(triple_rdms, certificate_tolerance)
    p4 = disappearance_threshold(four_rdms, certificate_tolerance)

    def safe_window(lower: float, upper: float | None) -> tuple[float, float] | None:
        if upper is None or upper <= lower:
            return None
        width = upper - lower
        pad = min(max(1e-5, 0.02 * width), 0.2 * width)
        return (lower + pad, upper - pad) if width > 2 * pad else None

    e3 = safe_window(p2, p3)
    # E4 is reserved for the region after all selected three-body certificates disappear.
    e4 = safe_window(max(p2, p3 or p2), p4)
    lower_margin = order_margins(pair_rdms, min(1.0, p2 + 1e-8)).min_pt_eigenvalue
    return ParentWindow(p2, p3, p4, lower_margin, e3, e4, parent2, parent3, parent4)


def rdms_from_legacy(features: np.ndarray, skip3_values: np.ndarray | None = None, spec: FeatureSpec | None = None):
    spec = spec or FeatureSpec(16)
    decoded = decode_correlators(features, spec)
    auxiliary = None if skip3_values is None else decode_skip3(skip3_values, Skip3Spec(spec.N))
    if auxiliary is not None and auxiliary.ndim == 5:
        auxiliary = auxiliary[None, :]
    pairs = [((i, j), reconstruct_block_rdm(decoded, (i, j))) for i, j in combinations(range(spec.N), 2)]
    triples = [((i, i + 1, i + 2), reconstruct_block_rdm(decoded, (i, i + 1, i + 2))) for i in range(spec.N - 2)]
    fours = [
        ((i, i + 1, i + 2, i + 3), reconstruct_block_rdm(decoded, (i, i + 1, i + 2, i + 3), skip3=auxiliary))
        for i in range(spec.N - 3)
    ]
    return pairs, triples, fours

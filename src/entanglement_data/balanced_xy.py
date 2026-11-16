from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.local_unitary import feature_slices


PAULI = np.asarray(
    [
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=np.complex128,
)
IDENTITY2 = np.eye(2, dtype=np.complex128)
FAMILIES = (
    "random_product",
    "convex_product_mixture",
    "classically_correlated_mixture",
    "noisy_separable_mixture",
)


@dataclass(frozen=True)
class SeparableCandidate:
    family: str
    seed: int
    weights: np.ndarray
    bloch: np.ndarray
    detail: dict[str, object]

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights)
        bloch = np.asarray(self.bloch)
        if weights.ndim != 1 or bloch.ndim != 3 or bloch.shape[0] != len(weights) or bloch.shape[-1] != 3:
            raise ValueError("invalid product-mixture proof arrays")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("mixture weights must be nonnegative and normalized")
        if np.max(np.linalg.norm(bloch, axis=-1)) > 1.0 + 1e-10:
            raise ValueError("single-site Bloch vectors leave the physical ball")

    @property
    def components(self) -> int:
        return len(self.weights)

    @property
    def N(self) -> int:
        return self.bloch.shape[1]

    @property
    def proof_json(self) -> str:
        return json.dumps(
            {
                "type": "explicit_convex_mixture_of_product_single_qubit_states",
                "family": self.family,
                "components": self.components,
                "detail": self.detail,
            },
            sort_keys=True,
        )


def _unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=(*shape, 3))
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values


def _random_bloch(
    rng: np.random.Generator,
    components: int,
    N: int,
    *,
    purity_low: float = 0.05,
    purity_high: float = 1.0,
) -> np.ndarray:
    direction = _unit_vectors(rng, (components, N))
    # A beta mixture covers both highly mixed and nearly pure local states.
    radius = purity_low + (purity_high - purity_low) * rng.beta(1.4, 0.9, size=(components, N, 1))
    return direction * radius


def generate_candidate(family: str, seed: int, *, N: int = 16) -> SeparableCandidate:
    rng = np.random.default_rng(seed)
    if family == "random_product":
        weights = np.ones(1, dtype=np.float64)
        bloch = _random_bloch(rng, 1, N, purity_low=0.15)
        detail = {"local_channel": "none", "mixture_rank": 1}
    elif family == "convex_product_mixture":
        components = int(rng.integers(2, 17))
        alpha = float(np.exp(rng.uniform(np.log(0.15), np.log(3.0))))
        weights = rng.dirichlet(np.full(components, alpha))
        bloch = _random_bloch(rng, components, N)
        detail = {"mixture_rank": components, "dirichlet_alpha": alpha, "local_channel": "none"}
    elif family == "classically_correlated_mixture":
        components = int(rng.integers(2, 7))
        axes = _unit_vectors(rng, (N,))
        base_pattern = rng.choice((-1.0, 1.0), size=(N,))
        signs = np.empty((components, N), dtype=np.float64)
        signs[0] = base_pattern
        if components > 1:
            signs[1] = -base_pattern
        for component in range(2, components):
            # Domain patterns retain strong classical correlations without
            # reducing every state to a two-branch GHZ-like mixture.
            boundaries = sorted(rng.choice(np.arange(1, N), size=int(rng.integers(1, 5)), replace=False).tolist())
            pattern = base_pattern.copy()
            flip = False
            start = 0
            for stop in (*boundaries, N):
                if flip:
                    pattern[start:stop] *= -1
                flip = not flip
                start = stop
            signs[component] = pattern
        radii = rng.uniform(0.72, 1.0, size=(components, N, 1))
        transverse = 0.08 * _unit_vectors(rng, (components, N))
        bloch = signs[:, :, None] * axes[None, :, :] * radii + transverse
        norm = np.linalg.norm(bloch, axis=-1, keepdims=True)
        bloch /= np.maximum(1.0, norm)
        alpha = float(np.exp(rng.uniform(np.log(0.25), np.log(1.2))))
        weights = rng.dirichlet(np.full(components, alpha))
        detail = {
            "mixture_rank": components,
            "dirichlet_alpha": alpha,
            "latent_patterns": "global_antipodal_plus_random_domains",
            "local_channel": "none",
        }
    elif family == "noisy_separable_mixture":
        components = int(rng.integers(2, 10))
        alpha = float(np.exp(rng.uniform(np.log(0.2), np.log(2.0))))
        weights = rng.dirichlet(np.full(components, alpha))
        bloch = _random_bloch(rng, components, N, purity_low=0.2)
        depolarization = rng.uniform(0.05, 0.55, size=(N,))
        dephasing = rng.uniform(0.0, 0.75, size=(N,))
        axes = _unit_vectors(rng, (N,))
        parallel = np.einsum("qia,ia->qi", bloch, axes)[:, :, None] * axes[None, :, :]
        bloch = (1.0 - depolarization[None, :, None]) * (
            (1.0 - dephasing[None, :, None]) * bloch + dephasing[None, :, None] * parallel
        )
        mixed_weight = float(rng.uniform(0.03, 0.30))
        weights = np.concatenate(((1.0 - mixed_weight) * weights, [mixed_weight]))
        bloch = np.concatenate((bloch, np.zeros((1, N, 3), dtype=np.float64)), axis=0)
        detail = {
            "mixture_rank_before_maximally_mixed_component": components,
            "dirichlet_alpha": alpha,
            "local_channel": "sitewise_depolarization_plus_dephasing",
            "maximally_mixed_product_weight": mixed_weight,
            "depolarization_range": [float(depolarization.min()), float(depolarization.max())],
            "dephasing_range": [float(dephasing.min()), float(dephasing.max())],
        }
    else:
        raise ValueError(f"unknown separable family {family}")
    return SeparableCandidate(family, seed, weights.astype(np.float64), bloch.astype(np.float64), detail)


def product_mixture_features(
    weights: np.ndarray, bloch: np.ndarray, *, output_dtype: np.dtype | type = np.float32
) -> np.ndarray:
    """Compute the historical T1/all-T2/contiguous-T3/T4 schema analytically."""

    weights = np.asarray(weights, dtype=np.float64)
    bloch = np.asarray(bloch, dtype=np.float64)
    if bloch.ndim != 3 or bloch.shape[0] != len(weights) or bloch.shape[-1] != 3:
        raise ValueError("bloch must have shape [Q,N,3] matching weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-12) or np.any(weights < 0):
        raise ValueError("invalid convex weights")
    N = bloch.shape[1]
    blocks = [np.einsum("q,qia->ia", weights, bloch, optimize=True).reshape(-1)]
    pair_blocks = []
    for left, right in combinations(range(N), 2):
        pair_blocks.append(np.einsum("q,qa,qb->ab", weights, bloch[:, left], bloch[:, right], optimize=True).reshape(-1))
    blocks.append(np.concatenate(pair_blocks))
    triples = []
    for start in range(N - 2):
        triples.append(
            np.einsum(
                "q,qa,qb,qc->abc", weights, bloch[:, start], bloch[:, start + 1], bloch[:, start + 2],
                optimize=True,
            ).reshape(-1)
        )
    blocks.append(np.concatenate(triples))
    fours = []
    for start in range(N - 3):
        fours.append(
            np.einsum(
                "q,qa,qb,qc,qd->abcd", weights, bloch[:, start], bloch[:, start + 1],
                bloch[:, start + 2], bloch[:, start + 3], optimize=True,
            ).reshape(-1)
        )
    blocks.append(np.concatenate(fours))
    result = np.concatenate(blocks).astype(output_dtype, copy=False)
    if result.shape != (FeatureSpec(N).num_features,):
        raise RuntimeError("constructed feature width does not match historical schema")
    return result


def feature_statistics(features: np.ndarray, spec: FeatureSpec) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != spec.num_features:
        raise ValueError("feature matrix does not match FeatureSpec")
    slices = feature_slices(spec)
    one = values[:, slices.one]
    two = values[:, slices.two]
    three = values[:, slices.three]
    pair = two.reshape(len(values), -1, 3, 3)
    nearest = pair[:, slices.pair_right - slices.pair_left == 1]
    output = {
        "T1_fro": np.linalg.norm(one, axis=1),
        "T1_rms": np.sqrt(np.mean(one**2, axis=1)),
        "T2_fro": np.linalg.norm(two, axis=1),
        "T2_rms": np.sqrt(np.mean(two**2, axis=1)),
        "T3_fro": np.linalg.norm(three, axis=1),
        "T3_rms": np.sqrt(np.mean(three**2, axis=1)),
        "nearest_T2_mean_fro": np.mean(np.linalg.norm(nearest, axis=(-2, -1)), axis=1),
        "correlator_abs_max": np.max(np.abs(values), axis=1),
    }
    return {key: item[0] if single else item for key, item in output.items()}


def select_diverse_overlap(
    candidate_statistics: np.ndarray,
    target_statistics: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select physically overlapping, diverse candidates without model scores."""

    candidate = np.log1p(np.asarray(candidate_statistics, dtype=np.float64))
    target = np.log1p(np.asarray(target_statistics, dtype=np.float64))
    if candidate.ndim != 2 or target.ndim != 2 or candidate.shape[1] != target.shape[1]:
        raise ValueError("candidate and target statistics must be compatible matrices")
    center = np.median(target, axis=0)
    scale = np.quantile(target, 0.75, axis=0) - np.quantile(target, 0.25, axis=0)
    scale = np.maximum(scale, 1e-8)
    standardized = (candidate - center) / scale
    target_distance = np.sqrt(np.mean(standardized**2, axis=1))
    eligible_count = min(len(candidate), max(count, int(np.ceil(0.65 * len(candidate)))))
    eligible = np.argsort(target_distance, kind="stable")[:eligible_count]
    chosen = [int(eligible[0])]
    min_distance = np.linalg.norm(standardized - standardized[chosen[0]], axis=1)
    eligible_mask = np.zeros(len(candidate), dtype=bool)
    eligible_mask[eligible] = True
    eligible_mask[chosen[0]] = False
    while len(chosen) < count:
        score = min_distance - 0.20 * target_distance
        score[~eligible_mask] = -np.inf
        next_index = int(np.argmax(score))
        chosen.append(next_index)
        eligible_mask[next_index] = False
        distance = np.linalg.norm(standardized - standardized[next_index], axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(chosen, dtype=np.int64), target_distance


def single_qubit_density(bloch: np.ndarray) -> np.ndarray:
    return 0.5 * (IDENTITY2 + np.einsum("a,aij->ij", np.asarray(bloch), PAULI))


def pair_ppt_minima(candidate: SeparableCandidate) -> np.ndarray:
    minima = []
    for left, right in combinations(range(candidate.N), 2):
        rho = np.zeros((4, 4), dtype=np.complex128)
        for weight, component in zip(candidate.weights, candidate.bloch):
            rho += weight * np.kron(single_qubit_density(component[left]), single_qubit_density(component[right]))
        partial_transpose = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
        minima.append(float(np.linalg.eigvalsh(partial_transpose).min()))
    return np.asarray(minima, dtype=np.float64)

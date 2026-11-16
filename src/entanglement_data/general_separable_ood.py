from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .xy.size_transfer import LocalRawSpec, connected_from_raw, local_masks


FAMILIES = (
    "iid_bloch_ball",
    "iid_pure_local",
    "iid_strongly_mixed_local",
    "iid_sparse_weights",
)
K_BANDS = ((2, 4), (5, 8), (9, 16), (17, 32))


@dataclass(frozen=True)
class GeneralSeparableSample:
    """Explicit constructive proof for a fully separable qubit state.

    ``bloch[mu, i]`` is sampled independently in direction for every mixture
    component ``mu`` and site ``i``.  No shared-axis/sign-flip construction is
    used by any family in this module.
    """

    family: str
    seed: int
    k_band: tuple[int, int]
    alpha: float
    weights: np.ndarray
    bloch: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        bloch = np.asarray(self.bloch, dtype=np.float64)
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family {self.family!r}")
        if weights.ndim != 1 or bloch.ndim != 3 or bloch.shape[0] != len(weights) or bloch.shape[2] != 3:
            raise ValueError("expected weights [K] and Bloch vectors [K,N,3]")
        if len(weights) < 2 or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise ValueError("invalid nontrivial convex weights")
        if not np.isfinite(bloch).all() or np.max(np.linalg.norm(bloch, axis=-1)) > 1.0 + 1e-12:
            raise ValueError("invalid one-qubit Bloch vectors")
        if not self.k_band[0] <= len(weights) <= self.k_band[1]:
            raise ValueError("K lies outside its declared stratum")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bloch", bloch)

    @property
    def K(self) -> int:
        return len(self.weights)

    @property
    def N(self) -> int:
        return self.bloch.shape[1]


def _unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=(*shape, 3))
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-14):
        raise RuntimeError("Gaussian direction sampler produced a near-zero vector")
    return values / norms


def production_manifest(total: int = 2000, *, master_seed: int = 2026091001) -> list[dict[str, Any]]:
    strata = [(family, band) for family in FAMILIES for band in K_BANDS]
    if total <= 0 or total % len(strata):
        raise ValueError(f"total must be a positive multiple of {len(strata)}")
    rows = []
    per_stratum = total // len(strata)
    for stratum_index, (family, band) in enumerate(strata):
        for ordinal in range(per_stratum):
            sample_id = stratum_index * per_stratum + ordinal
            rows.append({
                "sample_id": sample_id,
                "family": family,
                "k_band": band,
                "seed": int(master_seed + 1_000_003 * (sample_id + 1)),
            })
    rng = np.random.default_rng(master_seed + 97)
    permutation = rng.permutation(total)
    shuffled = [rows[int(index)] for index in permutation]
    for sample_id, row in enumerate(shuffled):
        row["sample_id"] = sample_id
    return shuffled


def generate_sample(
    family: str,
    seed: int,
    k_band: tuple[int, int],
    *,
    N: int = 32,
) -> GeneralSeparableSample:
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if N < 5:
        raise ValueError("N must exceed the observation radius")
    low, high = (int(value) for value in k_band)
    if low < 2 or high < low:
        raise ValueError("invalid K band")
    rng = np.random.default_rng(int(seed))
    K = int(rng.integers(low, high + 1))

    # This is the defining benchmark property: every (mu, i) gets an
    # independently sampled direction before any radius is applied.
    directions = _unit_vectors(rng, (K, N))
    if family == "iid_pure_local":
        radii = np.ones((K, N, 1), dtype=np.float64)
        alpha = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
    elif family == "iid_strongly_mixed_local":
        radii = rng.uniform(0.05, 0.55, size=(K, N, 1))
        alpha = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
    else:
        # r=u^(1/3) is uniform in the volume of the Bloch ball.
        radii = rng.random((K, N, 1)) ** (1.0 / 3.0)
        if family == "iid_sparse_weights":
            alpha = float(np.exp(rng.uniform(np.log(0.02), np.log(0.15))))
        else:
            alpha = float(np.exp(rng.uniform(np.log(0.1), np.log(10.0))))
    weights = rng.dirichlet(np.full(K, alpha, dtype=np.float64))
    return GeneralSeparableSample(family, int(seed), (low, high), alpha, weights, directions * radii)


def product_mixture_local_raw(
    sample: GeneralSeparableSample,
    *,
    R_obs: int = 4,
) -> dict[str, np.ndarray]:
    """Compute exact local moments directly from the stored product mixture."""

    spec = LocalRawSpec(sample.N, R_obs)
    weights, bloch = sample.weights, sample.bloch
    pair_mask, triple_mask, four_mask = local_masks(spec)
    t1 = np.einsum("q,qia->ia", weights, bloch, optimize=True)
    t2 = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for distance in range(1, spec.R_obs + 1):
        width = spec.N - distance
        t2[:width, distance - 1] = np.einsum(
            "q,qia,qib->iab", weights, bloch[:, :width], bloch[:, distance:], optimize=True,
        )
    t3 = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    width3 = spec.N - 2
    t3[:width3] = np.einsum(
        "q,qia,qib,qic->iabc",
        weights,
        bloch[:, :width3],
        bloch[:, 1 : 1 + width3],
        bloch[:, 2 : 2 + width3],
        optimize=True,
    )
    t4 = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    width4 = spec.N - 3
    t4[:width4] = np.einsum(
        "q,qia,qib,qic,qid->iabcd",
        weights,
        bloch[:, :width4],
        bloch[:, 1 : 1 + width4],
        bloch[:, 2 : 2 + width4],
        bloch[:, 3 : 3 + width4],
        optimize=True,
    )
    k2, k3 = connected_from_raw(t1, t2, t3, spec)
    return {
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T4": t4,
        "K2": k2,
        "K3": k3,
        "T2_mask": pair_mask,
        "T3_mask": triple_mask,
        "T4_mask": four_mask,
    }


def direction_diagnostics(sample: GeneralSeparableSample) -> dict[str, float | int]:
    norms = np.linalg.norm(sample.bloch, axis=-1)
    directions = sample.bloch / np.maximum(norms[..., None], 1e-15)
    ranks = [np.linalg.matrix_rank(directions[:, site], tol=1e-10) for site in range(sample.N)]
    pair_abs_cosine = np.abs(np.sum(directions[0] * directions[1], axis=-1))
    return {
        "minimum_site_direction_rank": int(min(ranks)),
        "maximum_site_direction_rank": int(max(ranks)),
        "mean_abs_cosine_mu0_mu1": float(pair_abs_cosine.mean()),
        "maximum_abs_cosine_mu0_mu1": float(pair_abs_cosine.max()),
        "maximum_bloch_norm": float(norms.max()),
        "minimum_bloch_norm": float(norms.min()),
        "effective_components": float(1.0 / np.square(sample.weights).sum()),
    }

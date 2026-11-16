from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorizedPoint:
    """One symmetry-broken product ground state on the XY factorization circle."""

    gamma: float
    h: float
    locus_angle: float
    branch: int
    seed: int

    def __post_init__(self) -> None:
        if self.branch not in (-1, 1):
            raise ValueError("branch must be +1 or -1")
        if self.gamma < 0.0 or self.h < 0.0:
            raise ValueError("this benchmark uses the first-quadrant XY factorization arc")
        if abs(self.gamma**2 + self.h**2 - 1.0) > 1e-12:
            raise ValueError("point does not lie on gamma^2+h^2=1")


def sample_factorized_points(
    count: int,
    *,
    master_seed: int,
    gamma_min: float = 0.02,
    gamma_max: float = 0.98,
) -> list[FactorizedPoint]:
    """Stratified, deterministic coverage of the historical first-quadrant arc.

    The construction deliberately has no classifier-dependent filtering.  It
    returns equal branch counts when ``count`` is even.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= gamma_min < gamma_max <= 1.0:
        raise ValueError("gamma bounds must satisfy 0 <= min < max <= 1")
    rng = np.random.default_rng(master_seed)
    theta_min, theta_max = math.asin(gamma_min), math.asin(gamma_max)
    theta = theta_min + (np.arange(count) + rng.random(count)) * (theta_max - theta_min) / count
    branches = np.where(np.arange(count) % 2 == 0, 1, -1)
    permutation = rng.permutation(count)
    points: list[FactorizedPoint] = []
    for output_index, source_index in enumerate(permutation):
        angle = float(theta[source_index])
        points.append(
            FactorizedPoint(
                gamma=float(math.sin(angle)),
                h=float(math.cos(angle)),
                locus_angle=angle,
                branch=int(branches[source_index]),
                seed=int(master_seed + output_index),
            )
        )
    return points


def factorized_local_bloch(gamma: float, h: float, branch: int) -> np.ndarray:
    """Return ``(<X>, <Y>, <Z>)`` in the historical Pauli convention."""

    if branch not in (-1, 1):
        raise ValueError("branch must be +1 or -1")
    if abs(gamma * gamma + h * h - 1.0) > 1e-10:
        raise ValueError("factorized point must lie on gamma^2+h^2=1")
    argument = float(np.clip(h / (1.0 + gamma), -1.0, 1.0))
    phi = 0.5 * math.acos(argument)
    return np.asarray([branch * math.sin(2.0 * phi), 0.0, math.cos(2.0 * phi)], dtype=np.float64)


def product_energy_pbc(N: int, gamma: float, h: float, local_bloch: np.ndarray) -> float:
    """Analytic expectation of the finite PBC XY Hamiltonian for a uniform product."""

    x, y, z = np.asarray(local_bloch, dtype=np.float64)
    bond = 0.5 * (1.0 + gamma) * x * x + 0.5 * (1.0 - gamma) * y * y
    return float(-N * bond - N * h * z)

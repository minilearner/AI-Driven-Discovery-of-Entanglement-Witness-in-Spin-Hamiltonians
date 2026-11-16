from __future__ import annotations

import random
from typing import Any

import h5py
import torch

from .production import _draw_dirichlet
from ..states.mps import random_mps
from ..states.noise import apply_local_noise, haar_local_unitaries
from ..states.separable import SeparableMixture


def recover_dv1_state(h5: h5py.File, row: int, *, device: torch.device | None = None):
    """Exactly reconstruct one D_v1 state from its frozen per-row metadata."""
    device = device or torch.device("cpu")
    N = int(h5["metadata/N"][row])
    seed = int(h5["metadata/seed"][row])
    generator_type = str(h5["metadata/generator_type"].asstr()[row])
    tg = torch.Generator(device=device).manual_seed(seed)
    if generator_type == "product_state":
        state = SeparableMixture.random(N, 1, generator=tg, device=device)
    elif generator_type == "classical_ghz_mixture":
        state = SeparableMixture.classical_ghz_mixture(N, generator=tg, device=device)
    elif generator_type == "separable_convex_mixture":
        K = int(h5["metadata/K"][row])
        alpha = float(h5["metadata/dirichlet_alpha"][row])
        state = SeparableMixture.random(N, K, generator=tg, device=device)
        state = SeparableMixture(_draw_dirichlet(K, alpha, seed + 17, device), state.bloch)
    elif generator_type == "noisy_random_mps":
        chi = int(h5["metadata/requested_chi_max"][row])
        state = random_mps(N, chi, generator=tg, device=device)
        unitaries, _ = haar_local_unitaries(N, generator=tg, device=device)
        state = state.apply_local_unitaries(unitaries)
        noise_type = str(h5["metadata/noise_type"].asstr()[row])
        strength = float(h5["metadata/noise_strength"][row])
        state = apply_local_noise(state, noise_type, strength)
        return state
    else:
        raise ValueError(f"unsupported D_v1 generator_type {generator_type!r}")
    # Production v1 always consumed the next generator values for a local Haar rotation.
    _, rotations = haar_local_unitaries(N, generator=tg, device=device)
    return state.apply_rotations(rotations)


def dv1_recovery_metadata(h5: h5py.File, row: int) -> dict[str, Any]:
    return {
        "state_id": str(h5["metadata/state_id"].asstr()[row]),
        "seed": int(h5["metadata/seed"][row]),
        "generator_type": str(h5["metadata/generator_type"].asstr()[row]),
        "noise_type": str(h5["metadata/noise_type"].asstr()[row]),
        "noise_strength": float(h5["metadata/noise_strength"][row]),
    }

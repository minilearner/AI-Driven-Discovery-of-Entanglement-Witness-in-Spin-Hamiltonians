from __future__ import annotations

import math

import torch

from .mps import PurifiedMPS


def haar_local_unitaries(
    N: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Haar SU(2) matrices and their corresponding SO(3) rotations."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    q = torch.randn(N, 4, generator=generator, device=device, dtype=real_dtype)
    q /= torch.linalg.vector_norm(q, dim=1, keepdim=True)
    w, x, y, z = q.unbind(dim=1)
    u = torch.empty((N, 2, 2), dtype=dtype, device=device)
    u[:, 0, 0] = w - 1j * z
    u[:, 0, 1] = -y - 1j * x
    u[:, 1, 0] = y - 1j * x
    u[:, 1, 1] = w + 1j * z
    r = torch.empty((N, 3, 3), dtype=real_dtype, device=device)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return u, r


def channel_kraus(name: str, strength: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("noise strength must be in [0,1]")
    i = torch.eye(2, dtype=dtype, device=device)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    p = strength
    if name == "dephasing":
        # Off-diagonal elements are attenuated by 1-p; p=1 is complete dephasing.
        return torch.stack((math.sqrt(1 - p / 2) * i, math.sqrt(p / 2) * z))
    if name == "depolarizing":
        return torch.stack(
            (math.sqrt(1 - 3 * p / 4) * i, math.sqrt(p / 4) * x, math.sqrt(p / 4) * y, math.sqrt(p / 4) * z)
        )
    if name == "amplitude_damping":
        k0 = torch.tensor([[1, 0], [0, math.sqrt(1 - p)]], dtype=dtype, device=device)
        k1 = torch.tensor([[0, math.sqrt(p)], [0, 0]], dtype=dtype, device=device)
        return torch.stack((k0, k1))
    raise ValueError(f"unknown noise channel: {name}")


def apply_local_noise(state: PurifiedMPS, name: str, strength: float) -> PurifiedMPS:
    kraus = channel_kraus(name, strength, device=state.device, dtype=state.dtype)
    out: list[torch.Tensor] = []
    for a in state.tensors:
        # B[l,s,(anc,k),r] = sum_t K[k,s,t] A[l,t,anc,r].
        b = torch.einsum("kst,ltar->lsakr", kraus, a)
        out.append(b.reshape(a.shape[0], 2, a.shape[2] * kraus.shape[0], a.shape[3]))
    return PurifiedMPS(out).normalize()


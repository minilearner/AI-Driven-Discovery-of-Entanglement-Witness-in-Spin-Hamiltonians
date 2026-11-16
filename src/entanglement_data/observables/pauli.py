from __future__ import annotations

import torch

AXIS_NAMES = ("X", "Y", "Z")


def pauli_matrices(*, device: torch.device, dtype: torch.dtype, include_identity: bool = False) -> torch.Tensor:
    mats = [
        torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
        torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
    ]
    if include_identity:
        mats.insert(0, torch.eye(2, dtype=dtype, device=device))
    return torch.stack(mats)


from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import QuantumState


class SeparableMixture(QuantumState):
    """Convex mixture of product one-qubit states, represented by Bloch vectors."""

    def __init__(self, weights: torch.Tensor, bloch: torch.Tensor):
        if weights.ndim != 1 or bloch.ndim != 3 or bloch.shape[0] != weights.shape[0] or bloch.shape[2] != 3:
            raise ValueError("expected weights [K] and bloch [K,N,3]")
        if torch.any(weights < 0):
            raise ValueError("mixture weights must be nonnegative")
        if not torch.isclose(weights.sum(), torch.tensor(1.0, device=weights.device, dtype=weights.dtype), atol=1e-10):
            raise ValueError("mixture weights must sum to one")
        if torch.any(torch.linalg.vector_norm(bloch, dim=-1) > 1.0 + 1e-10):
            raise ValueError("Bloch vector norm cannot exceed one")
        self.weights = weights
        self.bloch = bloch
        self.N = bloch.shape[1]

    @classmethod
    def random(
        cls,
        N: int,
        K: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> "SeparableMixture":
        raw_w = torch.rand(K, generator=generator, device=device, dtype=dtype)
        weights = raw_w / raw_w.sum()
        directions = torch.randn(K, N, 3, generator=generator, device=device, dtype=dtype)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-15)
        # Uniform radius in the Bloch ball, with a small pure-state component.
        radii = torch.rand(K, N, 1, generator=generator, device=device, dtype=dtype).pow(1.0 / 3.0)
        pure_mask = torch.rand(K, N, 1, generator=generator, device=device) < 0.15
        radii = torch.where(pure_mask, torch.ones_like(radii), radii)
        return cls(weights, directions * radii)

    @classmethod
    def classical_ghz_mixture(
        cls,
        N: int,
        *,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> "SeparableMixture":
        """Random-basis analogue of 1/2(|00..0><.| + |11..1><.|)."""
        directions = torch.randn(N, 3, generator=generator, device=device, dtype=dtype)
        directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-15)
        bloch = torch.stack((directions, -directions), dim=0)
        weights = torch.full((2,), 0.5, dtype=dtype, device=device)
        return cls(weights, bloch)

    def pauli_expectation(self, sites: Sequence[int], axes: Sequence[int]) -> torch.Tensor:
        if len(sites) != len(axes):
            raise ValueError("sites and axes must have equal length")
        values = torch.ones(self.weights.shape[0], dtype=self.bloch.dtype, device=self.bloch.device)
        for site, axis in zip(sites, axes):
            values *= self.bloch[:, site, axis]
        return torch.dot(self.weights, values)

    def expectation(self, sites: Sequence[int], operators: Sequence[torch.Tensor]) -> torch.Tensor:
        # General one-qubit O = c0 I + c.sigma, evaluated independently per component.
        if len(sites) != len(operators):
            raise ValueError("sites and operators must have equal length")
        values = torch.ones(self.weights.shape[0], dtype=torch.complex128, device=self.bloch.device)
        for site, op in zip(sites, operators):
            op = op.to(device=self.bloch.device, dtype=torch.complex128)
            c0 = torch.trace(op) / 2
            coeff = torch.stack(
                [
                    (op[0, 1] + op[1, 0]) / 2,
                    1j * (op[0, 1] - op[1, 0]) / 2,
                    (op[0, 0] - op[1, 1]) / 2,
                ]
            )
            values *= c0 + torch.einsum("ka,a->k", self.bloch[:, site].to(torch.complex128), coeff)
        return torch.dot(self.weights.to(torch.complex128), values)

    def apply_rotations(self, rotations: torch.Tensor) -> "SeparableMixture":
        """Apply SO(3) rotations corresponding to local SU(2) unitaries."""
        if rotations.shape != (self.N, 3, 3):
            raise ValueError("rotations must have shape [N,3,3]")
        rotated = torch.einsum("nij,knj->kni", rotations.to(self.bloch), self.bloch)
        return SeparableMixture(self.weights.clone(), rotated)

    def mean_local_bloch_norm(self) -> float:
        local = torch.einsum("k,kna->na", self.weights, self.bloch)
        return float(torch.linalg.vector_norm(local, dim=-1).mean().item())

    def purity(self) -> float:
        # Tr(rho_r rho_s) factorizes over sites.
        dots = torch.einsum("rna,sna->rsn", self.bloch, self.bloch)
        overlaps = ((1.0 + dots) / 2.0).clamp_min(0.0).prod(dim=-1)
        return float(torch.einsum("r,s,rs->", self.weights, self.weights, overlaps).item())

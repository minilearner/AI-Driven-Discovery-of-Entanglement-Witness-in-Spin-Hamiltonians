from __future__ import annotations

from dataclasses import dataclass

import torch

from ..observables.pauli import pauli_matrices
from ..states.base import QuantumState
from ..states.mps import PurifiedMPS, _transfer_left


@dataclass(frozen=True)
class NPTCertificate:
    certified: bool
    min_eigenvalue: float
    pair: tuple[int, int] | None
    checked_pairs: int


def two_site_density_matrix(state: QuantumState, i: int, j: int) -> torch.Tensor:
    if not 0 <= i < j < state.N:
        raise ValueError("require 0 <= i < j < N")
    if hasattr(state, "device"):
        device = state.device  # type: ignore[attr-defined]
        dtype = state.dtype  # type: ignore[attr-defined]
    else:
        device = state.bloch.device  # type: ignore[attr-defined]
        dtype = torch.complex128
    basis = pauli_matrices(device=device, dtype=dtype, include_identity=True)
    rho = torch.zeros((4, 4), dtype=dtype, device=device)
    for a in range(4):
        for b in range(4):
            if a == 0 and b == 0:
                coeff = torch.tensor(1.0, dtype=dtype, device=device)
            elif a == 0:
                coeff = state.expectation((j,), (basis[b],))
            elif b == 0:
                coeff = state.expectation((i,), (basis[a],))
            else:
                coeff = state.expectation((i, j), (basis[a], basis[b]))
            rho += coeff * torch.kron(basis[a], basis[b]) / 4
    return (rho + rho.conj().T) / 2


def partial_transpose_two_qubit(rho: torch.Tensor, subsystem: int = 0) -> torch.Tensor:
    shaped = rho.reshape(2, 2, 2, 2)  # (i,j,k,l) = <ij|rho|kl>
    if subsystem == 0:
        return shaped.permute(2, 1, 0, 3).reshape(4, 4)
    if subsystem == 1:
        return shaped.permute(0, 3, 2, 1).reshape(4, 4)
    raise ValueError("subsystem must be 0 or 1")


def _transfer_all_paulis(env: torch.Tensor, tensor: torch.Tensor, paulis: torch.Tensor) -> torch.Tensor:
    return torch.einsum("qab,ascr,pst,btcu->qpru", env, tensor.conj(), paulis, tensor)


def _transfer_identity_batch(env: torch.Tensor, tensor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    return torch.einsum("qab,ascr,st,btcu->qru", env, tensor.conj(), identity, tensor)


def _mps_npt_pairs(state: PurifiedMPS, max_distance: int):
    """Yield pair RDMs while reusing MPS environments and batched Pauli transfers."""
    paulis = pauli_matrices(device=state.device, dtype=state.dtype)
    identity = state.identity()
    left = state.left_environments()
    right = state.right_environments()
    local = torch.empty((state.N, 3), dtype=state.dtype, device=state.device)
    for i in range(state.N):
        env = _transfer_all_paulis(left[i].unsqueeze(0), state.tensors[i], paulis)[0]
        local[i] = torch.einsum("pab,ab->p", env, right[i + 1])
    for distance in range(1, max_distance + 1):
        for i in range(state.N - distance):
            j = i + distance
            env = _transfer_all_paulis(left[i].unsqueeze(0), state.tensors[i], paulis)[0]
            for k in range(i + 1, j):
                env = _transfer_identity_batch(env, state.tensors[k], identity)
            pair = _transfer_all_paulis(env, state.tensors[j], paulis)
            pair = torch.einsum("pqab,ab->pq", pair, right[j + 1])
            rho = torch.kron(identity, identity)
            for a in range(3):
                rho = rho + local[i, a] * torch.kron(paulis[a], identity)
                rho = rho + local[j, a] * torch.kron(identity, paulis[a])
                for b in range(3):
                    rho = rho + pair[a, b] * torch.kron(paulis[a], paulis[b])
            yield i, j, (rho / 4 + (rho / 4).conj().T) / 2


def mps_two_site_density_matrices(state: PurifiedMPS, max_distance: int = 1) -> list[tuple[int, int, torch.Tensor]]:
    """Extract all checked two-site RDMs with one set of cached MPS environments."""
    return list(_mps_npt_pairs(state, max_distance))


def certify_two_site_rdms(
    rdms: list[tuple[int, int, torch.Tensor]], *, tolerance: float = 1e-9
) -> NPTCertificate:
    minimum = float("inf")
    minimum_pair: tuple[int, int] | None = None
    for i, j, rho in rdms:
        eig_min = float(torch.linalg.eigvalsh(partial_transpose_two_qubit(rho)).min().real.item())
        if eig_min < minimum:
            minimum, minimum_pair = eig_min, (i, j)
    return NPTCertificate(minimum < -tolerance, minimum, minimum_pair, len(rdms))


def apply_channel_to_two_site_rdms(
    rdms: list[tuple[int, int, torch.Tensor]], kraus: torch.Tensor
) -> list[tuple[int, int, torch.Tensor]]:
    """Apply the same local CPTP channel to both qubits of every pair RDM."""
    two_site_kraus = [torch.kron(left, right) for left in kraus for right in kraus]
    output: list[tuple[int, int, torch.Tensor]] = []
    for i, j, rho in rdms:
        transformed = torch.zeros_like(rho)
        for operator in two_site_kraus:
            transformed += operator @ rho @ operator.conj().T
        transformed = (transformed + transformed.conj().T) / 2
        output.append((i, j, transformed))
    return output


def certify_two_site_npt(state: QuantumState, *, max_distance: int = 1, tolerance: float = 1e-9) -> NPTCertificate:
    minimum = float("inf")
    minimum_pair: tuple[int, int] | None = None
    checked = 0
    if isinstance(state, PurifiedMPS):
        pairs = _mps_npt_pairs(state, max_distance)
    else:
        pairs = (
            (i, i + distance, two_site_density_matrix(state, i, i + distance))
            for distance in range(1, max_distance + 1)
            for i in range(state.N - distance)
        )
    for i, j, rho in pairs:
        eig_min = float(torch.linalg.eigvalsh(partial_transpose_two_qubit(rho)).min().real.item())
        checked += 1
        if eig_min < minimum:
            minimum, minimum_pair = eig_min, (i, j)
    return NPTCertificate(minimum < -tolerance, minimum, minimum_pair, checked)

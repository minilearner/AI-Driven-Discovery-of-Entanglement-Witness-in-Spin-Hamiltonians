from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .base import QuantumState


def _transfer_left(env: torch.Tensor, tensor: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ab,ascr,st,btcu->ru", env, tensor.conj(), operator, tensor)


def _transfer_right(env: torch.Tensor, tensor: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ascr,st,btcu,ru->ab", tensor.conj(), operator, tensor, env)


@dataclass
class PurifiedMPS(QuantumState):
    """Open-boundary MPS with explicit physical and local purification indices."""

    tensors: list[torch.Tensor]

    def __post_init__(self) -> None:
        if not self.tensors:
            raise ValueError("MPS cannot be empty")
        for i, a in enumerate(self.tensors):
            if a.ndim != 4 or a.shape[1] != 2:
                raise ValueError("each tensor must have shape [Dl,2,anc,Dr]")
            if i == 0 and a.shape[0] != 1 or i == len(self.tensors) - 1 and a.shape[3] != 1:
                raise ValueError("open boundary dimensions must be one")
            if i and self.tensors[i - 1].shape[3] != a.shape[0]:
                raise ValueError("neighboring bond dimensions do not match")
        self.N = len(self.tensors)

    @property
    def device(self) -> torch.device:
        return self.tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.tensors[0].dtype

    @property
    def bond_dimensions(self) -> list[int]:
        return [self.tensors[0].shape[0]] + [a.shape[3] for a in self.tensors]

    def copy(self) -> "PurifiedMPS":
        return PurifiedMPS([a.clone() for a in self.tensors])

    def identity(self) -> torch.Tensor:
        return torch.eye(2, dtype=self.dtype, device=self.device)

    def left_environments(self) -> list[torch.Tensor]:
        envs = [torch.ones((1, 1), dtype=self.dtype, device=self.device)]
        ident = self.identity()
        for a in self.tensors:
            envs.append(_transfer_left(envs[-1], a, ident))
        return envs

    def right_environments(self) -> list[torch.Tensor]:
        envs: list[torch.Tensor | None] = [None] * (self.N + 1)
        envs[self.N] = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        ident = self.identity()
        for i in range(self.N - 1, -1, -1):
            envs[i] = _transfer_right(envs[i + 1], self.tensors[i], ident)  # type: ignore[arg-type]
        return envs  # type: ignore[return-value]

    def norm(self) -> torch.Tensor:
        return self.left_environments()[-1][0, 0].real

    def purity(self) -> float:
        """Return Tr(rho_physical^2) by a two-copy swap contraction.

        The contraction keeps four MPS bond indices and never constructs the
        exponentially large physical density matrix.
        """
        env = torch.ones((1, 1, 1, 1), dtype=self.dtype, device=self.device)
        for a in self.tensors:
            env = torch.einsum(
                "abcd,apxr,bqyu,cqxt,dpyv->rutv",
                env,
                a.conj(),
                a.conj(),
                a,
                a,
            )
        value = env.reshape(-1)[0]
        if abs(float(value.imag.item())) > 1e-9:
            raise FloatingPointError("purity contraction has a non-negligible imaginary residue")
        result = float(value.real.item())
        if result < -1e-9 or result > 1.0 + 1e-8:
            raise FloatingPointError(f"invalid purity {result}")
        return min(1.0, max(0.0, result))

    def normalize(self) -> "PurifiedMPS":
        nrm = self.norm()
        if nrm <= 0:
            raise ValueError("MPS has non-positive norm")
        self.tensors[-1] = self.tensors[-1] / torch.sqrt(nrm)
        return self

    def expectation(self, sites: Sequence[int], operators: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(sites) != len(operators):
            raise ValueError("sites and operators must have equal length")
        if list(sites) != sorted(sites) or len(set(sites)) != len(sites):
            raise ValueError("sites must be distinct and increasing")
        op_map = {i: op.to(device=self.device, dtype=self.dtype) for i, op in zip(sites, operators)}
        env = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        ident = self.identity()
        for i, a in enumerate(self.tensors):
            env = _transfer_left(env, a, op_map.get(i, ident))
        return env[0, 0]

    def reduced_density_matrix(self, sites: Sequence[int]) -> torch.Tensor:
        """Contract an arbitrary small physical RDM without forming the global state.

        The output basis follows the order supplied in ``sites``.  Internally we
        contract sites in chain order and permute the result back at the end.
        This routine is intended for the k<=5 certification blocks.
        """
        requested = tuple(int(i) for i in sites)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("sites must be a non-empty sequence of distinct indices")
        if min(requested) < 0 or max(requested) >= self.N:
            raise ValueError("site index out of range")
        ordered = tuple(sorted(requested))
        selected = set(ordered)
        # env axes: bra bond, ket bond, then (ket, bra) per selected site.
        env = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        for i, a in enumerate(self.tensors):
            if i in selected:
                env = torch.einsum("ab...,ascr,btcu->ru...ts", env, a.conj(), a)
            else:
                env = torch.einsum("ab...,ascr,bscu->ru...", env, a.conj(), a)
        tensor = env.reshape(*(2 for _ in range(2 * len(ordered))))
        # Current pairs are (ket_i, bra_i); matrix axes need all ket, then all bra.
        ket_axes = list(range(0, 2 * len(ordered), 2))
        bra_axes = list(range(1, 2 * len(ordered), 2))
        tensor = tensor.permute(*(ket_axes + bra_axes))
        if requested != ordered:
            pos = {site: j for j, site in enumerate(ordered)}
            order = [pos[site] for site in requested]
            tensor = tensor.permute(*(order + [len(order) + j for j in order]))
        dim = 2 ** len(requested)
        rho = tensor.reshape(dim, dim)
        rho = (rho + rho.conj().T) / 2
        trace = torch.trace(rho).real
        if trace <= 0:
            raise FloatingPointError("reduced density matrix has non-positive trace")
        return rho / trace

    def apply_local_unitaries(self, unitaries: torch.Tensor) -> "PurifiedMPS":
        if unitaries.shape != (self.N, 2, 2):
            raise ValueError("unitaries must have shape [N,2,2]")
        out = [torch.einsum("st,ltar->lsar", unitaries[i].to(a), a) for i, a in enumerate(self.tensors)]
        return PurifiedMPS(out)

    def apply_gauge(self, bond: int, gauge: torch.Tensor) -> "PurifiedMPS":
        """Apply A[bond-1] G and G^-1 A[bond] without changing the state."""
        if not 1 <= bond < self.N:
            raise ValueError("bond must be in [1,N-1]")
        out = [a.clone() for a in self.tensors]
        g = gauge.to(out[bond - 1])
        if g.shape != (out[bond - 1].shape[3], out[bond - 1].shape[3]):
            raise ValueError("gauge dimension does not match bond")
        out[bond - 1] = torch.einsum("lsar,ru->lsau", out[bond - 1], g)
        out[bond] = torch.einsum("lr,rsau->lsau", torch.linalg.inv(g), out[bond])
        return PurifiedMPS(out)

    def schmidt_values(self, cut: int, tolerance: float = 1e-12) -> torch.Tensor:
        """Schmidt values of a pure MPS at a cut, using mixed canonical form."""
        if any(a.shape[2] != 1 for a in self.tensors):
            raise ValueError("Schmidt values are only defined here for pure MPS")
        if not 1 <= cut < self.N:
            raise ValueError("cut must be in [1,N-1]")
        tensors = [a[:, :, 0, :].clone() for a in self.tensors]
        for i in range(cut):
            dl, d, dr = tensors[i].shape
            q, r = torch.linalg.qr(tensors[i].reshape(dl * d, dr), mode="reduced")
            tensors[i] = q.reshape(dl, d, q.shape[1])
            if i + 1 < self.N:
                tensors[i + 1] = torch.einsum("ab,bsr->asr", r, tensors[i + 1])
        center = None
        for i in range(self.N - 1, cut - 1, -1):
            dl, d, dr = tensors[i].shape
            q, r = torch.linalg.qr(tensors[i].reshape(dl, d * dr).T, mode="reduced")
            tensors[i] = q.T.reshape(q.shape[1], d, dr)
            center = r.T
            if i > cut:
                tensors[i - 1] = torch.einsum("lsr,ru->lsu", tensors[i - 1], center)
        assert center is not None
        values = torch.linalg.svdvals(center).real
        values = values[values > tolerance]
        return values / torch.linalg.vector_norm(values)


def random_mps(
    N: int,
    chi_max: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.complex128,
) -> PurifiedMPS:
    if N < 2 or chi_max < 1:
        raise ValueError("N >= 2 and chi_max >= 1 required")
    bonds = [1]
    for i in range(1, N):
        bonds.append(min(chi_max, 2 ** min(i, N - i)))
    bonds.append(1)
    tensors: list[torch.Tensor] = []
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    for i in range(N):
        shape = (bonds[i], 2, 1, bonds[i + 1])
        re = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
        im = torch.randn(shape, generator=generator, device=device, dtype=real_dtype)
        tensors.append((re + 1j * im).to(dtype))
    # Stable left-canonical QR sweep.
    for i in range(N - 1):
        dl, d, anc, dr = tensors[i].shape
        q, r = torch.linalg.qr(tensors[i].reshape(dl * d * anc, dr), mode="reduced")
        tensors[i] = q.reshape(dl, d, anc, q.shape[1])
        tensors[i + 1] = torch.einsum("ab,bscr->ascr", r, tensors[i + 1])
    return PurifiedMPS(tensors).normalize()


def mps_from_statevector(vector: torch.Tensor, N: int, chi_max: int | None = None) -> PurifiedMPS:
    if vector.numel() != 2**N:
        raise ValueError("statevector size must be 2**N")
    dtype, device = vector.dtype, vector.device
    work = vector.reshape(1, -1) / torch.linalg.vector_norm(vector)
    tensors: list[torch.Tensor] = []
    left_dim = 1
    for site in range(N - 1):
        work = work.reshape(left_dim * 2, -1)
        u, s, vh = torch.linalg.svd(work, full_matrices=False)
        rank = s.numel() if chi_max is None else min(s.numel(), chi_max)
        u, s, vh = u[:, :rank], s[:rank], vh[:rank]
        tensors.append(u.reshape(left_dim, 2, 1, rank).to(dtype=dtype, device=device))
        work = s[:, None] * vh
        left_dim = rank
    tensors.append(work.reshape(left_dim, 2, 1, 1).to(dtype=dtype, device=device))
    return PurifiedMPS(tensors).normalize()


def purified_mps_to_dense_density(state: PurifiedMPS) -> torch.Tensor:
    """Dense reference implementation for tests only; exponential in N."""
    if state.N > 10:
        raise ValueError("dense conversion is restricted to N <= 10")
    psi = state.tensors[0][0]
    # psi axes are p0,a0,bond; append p_i,a_i,bond.
    for a in state.tensors[1:]:
        psi = torch.tensordot(psi, a, dims=([-1], [0]))
    psi = psi.squeeze(-1)
    physical_axes = list(range(0, 2 * state.N, 2))
    ancilla_axes = list(range(1, 2 * state.N, 2))
    psi = psi.permute(*(physical_axes + ancilla_axes)).reshape(2**state.N, -1)
    return psi @ psi.conj().T

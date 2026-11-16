from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from itertools import product

import numpy as np
import torch

from ..states.mps import PurifiedMPS
from ..states.separable import SeparableMixture
from .pauli import AXIS_NAMES, pauli_matrices


@dataclass(frozen=True)
class Skip3Term:
    sites: tuple[int, int, int]
    axes: tuple[int, int, int]
    name: str


@dataclass(frozen=True)
class Skip3Spec:
    """The two missing three-site supports inside every contiguous 4-block."""

    N: int = 16

    @cached_property
    def supports(self) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for i in range(self.N - 3):
            out.extend(((i, i + 1, i + 3), (i, i + 2, i + 3)))
        return out

    @cached_property
    def terms(self) -> list[Skip3Term]:
        out: list[Skip3Term] = []
        for sites in self.supports:
            for axes in product(range(3), repeat=3):
                name = " ".join(f"{AXIS_NAMES[a]}_{s}" for s, a in zip(sites, axes))
                out.append(Skip3Term(sites, axes, name))
        return out

    @cached_property
    def names(self) -> list[str]:
        return [term.name for term in self.terms]

    @property
    def num_features(self) -> int:
        return 2 * max(0, self.N - 3) * 27

    @cached_property
    def schema_hash(self) -> str:
        payload = json.dumps(self.names, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checked_real(value: torch.Tensor, tolerance: float = 1e-9) -> np.ndarray:
    if float(value.imag.abs().max().item()) > tolerance:
        raise FloatingPointError("skip3 Hermitian expectations have a non-negligible imaginary residue")
    return value.real.detach().cpu().numpy()


def _mps_skip3(state: PurifiedMPS, spec: Skip3Spec) -> np.ndarray:
    if not spec.supports:
        return np.empty(0, dtype=np.float64)
    paulis = pauli_matrices(device=state.device, dtype=state.dtype)
    ident = state.identity()
    left = state.left_environments()
    right = state.right_environments()
    values: list[np.ndarray] = []
    for sites in spec.supports:
        selected = set(sites)
        env = left[sites[0]].unsqueeze(0)
        for site in range(sites[0], sites[-1] + 1):
            tensor = state.tensors[site]
            if site in selected:
                expanded = torch.einsum("qab,ascr,pst,btcu->qpru", env, tensor.conj(), paulis, tensor)
                env = expanded.reshape(-1, expanded.shape[-2], expanded.shape[-1])
            else:
                env = torch.einsum("qab,ascr,st,btcu->qru", env, tensor.conj(), ident, tensor)
        values.append(_checked_real(torch.einsum("qab,ab->q", env, right[sites[-1] + 1])))
    return np.concatenate(values)


def _separable_skip3(state: SeparableMixture, spec: Skip3Spec) -> np.ndarray:
    if not spec.supports:
        return np.empty(0, dtype=np.float64)
    blocks = []
    for i, j, k in spec.supports:
        tensor = torch.einsum(
            "r,ra,rb,rc->abc", state.weights, state.bloch[:, i], state.bloch[:, j], state.bloch[:, k]
        )
        blocks.append(tensor.detach().cpu().numpy().reshape(-1))
    return np.concatenate(blocks)


def compute_skip3_features(
    state: PurifiedMPS | SeparableMixture,
    spec: Skip3Spec | None = None,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    spec = spec or Skip3Spec(state.N)
    if state.N != spec.N:
        raise ValueError("state and skip3 specification have different N")
    if isinstance(state, PurifiedMPS):
        values = _mps_skip3(state, spec)
    elif isinstance(state, SeparableMixture):
        values = _separable_skip3(state, spec)
    else:
        raise TypeError(f"unsupported state type: {type(state)!r}")
    if values.shape != (spec.num_features,):
        raise RuntimeError(f"skip3 feature count mismatch: {values.shape}")
    return values.astype(output_dtype, copy=False)


def decode_skip3(values: np.ndarray, spec: Skip3Spec | None = None) -> np.ndarray:
    spec = spec or Skip3Spec(16)
    array = np.asarray(values)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != spec.num_features:
        raise ValueError(f"skip3 values must have shape [{spec.num_features}] or [B,{spec.num_features}]")
    decoded = array.astype(np.float64, copy=False).reshape(len(array), spec.N - 3, 2, 3, 3, 3)
    return decoded[0] if single else decoded


def transform_skip3(
    values: np.ndarray,
    rotations: np.ndarray,
    spec: Skip3Spec | None = None,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    spec = spec or Skip3Spec(16)
    array = np.asarray(values)
    single = array.ndim == 1
    if single:
        array = array[None, :]
    tensor = decode_skip3(array, spec)
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim == 3:
        matrices = np.broadcast_to(matrices[None], (len(array), spec.N, 3, 3))
    if matrices.shape != (len(array), spec.N, 3, 3):
        raise ValueError("rotation shape mismatch")
    out = np.empty_like(tensor)
    for support_index, sites in enumerate(spec.supports):
        block, branch = divmod(support_index, 2)
        i, j, k = sites
        out[:, block, branch] = np.einsum(
            "sam,sbn,scp,smnp->sabc",
            matrices[:, i], matrices[:, j], matrices[:, k], tensor[:, block, branch], optimize=True,
        )
    result = out.reshape(len(array), -1).astype(output_dtype, copy=False)
    return result[0] if single else result

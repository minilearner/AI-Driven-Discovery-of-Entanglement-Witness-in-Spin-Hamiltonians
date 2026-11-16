from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch

from ..states.mps import PurifiedMPS, _transfer_left
from ..states.separable import SeparableMixture
from .feature_spec import FeatureSpec
from .pauli import pauli_matrices


@dataclass
class FeatureResult:
    values: np.ndarray
    names: list[str]
    connected_values: np.ndarray | None = None
    connected_names: list[str] | None = None


def _real_scalar(value: torch.Tensor, tolerance: float = 1e-9) -> float:
    if abs(float(value.imag.item())) > tolerance:
        raise FloatingPointError(f"Hermitian expectation has imaginary residue {value.imag.item()}")
    return float(value.real.item())


def _transfer_all_paulis(env: torch.Tensor, tensor: torch.Tensor, paulis: torch.Tensor) -> torch.Tensor:
    """Append all X/Y/Z choices to a batch of left environments."""
    return torch.einsum("qab,ascr,pst,btcu->qpru", env, tensor.conj(), paulis, tensor)


def _transfer_identity_batch(env: torch.Tensor, tensor: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    return torch.einsum("qab,ascr,st,btcu->qru", env, tensor.conj(), identity, tensor)


def _checked_real_array(value: torch.Tensor, tolerance: float = 1e-9) -> list[float]:
    if float(value.imag.abs().max().item()) > tolerance:
        raise FloatingPointError("Hermitian expectations have a non-negligible imaginary residue")
    return value.real.detach().cpu().reshape(-1).tolist()


def _mps_features(state: PurifiedMPS, spec: FeatureSpec, output_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    paulis = pauli_matrices(device=state.device, dtype=state.dtype)
    ident = state.identity()
    left = state.left_environments()
    right = state.right_environments()
    values: list[float] = []

    # One-body terms.
    for i in range(state.N):
        m = _transfer_all_paulis(left[i].unsqueeze(0), state.tensors[i], paulis)[0]
        values.extend(_checked_real_array(torch.einsum("pab,ab->p", m, right[i + 1])))

    # All pairs: one identity propagation per fixed left endpoint/operator.
    if spec.all_two_body:
        pair_values: dict[tuple[int, int, int, int], float] = {}
        for i in range(state.N):
            m = _transfer_all_paulis(left[i].unsqueeze(0), state.tensors[i], paulis)[0]
            for j in range(i + 1, state.N):
                end = _transfer_all_paulis(m, state.tensors[j], paulis)
                pair_matrix = torch.einsum("pqab,ab->pq", end, right[j + 1])
                pair_list = _checked_real_array(pair_matrix)
                for a in range(3):
                    for b in range(3):
                        pair_values[(i, j, a, b)] = pair_list[3 * a + b]
                m = _transfer_identity_batch(m, state.tensors[j], ident)
        # Match FeatureSpec ordering: (i,j) first, then axes with b fastest.
        for i in range(state.N):
            for j in range(i + 1, state.N):
                for a in range(3):
                    for b in range(3):
                        values.append(pair_values[(i, j, a, b)])

    # Short contiguous blocks reuse the cached outer environments.
    if spec.contiguous_three_body:
        for i in range(state.N - 2):
            m = left[i].unsqueeze(0)
            for offset in range(3):
                expanded = _transfer_all_paulis(m, state.tensors[i + offset], paulis)
                m = expanded.reshape(-1, expanded.shape[-2], expanded.shape[-1])
            values.extend(_checked_real_array(torch.einsum("qab,ab->q", m, right[i + 3])))
    if spec.contiguous_four_body:
        for i in range(state.N - 3):
            m = left[i].unsqueeze(0)
            for offset in range(4):
                expanded = _transfer_all_paulis(m, state.tensors[i + offset], paulis)
                m = expanded.reshape(-1, expanded.shape[-2], expanded.shape[-1])
            values.extend(_checked_real_array(torch.einsum("qab,ab->q", m, right[i + 4])))
    return np.asarray(values, dtype=output_dtype)


def _separable_features(state: SeparableMixture, spec: FeatureSpec, output_dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    values = [float(state.pauli_expectation(term.sites, term.axes).item()) for term in spec.terms()]
    return np.asarray(values, dtype=output_dtype)


def _connected_two_three(raw: np.ndarray, spec: FeatureSpec) -> tuple[np.ndarray, list[str]]:
    """Connected 2/3-body cumulants from raw moments; no new contractions."""
    terms = spec.terms()
    moment = {(t.sites, t.axes): float(v) for t, v in zip(terms, raw)}
    out: list[float] = []
    names: list[str] = []
    for term in terms:
        if len(term.sites) == 2:
            i, j = term.sites
            a, b = term.axes
            value = moment[(term.sites, term.axes)] - moment[((i,), (a,))] * moment[((j,), (b,))]
        elif len(term.sites) == 3:
            i, j, k = term.sites
            a, b, c = term.axes
            value = (
                moment[(term.sites, term.axes)]
                - moment[((i, j), (a, b))] * moment[((k,), (c,))]
                - moment[((i, k), (a, c))] * moment[((j,), (b,))]
                - moment[((j, k), (b, c))] * moment[((i,), (a,))]
                + 2 * moment[((i,), (a,))] * moment[((j,), (b,))] * moment[((k,), (c,))]
            )
        else:
            continue
        out.append(value)
        names.append(f"connected({term.name})")
    return np.asarray(out, dtype=np.float32), names


def compute_features(
    state: PurifiedMPS | SeparableMixture,
    spec: FeatureSpec,
    *,
    connected: bool = False,
    output_dtype: np.dtype | type = np.float32,
) -> FeatureResult:
    if state.N != spec.N:
        raise ValueError("state and feature specification have different N")
    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("output_dtype must be float32 or float64")
    if isinstance(state, PurifiedMPS):
        values = _mps_features(state, spec, dtype)
    elif isinstance(state, SeparableMixture):
        values = _separable_features(state, spec, dtype)
    else:
        raise TypeError(f"unsupported state type: {type(state)!r}")
    if values.shape != (spec.num_features,):
        raise RuntimeError(f"feature count mismatch: got {values.size}, expected {spec.num_features}")
    connected_values = connected_names = None
    if connected:
        connected_values, connected_names = _connected_two_three(values, spec)
    return FeatureResult(values, spec.names(), connected_values, connected_names)

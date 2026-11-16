from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Mapping

import numpy as np

from entanglement_data.ml.invariant_transformer import structured_inv123_raw_t4_local_tokens
from entanglement_data.xy.state_space_exact import DensePureState


PAULIS = (
    np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
)


@dataclass(frozen=True)
class LocalRawSpec:
    """The literal Pauli estimators visible to the frozen local model."""

    N: int = 16
    R_obs: int = 4

    def __post_init__(self) -> None:
        if self.N <= self.R_obs or self.R_obs < 2:
            raise ValueError("require N > R_obs >= 2")

    @property
    def order_counts(self) -> dict[str, int]:
        return {
            "T1": 3 * self.N,
            "T2": 9 * sum(self.N - distance for distance in range(1, self.R_obs + 1)),
            "T3": 27 * (self.N - 2),
            "T4": 81 * (self.N - 3),
        }

    @property
    def raw_dimension(self) -> int:
        return sum(self.order_counts.values())

    @property
    def order_slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for name, count in self.order_counts.items():
            result[name] = slice(start, start + count)
            start += count
        return result


@lru_cache(maxsize=None)
def pauli_product_basis(order: int) -> np.ndarray:
    if order < 1 or order > 4:
        raise ValueError("only one- through four-body Pauli products are supported")
    matrices = []
    for axes in product(range(3), repeat=order):
        value = np.asarray([[1.0 + 0.0j]])
        for axis in axes:
            value = np.kron(value, PAULIS[axis])
        matrices.append(value)
    return np.stack(matrices)


def pauli_tensor_from_rdm(rho: np.ndarray, order: int) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.complex128)
    dimension = 1 << order
    if rho.shape != (dimension, dimension):
        raise ValueError("RDM dimension does not match Pauli order")
    values = np.einsum("ij,aji->a", rho, pauli_product_basis(order), optimize=True)
    if np.max(np.abs(values.imag)) > 2e-9:
        raise FloatingPointError("Pauli tensor has a non-negligible imaginary residual")
    return values.real.reshape((3,) * order)


def translation_invariant_blocks(state: DensePureState, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    if state.N != spec.N:
        raise ValueError("state and observable specifications have different N")
    one_block = pauli_tensor_from_rdm(state.reduced_density_matrix((0,)), 1)
    pair_blocks = np.stack([
        pauli_tensor_from_rdm(state.reduced_density_matrix((0, distance)), 2)
        for distance in range(1, spec.R_obs + 1)
    ])
    three_block = pauli_tensor_from_rdm(state.reduced_density_matrix((0, 1, 2)), 3)
    four_block = pauli_tensor_from_rdm(state.reduced_density_matrix((0, 1, 2, 3)), 4)
    one = np.repeat(one_block[None], spec.N, axis=0)
    two = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for distance in range(1, spec.R_obs + 1):
        two[: spec.N - distance, distance - 1] = pair_blocks[distance - 1]
    three = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    three[: spec.N - 2] = three_block
    four = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    four[: spec.N - 3] = four_block
    return {"one": one, "two": two, "three": three, "four": four}


def pack_raw_blocks(blocks: Mapping[str, np.ndarray], spec: LocalRawSpec) -> np.ndarray:
    one = np.asarray(blocks["one"])
    two = np.asarray(blocks["two"])
    three = np.asarray(blocks["three"])
    four = np.asarray(blocks["four"])
    prefix = one.shape[:-2]
    if one.shape[-2:] != (spec.N, 3):
        raise ValueError("one-body block has the wrong shape")
    if two.shape != (*prefix, spec.N, spec.R_obs, 3, 3):
        raise ValueError("two-body block has the wrong shape")
    if three.shape != (*prefix, spec.N, 3, 3, 3):
        raise ValueError("three-body block has the wrong shape")
    if four.shape != (*prefix, spec.N, 3, 3, 3, 3):
        raise ValueError("four-body block has the wrong shape")
    pieces = [one.reshape(*prefix, -1)]
    for distance in range(1, spec.R_obs + 1):
        pieces.append(two[..., : spec.N - distance, distance - 1, :, :].reshape(*prefix, -1))
    pieces.append(three[..., : spec.N - 2, :, :, :].reshape(*prefix, -1))
    pieces.append(four[..., : spec.N - 3, :, :, :, :].reshape(*prefix, -1))
    result = np.concatenate(pieces, axis=-1)
    if result.shape[-1] != spec.raw_dimension:
        raise RuntimeError("packed raw-observable dimension mismatch")
    return result


def unpack_raw_vector(values: np.ndarray, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    values = np.asarray(values)
    single = values.ndim == 1
    if single:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != spec.raw_dimension:
        raise ValueError(f"expected [B,{spec.raw_dimension}] raw observables")
    batch = len(values)
    cursor = 0
    one_count = 3 * spec.N
    one = values[:, cursor : cursor + one_count].reshape(batch, spec.N, 3)
    cursor += one_count
    two = np.zeros((batch, spec.N, spec.R_obs, 3, 3), dtype=values.dtype)
    for distance in range(1, spec.R_obs + 1):
        count = 9 * (spec.N - distance)
        two[:, : spec.N - distance, distance - 1] = values[:, cursor : cursor + count].reshape(
            batch, spec.N - distance, 3, 3
        )
        cursor += count
    three_count = 27 * (spec.N - 2)
    three = np.zeros((batch, spec.N, 3, 3, 3), dtype=values.dtype)
    three[:, : spec.N - 2] = values[:, cursor : cursor + three_count].reshape(
        batch, spec.N - 2, 3, 3, 3
    )
    cursor += three_count
    four_count = 81 * (spec.N - 3)
    four = np.zeros((batch, spec.N, 3, 3, 3, 3), dtype=values.dtype)
    four[:, : spec.N - 3] = values[:, cursor : cursor + four_count].reshape(
        batch, spec.N - 3, 3, 3, 3, 3
    )
    cursor += four_count
    if cursor != spec.raw_dimension:
        raise RuntimeError("raw-observable unpack cursor mismatch")
    result = {"one": one, "two": two, "three": three, "four": four}
    return {key: value[0] for key, value in result.items()} if single else result


def connected_blocks(blocks: Mapping[str, np.ndarray], spec: LocalRawSpec) -> tuple[np.ndarray, np.ndarray]:
    one = np.asarray(blocks["one"])
    two = np.asarray(blocks["two"])
    three = np.asarray(blocks["three"])
    if one.ndim == 2:
        one, two, three = one[None], two[None], three[None]
    batch = len(one)
    connected_two = np.zeros_like(two)
    for distance in range(1, spec.R_obs + 1):
        width = spec.N - distance
        connected_two[:, :width, distance - 1] = (
            two[:, :width, distance - 1]
            - one[:, :width, :, None] * one[:, distance : distance + width, None, :]
        )
    connected_three = np.zeros_like(three)
    width = spec.N - 2
    m0, m1, m2 = one[:, :width], one[:, 1 : width + 1], one[:, 2 : width + 2]
    t01 = two[:, :width, 0]
    t12 = two[:, 1 : width + 1, 0]
    t02 = two[:, :width, 1]
    connected_three[:, :width] = (
        three[:, :width]
        - t01[..., :, :, None] * m2[..., None, None, :]
        - t02[..., :, None, :] * m1[..., None, :, None]
        - t12[..., None, :, :] * m0[..., :, None, None]
        + 2.0 * m0[..., :, None, None] * m1[..., None, :, None] * m2[..., None, None, :]
    )
    return connected_two, connected_three


def invariant_tokens_from_raw(values: np.ndarray, spec: LocalRawSpec, *, dtype=np.float32) -> np.ndarray:
    blocks = unpack_raw_vector(values, spec)
    single = np.asarray(values).ndim == 1
    if single:
        blocks = {key: value[None] for key, value in blocks.items()}
    connected_two, connected_three = connected_blocks(blocks, spec)
    batch = len(blocks["one"])
    two_mask = np.zeros((batch, spec.N, spec.R_obs), dtype=bool)
    for distance in range(1, spec.R_obs + 1):
        two_mask[:, : spec.N - distance, distance - 1] = True
    three_mask = np.zeros((batch, spec.N), dtype=bool)
    three_mask[:, : spec.N - 2] = True
    four_mask = np.zeros((batch, spec.N), dtype=bool)
    four_mask[:, : spec.N - 3] = True
    result = structured_inv123_raw_t4_local_tokens(
        blocks["one"], blocks["two"], connected_two,
        blocks["three"], connected_three, blocks["four"],
        two_mask=two_mask, three_mask=three_mask, four_mask=four_mask,
        R_obs=spec.R_obs, output_dtype=dtype,
    )
    return result[0] if single else result


def rms_distances(values: np.ndarray, reference: np.ndarray, spec: LocalRawSpec) -> dict[str, float]:
    delta = np.asarray(values, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if delta.shape != (spec.raw_dimension,):
        raise ValueError("distance inputs have the wrong dimension")
    result = {
        f"d_{name}": float(np.sqrt(np.mean(np.square(delta[indices]))))
        for name, indices in spec.order_slices.items()
    }
    result["d_raw"] = float(np.sqrt(np.mean(np.square(delta))))
    return result


def sample_finite_shots(
    means: np.ndarray,
    n_shots: int,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 1 or np.any(means < -1.0 - 1e-9) or np.any(means > 1.0 + 1e-9):
        raise ValueError("Pauli means must be a one-dimensional vector in [-1,1]")
    probability = np.clip((means + 1.0) / 2.0, 0.0, 1.0)
    counts = rng.binomial(int(n_shots), probability, size=(int(repetitions), len(means)))
    return (2.0 * counts / float(n_shots) - 1.0).astype(np.float32)


def channel_anchor_means(tokens: np.ndarray, spec: LocalRawSpec) -> np.ndarray:
    """Average each physical channel over only its valid OBC anchors."""

    values = np.asarray(tokens)
    single = values.ndim == 2
    if single:
        values = values[None]
    if values.ndim != 3 or values.shape[1:] != (spec.N, 122):
        raise ValueError("expected full [B,N,122] invariant tokens")
    output = np.zeros((len(values), 117), dtype=np.float64)
    output[:, 0] = values[:, :, 0].mean(axis=1)
    channel = 1
    for distance in range(1, spec.R_obs + 1):
        output[:, channel : channel + 14] = values[:, : spec.N - distance, channel : channel + 14].mean(axis=1)
        channel += 14
    output[:, channel : channel + 20] = values[:, : spec.N - 2, channel : channel + 20].mean(axis=1)
    output[:, 77:117] = values[:, : spec.N - 3, 77:117].mean(axis=1)
    return output[0] if single else output

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np


Representation = Literal["Inv123", "C_geom3"]
Pooling = Literal["max", "mean", "max_mean"]


@dataclass(frozen=True)
class SpatialInvariantSpec:
    """Map flattened LU invariants to actual OBC lattice anchors."""

    representation: Representation
    N: int = 16

    @property
    def pair_channels(self) -> int:
        return 14 if self.representation == "Inv123" else 10

    @property
    def triple_channels(self) -> int:
        return 20

    @property
    def flat_dimension(self) -> int:
        return 1976 if self.representation == "Inv123" else 1496

    @property
    def local_channels(self) -> int:
        # K1 + distance-resolved pairs + contiguous K3 + 15 pair masks + K3 mask.
        return 1 + (self.N - 1) * self.pair_channels + self.triple_channels + self.N

    @property
    def distance_channels(self) -> int:
        # mean, max, and RMS for each invariant at a fixed physical separation.
        return 3 * self.pair_channels

    @cached_property
    def pair_sites(self) -> tuple[np.ndarray, np.ndarray]:
        left, right = np.triu_indices(self.N, 1)
        return left.astype(np.int64), right.astype(np.int64)

    @cached_property
    def pair_lookup(self) -> np.ndarray:
        lookup = np.full((self.N, self.N), -1, dtype=np.int64)
        left, right = self.pair_sites
        lookup[left, right] = np.arange(len(left))
        return lookup

    @cached_property
    def channel_names(self) -> list[str]:
        names = ["K1/bloch_norm"]
        for distance in range(1, self.N):
            names.extend(f"pair/r_{distance}/inv_{index}" for index in range(self.pair_channels))
        names.extend(f"contiguous_K3/inv_{index}" for index in range(self.triple_channels))
        names.extend(f"mask/pair_r_{distance}" for distance in range(1, self.N))
        names.append("mask/contiguous_K3")
        return names

    def local_channel_indices(self, R_obs: int | None = None) -> np.ndarray:
        cutoff = self.N - 1 if R_obs is None else int(R_obs)
        if not 1 <= cutoff <= self.N - 1: raise ValueError("invalid R_obs")
        pair_stop = 1 + cutoff * self.pair_channels
        triple_start = 1 + (self.N - 1) * self.pair_channels
        mask_start = triple_start + self.triple_channels
        return np.r_[np.arange(pair_stop), np.arange(triple_start,triple_start+self.triple_channels), np.arange(mask_start,mask_start+cutoff), mask_start+self.N-1].astype(np.int64)

    def local_valid_positions(self, R_obs: int | None = None) -> list[np.ndarray | None]:
        cutoff = self.N - 1 if R_obs is None else int(R_obs); valid: list[np.ndarray | None]=[np.arange(self.N)]
        for distance in range(1,cutoff+1): valid.extend([np.arange(self.N-distance)]*self.pair_channels)
        valid.extend([np.arange(self.N-2)]*self.triple_channels)
        valid.extend([None]*(cutoff+1))  # explicit masks remain unnormalized
        return valid


def _flat_blocks(values: np.ndarray, spec: SpatialInvariantSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != spec.flat_dimension:
        raise ValueError(f"expected [B,{spec.flat_dimension}], got {values.shape}")
    batch, N = len(values), spec.N
    pairs = N * (N - 1) // 2
    one = values[:, :N].reshape(batch, N, 1)
    if spec.representation == "Inv123":
        pair = values[:, N : N + 14 * pairs].reshape(batch, pairs, 14)
        triple = values[:, N + 14 * pairs :].reshape(batch, N - 2, 20)
    else:
        offset = N
        intrinsic = values[:, offset : offset + 4 * pairs].reshape(batch, pairs, 4); offset += 4 * pairs
        geometry = values[:, offset : offset + 6 * pairs].reshape(batch, pairs, 6); offset += 6 * pairs
        k3_intrinsic = values[:, offset : offset + 10 * (N - 2)].reshape(batch, N - 2, 10); offset += 10 * (N - 2)
        k3_cross = values[:, offset:].reshape(batch, N - 2, 10)
        pair = np.concatenate((intrinsic, geometry), axis=2)
        triple = np.concatenate((k3_intrinsic, k3_cross), axis=2)
    return one, pair, triple


def organize_local_invariants(
    values: np.ndarray,
    spec: SpatialInvariantSpec,
    *,
    R_obs: int | None = None,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return [B,N,C] with true physical position as the second axis.

    Pair channels are anchored on the left site and kept distinct by distance.
    Contiguous triples are also left anchored.  OBC boundaries are zero padded
    and explicitly identified by mask channels; no wrap-around observables are
    fabricated.
    """
    cutoff = spec.N - 1 if R_obs is None else int(R_obs)
    if not 1 <= cutoff <= spec.N - 1:
        raise ValueError("R_obs must lie in [1,N-1] or be None")
    one, pair, triple = _flat_blocks(values, spec)
    batch = len(one); out = np.zeros((batch, spec.N, spec.local_channels), dtype=output_dtype)
    out[:, :, 0] = one[:, :, 0]
    pair_start = 1
    mask_start = pair_start + (spec.N - 1) * spec.pair_channels + spec.triple_channels
    for distance in range(1, spec.N):
        channel = pair_start + (distance - 1) * spec.pair_channels
        if distance <= cutoff:
            anchors = np.arange(spec.N - distance)
            indices = spec.pair_lookup[anchors, anchors + distance]
            out[:, anchors, channel : channel + spec.pair_channels] = pair[:, indices]
            out[:, anchors, mask_start + distance - 1] = 1.0
    triple_start = pair_start + (spec.N - 1) * spec.pair_channels
    out[:, : spec.N - 2, triple_start : triple_start + spec.triple_channels] = triple
    out[:, : spec.N - 2, mask_start + spec.N - 1] = 1.0
    return out


def organize_distance_invariants(
    values: np.ndarray,
    spec: SpatialInvariantSpec,
    *,
    R_obs: int | None = None,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return [B,R_obs,3*P] pooled along position but not separation."""
    cutoff = spec.N - 1 if R_obs is None else int(R_obs)
    if not 1 <= cutoff <= spec.N - 1:
        raise ValueError("R_obs must lie in [1,N-1] or be None")
    _, pair, _ = _flat_blocks(values, spec); result = []
    for distance in range(1, cutoff + 1):
        anchors = np.arange(spec.N - distance); indices = spec.pair_lookup[anchors, anchors + distance]
        block = pair[:, indices]
        result.append(np.concatenate((block.mean(axis=1), block.max(axis=1), np.sqrt(np.square(block).mean(axis=1))), axis=1))
    return np.stack(result, axis=1).astype(output_dtype, copy=False)


def raw_observable_count(N: int, R_obs: int, *, include_K1: bool = True, include_contiguous_K3: bool = True) -> int:
    if not 1 <= R_obs <= N - 1:
        raise ValueError("invalid R_obs")
    count = 3 * N if include_K1 else 0
    count += 9 * sum(N - distance for distance in range(1, R_obs + 1))
    if include_contiguous_K3:
        count += 27 * (N - 2)
    return count



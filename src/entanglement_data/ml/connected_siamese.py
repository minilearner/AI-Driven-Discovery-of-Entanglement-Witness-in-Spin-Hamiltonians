from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np
import torch
from torch import nn

from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.local_unitary import feature_slices


Representation = Literal["T1K2K3", "T1T2T3K2K3"]


@dataclass(frozen=True)
class ConnectedSiameseSpec:
    """Canonical raw T1/T2/T3 measurements and a derived connected representation.

    The compact canonical input always contains literal T1, T2 (distances 1..R_obs)
    and contiguous T3.  K2 and the strict third cumulant K3 are recomputed from
    those raw tensors after every local rotation; they are not extra measurements.
    """

    N: int = 16
    R_obs: int = 4
    representation: Representation = "T1K2K3"

    def __post_init__(self) -> None:
        if self.N < 3:
            raise ValueError("N must be at least three")
        if not 2 <= self.R_obs < self.N:
            raise ValueError("R_obs must lie in [2,N-1] because K3 needs distance-two T2")
        if self.representation not in ("T1K2K3", "T1T2T3K2K3"):
            raise ValueError(f"unknown representation {self.representation}")

    @property
    def one_feature_count(self) -> int:
        return 3 * self.N

    @property
    def pair_support_count(self) -> int:
        return sum(self.N - distance for distance in range(1, self.R_obs + 1))

    @property
    def pair_feature_count(self) -> int:
        return 9 * self.pair_support_count

    @property
    def triple_feature_count(self) -> int:
        return 27 * (self.N - 2)

    @property
    def canonical_raw_feature_count(self) -> int:
        return self.one_feature_count + self.pair_feature_count + self.triple_feature_count

    @property
    def representation_feature_count(self) -> int:
        if self.representation == "T1K2K3":
            return self.one_feature_count + self.pair_feature_count + self.triple_feature_count
        return self.one_feature_count + 2 * self.pair_feature_count + 2 * self.triple_feature_count

    @property
    def physical_channels(self) -> int:
        return 3 + 9 * self.R_obs + 27 if self.representation == "T1K2K3" else 3 + 18 * self.R_obs + 54

    @property
    def mask_channels(self) -> int:
        # Availability depends only on support geometry, so raw and connected
        # families share one mask for every distance and one for triples.
        return self.R_obs + 1

    @property
    def spatial_channels(self) -> int:
        return self.physical_channels + self.mask_channels

    @property
    def network_tensor_entries(self) -> int:
        return self.N * self.spatial_channels

    @cached_property
    def pair_sites(self) -> tuple[np.ndarray, np.ndarray]:
        left, right = [], []
        for distance in range(1, self.R_obs + 1):
            left.extend(range(self.N - distance))
            right.extend(range(distance, self.N))
        return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)

    @cached_property
    def pair_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for distance in range(1, self.R_obs + 1):
            offsets.append(offsets[-1] + 9 * (self.N - distance))
        return tuple(offsets)


def _as_batch_numpy(values: np.ndarray, width: int) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    single = array.ndim == 1
    if single:
        array = array[None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"expected [{width}] or [B,{width}], got {array.shape}")
    return array, single


def extract_canonical_t1t2t3(
    values: np.ndarray,
    source_spec: FeatureSpec,
    target_spec: ConnectedSiameseSpec,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Extract literal T1, local T2, and contiguous T3 in canonical order."""

    if source_spec.N != target_spec.N:
        raise ValueError("source and target N differ")
    source, single = _as_batch_numpy(values, source_spec.num_features)
    slices = feature_slices(source_spec)
    one = source[:, slices.one]
    all_pairs = source[:, slices.two].reshape(len(source), -1, 3, 3)
    lookup = np.full((source_spec.N, source_spec.N), -1, dtype=np.int64)
    lookup[slices.pair_left, slices.pair_right] = np.arange(len(slices.pair_left))
    left, right = target_spec.pair_sites
    pairs = all_pairs[:, lookup[left, right]].reshape(len(source), -1)
    triples = source[:, slices.three]
    result = np.concatenate((one, pairs, triples), axis=1).astype(output_dtype, copy=False)
    if result.shape[1] != target_spec.canonical_raw_feature_count:
        raise RuntimeError("canonical T1/T2/T3 extraction dimension mismatch")
    return result[0] if single else result


def split_canonical(
    values: torch.Tensor, spec: ConnectedSiameseSpec
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != spec.canonical_raw_feature_count:
        raise ValueError(f"expected [B,{spec.canonical_raw_feature_count}], got {tuple(values.shape)}")
    one = values[:, : spec.one_feature_count].reshape(len(values), spec.N, 3)
    pairs = []
    base = spec.one_feature_count
    for distance in range(1, spec.R_obs + 1):
        start, stop = spec.pair_offsets[distance - 1 : distance + 1]
        pairs.append(values[:, base + start : base + stop].reshape(len(values), spec.N - distance, 3, 3))
    triple = values[:, base + spec.pair_feature_count :].reshape(len(values), spec.N - 2, 3, 3, 3)
    return one, pairs, triple


def connected_tensors(
    values: torch.Tensor, spec: ConnectedSiameseSpec
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """Return T1, T2, T3, K2 and the strict centered third cumulant K3."""

    one, pairs, triple = split_canonical(values, spec)
    k2 = []
    for distance, pair in enumerate(pairs, start=1):
        product = torch.einsum("nqa,nqb->nqab", one[:, : spec.N - distance], one[:, distance:])
        k2.append(pair - product)

    mi, mj, mk = one[:, :-2], one[:, 1:-1], one[:, 2:]
    tij, tjk, tik = pairs[0][:, :-1], pairs[0][:, 1:], pairs[1]
    k3 = (
        triple
        - torch.einsum("nqa,nqbc->nqabc", mi, tjk)
        - torch.einsum("nqb,nqac->nqabc", mj, tik)
        - torch.einsum("nqc,nqab->nqabc", mk, tij)
        + 2.0 * torch.einsum("nqa,nqb,nqc->nqabc", mi, mj, mk)
    )
    return one, pairs, triple, k2, k3


def rotate_canonical_raw(
    values: torch.Tensor, rotations: torch.Tensor, spec: ConnectedSiameseSpec
) -> torch.Tensor:
    """Rotate only literal raw tensors; connected tensors must be recomputed later."""

    if rotations.shape != (len(values), spec.N, 3, 3):
        raise ValueError(f"expected rotations [B,{spec.N},3,3], got {tuple(rotations.shape)}")
    one, pairs, triple = split_canonical(values, spec)
    rotated_one = torch.einsum("nqai,nqi->nqa", rotations, one)
    rotated_pairs = []
    for distance, pair in enumerate(pairs, start=1):
        rotated_pairs.append(
            torch.einsum(
                "nqai,nqbj,nqij->nqab",
                rotations[:, : spec.N - distance], rotations[:, distance:], pair,
            ).reshape(len(values), -1)
        )
    rotated_triple = torch.einsum(
        "nqai,nqbj,nqck,nqijk->nqabc",
        rotations[:, :-2], rotations[:, 1:-1], rotations[:, 2:], triple,
    ).reshape(len(values), -1)
    return torch.cat((rotated_one.reshape(len(values), -1), *rotated_pairs, rotated_triple), dim=1)


def canonical_to_spatial(values: torch.Tensor, spec: ConnectedSiameseSpec) -> torch.Tensor:
    """Build a site-anchored [B,N,C] tensor with shared geometric masks."""

    one, pairs, triple, k2, k3 = connected_tensors(values, spec)
    output = values.new_zeros((len(values), spec.N, spec.spatial_channels))
    channel = 0
    output[:, :, channel : channel + 3] = one
    channel += 3
    families: list[tuple[list[torch.Tensor], torch.Tensor]]
    if spec.representation == "T1K2K3":
        families = [(k2, k3)]
    else:
        families = [(pairs, triple), (k2, k3)]
    for pair_family, triple_family in families:
        for distance, block in enumerate(pair_family, start=1):
            output[:, : spec.N - distance, channel : channel + 9] = block.reshape(len(values), spec.N - distance, 9)
            channel += 9
        output[:, : spec.N - 2, channel : channel + 27] = triple_family.reshape(len(values), spec.N - 2, 27)
        channel += 27
    if channel != spec.physical_channels:
        raise RuntimeError("connected spatial channel accounting mismatch")
    for distance in range(1, spec.R_obs + 1):
        output[:, : spec.N - distance, channel + distance - 1] = 1.0
    output[:, : spec.N - 2, channel + spec.R_obs] = 1.0
    return output


class ConnectedSpatialMLP(nn.Module):
    def __init__(self, spec: ConnectedSiameseSpec, hidden: int = 128):
        super().__init__()
        self.spec = spec
        self.network = nn.Sequential(
            nn.Linear(spec.network_tensor_entries, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(canonical_to_spatial(values, self.spec).reshape(len(values), -1))

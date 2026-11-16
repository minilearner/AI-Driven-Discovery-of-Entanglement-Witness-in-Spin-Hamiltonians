from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from torch import nn

from entanglement_data.observables.feature_spec import FeatureSpec
from entanglement_data.observables.local_unitary import feature_slices


@dataclass(frozen=True)
class RawT2T3Spec:
    """OBC-style raw two- and three-body input used by the Siamese stage."""

    N: int = 16
    R_obs: int = 4

    def __post_init__(self) -> None:
        if self.N < 3:
            raise ValueError("N must be at least three")
        if not 1 <= self.R_obs < self.N:
            raise ValueError("R_obs must lie in [1,N-1]")

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
    def physical_feature_count(self) -> int:
        return self.pair_feature_count + self.triple_feature_count

    @property
    def physical_channels(self) -> int:
        return 9 * self.R_obs + 27

    @property
    def mask_channels(self) -> int:
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
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"expected [{width}] or [B,{width}], got {array.shape}")
    return array, single


def extract_raw_t2t3(
    values: np.ndarray,
    source_spec: FeatureSpec,
    target_spec: RawT2T3Spec,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Extract the 864 literal raw correlators in distance-major support order."""

    if source_spec.N != target_spec.N:
        raise ValueError("source and target N differ")
    source, single = _as_batch_numpy(values, source_spec.num_features)
    slices = feature_slices(source_spec)
    pair = source[:, slices.two].reshape(len(source), -1, 3, 3)
    lookup = np.full((source_spec.N, source_spec.N), -1, dtype=np.int64)
    lookup[slices.pair_left, slices.pair_right] = np.arange(len(slices.pair_left))
    left, right = target_spec.pair_sites
    chosen_pair = pair[:, lookup[left, right]].reshape(len(source), -1)
    triple = source[:, slices.three].reshape(len(source), -1)
    result = np.concatenate((chosen_pair, triple), axis=1).astype(output_dtype, copy=False)
    if result.shape[1] != target_spec.physical_feature_count:
        raise RuntimeError("raw T2+T3 extraction dimension mismatch")
    return result[0] if single else result


def split_raw_t2t3(values: torch.Tensor, spec: RawT2T3Spec) -> tuple[list[torch.Tensor], torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != spec.physical_feature_count:
        raise ValueError(f"expected [B,{spec.physical_feature_count}], got {tuple(values.shape)}")
    pairs = []
    for distance in range(1, spec.R_obs + 1):
        start, stop = spec.pair_offsets[distance - 1 : distance + 1]
        pairs.append(values[:, start:stop].reshape(len(values), spec.N - distance, 3, 3))
    triple = values[:, spec.pair_feature_count :].reshape(len(values), spec.N - 2, 3, 3, 3)
    return pairs, triple


def raw_t2t3_to_spatial(values: torch.Tensor, spec: RawT2T3Spec) -> torch.Tensor:
    """Map literal raw correlators to [B,N,C], adding only padding and masks."""

    pairs, triple = split_raw_t2t3(values, spec)
    output = values.new_zeros((len(values), spec.N, spec.spatial_channels))
    for distance, block in enumerate(pairs, start=1):
        channel = 9 * (distance - 1)
        output[:, : spec.N - distance, channel : channel + 9] = block.reshape(len(values), spec.N - distance, 9)
        output[:, : spec.N - distance, spec.physical_channels + distance - 1] = 1.0
    output[:, : spec.N - 2, 9 * spec.R_obs : spec.physical_channels] = triple.reshape(
        len(values), spec.N - 2, 27
    )
    output[:, : spec.N - 2, -1] = 1.0
    return output


def quaternion_to_so3_torch(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[-1] != 4:
        raise ValueError("quaternions must end in dimension four")
    q = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(dim=-1)
    output = q.new_empty((*q.shape[:-1], 3, 3))
    output[..., 0, 0] = 1 - 2 * (y.square() + z.square())
    output[..., 0, 1] = 2 * (x * y - z * w)
    output[..., 0, 2] = 2 * (x * z + y * w)
    output[..., 1, 0] = 2 * (x * y + z * w)
    output[..., 1, 1] = 1 - 2 * (x.square() + z.square())
    output[..., 1, 2] = 2 * (y * z - x * w)
    output[..., 2, 0] = 2 * (x * z - y * w)
    output[..., 2, 1] = 2 * (y * z + x * w)
    output[..., 2, 2] = 1 - 2 * (x.square() + y.square())
    return output


def random_local_so3(
    batch_size: int,
    spec: RawT2T3Spec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    quaternions = torch.randn(
        batch_size, spec.N, 4, device=device, dtype=dtype, generator=generator
    )
    return quaternion_to_so3_torch(quaternions)


def rotate_raw_t2t3(values: torch.Tensor, rotations: torch.Tensor, spec: RawT2T3Spec) -> torch.Tensor:
    """Apply independent local SO(3) rotations directly to raw T2 and T3 tensors."""

    if rotations.shape != (len(values), spec.N, 3, 3):
        raise ValueError(f"expected rotations [B,{spec.N},3,3], got {tuple(rotations.shape)}")
    pairs, triple = split_raw_t2t3(values, spec)
    rotated_pairs = []
    for distance, block in enumerate(pairs, start=1):
        left = rotations[:, : spec.N - distance]
        right = rotations[:, distance:]
        rotated_pairs.append(
            torch.einsum("nqai,nqcj,nqij->nqac", left, right, block).reshape(len(values), -1)
        )
    starts = rotations[:, : spec.N - 2]
    middle = rotations[:, 1 : spec.N - 1]
    ends = rotations[:, 2:]
    rotated_triple = torch.einsum(
        "nqai,nqbj,nqck,nqijk->nqabc", starts, middle, ends, triple
    ).reshape(len(values), -1)
    result = torch.cat((*rotated_pairs, rotated_triple), dim=1)
    if result.shape[1] != spec.physical_feature_count:
        raise RuntimeError("rotated raw T2+T3 dimension mismatch")
    return result


class RawSpatialMLP(nn.Module):
    def __init__(self, spec: RawT2T3Spec, hidden: int = 128):
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
        spatial = raw_t2t3_to_spatial(values, self.spec)
        return self.network(spatial.reshape(len(values), -1))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

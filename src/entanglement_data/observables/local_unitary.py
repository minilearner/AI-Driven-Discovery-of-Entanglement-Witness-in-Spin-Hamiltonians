from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .feature_spec import FeatureSpec


def random_unit_quaternions(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=(*shape, 4))
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    # q and -q are the same SO(3) rotation; a fixed hemisphere makes hashes stable.
    values *= np.where(values[..., :1] < 0, -1.0, 1.0)
    return values


def quaternion_to_so3(quaternions: np.ndarray) -> np.ndarray:
    """Active SO(3) rotation for q=(w,x,y,z)."""
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternions must end in dimension 4")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    out = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def quaternion_to_su2(quaternions: np.ndarray) -> np.ndarray:
    """SU(2) matrix U=wI-i(xX+yY+zZ), consistent with quaternion_to_so3."""
    q = np.asarray(quaternions, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError("quaternions must end in dimension 4")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    out = np.empty((*q.shape[:-1], 2, 2), dtype=np.complex128)
    out[..., 0, 0] = w - 1j * z
    out[..., 0, 1] = -y - 1j * x
    out[..., 1, 0] = y - 1j * x
    out[..., 1, 1] = w + 1j * z
    return out


def random_local_rotations(rng: np.random.Generator, N: int, batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    shape = (N,) if batch_size is None else (batch_size, N)
    quaternions = random_unit_quaternions(rng, shape)
    return quaternion_to_so3(quaternions), quaternions


def rotation_hash(rotations: np.ndarray) -> str:
    value = np.asarray(rotations, dtype=np.float64)
    return hashlib.sha256(value.tobytes()).hexdigest()


@dataclass(frozen=True)
class FeatureSlices:
    one: slice
    two: slice
    three: slice
    four: slice
    pair_left: np.ndarray
    pair_right: np.ndarray


def feature_slices(spec: FeatureSpec) -> FeatureSlices:
    if not (spec.all_two_body and spec.contiguous_three_body and spec.contiguous_four_body):
        raise ValueError("local-SU(2) augmentation requires the full configured feature schema")
    one_stop = 3 * spec.N
    two_stop = one_stop + 9 * spec.N * (spec.N - 1) // 2
    three_stop = two_stop + 27 * (spec.N - 2)
    pair_left, pair_right = zip(*((i, j) for i in range(spec.N) for j in range(i + 1, spec.N)))
    return FeatureSlices(
        one=slice(0, one_stop),
        two=slice(one_stop, two_stop),
        three=slice(two_stop, three_stop),
        four=slice(three_stop, spec.num_features),
        pair_left=np.asarray(pair_left, dtype=np.int64),
        pair_right=np.asarray(pair_right, dtype=np.int64),
    )


def transform_features(
    features: np.ndarray,
    rotations: np.ndarray,
    spec: FeatureSpec,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Apply local SO(3) rotations to all correlator tensors without changing schema order."""
    values = np.asarray(features)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != spec.num_features:
        raise ValueError(f"features must have shape [B,{spec.num_features}] or [{spec.num_features}]")
    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim == 3:
        matrices = np.broadcast_to(matrices[None, :, :, :], (len(values), spec.N, 3, 3))
    if matrices.shape != (len(values), spec.N, 3, 3):
        raise ValueError(f"rotations must have shape [{spec.N},3,3] or [B,{spec.N},3,3]")
    orthogonality = np.einsum("siam,sicm->siac", matrices, matrices)
    if not np.allclose(orthogonality, np.eye(3), atol=1e-8) or not np.allclose(np.linalg.det(matrices), 1.0, atol=1e-8):
        raise ValueError("rotations must be proper orthogonal matrices")
    slices = feature_slices(spec)
    work = values.astype(np.float64, copy=False)
    output = np.empty_like(work)

    one = work[:, slices.one].reshape(len(work), spec.N, 3)
    output[:, slices.one] = np.einsum("siam,sim->sia", matrices, one, optimize=True).reshape(len(work), -1)

    two = work[:, slices.two].reshape(len(work), len(slices.pair_left), 3, 3)
    left = matrices[:, slices.pair_left]
    right = matrices[:, slices.pair_right]
    output[:, slices.two] = np.einsum("spam,spbn,spmn->spab", left, right, two, optimize=True).reshape(len(work), -1)

    three = work[:, slices.three].reshape(len(work), spec.N - 2, 3, 3, 3)
    starts3 = np.arange(spec.N - 2)
    output[:, slices.three] = np.einsum(
        "stam,stbn,stcp,stmnp->stabc",
        matrices[:, starts3], matrices[:, starts3 + 1], matrices[:, starts3 + 2], three, optimize=True,
    ).reshape(len(work), -1)

    four = work[:, slices.four].reshape(len(work), spec.N - 3, 3, 3, 3, 3)
    starts4 = np.arange(spec.N - 3)
    output[:, slices.four] = np.einsum(
        "stam,stbn,stcp,stdq,stmnpq->stabcd",
        matrices[:, starts4], matrices[:, starts4 + 1], matrices[:, starts4 + 2], matrices[:, starts4 + 3], four,
        optimize=True,
    ).reshape(len(work), -1)
    cast = output.astype(output_dtype, copy=False)
    return cast[0] if single else cast


def matched_pair_augment(
    features: np.ndarray,
    pair_ids: np.ndarray,
    *,
    spec: FeatureSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply exactly one shared local rotation to both rows of every matched pair."""
    values = np.asarray(features)
    ids = np.asarray(pair_ids)
    if values.ndim != 2 or len(values) != len(ids):
        raise ValueError("features and pair_ids must have matching row counts")
    unique, counts = np.unique(ids, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("each pair_id must occur exactly twice")
    rotations_by_pair = {}
    hashes_by_pair = {}
    for pair_id in unique:
        local_seed = int(hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
        rotation, _ = random_local_rotations(np.random.default_rng(local_seed), spec.N)
        rotations_by_pair[pair_id] = rotation
        hashes_by_pair[pair_id] = rotation_hash(rotation)
    rotations = np.stack([rotations_by_pair[pair_id] for pair_id in ids])
    hashes = np.asarray([hashes_by_pair[pair_id] for pair_id in ids])
    return transform_features(values, rotations, spec), hashes

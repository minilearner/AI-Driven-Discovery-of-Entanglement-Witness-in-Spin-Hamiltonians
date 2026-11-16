from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .feature_spec import FeatureSpec
from .local_unitary import feature_slices
from .skip3 import Skip3Spec, decode_skip3


@dataclass(frozen=True)
class CorrelatorTensors:
    one: np.ndarray
    two: np.ndarray
    three: np.ndarray
    four: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    single_input: bool


@dataclass(frozen=True)
class LUInvariantSpec:
    N: int = 16

    @property
    def one_body_dimension(self) -> int:
        return self.N

    @property
    def two_body_dimension(self) -> int:
        return 14 * self.N * (self.N - 1) // 2

    @property
    def three_body_dimension(self) -> int:
        return 20 * max(0, self.N - 2)

    @property
    def four_body_dimension(self) -> int:
        return 80 * max(0, self.N - 3)

    @property
    def num_features(self) -> int:
        return self.one_body_dimension + self.two_body_dimension + self.three_body_dimension + self.four_body_dimension

    @property
    def inv12_stop(self) -> int:
        return self.one_body_dimension + self.two_body_dimension

    @property
    def inv123_stop(self) -> int:
        return self.inv12_stop + self.three_body_dimension

    @cached_property
    def names(self) -> list[str]:
        names: list[str] = []
        for i in range(self.N):
            names.append(f"onebody/site_{i}/bloch_norm")
        tensor2_names = (
            "sv_1", "sv_2", "sv_3", "determinant", "bloch_contraction",
            "left_contraction_norm", "right_contraction_norm",
        )
        for i in range(self.N):
            for j in range(i + 1, self.N):
                for tensor_name in ("raw_T2", "connected_C2"):
                    names.extend(f"twobody/sites_{i}_{j}/{tensor_name}/{name}" for name in tensor2_names)
        tensor3_names = ["frobenius_norm"] + [f"mode_{mode}_sv_{rank}" for mode in range(3) for rank in range(1, 4)]
        for i in range(self.N - 2):
            for tensor_name in ("raw_T3", "connected_K3"):
                names.extend(f"threebody/sites_{i}_{i+1}_{i+2}/{tensor_name}/{name}" for name in tensor3_names)
        tensor4_names = ["frobenius_norm"]
        tensor4_names += [f"mode_{mode}_sv_{rank}" for mode in range(4) for rank in range(1, 4)]
        tensor4_names += [
            f"bipartition_{partition}_sv_{rank}"
            for partition in ("01_23", "02_13", "03_12") for rank in range(1, 10)
        ]
        for i in range(self.N - 3):
            # R4_gt2 is an exact higher-order residual, not a centered or pure fourth cumulant.
            for tensor_name in ("raw_T4", "higher_order_four_body_residual_R4_gt2"):
                names.extend(f"fourbody/sites_{i}_{i+1}_{i+2}_{i+3}/{tensor_name}/{name}" for name in tensor4_names)
        if len(names) != self.num_features:
            raise RuntimeError(f"LU-invariant feature-name count {len(names)} != {self.num_features}")
        return names


def decode_correlators(features: np.ndarray, spec: FeatureSpec) -> CorrelatorTensors:
    values = np.asarray(features)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != spec.num_features:
        raise ValueError(f"features must have shape [{spec.num_features}] or [B,{spec.num_features}]")
    slices = feature_slices(spec)
    work = values.astype(np.float64, copy=False)
    return CorrelatorTensors(
        one=work[:, slices.one].reshape(len(work), spec.N, 3),
        two=work[:, slices.two].reshape(len(work), len(slices.pair_left), 3, 3),
        three=work[:, slices.three].reshape(len(work), spec.N - 2, 3, 3, 3),
        four=work[:, slices.four].reshape(len(work), spec.N - 3, 3, 3, 3, 3),
        pair_left=slices.pair_left,
        pair_right=slices.pair_right,
        single_input=single,
    )


def _pair_lookup(N: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lookup = np.full((N, N), -1, dtype=np.int64)
    lookup[left, right] = np.arange(len(left))
    lookup[right, left] = np.arange(len(left))
    return lookup


def connected_two_body(decoded: CorrelatorTensors) -> np.ndarray:
    left = decoded.one[:, decoded.pair_left]
    right = decoded.one[:, decoded.pair_right]
    return decoded.two - np.einsum("bpi,bpj->bpij", left, right)


def connected_third_cumulant(decoded: CorrelatorTensors) -> np.ndarray:
    N = decoded.one.shape[1]
    lookup = _pair_lookup(N, decoded.pair_left, decoded.pair_right)
    starts = np.arange(N - 2)
    m0, m1, m2 = decoded.one[:, starts], decoded.one[:, starts + 1], decoded.one[:, starts + 2]
    t01 = decoded.two[:, lookup[starts, starts + 1]]
    t02 = decoded.two[:, lookup[starts, starts + 2]]
    t12 = decoded.two[:, lookup[starts + 1, starts + 2]]
    return (
        decoded.three
        - np.einsum("sna,snbc->snabc", m0, t12)
        - np.einsum("snb,snac->snabc", m1, t02)
        - np.einsum("snc,snab->snabc", m2, t01)
        + 2.0 * np.einsum("sna,snb,snc->snabc", m0, m1, m2)
    )


def higher_order_four_body_residual(decoded: CorrelatorTensors) -> np.ndarray:
    """Exact R4_gt2 residual available from one-, two-, and four-body moments.

    This removes every four-body contribution constructible from first- and
    second-order cumulants.  It is LU covariant but is *not* the centered
    fourth moment, a pure fourth cumulant, or a complete connected tensor.
    """
    N = decoded.one.shape[1]
    lookup = _pair_lookup(N, decoded.pair_left, decoded.pair_right)
    starts = np.arange(N - 3)
    sites = [starts + offset for offset in range(4)]
    moments = [decoded.one[:, site] for site in sites]
    connected = connected_two_body(decoded)

    def c(left: int, right: int) -> np.ndarray:
        return connected[:, lookup[sites[left], sites[right]]]

    m0, m1, m2, m3 = moments
    c01, c02, c03 = c(0, 1), c(0, 2), c(0, 3)
    c12, c13, c23 = c(1, 2), c(1, 3), c(2, 3)
    residual = decoded.four.copy()
    residual -= np.einsum("sna,snb,snc,snd->snabcd", m0, m1, m2, m3)
    residual -= np.einsum("snab,snc,snd->snabcd", c01, m2, m3)
    residual -= np.einsum("snac,snb,snd->snabcd", c02, m1, m3)
    residual -= np.einsum("snad,snb,snc->snabcd", c03, m1, m2)
    residual -= np.einsum("snbc,sna,snd->snabcd", c12, m0, m3)
    residual -= np.einsum("snbd,sna,snc->snabcd", c13, m0, m2)
    residual -= np.einsum("sncd,sna,snb->snabcd", c23, m0, m1)
    residual -= np.einsum("snab,sncd->snabcd", c01, c23)
    residual -= np.einsum("snac,snbd->snabcd", c02, c13)
    residual -= np.einsum("snad,snbc->snabcd", c03, c12)
    return residual


def exact_fourth_cumulant_raw(
    decoded: CorrelatorTensors,
    skip3: np.ndarray,
    skip3_spec: Skip3Spec | None = None,
) -> np.ndarray:
    """Exact fourth-order connected cumulant from raw moments through order four.

    Unlike R4_gt2, this requires the two non-contiguous three-body moments in
    each contiguous four-site block.  ``skip3`` supplies exactly those 702
    auxiliary observables without modifying the historical 2559 schema.
    """
    N = decoded.one.shape[1]
    spec = skip3_spec or Skip3Spec(N)
    auxiliary = decode_skip3(skip3, spec)
    if auxiliary.ndim == 5:
        auxiliary = auxiliary[None, :]
    if len(auxiliary) != len(decoded.one):
        raise ValueError("legacy and skip3 batch sizes differ")
    lookup = _pair_lookup(N, decoded.pair_left, decoded.pair_right)
    starts = np.arange(N - 3)
    s0, s1, s2, s3 = (starts + offset for offset in range(4))
    m0, m1, m2, m3 = (decoded.one[:, site] for site in (s0, s1, s2, s3))
    t01 = decoded.two[:, lookup[s0, s1]]
    t02 = decoded.two[:, lookup[s0, s2]]
    t03 = decoded.two[:, lookup[s0, s3]]
    t12 = decoded.two[:, lookup[s1, s2]]
    t13 = decoded.two[:, lookup[s1, s3]]
    t23 = decoded.two[:, lookup[s2, s3]]
    t012 = decoded.three[:, starts]
    t123 = decoded.three[:, starts + 1]
    t013 = auxiliary[:, :, 0]
    t023 = auxiliary[:, :, 1]
    cumulant = decoded.four.copy()
    cumulant -= np.einsum("sna,snbcd->snabcd", m0, t123)
    cumulant -= np.einsum("snb,snacd->snabcd", m1, t023)
    cumulant -= np.einsum("snc,snabd->snabcd", m2, t013)
    cumulant -= np.einsum("snd,snabc->snabcd", m3, t012)
    cumulant -= np.einsum("snab,sncd->snabcd", t01, t23)
    cumulant -= np.einsum("snac,snbd->snabcd", t02, t13)
    cumulant -= np.einsum("snad,snbc->snabcd", t03, t12)
    cumulant += 2.0 * np.einsum("sna,snb,sncd->snabcd", m0, m1, t23)
    cumulant += 2.0 * np.einsum("sna,snc,snbd->snabcd", m0, m2, t13)
    cumulant += 2.0 * np.einsum("sna,snd,snbc->snabcd", m0, m3, t12)
    cumulant += 2.0 * np.einsum("snb,snc,snad->snabcd", m1, m2, t03)
    cumulant += 2.0 * np.einsum("snb,snd,snac->snabcd", m1, m3, t02)
    cumulant += 2.0 * np.einsum("snc,snd,snab->snabcd", m2, m3, t01)
    cumulant -= 6.0 * np.einsum("sna,snb,snc,snd->snabcd", m0, m1, m2, m3)
    return cumulant


def exact_fourth_cumulant_centered(
    decoded: CorrelatorTensors,
    skip3: np.ndarray,
    skip3_spec: Skip3Spec | None = None,
) -> np.ndarray:
    """Equivalent centered-moment construction of the exact K4 tensor."""
    N = decoded.one.shape[1]
    spec = skip3_spec or Skip3Spec(N)
    auxiliary = decode_skip3(skip3, spec)
    if auxiliary.ndim == 5:
        auxiliary = auxiliary[None, :]
    if len(auxiliary) != len(decoded.one):
        raise ValueError("legacy and skip3 batch sizes differ")
    lookup = _pair_lookup(N, decoded.pair_left, decoded.pair_right)
    starts = np.arange(N - 3)
    s0, s1, s2, s3 = (starts + offset for offset in range(4))
    m0, m1, m2, m3 = (decoded.one[:, site] for site in (s0, s1, s2, s3))
    t01 = decoded.two[:, lookup[s0, s1]]
    t02 = decoded.two[:, lookup[s0, s2]]
    t03 = decoded.two[:, lookup[s0, s3]]
    t12 = decoded.two[:, lookup[s1, s2]]
    t13 = decoded.two[:, lookup[s1, s3]]
    t23 = decoded.two[:, lookup[s2, s3]]
    t012 = decoded.three[:, starts]
    t123 = decoded.three[:, starts + 1]
    t013, t023 = auxiliary[:, :, 0], auxiliary[:, :, 1]
    centered = decoded.four.copy()
    centered -= np.einsum("sna,snbcd->snabcd", m0, t123)
    centered -= np.einsum("snb,snacd->snabcd", m1, t023)
    centered -= np.einsum("snc,snabd->snabcd", m2, t013)
    centered -= np.einsum("snd,snabc->snabcd", m3, t012)
    centered += np.einsum("sna,snb,sncd->snabcd", m0, m1, t23)
    centered += np.einsum("sna,snc,snbd->snabcd", m0, m2, t13)
    centered += np.einsum("sna,snd,snbc->snabcd", m0, m3, t12)
    centered += np.einsum("snb,snc,snad->snabcd", m1, m2, t03)
    centered += np.einsum("snb,snd,snac->snabcd", m1, m3, t02)
    centered += np.einsum("snc,snd,snab->snabcd", m2, m3, t01)
    centered -= 3.0 * np.einsum("sna,snb,snc,snd->snabcd", m0, m1, m2, m3)
    connected = connected_two_body(decoded)
    c01 = connected[:, lookup[s0, s1]]
    c02 = connected[:, lookup[s0, s2]]
    c03 = connected[:, lookup[s0, s3]]
    c12 = connected[:, lookup[s1, s2]]
    c13 = connected[:, lookup[s1, s3]]
    c23 = connected[:, lookup[s2, s3]]
    centered -= np.einsum("snab,sncd->snabcd", c01, c23)
    centered -= np.einsum("snac,snbd->snabcd", c02, c13)
    centered -= np.einsum("snad,snbc->snabcd", c03, c12)
    return centered


def exact_k4_invariants(
    features: np.ndarray,
    skip3: np.ndarray,
    spec: FeatureSpec,
    *,
    compact: bool = False,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return 520 full or 104 compact exact-K4 invariants."""
    decoded = decode_correlators(features, spec)
    tensor = exact_fourth_cumulant_raw(decoded, skip3, Skip3Spec(spec.N))
    full = _four_tensor_invariants(tensor)
    if compact:
        # Frobenius norm, largest singular value for each 1|3 and 2|2 unfolding.
        selected = np.asarray((0, 1, 4, 7, 10, 13, 22, 31), dtype=np.int64)
        full = full[..., selected]
    result = full.reshape(len(decoded.one), -1).astype(output_dtype, copy=False)
    return result[0] if decoded.single_input else result


def _two_tensor_invariants(tensor: np.ndarray, left_m: np.ndarray, right_m: np.ndarray) -> np.ndarray:
    singular = np.linalg.svd(tensor, compute_uv=False)
    determinant = np.linalg.det(tensor)[..., None]
    scalar = np.einsum("bpi,bpij,bpj->bp", left_m, tensor, right_m)[..., None]
    left_norm = np.linalg.norm(np.einsum("bpij,bpj->bpi", tensor, right_m), axis=-1)[..., None]
    right_norm = np.linalg.norm(np.einsum("bpij,bpi->bpj", tensor, left_m), axis=-1)[..., None]
    return np.concatenate((singular, determinant, scalar, left_norm, right_norm), axis=-1)


def _three_tensor_invariants(tensor: np.ndarray) -> np.ndarray:
    batch, blocks = tensor.shape[:2]
    values = [np.linalg.norm(tensor.reshape(batch, blocks, -1), axis=-1)[..., None]]
    for mode in range(3):
        unfolded = np.moveaxis(tensor, 2 + mode, 2).reshape(batch, blocks, 3, 9)
        values.append(np.linalg.svd(unfolded, compute_uv=False))
    return np.concatenate(values, axis=-1)


def compute_inv123_invariant_blocks(
    one: np.ndarray,
    raw_two: np.ndarray,
    connected_two: np.ndarray,
    raw_three: np.ndarray,
    connected_three: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the established Inv123 invariant blocks without changing them.

    This public block-level adapter is shared by the historical flattened
    feature path and by variable-length structured correlator inputs.  It does
    not define a new invariant family: the returned blocks are exactly the
    one-body norms, the 14 historical raw/connected two-body invariants, and
    the 20 historical raw/connected contiguous-three-body invariants.
    """
    one = np.asarray(one)
    raw_two = np.asarray(raw_two)
    connected_two = np.asarray(connected_two)
    raw_three = np.asarray(raw_three)
    connected_three = np.asarray(connected_three)
    pair_left = np.asarray(pair_left, dtype=np.int64)
    pair_right = np.asarray(pair_right, dtype=np.int64)
    if one.ndim != 3 or one.shape[-1] != 3:
        raise ValueError("one must have shape [B,N,3]")
    batch = len(one)
    if raw_two.shape != connected_two.shape or raw_two.shape != (batch, len(pair_left), 3, 3):
        raise ValueError("two-body blocks or pair supports have incompatible shapes")
    if raw_three.shape != connected_three.shape or raw_three.ndim != 5 or raw_three.shape[0] != batch or raw_three.shape[2:] != (3, 3, 3):
        raise ValueError("three-body blocks have incompatible shapes")
    if pair_left.shape != pair_right.shape or np.any(pair_left < 0) or np.any(pair_right >= one.shape[1]):
        raise ValueError("pair support indices are invalid")
    left_m = one[:, pair_left]
    right_m = one[:, pair_right]
    one_inv = np.linalg.norm(one, axis=-1)
    two_inv = (
        np.empty((batch, 0, 14), dtype=np.result_type(one, raw_two))
        if raw_two.shape[1] == 0
        else np.concatenate(
            (
                _two_tensor_invariants(raw_two, left_m, right_m),
                _two_tensor_invariants(connected_two, left_m, right_m),
            ),
            axis=-1,
        )
    )
    three_inv = (
        np.empty((batch, 0, 20), dtype=np.result_type(one, raw_three))
        if raw_three.shape[1] == 0
        else np.concatenate(
            (_three_tensor_invariants(raw_three), _three_tensor_invariants(connected_three)),
            axis=-1,
        )
    )
    return tuple(
        block.astype(output_dtype, copy=False) for block in (one_inv, two_inv, three_inv)
    )


def _four_tensor_invariants(tensor: np.ndarray) -> np.ndarray:
    batch, blocks = tensor.shape[:2]
    values = [np.linalg.norm(tensor.reshape(batch, blocks, -1), axis=-1)[..., None]]
    for mode in range(4):
        unfolded = np.moveaxis(tensor, 2 + mode, 2).reshape(batch, blocks, 3, 27)
        values.append(np.linalg.svd(unfolded, compute_uv=False))
    for permutation in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
        axes = (0, 1) + tuple(2 + axis for axis in permutation)
        unfolded = tensor.transpose(axes).reshape(batch, blocks, 9, 9)
        values.append(np.linalg.svd(unfolded, compute_uv=False))
    return np.concatenate(values, axis=-1)


def raw_t4_invariant_blocks(
    tensor: np.ndarray,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Return the 40 exact LU invariants of raw four-body tensors.

    ``tensor`` must end in four Cartesian Pauli axes and may contain any
    leading dimensions.  The returned order is Frobenius norm, the three
    singular values of each of the four 1|3 unfoldings, and the nine singular
    values of each of the 12|34, 13|24, and 14|23 unfoldings.  No centered or
    connected four-body quantity is constructed here.
    """
    values = np.asarray(tensor)
    if values.ndim < 4 or values.shape[-4:] != (3, 3, 3, 3):
        raise ValueError("raw T4 tensor must end in shape [3,3,3,3]")
    leading = values.shape[:-4]
    flattened = values.reshape(-1, 1, 3, 3, 3, 3)
    result = _four_tensor_invariants(flattened).reshape(*leading, 40)
    return result.astype(output_dtype, copy=False)


def compute_lu_invariant_features(
    features: np.ndarray,
    spec: FeatureSpec,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    decoded = decode_correlators(features, spec)
    invariant_spec = LUInvariantSpec(spec.N)
    connected2 = connected_two_body(decoded)
    left_m = decoded.one[:, decoded.pair_left]
    right_m = decoded.one[:, decoded.pair_right]
    one = np.linalg.norm(decoded.one, axis=-1)
    two = np.concatenate(
        (_two_tensor_invariants(decoded.two, left_m, right_m), _two_tensor_invariants(connected2, left_m, right_m)),
        axis=-1,
    ).reshape(len(decoded.one), -1)
    connected3 = connected_third_cumulant(decoded)
    three = np.concatenate(
        (_three_tensor_invariants(decoded.three), _three_tensor_invariants(connected3)), axis=-1
    ).reshape(len(decoded.one), -1)
    residual4 = higher_order_four_body_residual(decoded)
    four = np.concatenate(
        (_four_tensor_invariants(decoded.four), _four_tensor_invariants(residual4)), axis=-1
    ).reshape(len(decoded.one), -1)
    result = np.concatenate((one, two, three, four), axis=-1)
    if result.shape[1] != invariant_spec.num_features or len(invariant_spec.names) != invariant_spec.num_features:
        raise RuntimeError("LU-invariant feature dimension mismatch")
    result = result.astype(output_dtype, copy=False)
    return result[0] if decoded.single_input else result


def compute_inv123_features(
    features: np.ndarray,
    spec: FeatureSpec,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Compute exactly the Inv123 prefix without evaluating four-body blocks.

    The returned columns are identical to the first ``inv123_stop`` columns of
    :func:`compute_lu_invariant_features`.  This evaluation-only path avoids
    unnecessary four-body SVDs when frozen models consume only Inv123.
    """
    decoded = decode_correlators(features, spec)
    invariant_spec = LUInvariantSpec(spec.N)
    one, two, three = compute_inv123_invariant_blocks(
        decoded.one,
        decoded.two,
        connected_two_body(decoded),
        decoded.three,
        connected_third_cumulant(decoded),
        decoded.pair_left,
        decoded.pair_right,
        output_dtype=output_dtype,
    )
    result = np.concatenate(
        (one, two.reshape(len(decoded.one), -1), three.reshape(len(decoded.one), -1)), axis=-1
    )
    if result.shape[1] != invariant_spec.inv123_stop:
        raise RuntimeError("Inv123 feature dimension mismatch")
    result = result.astype(output_dtype, copy=False)
    return result[0] if decoded.single_input else result

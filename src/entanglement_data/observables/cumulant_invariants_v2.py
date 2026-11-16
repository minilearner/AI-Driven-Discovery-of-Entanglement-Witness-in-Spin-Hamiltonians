from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .feature_spec import FeatureSpec
from .lu_invariants import (
    _four_tensor_invariants,
    _pair_lookup,
    _three_tensor_invariants,
    connected_third_cumulant,
    connected_two_body,
    decode_correlators,
    exact_fourth_cumulant_raw,
)
from .skip3 import Skip3Spec


@dataclass(frozen=True)
class CumulantInvariantV2Spec:
    N: int = 16

    @property
    def pairs(self) -> int:
        return self.N * (self.N - 1) // 2

    @cached_property
    def dimensions(self) -> dict[str, int]:
        return {
            "K1": self.N,
            "K2_intrinsic": 4 * self.pairs,
            "K2_geometry": 6 * self.pairs,
            "K2_chirality": 2 * self.pairs,
            "K3_intrinsic": 10 * (self.N - 2),
            "K3_cross": 10 * (self.N - 2),
            "K4_compact_intrinsic": 8 * (self.N - 3),
            "K4_cross": 10 * (self.N - 3),
        }

    @cached_property
    def names(self) -> dict[str, list[str]]:
        names: dict[str, list[str]] = {key: [] for key in self.dimensions}
        names["K1"] = [f"K1/site_{i}/bloch_norm" for i in range(self.N)]
        for i in range(self.N):
            for j in range(i + 1, self.N):
                prefix = f"K2/sites_{i}_{j}"
                names["K2_intrinsic"].extend(f"{prefix}/{x}" for x in ("sv_1", "sv_2", "sv_3", "determinant"))
                names["K2_geometry"].extend(f"{prefix}/{x}" for x in (
                    "m_i_C_m_j", "m_i_CCt_m_i", "m_j_CtC_m_j", "m_i_CCt2_m_i", "m_j_CtC2_m_j", "m_i_C_CtC_m_j"
                ))
                names["K2_chirality"].extend(f"{prefix}/{x}" for x in ("left_chirality", "right_chirality"))
        k3_intrinsic = ["frobenius_norm"] + [f"mode_{mode}_sv_{rank}" for mode in range(3) for rank in range(1, 4)]
        k3_cross = (
            "K3_mmm", "contract_m_i_fro", "contract_m_j_fro", "contract_m_k_fro",
            "K3_Cij_mk", "K3_Cik_mj", "K3_Cjk_mi", "K3_Cij_vector_norm", "K3_Cik_vector_norm", "K3_Cjk_vector_norm",
        )
        k4_intrinsic = ("frobenius_norm", "mode_0_sv_1", "mode_1_sv_1", "mode_2_sv_1", "mode_3_sv_1", "partition_01_23_sv_1", "partition_02_13_sv_1", "partition_03_12_sv_1")
        k4_cross = (
            "K4_mmmm", "K4_C01_C23", "K4_C02_C13", "K4_C03_C12",
            "K4_C01_m2_m3", "K4_C02_m1_m3", "K4_C03_m1_m2", "K4_C12_m0_m3", "K4_C13_m0_m2", "K4_C23_m0_m1",
        )
        for i in range(self.N - 2):
            prefix = f"K3/sites_{i}_{i+1}_{i+2}"
            names["K3_intrinsic"].extend(f"{prefix}/{x}" for x in k3_intrinsic)
            names["K3_cross"].extend(f"{prefix}/{x}" for x in k3_cross)
        for i in range(self.N - 3):
            prefix = f"K4_exact/sites_{i}_{i+1}_{i+2}_{i+3}"
            names["K4_compact_intrinsic"].extend(f"{prefix}/{x}" for x in k4_intrinsic)
            names["K4_cross"].extend(f"{prefix}/{x}" for x in k4_cross)
        for key, expected in self.dimensions.items():
            if len(names[key]) != expected:
                raise RuntimeError(f"{key} name count {len(names[key])} != {expected}")
        return names


def compute_cumulant_invariant_blocks_v2(
    features: np.ndarray,
    skip3: np.ndarray,
    spec: FeatureSpec | None = None,
    *,
    output_dtype: np.dtype | type = np.float32,
) -> dict[str, np.ndarray]:
    spec = spec or FeatureSpec(16)
    values = np.asarray(features)
    single = values.ndim == 1
    decoded = decode_correlators(values, spec)
    batch = len(decoded.one)
    connected2 = connected_two_body(decoded)
    connected3 = connected_third_cumulant(decoded)
    lookup = _pair_lookup(spec.N, decoded.pair_left, decoded.pair_right)
    left_m = decoded.one[:, decoded.pair_left]
    right_m = decoded.one[:, decoded.pair_right]

    sv = np.linalg.svd(connected2, compute_uv=False)
    determinant = np.linalg.det(connected2)[..., None]
    K2_intrinsic = np.concatenate((sv, determinant), axis=-1).reshape(batch, -1)
    Cb = np.einsum("spab,spb->spa", connected2, right_m)
    Cta = np.einsum("spab,spa->spb", connected2, left_m)
    A_a = np.einsum("spab,spb->spa", connected2, Cta)
    B_b = np.einsum("spab,spa->spb", connected2, Cb)
    A2_a = np.einsum("spab,spb->spa", connected2, np.einsum("spab,spa->spb", connected2, A_a))
    B2_b = np.einsum("spab,spa->spb", connected2, np.einsum("spab,spb->spa", connected2, B_b))
    geometry = np.stack(
        (
            np.einsum("spa,spa->sp", left_m, Cb),
            np.einsum("spa,spa->sp", left_m, A_a),
            np.einsum("spb,spb->sp", right_m, B_b),
            np.einsum("spa,spa->sp", left_m, A2_a),
            np.einsum("spb,spb->sp", right_m, B2_b),
            np.einsum("spa,spab,spb->sp", left_m, connected2, B_b),
        ),
        axis=-1,
    ).reshape(batch, -1)
    chirality = np.stack(
        (
            np.einsum("spa,spa->sp", left_m, np.cross(A_a, A2_a)),
            np.einsum("spb,spb->sp", right_m, np.cross(B_b, B2_b)),
        ),
        axis=-1,
    ).reshape(batch, -1)

    K3_intrinsic = _three_tensor_invariants(connected3).reshape(batch, -1)
    starts3 = np.arange(spec.N - 2)
    m0, m1, m2 = (decoded.one[:, starts3 + offset] for offset in range(3))
    c01 = connected2[:, lookup[starts3, starts3 + 1]]
    c02 = connected2[:, lookup[starts3, starts3 + 2]]
    c12 = connected2[:, lookup[starts3 + 1, starts3 + 2]]
    K3_cross = np.stack(
        (
            np.einsum("snabc,sna,snb,snc->sn", connected3, m0, m1, m2),
            np.linalg.norm(np.einsum("snabc,sna->snbc", connected3, m0), axis=(-2, -1)),
            np.linalg.norm(np.einsum("snabc,snb->snac", connected3, m1), axis=(-2, -1)),
            np.linalg.norm(np.einsum("snabc,snc->snab", connected3, m2), axis=(-2, -1)),
            np.einsum("snabc,snab,snc->sn", connected3, c01, m2),
            np.einsum("snabc,snac,snb->sn", connected3, c02, m1),
            np.einsum("snabc,snbc,sna->sn", connected3, c12, m0),
            np.linalg.norm(np.einsum("snabc,snab->snc", connected3, c01), axis=-1),
            np.linalg.norm(np.einsum("snabc,snac->snb", connected3, c02), axis=-1),
            np.linalg.norm(np.einsum("snabc,snbc->sna", connected3, c12), axis=-1),
        ),
        axis=-1,
    ).reshape(batch, -1)

    K4 = exact_fourth_cumulant_raw(decoded, skip3, Skip3Spec(spec.N))
    full4 = _four_tensor_invariants(K4)
    selected = np.asarray((0, 1, 4, 7, 10, 13, 22, 31), dtype=np.int64)
    K4_intrinsic = full4[..., selected].reshape(batch, -1)
    starts4 = np.arange(spec.N - 3)
    q0, q1, q2, q3 = (decoded.one[:, starts4 + offset] for offset in range(4))
    q01 = connected2[:, lookup[starts4, starts4 + 1]]
    q02 = connected2[:, lookup[starts4, starts4 + 2]]
    q03 = connected2[:, lookup[starts4, starts4 + 3]]
    q12 = connected2[:, lookup[starts4 + 1, starts4 + 2]]
    q13 = connected2[:, lookup[starts4 + 1, starts4 + 3]]
    q23 = connected2[:, lookup[starts4 + 2, starts4 + 3]]
    K4_cross = np.stack(
        (
            np.einsum("snabcd,sna,snb,snc,snd->sn", K4, q0, q1, q2, q3),
            np.einsum("snabcd,snab,sncd->sn", K4, q01, q23),
            np.einsum("snabcd,snac,snbd->sn", K4, q02, q13),
            np.einsum("snabcd,snad,snbc->sn", K4, q03, q12),
            np.einsum("snabcd,snab,snc,snd->sn", K4, q01, q2, q3),
            np.einsum("snabcd,snac,snb,snd->sn", K4, q02, q1, q3),
            np.einsum("snabcd,snad,snb,snc->sn", K4, q03, q1, q2),
            np.einsum("snabcd,snbc,sna,snd->sn", K4, q12, q0, q3),
            np.einsum("snabcd,snbd,sna,snc->sn", K4, q13, q0, q2),
            np.einsum("snabcd,sncd,sna,snb->sn", K4, q23, q0, q1),
        ),
        axis=-1,
    ).reshape(batch, -1)
    blocks = {
        "K1": np.linalg.norm(decoded.one, axis=-1),
        "K2_intrinsic": K2_intrinsic,
        "K2_geometry": geometry,
        "K2_chirality": chirality,
        "K3_intrinsic": K3_intrinsic,
        "K3_cross": K3_cross,
        "K4_compact_intrinsic": K4_intrinsic,
        "K4_cross": K4_cross,
    }
    expected = CumulantInvariantV2Spec(spec.N).dimensions
    for key, value in blocks.items():
        if value.shape != (batch, expected[key]) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {key} block {value.shape}")
        blocks[key] = value.astype(output_dtype, copy=False)[0] if single else value.astype(output_dtype, copy=False)
    return blocks

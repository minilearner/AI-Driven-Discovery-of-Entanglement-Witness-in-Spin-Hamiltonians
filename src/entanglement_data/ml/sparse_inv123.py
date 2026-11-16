from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PhysicalFeatureGroup:
    """One translation-tied group of literal Inv123 scalar features."""

    key: str
    feature_family: str
    physical_support_type: str
    distance_r: int | None
    invariant_index: int
    invariant_name: str
    flat_indices: tuple[int, ...]

    @property
    def scalar_count(self) -> int:
        return len(self.flat_indices)


class Inv123PhysicalSchema:
    """Exact 1976-feature N=16 Inv123 physical schema.

    The schema contains 16 one-site norms, 120 pair supports times 14
    invariants, and 14 contiguous triples times 20 invariants.  Padding and
    mask entries used by the historical spatial tensor are deliberately absent.
    """

    pair_invariant_names = (
        "raw_T2/sv_1", "raw_T2/sv_2", "raw_T2/sv_3",
        "raw_T2/determinant", "raw_T2/bloch_contraction",
        "raw_T2/left_contraction_norm", "raw_T2/right_contraction_norm",
        "connected_C2/sv_1", "connected_C2/sv_2", "connected_C2/sv_3",
        "connected_C2/determinant", "connected_C2/bloch_contraction",
        "connected_C2/left_contraction_norm", "connected_C2/right_contraction_norm",
    )
    triple_invariant_names = (
        "raw_T3/frobenius_norm",
        "raw_T3/mode_0_sv_1", "raw_T3/mode_0_sv_2", "raw_T3/mode_0_sv_3",
        "raw_T3/mode_1_sv_1", "raw_T3/mode_1_sv_2", "raw_T3/mode_1_sv_3",
        "raw_T3/mode_2_sv_1", "raw_T3/mode_2_sv_2", "raw_T3/mode_2_sv_3",
        "connected_K3/frobenius_norm",
        "connected_K3/mode_0_sv_1", "connected_K3/mode_0_sv_2", "connected_K3/mode_0_sv_3",
        "connected_K3/mode_1_sv_1", "connected_K3/mode_1_sv_2", "connected_K3/mode_1_sv_3",
        "connected_K3/mode_2_sv_1", "connected_K3/mode_2_sv_2", "connected_K3/mode_2_sv_3",
    )

    def __init__(self, N: int = 16):
        if N < 3:
            raise ValueError("N must be at least 3")
        self.N = int(N)
        self.one_body_dimension = self.N
        self.pair_count = self.N * (self.N - 1) // 2
        self.pair_dimension = 14 * self.pair_count
        self.triple_dimension = 20 * (self.N - 2)
        self.flat_dimension = self.one_body_dimension + self.pair_dimension + self.triple_dimension
        left, right = np.triu_indices(self.N, 1)
        self.pair_left = left.astype(np.int64)
        self.pair_right = right.astype(np.int64)
        self.k1_indices = np.arange(self.N, dtype=np.int64)
        groups: list[PhysicalFeatureGroup] = []
        for distance in range(1, self.N):
            supports = np.flatnonzero(self.pair_right - self.pair_left == distance)
            for invariant_index, name in enumerate(self.pair_invariant_names):
                indices = tuple((self.N + 14 * supports + invariant_index).astype(int).tolist())
                groups.append(PhysicalFeatureGroup(
                    key=f"pair/r_{distance}/inv_{invariant_index}",
                    feature_family="pair", physical_support_type="pair",
                    distance_r=distance, invariant_index=invariant_index,
                    invariant_name=name, flat_indices=indices,
                ))
        triple_start = self.N + self.pair_dimension
        for invariant_index, name in enumerate(self.triple_invariant_names):
            indices = tuple((triple_start + 20 * np.arange(self.N - 2) + invariant_index).astype(int).tolist())
            groups.append(PhysicalFeatureGroup(
                key=f"triple/contiguous/inv_{invariant_index}",
                feature_family="triple", physical_support_type="contiguous_triple",
                distance_r=None, invariant_index=invariant_index,
                invariant_name=name, flat_indices=indices,
            ))
        self.groups = tuple(groups)
        self.group_by_key = {group.key: group for group in self.groups}
        if len(self.groups) != 15 * 14 + 20 or len(self.group_by_key) != len(self.groups):
            raise RuntimeError("structured Inv123 group construction failed")
        covered = list(self.k1_indices)
        for group in self.groups:
            covered.extend(group.flat_indices)
        if sorted(covered) != list(range(self.flat_dimension)):
            raise RuntimeError("structured groups do not exactly cover Inv123")

    def groups_within_distance(self, R_obs: int) -> tuple[PhysicalFeatureGroup, ...]:
        cutoff = int(R_obs)
        if not 1 <= cutoff <= self.N - 1:
            raise ValueError("R_obs must lie in [1,N-1]")
        return tuple(group for group in self.groups if group.feature_family == "triple" or group.distance_r <= cutoff)

    def indices_for_groups(self, keys: Iterable[str], *, include_k1: bool = True) -> np.ndarray:
        indices = list(self.k1_indices) if include_k1 else []
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            if key not in self.group_by_key:
                raise KeyError(key)
            indices.extend(self.group_by_key[key].flat_indices)
        return np.asarray(sorted(indices), dtype=np.int64)

    def indices_for_R_obs(self, R_obs: int) -> np.ndarray:
        return self.indices_for_groups(group.key for group in self.groups_within_distance(R_obs))

    def physical_dimension_for_R_obs(self, R_obs: int) -> int:
        return int(len(self.indices_for_R_obs(R_obs)))

    def group_costs(self) -> np.ndarray:
        return np.asarray([group.scalar_count for group in self.groups], dtype=np.float32)

    def feature_group_map(self) -> np.ndarray:
        """Map flat features to gate index; K1 entries use -1 and stay open."""
        result = np.full(self.flat_dimension, -1, dtype=np.int64)
        for gate_index, group in enumerate(self.groups):
            result[np.asarray(group.flat_indices, dtype=np.int64)] = gate_index
        if not np.all(result[self.N:] >= 0):
            raise RuntimeError("non-K1 feature lacks a structured gate")
        return result

    def historical_flat_feature_names(self) -> tuple[str, ...]:
        """Return names in the exact order used by the frozen Inv123 caches."""
        names = [f"onebody/site_{site}/bloch_norm" for site in range(self.N)]
        for left, right in zip(self.pair_left.tolist(), self.pair_right.tolist()):
            names.extend(
                f"twobody/sites_{left}_{right}/{name}"
                for name in self.pair_invariant_names
            )
        for left in range(self.N - 2):
            names.extend(
                f"threebody/sites_{left}_{left + 1}_{left + 2}/{name}"
                for name in self.triple_invariant_names
            )
        if len(names) != self.flat_dimension:
            raise RuntimeError("historical Inv123 feature-name construction failed")
        return tuple(names)

    def raw_pauli_observable_count(self, keys: Iterable[str], *, include_k1: bool = True) -> int:
        """Count the union of raw Pauli expectations needed by selected invariants.

        A selected invariant requires the complete tensor on every translated
        support because singular values/contractions are nonlinear functions of
        tensor components.  Connected K3 additionally requires its one- and
        two-body submoments.  Identity expectations are not counted.
        """
        selected = [self.group_by_key[key] for key in dict.fromkeys(keys)]
        one_sites = set(range(self.N)) if include_k1 else set()
        pair_supports: set[tuple[int, int]] = set()
        triple_supports: set[tuple[int, int, int]] = set()
        for group in selected:
            if group.feature_family == "pair":
                distance = int(group.distance_r)
                pair_supports.update((left, left + distance) for left in range(self.N - distance))
                if "bloch_contraction" in group.invariant_name or group.invariant_name.startswith("connected_C2"):
                    one_sites.update(range(self.N))
            else:
                triple_supports.update((left, left + 1, left + 2) for left in range(self.N - 2))
                if group.invariant_name.startswith("connected_K3"):
                    one_sites.update(range(self.N))
                    for left in range(self.N - 2):
                        pair_supports.update(((left, left + 1), (left + 1, left + 2), (left, left + 2)))
        return 3 * len(one_sites) + 9 * len(pair_supports) + 27 * len(triple_supports)


class CompactInv123MLP(nn.Module):
    """True compact MLP consuming only selected literal physical features."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        if input_dim < 1 or hidden < 1:
            raise ValueError("input_dim and hidden must be positive")
        self.input_dim = int(input_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"expected [B,{self.input_dim}], got {tuple(values.shape)}")
        return self.network(values)


class HardConcreteGates(nn.Module):
    """Louizos-style stretched binary-concrete gates."""

    def __init__(
        self, count: int, *, beta: float = 2.0 / 3.0,
        gamma: float = -0.1, zeta: float = 1.1, initial_nonzero: float = 0.9,
    ):
        super().__init__()
        if count < 1 or not (0.0 < initial_nonzero < 1.0):
            raise ValueError("invalid gate configuration")
        if not gamma < 0.0 < 1.0 < zeta:
            raise ValueError("hard-concrete stretch must satisfy gamma<0<1<zeta")
        self.beta, self.gamma, self.zeta = float(beta), float(gamma), float(zeta)
        offset = self.beta * np.log(-self.gamma / self.zeta)
        initial_log_alpha = np.log(initial_nonzero / (1.0 - initial_nonzero)) + offset
        self.log_alpha = nn.Parameter(torch.full((count,), float(initial_log_alpha)))

    def nonzero_probability(self) -> torch.Tensor:
        offset = self.beta * np.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - offset)

    def deterministic(self) -> torch.Tensor:
        stretched = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        return stretched.clamp(0.0, 1.0)

    def forward(self, *, stochastic: bool | None = None) -> torch.Tensor:
        use_stochastic = self.training if stochastic is None else bool(stochastic)
        if not use_stochastic:
            return self.deterministic()
        uniform = torch.rand_like(self.log_alpha).clamp_(1e-6, 1.0 - 1e-6)
        concrete = torch.sigmoid((torch.log(uniform) - torch.log1p(-uniform) + self.log_alpha) / self.beta)
        return (concrete * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)


class StructuredSparseInv123MLP(nn.Module):
    """Full physical input with translation-tied stochastic pair/K3 gates."""

    def __init__(self, schema: Inv123PhysicalSchema, *, hidden: int = 128, initial_nonzero: float = 0.9):
        super().__init__()
        self.schema = schema
        self.gates = HardConcreteGates(len(schema.groups), initial_nonzero=initial_nonzero)
        self.classifier = CompactInv123MLP(schema.flat_dimension, hidden=hidden)
        self.register_buffer("feature_group", torch.as_tensor(schema.feature_group_map(), dtype=torch.long))
        self.register_buffer("group_cost", torch.as_tensor(schema.group_costs()))

    def feature_gates(self, *, stochastic: bool | None = None) -> torch.Tensor:
        group_values = self.gates(stochastic=stochastic)
        ones = torch.ones(1, device=group_values.device, dtype=group_values.dtype)
        return torch.cat((ones, group_values))[self.feature_group + 1]

    def expected_physical_count(self, *, include_k1: bool = True) -> torch.Tensor:
        base = float(self.schema.N if include_k1 else 0)
        return self.gates.nonzero_probability().dot(self.group_cost) + base

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.schema.flat_dimension:
            raise ValueError(f"expected [B,{self.schema.flat_dimension}], got {tuple(values.shape)}")
        return self.classifier(values * self.feature_gates())


def select_group_keys_for_budget(
    schema: Inv123PhysicalSchema,
    scores: Sequence[float] | np.ndarray,
    target_physical_count: int,
    *, include_k1: bool = True,
) -> tuple[str, ...]:
    """Maximum-score structured subset at the closest attainable scalar budget."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(schema.groups),) or not np.isfinite(values).all():
        raise ValueError("one finite score is required for every structured group")
    fixed = schema.N if include_k1 else 0
    target = int(target_physical_count) - fixed
    if target < 0:
        raise ValueError("target is below the fixed K1 budget")
    maximum = max(target + schema.N, target)
    neg_inf = -np.inf
    dp = np.full(maximum + 1, neg_inf, dtype=np.float64)
    dp[0] = 0.0
    parents: list[np.ndarray] = []
    choices: list[np.ndarray] = []
    for group, score in zip(schema.groups, values):
        cost = group.scalar_count
        previous = dp.copy()
        parent = np.arange(maximum + 1, dtype=np.int32)
        chosen = np.zeros(maximum + 1, dtype=np.bool_)
        for total in range(cost, maximum + 1):
            candidate = previous[total - cost] + score
            if np.isfinite(candidate) and candidate > dp[total]:
                dp[total] = candidate
                parent[total] = total - cost
                chosen[total] = True
        parents.append(parent)
        choices.append(chosen)
    reachable = np.flatnonzero(np.isfinite(dp))
    if not len(reachable):
        raise RuntimeError("no structured budget is reachable")
    distance = np.abs(reachable - target)
    candidates = reachable[distance == distance.min()]
    total = int(candidates[np.argmax(dp[candidates])])
    selected: list[str] = []
    for index in range(len(schema.groups) - 1, -1, -1):
        if choices[index][total]:
            selected.append(schema.groups[index].key)
            total = int(parents[index][total])
    return tuple(reversed(selected))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

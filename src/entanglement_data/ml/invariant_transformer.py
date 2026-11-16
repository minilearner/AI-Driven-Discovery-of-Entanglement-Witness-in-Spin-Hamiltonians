from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from entanglement_data.ml.spatial_invariants import SpatialInvariantSpec
from entanglement_data.ml.correlator_transformer import TransformerBlock
from entanglement_data.ml.sparse_inv123 import Inv123PhysicalSchema
from entanglement_data.ml.sparse_inv123 import HardConcreteGates
from entanglement_data.observables.lu_invariants import compute_inv123_invariant_blocks
from entanglement_data.observables.lu_invariants import raw_t4_invariant_blocks


@dataclass(frozen=True)
class InvariantTransformerConfig:
    physical_channels: int
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    dim_feedforward: int = 192
    dropout: float = 0.1
    position_encoding: str = "rope"
    pooling: str = "cls"

    def __post_init__(self) -> None:
        if self.physical_channels < 1:
            raise ValueError("physical_channels must be positive")
        if self.layers != 4 or self.heads != 4 or self.d_model != 128 or self.dim_feedforward != 192:
            raise ValueError("the invariant experiment freezes the Raw RoPE Transformer architecture")
        if self.position_encoding != "rope" or self.pooling != "cls":
            raise ValueError("only RoPE with CLS pooling is allowed")


class InvariantTokenTransformer(nn.Module):
    """Raw-baseline-matched RoPE Transformer over exact invariant tokens."""

    def __init__(self, config: InvariantTransformerConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.physical_channels, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.d_model,
                config.heads,
                config.dim_feedforward,
                config.dropout,
                position_encoding=config.position_encoding,
            )
            for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 128), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1)
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def encode(
        self,
        tokens: torch.Tensor,
        site_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.config.physical_channels:
            raise ValueError(
                f"expected [B,N,{self.config.physical_channels}], got {tuple(tokens.shape)}"
            )
        if site_mask is None:
            site_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if site_mask.shape != tokens.shape[:2]:
            raise ValueError("site_mask shape does not match tokens")
        values = self.input_projection(tokens)
        valid = site_mask.bool()
        values = torch.cat((self.cls_token.expand(len(values), -1, -1), values), dim=1)
        valid = torch.cat(
            (torch.ones((len(values), 1), dtype=torch.bool, device=values.device), valid), dim=1
        )
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            values, attention = block(values, valid, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        return self.final_norm(values[:, 0]), attentions

    def forward(
        self,
        tokens: torch.Tensor,
        site_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        encoded, attentions = self.encode(tokens, site_mask, return_attention=return_attention)
        logits = self.classifier(encoded).squeeze(-1)
        return (logits, attentions) if return_attention else logits


class NormalizedInvariantTransformer(nn.Module):
    """Apply frozen translation-tied train statistics before the Transformer."""

    def __init__(
        self,
        model: InvariantTokenTransformer,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
    ):
        super().__init__()
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1)
        std_tensor = torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1)
        if mean_tensor.shape[-1] != model.config.physical_channels or std_tensor.shape != mean_tensor.shape:
            raise ValueError("normalization shape does not match model channels")
        if torch.any(std_tensor <= 0):
            raise ValueError("normalization standard deviations must be positive")
        self.model = model
        self.register_buffer("input_mean", mean_tensor)
        self.register_buffer("input_std", std_tensor)

    def forward(
        self,
        tokens: torch.Tensor,
        site_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(
            (tokens - self.input_mean) / self.input_std,
            site_mask,
            return_attention=return_attention,
        )


def translation_tied_gate_manifest(R_obs: int = 4) -> list[dict[str, Any]]:
    """One shared gate per physical invariant channel, excluding masks."""
    rows = []
    for channel in local_inv123_channel_manifest(R_obs):
        if channel["mask"]:
            continue
        if channel["family"] == "one_body":
            cost_n16, cost_n32 = 16, 32
            key = "one/inv_bloch_norm"
        elif channel["family"] == "pair":
            distance = int(channel["pair_distance"])
            cost_n16, cost_n32 = 16 - distance, 32 - distance
            key = f"pair/r_{distance}/inv_{channel['invariant_mathematical_name']}"
        else:
            cost_n16, cost_n32 = 14, 30
            key = f"triple/contiguous/inv_{channel['invariant_mathematical_name']}"
        rows.append({
            "group_index": len(rows), "key": key, "token_channel": channel["channel"],
            "family": channel["family"], "raw_or_connected": channel["raw_or_connected"],
            "pair_distance": channel["pair_distance"],
            "invariant_mathematical_name": channel["invariant_mathematical_name"],
            "svd_based": channel["svd_based"],
            "N16_scalar_occurrences": cost_n16, "N32_scalar_occurrences": cost_n32,
        })
    if len(rows) != 77 or [row["token_channel"] for row in rows] != list(range(77)):
        raise RuntimeError("translation-tied gate manifest diverged from local Inv123 tokens")
    return rows


def channel_gate_manifest(R_obs: int = 4) -> list[dict[str, Any]]:
    """Return one equal-cost gate per scalar physical token channel.

    This is the Transformer-native sparsity schema.  Unlike the historical
    occurrence-budget experiment, every shared scalar channel costs exactly
    one regardless of lattice support count or pair distance.  Boundary masks
    are deliberately absent because they are never gated.
    """
    rows = []
    for row in translation_tied_gate_manifest(R_obs):
        rows.append({
            **row,
            "channel_cost": 1,
            "historical_occurrence_cost_not_used": row["N16_scalar_occurrences"],
        })
    if [row["token_channel"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("channel gate manifest must follow scalar token-channel order")
    return rows


class TranslationTiedGatedInvariantTransformer(nn.Module):
    """Label-supervised hard-concrete gates shared across lattice anchors."""

    def __init__(
        self,
        model: InvariantTokenTransformer,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        *,
        R_obs: int = 4,
        initial_nonzero: float = 0.9,
    ):
        super().__init__()
        if model.config.physical_channels != len(local_inv123_channel_manifest(R_obs)):
            raise ValueError("gated model requires the complete local Inv123 token schema")
        self.model = model
        self.gate_manifest = translation_tied_gate_manifest(R_obs)
        self.gates = HardConcreteGates(len(self.gate_manifest), initial_nonzero=initial_nonzero)
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("input_std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer(
            "group_cost_n16",
            torch.as_tensor([row["N16_scalar_occurrences"] for row in self.gate_manifest], dtype=torch.float32),
        )

    def feature_gates(self, *, stochastic: bool | None = None) -> torch.Tensor:
        physical = self.gates(stochastic=stochastic)
        masks = torch.ones(5, dtype=physical.dtype, device=physical.device)
        return torch.cat((physical, masks))

    def expected_physical_count_n16(self) -> torch.Tensor:
        return self.gates.nonzero_probability().dot(self.group_cost_n16)

    def forward(self, tokens: torch.Tensor, site_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = (tokens - self.input_mean) / self.input_std
        return self.model(normalized * self.feature_gates(), site_mask)


class ChannelGatedInvariantTransformer(nn.Module):
    """Hard-concrete gates with unit cost per shared scalar token channel."""

    def __init__(
        self,
        model: InvariantTokenTransformer,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        *,
        R_obs: int = 4,
        initial_nonzero: float = 0.9,
    ):
        super().__init__()
        manifest = local_inv123_channel_manifest(R_obs)
        if model.config.physical_channels != len(manifest):
            raise ValueError("channel-gated model requires the complete local Inv123 schema")
        self.model = model
        self.gate_manifest = channel_gate_manifest(R_obs)
        self.gates = HardConcreteGates(len(self.gate_manifest), initial_nonzero=initial_nonzero)
        self.mask_channels = tuple(row["channel"] for row in manifest if row["mask"])
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("input_std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1))
        if self.input_mean.shape[-1] != len(manifest) or self.input_std.shape != self.input_mean.shape:
            raise ValueError("normalization shape does not match the complete local schema")

    def feature_gates(self, *, stochastic: bool | None = None) -> torch.Tensor:
        physical = self.gates(stochastic=stochastic)
        masks = torch.ones(len(self.mask_channels), dtype=physical.dtype, device=physical.device)
        return torch.cat((physical, masks))

    def expected_physical_channels(self) -> torch.Tensor:
        return self.gates.nonzero_probability().sum()

    def forward(self, tokens: torch.Tensor, site_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = (tokens - self.input_mean) / self.input_std
        return self.model(normalized * self.feature_gates(), site_mask)


class SelectiveChannelGatedInvariantTransformer(nn.Module):
    """Gate an explicit subset while keeping context channels always enabled."""

    def __init__(
        self,
        model: InvariantTokenTransformer,
        mean: np.ndarray | torch.Tensor,
        std: np.ndarray | torch.Tensor,
        gate_positions: tuple[int, ...] | list[int] | np.ndarray,
        *,
        initial_nonzero: float = 0.9,
    ):
        super().__init__()
        positions = tuple(int(value) for value in gate_positions)
        if not positions or len(positions) != len(set(positions)):
            raise ValueError("gate_positions must be non-empty and unique")
        if min(positions) < 0 or max(positions) >= model.config.physical_channels:
            raise ValueError("gate position lies outside the model input")
        self.model = model
        self.gates = HardConcreteGates(len(positions), initial_nonzero=initial_nonzero)
        self.register_buffer("gate_positions", torch.as_tensor(positions, dtype=torch.long))
        self.register_buffer("input_mean", torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, -1))
        self.register_buffer("input_std", torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, -1))
        if self.input_mean.shape[-1] != model.config.physical_channels or self.input_std.shape != self.input_mean.shape:
            raise ValueError("normalization shape does not match model input")

    def feature_gates(self, *, stochastic: bool | None = None) -> torch.Tensor:
        result = torch.ones(
            self.model.config.physical_channels,
            dtype=self.input_mean.dtype,
            device=self.input_mean.device,
        )
        result[self.gate_positions] = self.gates(stochastic=stochastic)
        return result

    def expected_gated_channels(self) -> torch.Tensor:
        return self.gates.nonzero_probability().sum()

    def forward(self, tokens: torch.Tensor, site_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = (tokens - self.input_mean) / self.input_std
        return self.model(normalized * self.feature_gates(), site_mask)


def rank_physical_token_channels(scores: np.ndarray, *, R_obs: int = 4) -> tuple[int, ...]:
    """Rank all physical channels by descending finite selector score."""
    manifest = channel_gate_manifest(R_obs)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(manifest),) or not np.isfinite(values).all():
        raise ValueError("one finite score is required per physical token channel")
    # Stable channel-index tie breaking makes rankings exactly reproducible.
    return tuple(np.lexsort((np.arange(len(values)), -values)).tolist())


def nested_channel_selections(
    ranking: tuple[int, ...] | list[int] | np.ndarray,
    budgets: tuple[int, ...] | list[int] | np.ndarray,
    *,
    R_obs: int = 4,
) -> dict[int, tuple[int, ...]]:
    """Build strictly nested top-K physical sets; masks are not included."""
    physical = len(channel_gate_manifest(R_obs))
    ordered = tuple(int(value) for value in ranking)
    if len(ordered) != physical or set(ordered) != set(range(physical)):
        raise ValueError("ranking must be a permutation of all physical channels")
    normalized = tuple(sorted(set(int(value) for value in budgets)))
    if not normalized or normalized[0] < 1 or normalized[-1] > physical:
        raise ValueError("budgets must lie within the physical channel count")
    return {budget: ordered[:budget] for budget in normalized}


def select_translation_tied_groups_for_budget(
    scores: np.ndarray,
    target_n16_occurrences: int = 200,
    *,
    R_obs: int = 4,
) -> tuple[int, ...]:
    """Maximum-score shared-group subset at the closest attainable N16 budget."""
    manifest = translation_tied_gate_manifest(R_obs)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(manifest),) or not np.isfinite(values).all():
        raise ValueError("one finite score is required per shared group")
    costs = np.asarray([row["N16_scalar_occurrences"] for row in manifest], dtype=np.int64)
    maximum = int(target_n16_occurrences + costs.max())
    dp = np.full(maximum + 1, -np.inf)
    dp[0] = 0.0
    parents: list[np.ndarray] = []
    choices: list[np.ndarray] = []
    for cost, score in zip(costs, values):
        previous = dp.copy()
        parent = np.arange(maximum + 1, dtype=np.int32)
        chosen = np.zeros(maximum + 1, dtype=bool)
        for total in range(int(cost), maximum + 1):
            candidate = previous[total - cost] + score
            if np.isfinite(candidate) and candidate > dp[total]:
                dp[total] = candidate
                parent[total] = total - cost
                chosen[total] = True
        parents.append(parent); choices.append(chosen)
    reachable = np.flatnonzero(np.isfinite(dp))
    distance = np.abs(reachable - int(target_n16_occurrences))
    candidates = reachable[distance == distance.min()]
    total = int(candidates[np.argmax(dp[candidates])])
    selected = []
    for index in range(len(manifest) - 1, -1, -1):
        if choices[index][total]:
            selected.append(index)
            total = int(parents[index][total])
    return tuple(reversed(selected))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def is_svd_invariant(name: str) -> bool:
    return "sv_" in name


def local_inv123_channel_manifest(R_obs: int = 4) -> list[dict[str, Any]]:
    """Generate the length-independent token manifest from the frozen schema."""
    if R_obs < 1:
        raise ValueError("R_obs must be positive")
    rows: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        rows.append({"channel": len(rows), **values})

    add(
        family="one_body", raw_or_connected="raw", pair_distance=None,
        invariant_mathematical_name="bloch_norm", svd_based=False, mask=False,
    )
    for distance in range(1, R_obs + 1):
        for name in Inv123PhysicalSchema.pair_invariant_names:
            add(
                family="pair", raw_or_connected="connected" if name.startswith("connected_C2") else "raw",
                pair_distance=distance, invariant_mathematical_name=name,
                svd_based=is_svd_invariant(name), mask=False,
            )
    for name in Inv123PhysicalSchema.triple_invariant_names:
        add(
            family="contiguous_three_body",
            raw_or_connected="connected" if name.startswith("connected_K3") else "raw",
            pair_distance=None, invariant_mathematical_name=name,
            svd_based=is_svd_invariant(name), mask=False,
        )
    for distance in range(1, R_obs + 1):
        add(
            family="boundary_mask", raw_or_connected="mask", pair_distance=distance,
            invariant_mathematical_name=f"pair_r_{distance}_valid", svd_based=False, mask=True,
        )
    add(
        family="boundary_mask", raw_or_connected="mask", pair_distance=None,
        invariant_mathematical_name="contiguous_three_body_valid", svd_based=False, mask=True,
    )
    return rows


def raw_t4_invariant_manifest() -> list[dict[str, Any]]:
    """Manifest for the deliberately restricted raw-T4 candidate family."""
    rows: list[dict[str, Any]] = []

    def add(
        *, family: str, partition: str, singular_value_index: int | None,
        svd_based: bool, mathematical_definition: str,
    ) -> None:
        rows.append({
            "channel_id": len(rows),
            "family": family,
            "partition": partition,
            "singular_value_index": singular_value_index,
            "SVD_based": svd_based,
            "mathematical_definition": mathematical_definition,
            "anchor_convention": "left anchor i on contiguous sites (i,i+1,i+2,i+3)",
            "raw_or_connected": "raw",
        })

    add(
        family="T4_Frobenius", partition="1234", singular_value_index=None,
        svd_based=False, mathematical_definition="||T4||_F",
    )
    for mode in range(4):
        partition = f"{mode + 1}|{''.join(str(index + 1) for index in range(4) if index != mode)}"
        for rank in range(1, 4):
            add(
                family="T4_1|3_SVD", partition=partition,
                singular_value_index=rank, svd_based=True,
                mathematical_definition=f"s_{rank}(T4 unfolding {partition})",
            )
    for partition in ("12|34", "13|24", "14|23"):
        for rank in range(1, 10):
            add(
                family="T4_2|2_SVD", partition=partition,
                singular_value_index=rank, svd_based=True,
                mathematical_definition=f"s_{rank}(T4 unfolding {partition})",
            )
    if len(rows) != 40:
        raise RuntimeError("raw T4 candidate manifest must contain 40 channels")
    return rows


def local_inv123_raw_t4_channel_manifest(R_obs: int = 4) -> list[dict[str, Any]]:
    """Combined 77 old physical, 40 raw-T4, and five mask channels."""
    base = local_inv123_channel_manifest(R_obs)
    physical = [{**row, "channel": index} for index, row in enumerate(row for row in base if not row["mask"])]
    offset = len(physical)
    t4 = []
    for index, row in enumerate(raw_t4_invariant_manifest()):
        t4.append({
            "channel": offset + index,
            "family": "contiguous_four_body",
            "raw_or_connected": "raw",
            "pair_distance": None,
            "invariant_mathematical_name": row["mathematical_definition"],
            "svd_based": row["SVD_based"],
            "mask": False,
            "t4_manifest": row,
        })
    masks = []
    for row in (row for row in base if row["mask"]):
        masks.append({**row, "channel": offset + len(t4) + len(masks)})
    result = physical + t4 + masks
    if len(result) != 122 or [row["channel"] for row in result] != list(range(122)):
        raise RuntimeError("Local Inv123 + raw-T4 manifest layout diverged")
    return result


def structured_raw_t4_local_tokens(
    raw_four: np.ndarray,
    *,
    N: int | None = None,
    four_mask: np.ndarray | None = None,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Left-anchor raw-T4 invariants and zero-pad invalid OBC anchors.

    No sixth geometric mask is added: the experimental protocol freezes the
    five historical masks.  Invalid T4 anchors are therefore represented only
    by zero-valued T4 channels.
    """
    values = np.asarray(raw_four)
    if values.ndim != 6 or values.shape[-4:] != (3, 3, 3, 3):
        raise ValueError("raw_four must have shape [B,A,3,3,3,3]")
    batch, anchors = values.shape[:2]
    lattice = anchors if N is None else int(N)
    if lattice < anchors:
        raise ValueError("N cannot be smaller than the supplied T4 anchor dimension")
    invariant = raw_t4_invariant_blocks(values, output_dtype=output_dtype)
    if four_mask is not None:
        mask = np.asarray(four_mask, dtype=bool)
        if mask.shape == (batch, lattice):
            mask = mask[:, :anchors]
        if mask.shape != (batch, anchors):
            raise ValueError("four_mask must have shape [B,A] or [B,N]")
        invariant = invariant * mask[..., None]
    output = np.zeros((batch, lattice, 40), dtype=output_dtype)
    output[:, :anchors] = invariant
    return output


def structured_inv123_raw_t4_local_tokens(
    one: np.ndarray,
    raw_two: np.ndarray,
    connected_two: np.ndarray,
    raw_three: np.ndarray,
    connected_three: np.ndarray,
    raw_four: np.ndarray,
    *,
    two_mask: np.ndarray | None = None,
    three_mask: np.ndarray | None = None,
    four_mask: np.ndarray | None = None,
    R_obs: int = 4,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Build ``[B,N,122]`` Local Inv123 plus raw-T4 tokens."""
    base = structured_inv123_local_tokens(
        one, raw_two, connected_two, raw_three, connected_three,
        two_mask=two_mask, three_mask=three_mask, R_obs=R_obs,
        output_dtype=output_dtype,
    )
    t4 = structured_raw_t4_local_tokens(
        raw_four, N=base.shape[1], four_mask=four_mask, output_dtype=output_dtype,
    )
    mask_count = sum(row["mask"] for row in local_inv123_channel_manifest(R_obs))
    physical_stop = base.shape[-1] - mask_count
    result = np.concatenate((base[:, :, :physical_stop], t4, base[:, :, physical_stop:]), axis=-1)
    if result.shape[-1] != len(local_inv123_raw_t4_channel_manifest(R_obs)):
        raise RuntimeError("Local Inv123 + raw-T4 token width mismatch")
    return result


def local_inv123_channel_indices(N: int, R_obs: int = 4) -> np.ndarray:
    """Select the existing spatial channels; never infer flattened feature indices."""
    return SpatialInvariantSpec("Inv123", N=N).local_channel_indices(R_obs)


def project_full_spatial_inv123(tokens: np.ndarray, R_obs: int = 4) -> np.ndarray:
    values = np.asarray(tokens)
    if values.ndim != 3:
        raise ValueError("tokens must have shape [B,N,C]")
    spec = SpatialInvariantSpec("Inv123", N=values.shape[1])
    if values.shape[2] != spec.local_channels:
        raise ValueError("full spatial Inv123 channel count is inconsistent")
    return values[:, :, local_inv123_channel_indices(spec.N, R_obs)]


def structured_inv123_local_tokens(
    one: np.ndarray,
    raw_two: np.ndarray,
    connected_two: np.ndarray,
    raw_three: np.ndarray,
    connected_three: np.ndarray,
    *,
    two_mask: np.ndarray | None = None,
    three_mask: np.ndarray | None = None,
    R_obs: int = 4,
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Build length-scalable tokens using the established Inv123 block builder.

    ``raw_two`` and ``connected_two`` use ``[B,N,R,3,3]`` with forward
    distances 1..R.  The OBC policy matches the raw Transformer: cyclic
    wrap-around supports are omitted rather than fabricated.
    """
    one = np.asarray(one)
    raw_two = np.asarray(raw_two)
    connected_two = np.asarray(connected_two)
    raw_three = np.asarray(raw_three)
    connected_three = np.asarray(connected_three)
    if one.ndim != 3:
        raise ValueError("one must have shape [B,N,3]")
    batch, N = one.shape[:2]
    if raw_two.shape != connected_two.shape or raw_two.shape[:3] != (batch, N, raw_two.shape[2]):
        raise ValueError("two-body structured arrays are incompatible")
    if raw_two.shape[2] < R_obs or N <= R_obs:
        raise ValueError("structured input does not contain the requested pair range")
    if raw_three.shape != connected_three.shape or raw_three.shape[:2] != (batch, N):
        raise ValueError("three-body structured arrays are incompatible")
    manifest = local_inv123_channel_manifest(R_obs)
    output = np.zeros((batch, N, len(manifest)), dtype=output_dtype)
    output[:, :, 0] = np.linalg.norm(one, axis=-1)
    channel = 1
    for distance in range(1, R_obs + 1):
        width = N - distance
        left = np.arange(width, dtype=np.int64)
        right = left + distance
        _, pair_inv, _ = compute_inv123_invariant_blocks(
            one,
            raw_two[:, :width, distance - 1],
            connected_two[:, :width, distance - 1],
            raw_three[:, :0],
            connected_three[:, :0],
            left,
            right,
            output_dtype=output_dtype,
        )
        valid = np.ones((batch, width), dtype=bool)
        if two_mask is not None:
            valid &= np.asarray(two_mask)[:, :width, distance - 1].astype(bool)
        output[:, :width, channel : channel + 14] = pair_inv * valid[..., None]
        channel += 14
    triple_width = N - 2
    empty_left = np.empty(0, dtype=np.int64)
    _, _, triple_inv = compute_inv123_invariant_blocks(
        one,
        raw_two[:, :0, 0],
        connected_two[:, :0, 0],
        raw_three[:, :triple_width],
        connected_three[:, :triple_width],
        empty_left,
        empty_left,
        output_dtype=output_dtype,
    )
    triple_valid = np.ones((batch, triple_width), dtype=bool)
    if three_mask is not None:
        triple_valid &= np.asarray(three_mask)[:, :triple_width].astype(bool)
    output[:, :triple_width, channel : channel + 20] = triple_inv * triple_valid[..., None]
    channel += 20
    for distance in range(1, R_obs + 1):
        width = N - distance
        valid = np.ones((batch, width), dtype=bool)
        if two_mask is not None:
            valid &= np.asarray(two_mask)[:, :width, distance - 1].astype(bool)
        output[:, :width, channel] = valid
        channel += 1
    output[:, :triple_width, channel] = triple_valid
    if channel + 1 != len(manifest):
        raise RuntimeError("local Inv123 manifest and token layout diverged")
    return output


def config_dict(config: InvariantTransformerConfig) -> dict[str, Any]:
    return asdict(config)

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from entanglement_data.ml.connected_siamese import ConnectedSiameseSpec, split_canonical


PositionEncoding = Literal["none", "sinusoidal", "rope"]
Pooling = Literal["cls", "mean"]


@dataclass(frozen=True)
class CorrelatorTokenSpec:
    """Site-anchored raw T1/T2/T3 token schema.

    The 66 physical channels are T1 (3), forward T2 at distances 1..4
    (4*9), and forward contiguous T3 (27).  Five separate availability
    masks distinguish geometric padding from a physical correlator equal to
    zero.  Supports crossing the right boundary are intentionally omitted,
    even when the source state used periodic boundary conditions.
    """

    N: int
    R_obs: int = 4

    def __post_init__(self) -> None:
        if self.N < 5:
            raise ValueError("N must be at least five")
        if self.R_obs != 4:
            raise ValueError("the frozen baseline schema requires R_obs=4")

    @property
    def physical_channels(self) -> int:
        return 3 + 9 * self.R_obs + 27

    @property
    def mask_channels(self) -> int:
        return self.R_obs + 1

    @property
    def compact_feature_count(self) -> int:
        return 3 * self.N + 9 * sum(self.N - r for r in range(1, self.R_obs + 1)) + 27 * (self.N - 2)


def canonical_to_tokens(
    values: torch.Tensor, spec: CorrelatorTokenSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert compact canonical T1/T2/T3 to physical tokens and masks."""

    connected_spec = ConnectedSiameseSpec(N=spec.N, R_obs=spec.R_obs)
    if values.ndim != 2 or values.shape[1] != connected_spec.canonical_raw_feature_count:
        raise ValueError(
            f"expected [B,{connected_spec.canonical_raw_feature_count}], got {tuple(values.shape)}"
        )
    one, pairs, triple = split_canonical(values, connected_spec)
    tokens = values.new_zeros((len(values), spec.N, spec.physical_channels))
    masks = values.new_zeros((len(values), spec.N, spec.mask_channels))
    tokens[:, :, :3] = one
    channel = 3
    for distance, block in enumerate(pairs, start=1):
        width = spec.N - distance
        tokens[:, :width, channel : channel + 9] = block.reshape(len(values), width, 9)
        masks[:, :width, distance - 1] = 1.0
        channel += 9
    tokens[:, : spec.N - 2, channel:] = triple.reshape(len(values), spec.N - 2, 27)
    masks[:, : spec.N - 2, -1] = 1.0
    site_mask = torch.ones((len(values), spec.N), dtype=torch.bool, device=values.device)
    return tokens, masks, site_mask


def arrays_to_tokens(
    one: np.ndarray,
    pairs: np.ndarray,
    triple: np.ndarray,
    *,
    R_obs: int = 4,
    output_dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert already structured arrays to the same OBC-style token schema.

    Input pair/triple arrays may include cyclic PBC anchors.  This function
    deliberately applies the frozen forward-support boundary rule and ignores
    those wrap-around entries so N=16 and N=32 have identical semantics.
    """

    one = np.asarray(one)
    pairs = np.asarray(pairs)
    triple = np.asarray(triple)
    if one.ndim != 3 or one.shape[-1] != 3:
        raise ValueError(f"expected T1 [B,N,3], got {one.shape}")
    batch, sites, _ = one.shape
    if pairs.shape != (batch, sites, R_obs, 3, 3):
        raise ValueError(f"expected T2 [B,N,{R_obs},3,3], got {pairs.shape}")
    if triple.shape != (batch, sites, 3, 3, 3):
        raise ValueError(f"expected T3 [B,N,3,3,3], got {triple.shape}")
    spec = CorrelatorTokenSpec(sites, R_obs)
    tokens = np.zeros((batch, sites, spec.physical_channels), dtype=output_dtype)
    masks = np.zeros((batch, sites, spec.mask_channels), dtype=output_dtype)
    tokens[:, :, :3] = one
    channel = 3
    for distance in range(1, R_obs + 1):
        width = sites - distance
        tokens[:, :width, channel : channel + 9] = pairs[:, :width, distance - 1].reshape(batch, width, 9)
        masks[:, :width, distance - 1] = 1.0
        channel += 9
    tokens[:, : sites - 2, channel:] = triple[:, : sites - 2].reshape(batch, sites - 2, 27)
    masks[:, : sites - 2, -1] = 1.0
    site_mask = np.ones((batch, sites), dtype=bool)
    return tokens, masks, site_mask


def rotate_correlator_tokens(
    tokens: torch.Tensor,
    rotations: torch.Tensor,
    spec: CorrelatorTokenSpec,
) -> torch.Tensor:
    """Apply site-local SO(3) rotations to the frozen raw token channels.

    Availability masks and lattice positions are intentionally not arguments:
    local spin-basis rotations change only physical T1/T2/T3 components.
    """

    expected = (len(tokens), spec.N, spec.physical_channels)
    if tokens.shape != expected:
        raise ValueError(f"expected tokens {expected}, got {tuple(tokens.shape)}")
    if rotations.shape != (len(tokens), spec.N, 3, 3):
        raise ValueError(f"expected rotations [B,{spec.N},3,3], got {tuple(rotations.shape)}")
    output = torch.zeros_like(tokens)
    one = tokens[:, :, :3]
    output[:, :, :3] = torch.einsum("nqai,nqi->nqa", rotations, one)
    channel = 3
    for distance in range(1, spec.R_obs + 1):
        width = spec.N - distance
        pair = tokens[:, :width, channel : channel + 9].reshape(len(tokens), width, 3, 3)
        rotated_pair = torch.einsum(
            "nqai,nqbj,nqij->nqab",
            rotations[:, :width], rotations[:, distance:], pair,
        )
        output[:, :width, channel : channel + 9] = rotated_pair.reshape(len(tokens), width, 9)
        channel += 9
    triple = tokens[:, : spec.N - 2, channel:].reshape(len(tokens), spec.N - 2, 3, 3, 3)
    rotated_triple = torch.einsum(
        "nqai,nqbj,nqck,nqijk->nqabc",
        rotations[:, :-2], rotations[:, 1:-1], rotations[:, 2:], triple,
    )
    output[:, : spec.N - 2, channel:] = rotated_triple.reshape(len(tokens), spec.N - 2, 27)
    return output


class CorrelatorSequenceDataset(Dataset):
    """Small adapter for fixed- or mixed-length site-token sequences."""

    def __init__(
        self,
        tokens: list[np.ndarray] | np.ndarray,
        feature_masks: list[np.ndarray] | np.ndarray,
        labels: np.ndarray,
    ):
        if len(tokens) != len(feature_masks) or len(tokens) != len(labels):
            raise ValueError("tokens, feature masks, and labels must have equal length")
        self.tokens = tokens
        self.feature_masks = feature_masks
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.as_tensor(np.asarray(self.tokens[index]), dtype=torch.float32)
        masks = torch.as_tensor(np.asarray(self.feature_masks[index]), dtype=torch.float32)
        if tokens.ndim != 2 or tokens.shape[1] != 66:
            raise ValueError(f"item {index} has invalid token shape {tuple(tokens.shape)}")
        if masks.shape != (tokens.shape[0], 5):
            raise ValueError(f"item {index} has invalid feature-mask shape {tuple(masks.shape)}")
        return tokens, masks, torch.tensor(self.labels[index], dtype=torch.float32)


def pad_correlator_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad mixed N batches and return tokens, feature masks, site mask, labels, lengths."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    lengths = torch.tensor([item[0].shape[0] for item in batch], dtype=torch.int64)
    max_length = int(lengths.max())
    tokens = torch.zeros((len(batch), max_length, 66), dtype=batch[0][0].dtype)
    masks = torch.zeros((len(batch), max_length, 5), dtype=batch[0][1].dtype)
    site_mask = torch.zeros((len(batch), max_length), dtype=torch.bool)
    labels = torch.stack([item[2] for item in batch])
    for index, (item_tokens, item_masks, _) in enumerate(batch):
        length = item_tokens.shape[0]
        tokens[index, :length] = item_tokens
        masks[index, :length] = item_masks
        site_mask[index, :length] = True
    return tokens, masks, site_mask, labels, lengths


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(values: torch.Tensor, positions: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to [B,H,L,D] query or key tensors."""

    if values.ndim != 4 or values.shape[-1] % 2:
        raise ValueError("RoPE expects [B,H,L,D] with even head dimension")
    head_dim = values.shape[-1]
    inv_freq = torch.pow(
        values.new_tensor(base),
        -torch.arange(0, head_dim, 2, device=values.device, dtype=values.dtype) / head_dim,
    )
    angles = positions.to(device=values.device, dtype=values.dtype)[:, None] * inv_freq[None, :]
    cos = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None, :, :]
    return values * cos + _rotate_half(values) * sin


def sinusoidal_encoding(length: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if width % 2:
        raise ValueError("sinusoidal width must be even")
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10_000.0) / width)
    )
    output = torch.zeros((length, width), device=device, dtype=dtype)
    output[:, 0::2] = torch.sin(positions * frequencies)
    output[:, 1::2] = torch.cos(positions * frequencies)
    return output


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, position_encoding: PositionEncoding):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if (d_model // heads) % 2 and position_encoding == "rope":
            raise ValueError("RoPE head dimension must be even")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.position_encoding = position_encoding
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self, values: torch.Tensor, valid_tokens: torch.Tensor, *, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, _ = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.position_encoding == "rope":
            positions = torch.arange(length, device=values.device)
            query = apply_rope(query, positions)
            key = apply_rope(key, positions)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid_tokens[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = attention.masked_fill(~valid_tokens[:, None, :, None], 0.0)
        mixed = torch.matmul(self.attention_dropout(attention), value)
        mixed = mixed.transpose(1, 2).reshape(batch, length, self.d_model)
        output = self.output_dropout(self.output(mixed))
        return output, attention if return_attention else None


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dim_feedforward: int,
        dropout: float,
        position_encoding: PositionEncoding,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = RoPESelfAttention(d_model, heads, dropout, position_encoding)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, values: torch.Tensor, valid_tokens: torch.Tensor, *, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attended, attention = self.attention(self.norm1(values), valid_tokens, return_attention=return_attention)
        values = values + attended
        values = values + self.ffn(self.norm2(values))
        values = values.masked_fill(~valid_tokens[:, :, None], 0.0)
        return values, attention


@dataclass(frozen=True)
class CorrelatorTransformerConfig:
    physical_channels: int = 66
    mask_channels: int = 5
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    dim_feedforward: int = 192
    dropout: float = 0.1
    position_encoding: PositionEncoding = "rope"
    pooling: Pooling = "cls"

    def __post_init__(self) -> None:
        if self.position_encoding not in ("none", "sinusoidal", "rope"):
            raise ValueError(f"unknown position encoding: {self.position_encoding}")
        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"unknown pooling: {self.pooling}")


class CorrelatorTransformer(nn.Module):
    """Variable-length entanglement classifier without chain flattening."""

    def __init__(self, config: CorrelatorTransformerConfig | None = None):
        super().__init__()
        self.config = config or CorrelatorTransformerConfig()
        cfg = self.config
        self.physical_embedding = nn.Linear(cfg.physical_channels, cfg.d_model)
        self.mask_embedding = nn.Linear(cfg.mask_channels, cfg.d_model, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.pooling == "cls" else None
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                cfg.d_model, cfg.heads, cfg.dim_feedforward, cfg.dropout, cfg.position_encoding
            )
            for _ in range(cfg.layers)
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Linear(cfg.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def encode(
        self,
        tokens: torch.Tensor,
        feature_mask: torch.Tensor,
        site_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the pooled sequence representation before classification.

        This is deliberately the same representation that the historical
        ``forward`` method fed to ``self.classifier``.  Exposing it makes
        latent LU-orbit pooling possible without changing the frozen
        backbone parameterization or checkpoint keys.
        """
        cfg = self.config
        if tokens.ndim != 3 or tokens.shape[-1] != cfg.physical_channels:
            raise ValueError(f"expected physical tokens [B,N,{cfg.physical_channels}], got {tuple(tokens.shape)}")
        if feature_mask.shape != (*tokens.shape[:2], cfg.mask_channels):
            raise ValueError("feature mask shape does not match tokens")
        if site_mask is None:
            site_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if site_mask.shape != tokens.shape[:2]:
            raise ValueError("site mask shape does not match tokens")
        values = self.physical_embedding(tokens) + self.mask_embedding(feature_mask)
        valid_tokens = site_mask.bool()
        if cfg.position_encoding == "sinusoidal":
            values = values + sinusoidal_encoding(
                values.shape[1], cfg.d_model, device=values.device, dtype=values.dtype
            )[None, :, :]
        if cfg.pooling == "cls":
            assert self.cls_token is not None
            values = torch.cat((self.cls_token.expand(len(values), -1, -1), values), dim=1)
            valid_tokens = torch.cat(
                (torch.ones((len(values), 1), dtype=torch.bool, device=values.device), valid_tokens), dim=1
            )
        values = self.embedding_dropout(values)
        attentions = []
        for block in self.blocks:
            values, attention = block(values, valid_tokens, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        values = self.final_norm(values)
        if cfg.pooling == "cls":
            pooled = values[:, 0]
        else:
            weights = valid_tokens.to(values.dtype)
            pooled = (values * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        if return_attention:
            return pooled, torch.stack(attentions, dim=0)
        return pooled

    def forward(
        self,
        tokens: torch.Tensor,
        feature_mask: torch.Tensor,
        site_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(
            tokens, feature_mask, site_mask, return_attention=return_attention
        )
        if return_attention:
            pooled, attentions = encoded
            return self.classifier(pooled).squeeze(-1), attentions
        return self.classifier(encoded).squeeze(-1)

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

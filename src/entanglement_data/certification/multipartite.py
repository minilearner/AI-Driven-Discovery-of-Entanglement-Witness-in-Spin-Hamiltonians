from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from ..states.mps import PurifiedMPS


def inequivalent_bipartitions(k: int) -> list[tuple[int, ...]]:
    """Return one representative of every A|A^c nontrivial cut."""
    if k < 2:
        raise ValueError("k must be at least two")
    cuts: list[tuple[int, ...]] = []
    full = (1 << k) - 1
    for mask in range(1, full):
        comp = full ^ mask
        if mask < comp:
            cuts.append(tuple(i for i in range(k) if mask & (1 << i)))
    return cuts


def partial_transpose(rho: torch.Tensor, k: int, subsystem: tuple[int, ...]) -> torch.Tensor:
    dim = 2**k
    if rho.shape != (dim, dim):
        raise ValueError(f"rho must have shape {(dim, dim)}")
    axes = list(range(2 * k))
    for i in subsystem:
        if not 0 <= i < k:
            raise ValueError("subsystem index out of range")
        axes[i], axes[k + i] = axes[k + i], axes[i]
    return rho.reshape(*(2 for _ in range(2 * k))).permute(*axes).reshape(dim, dim)


def ccnr_norm(rho: torch.Tensor, k: int, subsystem: tuple[int, ...]) -> float:
    """Trace norm of the standard realignment across subsystem|complement."""
    left = tuple(subsystem)
    right = tuple(i for i in range(k) if i not in left)
    if not left or not right:
        raise ValueError("CCNR requires a nontrivial bipartition")
    # rho axes are ket_0..ket_k-1, bra_0..bra_k-1.  Realignment groups
    # (ket_A, bra_A) as rows and (ket_B, bra_B) as columns.
    axes = list(left) + [k + i for i in left] + list(right) + [k + i for i in right]
    matrix = rho.reshape(*(2 for _ in range(2 * k))).permute(*axes)
    matrix = matrix.reshape(4 ** len(left), 4 ** len(right))
    return float(torch.linalg.svdvals(matrix).sum().real.item())


@dataclass
class BlockCertificate:
    block_sites: np.ndarray
    block_source: np.ndarray
    pt_min_eigenvalues: np.ndarray
    ccnr_norms: np.ndarray
    block_min_pt: np.ndarray
    block_max_ccnr: np.ndarray
    block_npt: np.ndarray
    block_ccnr: np.ndarray
    best_npt_cut: np.ndarray
    best_ccnr_cut: np.ndarray
    any_npt: bool
    any_ccnr: bool
    npt_block_count: int
    ccnr_block_count: int
    global_min_pt: float
    global_max_ccnr: float
    best_npt_block: int
    best_ccnr_block: int


def certify_blocks(
    state: PurifiedMPS,
    blocks: list[tuple[int, ...]],
    sources: list[str],
    *,
    tol_pt: float = 1e-9,
    tol_ccnr: float = 1e-8,
) -> BlockCertificate:
    if len(blocks) != len(sources) or not blocks:
        raise ValueError("blocks and sources must be non-empty and have equal length")
    k = len(blocks[0])
    if any(len(block) != k for block in blocks):
        raise ValueError("all blocks must have equal size")
    cuts = inequivalent_bipartitions(k)
    pt = np.empty((len(blocks), len(cuts)), dtype=np.float64)
    ccnr = np.empty_like(pt)
    for b, block in enumerate(blocks):
        rho = state.reduced_density_matrix(block)
        for c, cut in enumerate(cuts):
            pt[b, c] = float(torch.linalg.eigvalsh(partial_transpose(rho, k, cut)).min().real.item())
            ccnr[b, c] = ccnr_norm(rho, k, cut)
    block_min = pt.min(axis=1)
    block_max = ccnr.max(axis=1)
    block_npt = block_min < -tol_pt
    block_ccnr = block_max > 1.0 + tol_ccnr
    best_pt_block = int(np.argmin(block_min))
    best_ccnr_block = int(np.argmax(block_max))
    return BlockCertificate(
        block_sites=np.asarray(blocks, dtype=np.int8),
        block_source=np.asarray(sources, dtype=object),
        pt_min_eigenvalues=pt,
        ccnr_norms=ccnr,
        block_min_pt=block_min,
        block_max_ccnr=block_max,
        block_npt=block_npt,
        block_ccnr=block_ccnr,
        best_npt_cut=np.argmin(pt, axis=1).astype(np.int8),
        best_ccnr_cut=np.argmax(ccnr, axis=1).astype(np.int8),
        any_npt=bool(block_npt.any()),
        any_ccnr=bool(block_ccnr.any()),
        npt_block_count=int(block_npt.sum()),
        ccnr_block_count=int(block_ccnr.sum()),
        global_min_pt=float(block_min[best_pt_block]),
        global_max_ccnr=float(block_max[best_ccnr_block]),
        best_npt_block=best_pt_block,
        best_ccnr_block=best_ccnr_block,
    )


def circular_span(block: tuple[int, ...], N: int) -> int:
    points = sorted(block)
    gaps = [points[(i + 1) % len(points)] - points[i] for i in range(len(points) - 1)]
    gaps.append(points[0] + N - points[-1])
    return N - max(gaps)


def sample_certificate_blocks(N: int, k: int, K: int, seed: int) -> tuple[list[tuple[int, ...]], list[str]]:
    """All PBC-contiguous blocks plus span-stratified nonlocal subsets."""
    contiguous = [tuple((i + d) % N for d in range(k)) for i in range(N)]
    contiguous_sets = {tuple(sorted(x)) for x in contiguous}
    candidates = [x for x in combinations(range(N), k) if x not in contiguous_sets]
    spans = np.asarray([circular_span(x, N) for x in candidates])
    # Tertiles are defined over the available discrete span distribution.
    q1, q2 = np.quantile(spans, [1 / 3, 2 / 3])
    strata = {
        "random_short": [x for x, s in zip(candidates, spans) if s <= q1],
        "random_medium": [x for x, s in zip(candidates, spans) if q1 < s <= q2],
        "random_long": [x for x, s in zip(candidates, spans) if s > q2],
    }
    rng = np.random.default_rng(seed)
    quotas = [K // 3 + (1 if i < K % 3 else 0) for i in range(3)]
    chosen: list[tuple[int, ...]] = []
    sources: list[str] = []
    for quota, (name, pool) in zip(quotas, strata.items()):
        indices = rng.choice(len(pool), size=min(quota, len(pool)), replace=False)
        chosen.extend(pool[int(i)] for i in indices)
        sources.extend([name] * len(indices))
    if len(chosen) < K:
        remaining = [x for x in candidates if x not in set(chosen)]
        indices = rng.choice(len(remaining), size=K - len(chosen), replace=False)
        chosen.extend(remaining[int(i)] for i in indices)
        sources.extend(["random_medium"] * len(indices))
    return contiguous + chosen, ["contiguous"] * N + sources

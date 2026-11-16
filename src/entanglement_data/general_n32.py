from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product
from typing import Any, Sequence

import numpy as np

from .certification.pairwise_invisible import PAULI, ccnr_norm, inequivalent_cuts, partial_transpose
from .xy.size_transfer import LocalRawSpec, connected_from_raw, local_masks


PPT_TOLERANCE = 1e-6
CERTIFICATE_TOLERANCE = 1e-6
ROBUST_MARGIN = 1e-5


def _kron_all(operators: Sequence[np.ndarray]) -> np.ndarray:
    result = np.asarray([[1.0 + 0.0j]])
    for operator in operators:
        result = np.kron(result, operator)
    return result


def density_from_state(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.complex128)
    vector = vector / np.linalg.norm(vector)
    return np.outer(vector, vector.conj())


def partial_trace(rho: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    k = int(round(math.log2(len(rho))))
    keep = tuple(sorted(int(value) for value in keep))
    traced = tuple(value for value in range(k) if value not in keep)
    permutation = keep + traced + tuple(k + value for value in keep) + tuple(k + value for value in traced)
    kept_dimension, traced_dimension = 2 ** len(keep), 2 ** len(traced)
    tensor = rho.reshape((2,) * (2 * k)).transpose(permutation)
    tensor = tensor.reshape(kept_dimension, traced_dimension, kept_dimension, traced_dimension)
    return np.trace(tensor, axis1=1, axis2=3)


@lru_cache(maxsize=4)
def pauli_basis(k: int) -> np.ndarray:
    return np.stack([_kron_all([PAULI[axis] for axis in axes]) for axes in product(range(4), repeat=k)])


def pauli_coefficients(rho: np.ndarray) -> np.ndarray:
    k = int(round(math.log2(len(rho))))
    return np.einsum("aij,ji->a", pauli_basis(k), rho, optimize=True).real.reshape((4,) * k)


def pauli_imaginary_residual(rho: np.ndarray) -> float:
    k = int(round(math.log2(len(rho))))
    values = np.einsum("aij,ji->a", pauli_basis(k), rho, optimize=True)
    return float(np.max(np.abs(values.imag)))


def bloch_to_rho(bloch: np.ndarray) -> np.ndarray:
    return (PAULI[0] + sum(float(bloch[axis]) * PAULI[axis + 1] for axis in range(3))) / 2


class MomentState:
    N: int

    def moment(self, sites: Sequence[int], axes: Sequence[int]) -> float:
        raise NotImplementedError

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        raise NotImplementedError


class ProductMixtureState(MomentState):
    """Constructively fully separable mixture of product qubit states."""

    def __init__(self, bloch: np.ndarray, weights: np.ndarray):
        self.bloch = np.asarray(bloch, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.bloch.ndim != 3 or self.bloch.shape[2] != 3:
            raise ValueError("bloch must have shape [components,N,3]")
        if self.weights.shape != (len(self.bloch),) or np.any(self.weights < 0):
            raise ValueError("invalid mixture weights")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("mixture weights must sum to one")
        if np.max(np.linalg.norm(self.bloch, axis=2)) > 1.0 + 1e-10:
            raise ValueError("unphysical Bloch vector")
        self.N = self.bloch.shape[1]
        self.max_imaginary_residual = 0.0

    def moment(self, sites: Sequence[int], axes: Sequence[int]) -> float:
        factors = np.ones(len(self.bloch), dtype=np.float64)
        for site, axis in zip(sites, axes):
            factors *= self.bloch[:, int(site), int(axis)]
        return float(self.weights @ factors)

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        sites = tuple(int(site) for site in sites)
        rho = np.zeros((2 ** len(sites),) * 2, dtype=np.complex128)
        for weight, component in zip(self.weights, self.bloch):
            rho += weight * _kron_all([bloch_to_rho(component[site]) for site in sites])
        return (rho + rho.conj().T) / 2


class EmbeddedBlockState(MomentState):
    """At most four entangled qubits embedded in a product N-qubit background."""

    def __init__(self, N: int, block_sites: Sequence[int], block_rho: np.ndarray, background_bloch: np.ndarray):
        self.N = int(N)
        self.block_sites = tuple(int(site) for site in block_sites)
        self.block_rho = np.asarray(block_rho, dtype=np.complex128)
        self.background_bloch = np.asarray(background_bloch, dtype=np.float64)
        k = len(self.block_sites)
        if k not in (2, 3, 4) or self.block_sites != tuple(range(self.block_sites[0], self.block_sites[0] + k)):
            raise ValueError("block must be a contiguous support of size 2, 3, or 4")
        if self.block_rho.shape != (2**k, 2**k) or self.background_bloch.shape != (N, 3):
            raise ValueError("block/background shape mismatch")
        if abs(np.trace(self.block_rho) - 1.0) > 1e-9 or np.linalg.eigvalsh(self.block_rho).min() < -1e-9:
            raise ValueError("unphysical block density matrix")
        self._block_position = {site: position for position, site in enumerate(self.block_sites)}
        self._coefficients = pauli_coefficients(self.block_rho)
        self.max_imaginary_residual = pauli_imaginary_residual(self.block_rho)
        self._reduced_cache: dict[tuple[int, ...], np.ndarray] = {}

    def moment(self, sites: Sequence[int], axes: Sequence[int]) -> float:
        block_axes = [0] * len(self.block_sites)
        outside = 1.0
        for site, axis in zip(sites, axes):
            site, axis = int(site), int(axis)
            if site in self._block_position:
                block_axes[self._block_position[site]] = axis + 1
            else:
                outside *= self.background_bloch[site, axis]
        return float(outside * self._coefficients[tuple(block_axes)])

    def _block_reduced(self, positions: tuple[int, ...]) -> np.ndarray:
        if positions not in self._reduced_cache:
            self._reduced_cache[positions] = partial_trace(self.block_rho, positions)
        return self._reduced_cache[positions]

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        sites = tuple(sorted(int(site) for site in sites))
        selected_block = tuple(self._block_position[site] for site in sites if site in self._block_position)
        components: list[np.ndarray] = []
        inserted_block = False
        for site in sites:
            if site in self._block_position:
                if not inserted_block:
                    components.append(self._block_reduced(selected_block))
                    inserted_block = True
            else:
                components.append(bloch_to_rho(self.background_bloch[site]))
        return _kron_all(components)


def random_unit_vector(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(count, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def random_background(rng: np.random.Generator, N: int, max_radius: float = 0.82) -> np.ndarray:
    return random_unit_vector(rng, N) * rng.uniform(0.05, max_radius, size=(N, 1))


def haar_unitary(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 0, phases / np.abs(phases), 1.0)
    return q * phases.conj()[None]


def local_rotate(rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(round(math.log2(len(rho))))
    unitary = _kron_all([haar_unitary(rng, 2) for _ in range(k)])
    return unitary @ rho @ unitary.conj().T


def ghz_state(k: int) -> np.ndarray:
    vector = np.zeros(2**k, dtype=np.complex128)
    vector[0] = vector[-1] = 1 / math.sqrt(2)
    return vector


def star_graph_state() -> np.ndarray:
    vector = np.ones(16, dtype=np.complex128) / 4
    for value in range(16):
        bits = [(value >> (3 - index)) & 1 for index in range(4)]
        vector[value] *= (-1) ** (bits[0] * (bits[1] + bits[2] + bits[3]))
    return vector


def apply_two_qubit_gate(vector: np.ndarray, gate: np.ndarray, left: int, right: int, k: int) -> np.ndarray:
    tensor = vector.reshape((2,) * k)
    other = tuple(site for site in range(k) if site not in (left, right))
    order = (left, right) + other
    inverse = np.argsort(order)
    moved = tensor.transpose(order).reshape(4, -1)
    return (gate @ moved).reshape((2, 2) + (2,) * len(other)).transpose(inverse).reshape(-1)


def random_circuit_state(rng: np.random.Generator, k: int, depth: int) -> np.ndarray:
    vector = np.zeros(2**k, dtype=np.complex128)
    vector[0] = 1.0
    vector = _kron_all([haar_unitary(rng, 2)[:, 0] for _ in range(k)]).reshape(-1)
    for layer in range(depth):
        start = layer % 2
        for left in range(start, k - 1, 2):
            vector = apply_two_qubit_gate(vector, haar_unitary(rng, 4), left, left + 1, k)
    return vector / np.linalg.norm(vector)


def apply_local_channel(rho: np.ndarray, kraus: Sequence[np.ndarray]) -> np.ndarray:
    k = int(round(math.log2(len(rho))))
    result = rho.copy()
    for site in range(k):
        updated = np.zeros_like(result)
        for operator in kraus:
            full = _kron_all([operator if position == site else PAULI[0] for position in range(k)])
            updated += full @ result @ full.conj().T
        result = updated
    return (result + result.conj().T) / 2


def dephasing_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    project0 = np.asarray([[1, 0], [0, 0]], dtype=np.complex128)
    project1 = np.asarray([[0, 0], [0, 1]], dtype=np.complex128)
    kraus = [math.sqrt(1 - strength) * PAULI[0], math.sqrt(strength) * project0, math.sqrt(strength) * project1]
    return apply_local_channel(rho, kraus)


def amplitude_damping_channel(rho: np.ndarray, strength: float) -> np.ndarray:
    kraus = [
        np.asarray([[1, 0], [0, math.sqrt(1 - strength)]], dtype=np.complex128),
        np.asarray([[0, math.sqrt(strength)], [0, 0]], dtype=np.complex128),
    ]
    return apply_local_channel(rho, kraus)


def white_noise(rho: np.ndarray, strength: float) -> np.ndarray:
    return (1 - strength) * rho + strength * np.eye(len(rho), dtype=np.complex128) / len(rho)


def local_raw(state: MomentState, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    pair_mask, triple_mask, four_mask = local_masks(spec)
    t1 = np.empty((spec.N, 3), dtype=np.float64)
    for site in range(spec.N):
        for axis in range(3):
            t1[site, axis] = state.moment((site,), (axis,))
    t2 = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for site, distance0 in zip(*np.nonzero(pair_mask)):
        distance = distance0 + 1
        t2[site, distance0] = np.asarray(
            [state.moment((site, site + distance), axes) for axes in product(range(3), repeat=2)]
        ).reshape(3, 3)
    t3 = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    for site in np.flatnonzero(triple_mask):
        t3[site] = np.asarray(
            [state.moment((site, site + 1, site + 2), axes) for axes in product(range(3), repeat=3)]
        ).reshape(3, 3, 3)
    t4 = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    for site in np.flatnonzero(four_mask):
        t4[site] = np.asarray(
            [state.moment((site, site + 1, site + 2, site + 3), axes) for axes in product(range(3), repeat=4)]
        ).reshape(3, 3, 3, 3)
    k2, k3 = connected_from_raw(t1, t2, t3, spec)
    return {
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "T4": t4,
        "K2": k2,
        "K3": k3,
        "T2_mask": pair_mask,
        "T3_mask": triple_mask,
        "T4_mask": four_mask,
    }


@dataclass
class SupportCertificates:
    supports: np.ndarray
    min_pt: np.ndarray
    max_ccnr: np.ndarray
    best_pt_cut: np.ndarray
    best_ccnr_cut: np.ndarray

    @property
    def npt_margin(self) -> float:
        return float(max(0.0, -np.min(self.min_pt)))

    @property
    def ccnr_margin(self) -> float:
        return float(max(0.0, np.max(self.max_ccnr) - 1.0))

    @property
    def certified(self) -> bool:
        return self.npt_margin > CERTIFICATE_TOLERANCE or self.ccnr_margin > CERTIFICATE_TOLERANCE


def support_certificates(state: MomentState, supports: Sequence[tuple[int, ...]], *, ccnr: bool) -> SupportCertificates:
    k = len(supports[0])
    cuts = inequivalent_cuts(k)
    min_pt = np.empty(len(supports), dtype=np.float64)
    max_ccnr = np.full(len(supports), np.nan, dtype=np.float64)
    best_pt_cut = np.empty(len(supports), dtype=np.int8)
    best_ccnr_cut = np.full(len(supports), -1, dtype=np.int8)
    for index, support in enumerate(supports):
        rho = state.reduced_density_matrix(support)
        pt_values = np.asarray([np.linalg.eigvalsh(partial_transpose(rho, cut))[0] for cut in cuts])
        min_pt[index], best_pt_cut[index] = float(pt_values.min()), int(pt_values.argmin())
        if ccnr:
            ccnr_values = np.asarray([ccnr_norm(rho, cut) for cut in cuts])
            max_ccnr[index], best_ccnr_cut[index] = float(ccnr_values.max()), int(ccnr_values.argmax())
    return SupportCertificates(np.asarray(supports, dtype=np.int8), min_pt, max_ccnr, best_pt_cut, best_ccnr_cut)


def pair_supports(N: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(N), 2))


def contiguous_supports(N: int, k: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(start, start + k)) for start in range(N - k + 1))


def hierarchical_certificates(state: MomentState, *, strict_separable: bool = False) -> dict[str, Any]:
    pairs = support_certificates(state, pair_supports(state.N), ccnr=False)
    any_e2 = bool(np.min(pairs.min_pt) < -PPT_TOLERANCE)
    triples = None
    fours = None
    if strict_separable:
        triples = support_certificates(state, contiguous_supports(state.N, 3), ccnr=True)
        fours = support_certificates(state, contiguous_supports(state.N, 4), ccnr=True)
        class_name = "strict_separable"
    elif any_e2:
        class_name = "E2"
    else:
        triples = support_certificates(state, contiguous_supports(state.N, 3), ccnr=True)
        if triples.certified:
            class_name = "E3"
        else:
            fours = support_certificates(state, contiguous_supports(state.N, 4), ccnr=True)
            class_name = "E4" if fours.certified else "unknown"
    selected = pairs if class_name == "E2" else triples if class_name == "E3" else fours if class_name == "E4" else None
    certificate_type, support, cut, margin = "constructive_separable", (), (), math.nan
    if selected is not None:
        if selected.npt_margin >= selected.ccnr_margin:
            index = int(np.argmin(selected.min_pt))
            certificate_type = "NPT"
            cut_index = int(selected.best_pt_cut[index])
            margin = selected.npt_margin
        else:
            index = int(np.argmax(selected.max_ccnr))
            certificate_type = "CCNR"
            cut_index = int(selected.best_ccnr_cut[index])
            margin = selected.ccnr_margin
        support = tuple(int(value) for value in selected.supports[index])
        cut = inequivalent_cuts(len(support))[cut_index]
    return {
        "class_name": class_name,
        "pairs": pairs,
        "triples": triples,
        "fours": fours,
        "certificate_type": certificate_type,
        "certificate_support": support,
        "certificate_cut": cut,
        "certificate_margin": margin,
    }


def _block_certificate_values(rho: np.ndarray) -> tuple[float, float, float, float, float]:
    k = int(round(math.log2(len(rho))))
    pair_min = math.inf
    for support in combinations(range(k), 2):
        pair = partial_trace(rho, support)
        pair_min = min(pair_min, float(np.linalg.eigvalsh(partial_transpose(pair, (0,)))[0]))
    triple_npt, triple_ccnr = 0.0, 0.0
    if k >= 3:
        triple_supports = tuple(combinations(range(k), 3)) if k == 3 else ((0, 1, 2), (1, 2, 3))
        for support in triple_supports:
            reduced = partial_trace(rho, support)
            for cut in inequivalent_cuts(3):
                triple_npt = max(triple_npt, -float(np.linalg.eigvalsh(partial_transpose(reduced, cut))[0]))
                triple_ccnr = max(triple_ccnr, ccnr_norm(reduced, cut) - 1.0)
    four_npt, four_ccnr = 0.0, 0.0
    if k == 4:
        for cut in inequivalent_cuts(4):
            four_npt = max(four_npt, -float(np.linalg.eigvalsh(partial_transpose(rho, cut))[0]))
            four_ccnr = max(four_ccnr, ccnr_norm(rho, cut) - 1.0)
    return pair_min, triple_npt, triple_ccnr, four_npt, four_ccnr


def _random_block_location(rng: np.random.Generator, N: int, k: int) -> tuple[int, ...]:
    start = int(rng.integers(0, N - k + 1))
    return tuple(range(start, start + k))


def _embedded(rng: np.random.Generator, rho: np.ndarray, N: int) -> EmbeddedBlockState:
    k = int(round(math.log2(len(rho))))
    return EmbeddedBlockState(N, _random_block_location(rng, N, k), local_rotate(rho, rng), random_background(rng, N))


def _product_state(rng: np.random.Generator, N: int, family: str) -> tuple[ProductMixtureState, dict[str, Any]]:
    if family == "random_product":
        bloch = random_unit_vector(rng, N)[None]
        weights = np.ones(1)
        noise = 0.0
        components = 1
    else:
        components = int(rng.integers(6, 13))
        weights = rng.dirichlet(np.ones(components))
        axes = random_unit_vector(rng, N)
        signs = rng.choice((-1.0, 1.0), size=(components, N, 1))
        bloch = signs * axes[None]
        noise = 0.0
        if family == "noisy_separable_mixture":
            noise = float(rng.uniform(0.2, 0.75))
            bloch *= 1.0 - noise
    return ProductMixtureState(bloch, weights), {
        "noise_strength": noise,
        "circuit_depth": 0,
        "block_start": -1,
        "block_size": 0,
        "constructive_proof": f"convex_mixture_of_{components}_product_states",
    }


def _thermal_pair(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    xx, yy, zz = (_kron_all((PAULI[axis], PAULI[axis])) for axis in (1, 2, 3))
    for _ in range(100):
        coupling = float(rng.uniform(0.8, 1.3))
        field = float(rng.uniform(-0.25, 0.25))
        beta = float(rng.uniform(1.0, 4.0))
        hamiltonian = coupling * (xx + yy + zz) + field * (_kron_all((PAULI[3], PAULI[0])) + _kron_all((PAULI[0], PAULI[3])))
        values, vectors = np.linalg.eigh(hamiltonian)
        weights = np.exp(-beta * (values - values.min()))
        rho = (vectors * (weights / weights.sum())[None]) @ vectors.conj().T
        if _block_certificate_values(rho)[0] < -ROBUST_MARGIN:
            return rho, beta
    raise RuntimeError("failed to sample robust thermal E2 pair")


def _random_circuit_e4(rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
    for _ in range(200):
        depth = int(rng.integers(4, 9))
        parent = density_from_state(random_circuit_state(rng, 4, depth))
        candidates = []
        for noise in np.linspace(0.02, 0.92, 91):
            rho = white_noise(parent, float(noise))
            pair, npt3, ccnr3, npt4, ccnr4 = _block_certificate_values(rho)
            if pair >= PPT_TOLERANCE and max(npt3, ccnr3) <= CERTIFICATE_TOLERANCE and max(npt4, ccnr4) >= ROBUST_MARGIN:
                candidates.append((rho, float(noise)))
        if candidates:
            rho, noise = candidates[len(candidates) // 2]
            return rho, depth, noise
    raise RuntimeError("failed to find an E4 window for random circuit parent")


def generate_family_state(family: str, seed: int, N: int = 32) -> tuple[MomentState, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if family in {"random_product", "classical_correlated_separable_mixture", "noisy_separable_mixture"}:
        state, meta = _product_state(rng, N, family)
        return state, meta

    noise = 0.0
    depth = 0
    generator = family
    if family == "bell_pair_embedding":
        rho = density_from_state(np.asarray([1, 0, 0, 1], dtype=np.complex128) / math.sqrt(2))
    elif family == "finite_depth_random_circuit":
        depth = int(rng.integers(1, 4))
        for _ in range(100):
            rho = density_from_state(random_circuit_state(rng, 2, depth))
            if _block_certificate_values(rho)[0] < -ROBUST_MARGIN:
                break
        else:
            raise RuntimeError("failed to sample circuit E2 state")
    elif family == "thermal_entangled_pair":
        rho, beta = _thermal_pair(rng)
        generator = f"two_site_Heisenberg_Gibbs_beta_{beta:.6g}"
    elif family == "white_noise_path_E3":
        noise = float(rng.uniform(0.04, 0.42))
        rho = white_noise(density_from_state(ghz_state(3)), noise)
    elif family == "product_endpoint_path_E3":
        base_noise = float(rng.uniform(0.025, 0.08))
        endpoint_weight = float(rng.uniform(0.08, 0.55))
        parent = white_noise(density_from_state(ghz_state(3)), base_noise)
        endpoint = density_from_state(np.eye(8, dtype=np.complex128)[:, int(rng.choice((0, 7)))])
        rho = (1 - endpoint_weight) * parent + endpoint_weight * endpoint
        noise = endpoint_weight
    elif family == "diversity_repaired_E3":
        parent = white_noise(density_from_state(ghz_state(3)), float(rng.uniform(0.02, 0.08)))
        mechanism = str(rng.choice(("dephasing", "amplitude_damping", "white_noise")))
        noise = float(rng.uniform(0.05, 0.32))
        if mechanism == "dephasing":
            rho = dephasing_channel(parent, noise)
        elif mechanism == "amplitude_damping":
            rho = amplitude_damping_channel(parent, noise)
        else:
            rho = white_noise(parent, noise)
        generator = mechanism
    elif family == "cluster_like_star_graph_E4":
        noise = float(rng.uniform(0.04, 0.36))
        rho = white_noise(density_from_state(star_graph_state()), noise)
    elif family == "random_circuit_multipartite_E4":
        rho, depth, noise = _random_circuit_e4(rng)
    elif family == "dephased_multipartite_E4":
        base_noise = float(rng.uniform(0.025, 0.08))
        noise = float(rng.uniform(0.05, 0.42))
        rho = dephasing_channel(white_noise(density_from_state(ghz_state(4)), base_noise), noise)
    else:
        raise ValueError(f"unknown family {family}")
    state = _embedded(rng, rho, N)
    return state, {
        "noise_strength": noise,
        "circuit_depth": depth,
        "block_start": state.block_sites[0],
        "block_size": len(state.block_sites),
        "constructive_proof": "not_applicable",
        "generator_detail": generator,
    }


CLASS_FAMILIES: dict[str, tuple[tuple[str, int], ...]] = {
    "strict_separable": (
        ("random_product", 100),
        ("classical_correlated_separable_mixture", 150),
        ("noisy_separable_mixture", 150),
    ),
    "E2": (
        ("bell_pair_embedding", 150),
        ("finite_depth_random_circuit", 150),
        ("thermal_entangled_pair", 100),
    ),
    "E3": (
        ("white_noise_path_E3", 200),
        ("product_endpoint_path_E3", 200),
        ("diversity_repaired_E3", 200),
    ),
    "E4": (
        ("cluster_like_star_graph_E4", 200),
        ("random_circuit_multipartite_E4", 200),
        ("dephased_multipartite_E4", 200),
    ),
}


def production_manifest(master_seed: int = 20260820) -> list[dict[str, Any]]:
    rows = []
    seed_sequence = np.random.SeedSequence(master_seed)
    total = sum(count for families in CLASS_FAMILIES.values() for _, count in families)
    child_seeds = seed_sequence.spawn(total)
    cursor = 0
    for class_name, families in CLASS_FAMILIES.items():
        for family, count in families:
            for family_index in range(count):
                seed = int(child_seeds[cursor].generate_state(1, dtype=np.uint64)[0])
                rows.append({"class_name": class_name, "family": family, "family_index": family_index, "seed": seed})
                cursor += 1
    rng = np.random.default_rng(master_seed + 1)
    rng.shuffle(rows)
    for sample_id, row in enumerate(rows):
        row["sample_id"] = sample_id
    return rows

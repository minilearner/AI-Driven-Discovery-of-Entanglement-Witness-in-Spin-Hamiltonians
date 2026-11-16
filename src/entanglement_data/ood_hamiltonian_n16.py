from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .diagnostics.observable_radius import LocalRawSpec, pack_raw_blocks
from .general_n32 import (
    MomentState,
    ProductMixtureState,
    contiguous_supports,
    hierarchical_certificates,
    pair_supports,
    pauli_coefficients,
    support_certificates,
)
from .random_hamiltonian import (
    LocalHamiltonianTemplate,
    product_ground_state,
    sample_local_hamiltonian_template,
    total_bond_couplings,
)
from .states.mps import PurifiedMPS
from .xy.size_transfer import local_masks


AXIS_NAMES = ("Sx", "Sy", "Sz")


@dataclass(frozen=True)
class PauliTerm:
    """One real coefficient multiplying a product of project Pauli matrices."""

    sites: tuple[int, ...]
    axes: tuple[int, ...]
    coefficient: float

    def __post_init__(self) -> None:
        if len(self.sites) not in (1, 2) or len(self.sites) != len(self.axes):
            raise ValueError("only one- and two-site terms are supported")
        if tuple(sorted(self.sites)) != self.sites or len(set(self.sites)) != len(self.sites):
            raise ValueError("term sites must be distinct and increasing")
        if any(axis not in (0, 1, 2) for axis in self.axes):
            raise ValueError("Pauli axes must be 0=X, 1=Y, or 2=Z")


@dataclass(frozen=True)
class HamiltonianPoint:
    family: str
    N: int
    parameters: dict[str, Any]
    terms: tuple[PauliTerm, ...]
    point_seed: int
    separability_proof: str | None = None
    initial_product_vectors: tuple[tuple[complex, complex], ...] | None = None

    @property
    def natural_scale(self) -> float:
        if self.family == "xxz":
            return max(abs(float(self.parameters["J"])), 1.0)
        if self.family == "j1j2":
            return abs(float(self.parameters["J1"]))
        if self.family == "interleaved_xxz_r2":
            return abs(float(self.parameters["J2"]))
        return max(float(np.mean(self.parameters.get("fields", [1.0]))), 1.0)


def _site_bit(values: np.ndarray, site: int, N: int) -> np.ndarray:
    return (values >> (N - 1 - site)) & 1


def sparse_pauli_hamiltonian(point: HamiltonianPoint) -> sp.csr_matrix:
    """Construct the exact sparse matrix in site-0-most-significant basis order."""

    dimension = 1 << point.N
    basis = np.arange(dimension, dtype=np.int64)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for term in point.terms:
        target = basis.copy()
        phase = np.ones(dimension, dtype=np.complex128)
        for site, axis in zip(term.sites, term.axes):
            bit = _site_bit(basis, site, point.N)
            if axis == 0:  # X
                target ^= 1 << (point.N - 1 - site)
            elif axis == 1:  # Y|0>=i|1>, Y|1>=-i|0>
                target ^= 1 << (point.N - 1 - site)
                phase *= 1j * (1.0 - 2.0 * bit)
            else:  # Z
                phase *= 1.0 - 2.0 * bit
        rows.append(target)
        cols.append(basis)
        data.append(float(term.coefficient) * phase)
    matrix = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    antihermitian = matrix - matrix.getH()
    if antihermitian.nnz and np.max(np.abs(antihermitian.data)) > 2.0e-12:
        raise FloatingPointError("Hamiltonian construction produced a non-Hermitian matrix")
    return (matrix + matrix.getH()) * 0.5


def xxz_point(N: int, *, J: float, delta: float, h: float, seed: int) -> HamiltonianPoint:
    """XXZ in the S=sigma/2 convention stated in the benchmark protocol."""

    terms: list[PauliTerm] = []
    for site in range(N - 1):
        terms.extend(
            (
                PauliTerm((site, site + 1), (0, 0), J / 4.0),
                PauliTerm((site, site + 1), (1, 1), J / 4.0),
                PauliTerm((site, site + 1), (2, 2), J * delta / 4.0),
            )
        )
    for site in range(N):
        terms.append(PauliTerm((site,), (2,), -h / 2.0))
    # A bond-wise positive-semidefinite decomposition proves polarization when
    # |h| >= max(0, J*Delta + |J|); a small margin avoids a degenerate boundary.
    saturation_bound = max(0.0, J * delta + abs(J))
    proof = None
    if abs(J) < 1.0e-15 and abs(h) > 0:
        proof = "decoupled_spin_product_ground_state"
    elif abs(h) > saturation_bound + 1.0e-8:
        proof = "bondwise_psd_fully_polarized_ground_state"
    if proof:
        vectors = tuple(((1.0 + 0j, 0j) if h >= 0 else (0j, 1.0 + 0j)) for _ in range(N))
    elif J > 0:
        vectors = tuple(((1.0 + 0j, 0j) if site % 2 == 0 else (0j, 1.0 + 0j)) for site in range(N))
    else:
        vectors = tuple(((1.0 + 0j, 0j) if h >= 0 else (0j, 1.0 + 0j)) for _ in range(N))
    return HamiltonianPoint(
        "xxz", N, {"J": float(J), "Delta": float(delta), "h": float(h), "saturation_bound": saturation_bound},
        tuple(terms), int(seed), proof, vectors,
    )


def j1j2_point(N: int, *, J2: float, h: float, seed: int, J1: float = 1.0) -> HamiltonianPoint:
    terms: list[PauliTerm] = []
    for distance, coupling in ((1, J1), (2, J2)):
        for site in range(N - distance):
            for axis in range(3):
                terms.append(PauliTerm((site, site + distance), (axis, axis), coupling / 4.0))
    for site in range(N):
        terms.append(PauliTerm((site,), (2,), -h / 2.0))
    saturation_bound = 2.0 * max(J1, 0.0) + 2.0 * max(J2, 0.0)
    proof = "bondwise_psd_fully_polarized_ground_state" if abs(h) > saturation_bound + 1.0e-8 else None
    if proof:
        vectors = tuple(((1.0 + 0j, 0j) if h >= 0 else (0j, 1.0 + 0j)) for _ in range(N))
    else:
        vectors = tuple(((1.0 + 0j, 0j) if site % 2 == 0 else (0j, 1.0 + 0j)) for site in range(N))
    return HamiltonianPoint(
        "j1j2", N, {"J1": float(J1), "J2": float(J2), "h": float(h), "saturation_bound": saturation_bound},
        tuple(terms), int(seed), proof, vectors,
    )


def interleaved_xxz_r2_point(
    N: int,
    *,
    epsilon: float,
    delta1: float,
    delta2: float,
    h: float,
    seed: int,
    J1: float = 1.0,
    J2: float = 1.0,
) -> HamiltonianPoint:
    """Anisotropic r=2-dominant XXZ support family in the S=sigma/2 convention.

    This is deliberately distinct from the isotropic J1-J2 OOD benchmark:
    ``delta1`` and ``delta2`` are sampled away from a jointly fixed value of
    one, and the nearest-neighbour coupling is scaled by ``epsilon <= 0.30``.
    """

    if not 0.0 <= epsilon <= 0.30:
        raise ValueError("epsilon must lie in the frozen [0, 0.30] support range")
    if J1 < 0 or J2 <= 0 or delta1 < 0 or delta2 < 0:
        raise ValueError("the constructive saturation proof assumes antiferromagnetic non-negative couplings")
    terms: list[PauliTerm] = []
    for distance, coupling, delta in (
        (2, float(J2), float(delta2)),
        (1, float(epsilon * J1), float(delta1)),
    ):
        for site in range(N - distance):
            terms.extend(
                (
                    PauliTerm((site, site + distance), (0, 0), coupling / 4.0),
                    PauliTerm((site, site + distance), (1, 1), coupling / 4.0),
                    PauliTerm((site, site + distance), (2, 2), coupling * delta / 4.0),
                )
            )
    for site in range(N):
        terms.append(PauliTerm((site,), (2,), -float(h) / 2.0))
    # For each interaction range, J*(Delta+1) is the bulk field threshold
    # yielded by the same bond-wise PSD decomposition used by xxz_point.
    # Summing the r=1 and r=2 bounds is conservative at the OBC edges.
    saturation_bound = float(J2) * (float(delta2) + 1.0) + float(epsilon * J1) * (float(delta1) + 1.0)
    proof = "bondwise_psd_fully_polarized_ground_state" if abs(h) > saturation_bound + 1.0e-8 else None
    if proof:
        vectors = tuple(((1.0 + 0j, 0j) if h >= 0 else (0j, 1.0 + 0j)) for _ in range(N))
    else:
        # Stagger independently along the even and odd r=2 subchains.
        vectors = tuple(((1.0 + 0j, 0j) if site % 4 < 2 else (0j, 1.0 + 0j)) for site in range(N))
    return HamiltonianPoint(
        "interleaved_xxz_r2",
        N,
        {
            "J1": float(J1),
            "J2": float(J2),
            "epsilon": float(epsilon),
            "Delta1": float(delta1),
            "Delta2": float(delta2),
            "h": float(h),
            "saturation_bound": saturation_bound,
        },
        tuple(terms),
        int(seed),
        proof,
        vectors,
    )


def dh_point(N: int, *, seed: int, lambda_ent: float | None = None) -> HamiltonianPoint:
    template = sample_local_hamiltonian_template(N, seed=int(seed))
    value = template.lambda_initial if lambda_ent is None else float(lambda_ent)
    bonds = total_bond_couplings(template, value)
    terms: list[PauliTerm] = []
    for site in range(N):
        for axis in range(3):
            terms.append(PauliTerm((site,), (axis,), -template.fields[site] * template.local_axes[site, axis]))
    for site in range(N - 1):
        for left in range(3):
            for right in range(3):
                coefficient = float(bonds[site, left, right])
                if abs(coefficient) > 1.0e-15:
                    terms.append(PauliTerm((site, site + 1), (left, right), coefficient))
    product = product_ground_state(template)
    vectors = tuple(tuple(complex(x) for x in tensor[0, :, 0, 0].numpy()) for tensor in product.tensors)
    parameters = {
        "lambda_ent": value,
        "lambda_initial": template.lambda_initial,
        "lambda_bin": template.lambda_bin,
        "profile_type": template.profile_type,
        "template_seed": template.template_seed,
        "local_axes": template.local_axes.tolist(),
        "fields": template.fields.tolist(),
        "commuting_couplings": template.commuting_couplings.tolist(),
        "noncommuting_couplings": template.noncommuting_couplings.tolist(),
    }
    proof = "commuting_product_parent_ground_state" if abs(value) < 1.0e-15 else None
    return HamiltonianPoint("dh_family", N, parameters, tuple(terms), int(seed), proof, vectors)


def point_from_parameters(family: str, N: int, parameters: dict[str, Any], seed: int) -> HamiltonianPoint:
    if family == "xxz":
        return xxz_point(N, J=parameters["J"], delta=parameters["Delta"], h=parameters["h"], seed=seed)
    if family == "j1j2":
        return j1j2_point(N, J1=parameters.get("J1", 1.0), J2=parameters["J2"], h=parameters["h"], seed=seed)
    if family == "interleaved_xxz_r2":
        return interleaved_xxz_r2_point(
            N,
            J1=parameters.get("J1", 1.0),
            J2=parameters.get("J2", 1.0),
            epsilon=parameters["epsilon"],
            delta1=parameters["Delta1"],
            delta2=parameters["Delta2"],
            h=parameters["h"],
            seed=seed,
        )
    if family == "dh_family":
        return dh_point(N, seed=seed, lambda_ent=parameters.get("lambda_ent"))
    raise ValueError(f"unknown family {family}")


def _latin_hypercube(count: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = (np.arange(count)[:, None] + rng.random((count, dimension))) / count
    for column in range(dimension):
        rng.shuffle(values[:, column])
    return values


def sample_parameter_points(family: str, N: int, count: int, *, seed: int, thermal: bool = False) -> list[HamiltonianPoint]:
    """Structured anchors followed by deterministic Latin-hypercube coverage."""

    if count < 1:
        return []
    points: list[HamiltonianPoint] = []
    if family == "xxz":
        anchors = [
            (1.0, -2.0, 0.0), (1.0, -1.0, 0.15), (1.0, -0.5, 0.0),
            (1.0, 0.0, 0.5), (1.0, 0.5, 1.0), (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.5), (1.0, 1.5, 0.3), (1.0, 2.5, 0.0),
            (-1.0, 1.0, 0.2), (-1.0, 0.0, 0.7), (-1.0, 2.0, 0.4),
            (0.0, 1.0, 0.7), (0.0, -1.0, -1.3), (1.0, 1.0, 3.2),
        ]
        if thermal:
            anchors = [anchors[i] for i in (0, 3, 5, 6, 8, 9, 12)]
        for index, (J, delta, h) in enumerate(anchors[:count]):
            points.append(xxz_point(N, J=J, delta=delta, h=h, seed=seed + index))
        lhs = _latin_hypercube(max(0, count - len(points)), 3, seed + 701)
        for offset, row in enumerate(lhs, start=len(points)):
            J = -1.0 if row[0] < 0.32 else 1.0
            delta = -2.5 + 5.5 * row[1]
            h = -3.5 + 7.0 * row[2]
            points.append(xxz_point(N, J=J, delta=delta, h=h, seed=seed + offset))
    elif family == "j1j2":
        anchors = [(0.0, 0.0), (0.15, 0.3), (0.3, 0.0), (0.5, 0.0), (0.6, 0.7), (1.0, 0.0),
                   (1.5, 1.0), (-0.4, 0.0), (0.5, 3.2)]
        for index, (J2, h) in enumerate(anchors[:count]):
            points.append(j1j2_point(N, J2=J2, h=h, seed=seed + index))
        lhs = _latin_hypercube(max(0, count - len(points)), 2, seed + 809)
        for offset, row in enumerate(lhs, start=len(points)):
            points.append(j1j2_point(N, J2=-0.5 + 2.0 * row[0], h=-4.0 + 8.0 * row[1], seed=seed + offset))
    elif family == "dh_family":
        for index in range(count):
            point_seed = seed + index
            template = sample_local_hamiltonian_template(N, seed=point_seed)
            value = 0.0 if index % 8 == 0 else template.lambda_initial
            points.append(dh_point(N, seed=point_seed, lambda_ent=value))
    else:
        raise ValueError(f"unknown family {family}")
    return points[:count]


def mps_to_statevector(state: PurifiedMPS) -> np.ndarray:
    if any(tensor.shape[2] != 1 for tensor in state.tensors):
        raise ValueError("only pure MPS can be converted to a statevector")
    work = state.tensors[0][0, :, 0, :].detach().cpu().numpy()
    for tensor in state.tensors[1:]:
        work = np.tensordot(work, tensor[:, :, 0, :].detach().cpu().numpy(), axes=([-1], [0]))
    vector = np.asarray(work[..., 0], dtype=np.complex128).reshape(-1)
    return vector / np.linalg.norm(vector)


def _tenpy_model(point: HamiltonianPoint):
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    class _ExplicitPauliModel(CouplingMPOModel):
        default_lattice = Chain
        force_default_lattice = True

        def init_sites(self, model_params):
            return SpinHalfSite(conserve=None)

        def init_terms(self, model_params):
            for sites, axes, coefficient in model_params["terms"]:
                strength = float(coefficient) * (2.0 ** len(sites))
                if len(sites) == 1:
                    self.add_onsite_term(strength, int(sites[0]), AXIS_NAMES[int(axes[0])])
                else:
                    self.add_coupling_term(
                        strength, int(sites[0]), int(sites[1]),
                        AXIS_NAMES[int(axes[0])], AXIS_NAMES[int(axes[1])],
                    )

    return _ExplicitPauliModel({
        "L": point.N,
        "terms": [(term.sites, term.axes, term.coefficient) for term in point.terms],
        "bc_MPS": "finite",
        "bc_x": "open",
    })


def solve_ground_state_dmrg(
    point: HamiltonianPoint,
    *,
    chi_max: int = 128,
    svd_min: float = 1.0e-10,
    max_sweeps: int = 40,
    energy_tol: float = 1.0e-10,
) -> tuple[PurifiedMPS, dict[str, Any]]:
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS

    from .xy.model import _tenpy_pure_to_torch

    started = time.perf_counter()
    model = _tenpy_model(point)
    if point.initial_product_vectors is not None:
        initial: Sequence[Any] = [np.asarray(value, dtype=np.complex128) for value in point.initial_product_vectors]
    else:
        initial = ["up" if site % 2 == 0 else "down" for site in range(point.N)]
    psi = MPS.from_product_state(model.lat.mps_sites(), initial, bc="finite", dtype=np.complex128)
    result = dmrg.run(psi, model, {
        "mixer": True,
        "max_sweeps": int(max_sweeps),
        "max_E_err": float(energy_tol),
        "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
        "combine": True,
    })
    state = _tenpy_pure_to_torch(psi)
    vector = mps_to_statevector(state)
    hamiltonian = sparse_pauli_hamiltonian(point)
    hpsi = hamiltonian @ vector
    energy = float(np.vdot(vector, hpsi).real)
    variance = max(0.0, float(np.vdot(hpsi, hpsi).real - energy * energy))
    sweep = result.get("sweep_statistics", {})
    energies = sweep.get("E", [])
    delta_e = abs(float(sweep.get("Delta_E", [math.inf])[-1])) if len(energies) else math.inf
    delta_s = abs(float(sweep.get("Delta_S", [math.inf])[-1])) if len(energies) else math.inf
    errors = result.get("bond_statistics", {}).get("err", [])
    truncation = float(sum(float(getattr(error, "eps", 0.0)) for error in errors))
    diagnostics = {
        "solver": "tenpy_two_site_dmrg_obc",
        "tenpy_reported_energy": float(result.get("E", math.nan)),
        "energy": energy,
        "energy_variance": variance,
        "variance_per_site": variance / point.N,
        "relative_energy_change": delta_e / max(abs(energy), 1.0),
        "entropy_change": delta_s,
        "sweeps": len(energies),
        "chi_max_requested": int(chi_max),
        "final_max_bond_dimension": int(max(state.bond_dimensions)),
        "truncation_error": truncation,
        "norm_error": abs(float(state.norm()) - 1.0),
        "converged": bool(np.isfinite(energy) and variance / point.N < 1.0e-8),
        "runtime_seconds": time.perf_counter() - started,
    }
    return state, diagnostics


class CachedMomentState(MomentState):
    """RDM-centric state adapter; moments are always derived from cached RDMs."""

    def __init__(self, N: int, callback: Callable[[tuple[int, ...]], np.ndarray]):
        self.N = int(N)
        self._callback = callback
        self._rdms: dict[tuple[int, ...], np.ndarray] = {}
        self._pauli: dict[tuple[int, ...], np.ndarray] = {}

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        key = tuple(int(site) for site in sites)
        if key not in self._rdms:
            rho = np.asarray(self._callback(key), dtype=np.complex128)
            rho = (rho + rho.conj().T) / 2.0
            trace = float(np.trace(rho).real)
            if trace <= 0:
                raise FloatingPointError("RDM has non-positive trace")
            self._rdms[key] = rho / trace
        return self._rdms[key]

    def moment(self, sites: Sequence[int], axes: Sequence[int]) -> float:
        key = tuple(int(site) for site in sites)
        if key not in self._pauli:
            coefficients = pauli_coefficients(self.reduced_density_matrix(key))
            self._pauli[key] = coefficients[(slice(1, 4),) * len(key)]
        return float(self._pauli[key][tuple(int(axis) for axis in axes)])


class ClassicalAxisGibbsState(MomentState):
    """Exact diagonal Gibbs mixture in independently rotated local product bases."""

    def __init__(self, axes: np.ndarray, fields: np.ndarray, couplings: np.ndarray, temperature: float):
        self.axes = np.asarray(axes, dtype=np.float64)
        self.fields = np.asarray(fields, dtype=np.float64)
        self.couplings = np.asarray(couplings, dtype=np.float64)
        self.N = len(self.fields)
        basis = np.arange(1 << self.N, dtype=np.uint32)
        bits = ((basis[:, None] >> np.arange(self.N - 1, -1, -1)) & 1).astype(np.int8)
        self.signs = 1.0 - 2.0 * bits
        energy = -self.signs @ self.fields - np.sum(
            self.couplings[None, :] * self.signs[:, :-1] * self.signs[:, 1:], axis=1
        )
        shifted = energy - energy.min()
        weights = np.exp(-shifted / float(temperature))
        self.weights = weights / weights.sum()
        self.energy = float(self.weights @ energy)
        self.energy_variance = float(self.weights @ np.square(energy - self.energy))
        self._spin_correlations: dict[tuple[int, ...], float] = {}
        self._rdms: dict[tuple[int, ...], np.ndarray] = {}

    def moment(self, sites: Sequence[int], axes: Sequence[int]) -> float:
        sites = tuple(int(site) for site in sites)
        if sites not in self._spin_correlations:
            signs = np.prod(self.signs[:, sites], axis=1)
            self._spin_correlations[sites] = float(self.weights @ signs)
        direction = float(np.prod([self.axes[site, int(axis)] for site, axis in zip(sites, axes)]))
        return direction * self._spin_correlations[sites]

    def reduced_density_matrix(self, sites: Sequence[int]) -> np.ndarray:
        sites = tuple(int(site) for site in sites)
        if sites not in self._rdms:
            pauli = (
                np.eye(2, dtype=np.complex128),
                np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
                np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
                np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
            )
            rho = np.zeros((1 << len(sites),) * 2, dtype=np.complex128)
            for labels in product(range(4), repeat=len(sites)):
                active = tuple(index for index, label in enumerate(labels) if label)
                coefficient = 1.0 if not active else self.moment(
                    tuple(sites[index] for index in active), tuple(labels[index] - 1 for index in active)
                )
                operator = np.asarray([[1.0 + 0j]])
                for label in labels:
                    operator = np.kron(operator, pauli[label])
                rho += coefficient * operator
            self._rdms[sites] = (rho + rho.conj().T) / (2.0 * (1 << len(sites)))
        return self._rdms[sites]


def moment_state_from_mps(state: PurifiedMPS) -> CachedMomentState:
    return CachedMomentState(
        state.N,
        lambda sites: state.reduced_density_matrix(sites).detach().cpu().numpy(),
    )


def statevector_rdm(vector: np.ndarray, N: int, sites: Sequence[int]) -> np.ndarray:
    sites = tuple(int(site) for site in sites)
    rest = tuple(site for site in range(N) if site not in sites)
    matrix = np.asarray(vector, dtype=np.complex128).reshape((2,) * N).transpose(sites + rest)
    matrix = matrix.reshape(1 << len(sites), -1)
    return matrix @ matrix.conj().T


def density_matrix_rdm(rho: np.ndarray, N: int, sites: Sequence[int]) -> np.ndarray:
    sites = tuple(int(site) for site in sites)
    rest = tuple(site for site in range(N) if site not in sites)
    k = len(sites)
    tensor = np.asarray(rho, dtype=np.complex128).reshape((2,) * (2 * N))
    order = sites + rest + tuple(N + site for site in sites) + tuple(N + site for site in rest)
    tensor = tensor.transpose(order).reshape(1 << k, 1 << (N - k), 1 << k, 1 << (N - k))
    return np.trace(tensor, axis1=1, axis2=3)


def exact_ground_state(point: HamiltonianPoint) -> tuple[float, CachedMomentState, np.ndarray]:
    hamiltonian = sparse_pauli_hamiltonian(point)
    values, vectors = eigsh(hamiltonian, k=1, which="SA", tol=1.0e-12, maxiter=20000)
    vector = vectors[:, 0] / np.linalg.norm(vectors[:, 0])
    return float(values[0].real), CachedMomentState(point.N, lambda sites: statevector_rdm(vector, point.N, sites)), vector


def local_raw_from_state(state: MomentState, spec: LocalRawSpec) -> dict[str, np.ndarray]:
    pair_mask, triple_mask, four_mask = local_masks(spec)
    one = np.empty((spec.N, 3), dtype=np.float64)
    for site in range(spec.N):
        one[site] = state._pauli[(site,)] if isinstance(state, CachedMomentState) and (site,) in state._pauli else [
            state.moment((site,), (axis,)) for axis in range(3)
        ]
    two = np.zeros((spec.N, spec.R_obs, 3, 3), dtype=np.float64)
    for site, distance0 in zip(*np.nonzero(pair_mask)):
        sites = (int(site), int(site + distance0 + 1))
        two[site, distance0] = np.asarray(
            [state.moment(sites, axes) for axes in np.ndindex(3, 3)]
        ).reshape(3, 3)
    three = np.zeros((spec.N, 3, 3, 3), dtype=np.float64)
    for site in np.flatnonzero(triple_mask):
        sites = (int(site), int(site + 1), int(site + 2))
        three[site] = np.asarray([state.moment(sites, axes) for axes in np.ndindex(3, 3, 3)]).reshape(3, 3, 3)
    four = np.zeros((spec.N, 3, 3, 3, 3), dtype=np.float64)
    for site in np.flatnonzero(four_mask):
        sites = (int(site), int(site + 1), int(site + 2), int(site + 3))
        four[site] = np.asarray([state.moment(sites, axes) for axes in np.ndindex(3, 3, 3, 3)]).reshape(3, 3, 3, 3)
    return {"one": one, "two": two, "three": three, "four": four}


def raw_vector_from_state(state: MomentState, spec: LocalRawSpec) -> np.ndarray:
    return pack_raw_blocks(local_raw_from_state(state, spec), spec)


def certificate_summary(state: MomentState, *, strict_separable: bool = False) -> dict[str, Any]:
    result = hierarchical_certificates(state, strict_separable=strict_separable)
    return {
        "class_name": result["class_name"],
        "certificate_type": result["certificate_type"],
        "certificate_support": list(result["certificate_support"]),
        "certificate_cut": list(result["certificate_cut"]),
        "certificate_margin": float(result["certificate_margin"]),
        "pair_npt_margin": float(result["pairs"].npt_margin),
        "triple_margin": None if result["triples"] is None else float(max(result["triples"].npt_margin, result["triples"].ccnr_margin)),
        "four_margin": None if result["fours"] is None else float(max(result["fours"].npt_margin, result["fours"].ccnr_margin)),
    }


def all_required_supports(N: int) -> tuple[tuple[int, ...], ...]:
    return tuple((site,) for site in range(N)) + pair_supports(N) + contiguous_supports(N, 3) + contiguous_supports(N, 4)


def _lanczos_basis(hamiltonian: sp.csr_matrix, initial: np.ndarray, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(initial, dtype=np.complex128)
    q /= np.linalg.norm(q)
    size = len(q)
    basis = np.empty((size, dimension), dtype=np.complex128)
    alpha = np.empty(dimension, dtype=np.float64)
    beta = np.empty(max(0, dimension - 1), dtype=np.float64)
    previous = np.zeros_like(q)
    previous_beta = 0.0
    used = dimension
    for index in range(dimension):
        basis[:, index] = q
        work = hamiltonian @ q - previous_beta * previous
        alpha[index] = float(np.vdot(q, work).real)
        work -= alpha[index] * q
        # One explicit reorthogonalization keeps m=120 stable on N=16.
        coefficients = basis[:, : index + 1].conj().T @ work
        work -= basis[:, : index + 1] @ coefficients
        if index + 1 == dimension:
            break
        value = float(np.linalg.norm(work))
        if value < 1.0e-13:
            used = index + 1
            break
        beta[index] = value
        previous, q, previous_beta = q, work / value, value
    return basis[:, :used], alpha[:used], beta[: max(0, used - 1)]


def _thermal_vector(basis: np.ndarray, alpha: np.ndarray, beta_lanczos: np.ndarray, beta_thermal: float, shift: float) -> tuple[np.ndarray, float]:
    tridiagonal = np.diag(alpha)
    if len(beta_lanczos):
        tridiagonal += np.diag(beta_lanczos, 1) + np.diag(beta_lanczos, -1)
    values, vectors = np.linalg.eigh(tridiagonal)
    exponents = -0.5 * beta_thermal * (values - shift)
    if float(np.max(exponents)) > 650.0 or float(np.min(exponents)) < -745.0:
        raise FloatingPointError("thermal Lanczos exponential is outside the safe float64 range")
    weights = np.exp(exponents) * vectors[0].conj()
    coefficients = vectors @ weights
    thermal = basis @ coefficients
    norm = float(np.vdot(thermal, thermal).real)
    return thermal, norm


@dataclass
class ThermalTypicalityResult:
    states: dict[float, CachedMomentState]
    block_states: dict[float, list[CachedMomentState]]
    diagnostics: dict[float, dict[str, Any]]


def thermal_lanczos_rdms(
    point: HamiltonianPoint,
    temperatures: Sequence[float],
    *,
    random_vectors: int,
    lanczos_dimension: int,
    seed: int,
    supports: Sequence[tuple[int, ...]] | None = None,
    uncertainty_blocks: int = 4,
) -> ThermalTypicalityResult:
    """RDM-first canonical thermal typicality using a Lanczos exponential."""

    if any(value <= 0 for value in temperatures):
        raise ValueError("thermal temperatures must be positive")
    if random_vectors < 2 or lanczos_dimension < 4:
        raise ValueError("FTLM requires at least R=2 and m=4")
    supports = all_required_supports(point.N) if supports is None else tuple(supports)
    hamiltonian = sparse_pauli_hamiltonian(point)
    dimension = hamiltonian.shape[0]
    # The previous rigorous norm bound was much too loose for the general D_H
    # family and caused every low-temperature weight to underflow.  With the
    # audited parameter ranges, the unshifted exponent remains safely below
    # float64 overflow (checked explicitly in _thermal_vector).
    energy_shift = 0.0
    rng = np.random.default_rng(seed)
    rdm_rows: dict[float, dict[tuple[int, ...], list[np.ndarray]]] = {
        float(T): {tuple(s): [] for s in supports} for T in temperatures
    }
    energy_rows: dict[float, list[float]] = {float(T): [] for T in temperatures}
    norm_rows: dict[float, list[float]] = {float(T): [] for T in temperatures}
    started = time.perf_counter()
    for _ in range(random_vectors):
        phases = rng.integers(0, 4, size=dimension)
        initial = np.asarray((1j) ** phases, dtype=np.complex128) / math.sqrt(dimension)
        basis, alpha, offdiag = _lanczos_basis(hamiltonian, initial, lanczos_dimension)
        for temperature in temperatures:
            T = float(temperature)
            vector, norm = _thermal_vector(basis, alpha, offdiag, 1.0 / T, energy_shift)
            norm_rows[T].append(norm)
            energy_rows[T].append(float(np.vdot(vector, hamiltonian @ vector).real))
            for support in supports:
                rdm_rows[T][tuple(support)].append(statevector_rdm(vector, point.N, support))
    states: dict[float, CachedMomentState] = {}
    block_states: dict[float, list[CachedMomentState]] = {}
    diagnostics: dict[float, dict[str, Any]] = {}
    blocks = [indices for indices in np.array_split(np.arange(random_vectors), min(uncertainty_blocks, random_vectors)) if len(indices)]
    for temperature in temperatures:
        T = float(temperature)
        total_norm = float(np.sum(norm_rows[T]))

        def aggregate(sites: tuple[int, ...], chosen: np.ndarray | None = None) -> np.ndarray:
            values = rdm_rows[T][sites]
            indices: Iterable[int] = range(len(values)) if chosen is None else (int(i) for i in chosen)
            rho = sum((values[i] for i in indices), np.zeros_like(values[0]))
            return rho / float(np.trace(rho).real)

        states[T] = CachedMomentState(point.N, lambda sites, T=T: aggregate(sites))
        block_states[T] = [CachedMomentState(point.N, lambda sites, block=block, T=T: aggregate(sites, block)) for block in blocks]
        energy = float(np.sum(energy_rows[T]) / total_norm)
        block_energy = [
            float(np.sum(np.asarray(energy_rows[T])[block]) / np.sum(np.asarray(norm_rows[T])[block])) for block in blocks
        ]
        max_rdm_uncertainty = 0.0
        min_eigenvalue = math.inf
        for support in supports:
            full = states[T].reduced_density_matrix(support)
            min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(full).min()))
            estimates = np.stack([state.reduced_density_matrix(support) for state in block_states[T]])
            if len(estimates) > 1:
                max_rdm_uncertainty = max(
                    max_rdm_uncertainty,
                    float(np.max(np.linalg.norm(estimates - full, axis=(1, 2)))),
                )
        diagnostics[T] = {
            "solver": "canonical_thermal_typicality_lanczos_rdm",
            "random_vectors": int(random_vectors),
            "lanczos_dimension": int(lanczos_dimension),
            "uncertainty_blocks": len(blocks),
            "energy": energy,
            "energy_block_std": float(np.std(block_energy, ddof=1)) if len(block_energy) > 1 else 0.0,
            "max_block_rdm_frobenius_deviation": max_rdm_uncertainty,
            "minimum_rdm_eigenvalue": min_eigenvalue,
            "energy_shift": energy_shift,
            "runtime_seconds": time.perf_counter() - started,
        }
    return ThermalTypicalityResult(states, block_states, diagnostics)


def _order_margin(state: MomentState, order: int) -> float:
    supports = pair_supports(state.N) if order == 2 else contiguous_supports(state.N, order)
    certificate = support_certificates(state, supports, ccnr=order > 2)
    return float(max(certificate.npt_margin, certificate.ccnr_margin))


def safe_thermal_certificate(
    state: MomentState,
    block_states: Sequence[MomentState],
    *,
    strict_separable_proof: str | None,
    numerical_floor: float = 1.0e-5,
) -> dict[str, Any]:
    if strict_separable_proof:
        summary = certificate_summary(state, strict_separable=True)
        summary.update({
            "class_name": "strict_separable",
            "constructive_proof": strict_separable_proof,
            "certificate_sigma": 0.0,
            "acceptance_threshold": 0.0,
            "stochastic_certificate_accepted": True,
        })
        return summary
    summary = certificate_summary(state)
    candidate = summary["class_name"]
    if candidate == "unknown":
        summary.update({"certificate_sigma": math.nan, "acceptance_threshold": numerical_floor,
                        "stochastic_certificate_accepted": False})
        return summary
    order = {"E2": 2, "E3": 3, "E4": 4}[candidate]
    margins = np.asarray([_order_margin(block, order) for block in block_states], dtype=np.float64)
    sigma = float(np.std(margins, ddof=1)) if len(margins) > 1 else math.inf
    threshold = max(float(numerical_floor), 5.0 * sigma)
    accepted = bool(summary["certificate_margin"] > threshold)
    summary.update({
        "raw_candidate_class": candidate,
        "certificate_sigma": sigma,
        "acceptance_threshold": threshold,
        "stochastic_certificate_accepted": accepted,
    })
    if not accepted:
        summary["class_name"] = "unknown"
    return summary


def exact_thermal_state(point: HamiltonianPoint, temperature: float) -> tuple[CachedMomentState, dict[str, float]]:
    matrix = sparse_pauli_hamiltonian(point).toarray()
    values, vectors = np.linalg.eigh(matrix)
    shifted = values - values.min()
    weights = np.exp(-shifted / float(temperature))
    weights /= weights.sum()
    rho = (vectors * weights[None]) @ vectors.conj().T
    state = CachedMomentState(point.N, lambda sites: density_matrix_rdm(rho, point.N, sites))
    return state, {"energy": float(weights @ values), "partition_shifted": float(np.exp(-shifted / temperature).sum())}


def constructive_thermal_state(point: HamiltonianPoint, temperature: float) -> tuple[MomentState, dict[str, Any]] | None:
    """Return an exact constructive Gibbs state for simple proven controls."""

    if point.family == "xxz" and abs(float(point.parameters["J"])) < 1.0e-15:
        h = float(point.parameters["h"])
        bloch = np.zeros((1, point.N, 3), dtype=np.float64)
        bloch[0, :, 2] = math.tanh(h / (2.0 * float(temperature)))
        state = ProductMixtureState(bloch, np.ones(1, dtype=np.float64))
        energy = -0.5 * h * float(bloch[0, :, 2].sum())
        return state, {
            "solver": "analytic_decoupled_spin_gibbs_product",
            "energy": energy,
            "energy_variance": math.nan,
            "max_block_rdm_frobenius_deviation": 0.0,
            "minimum_rdm_eigenvalue": float(((1.0 - abs(bloch[0, 0, 2])) / 2.0) ** 4),
            "random_vectors": 0,
            "lanczos_dimension": 0,
            "uncertainty_blocks": 0,
            "converged": True,
            "constructive_proof": "tensor_product_of_exact_single_spin_gibbs_states",
        }
    if point.family == "dh_family" and abs(float(point.parameters["lambda_ent"])) < 1.0e-15:
        state = ClassicalAxisGibbsState(
            np.asarray(point.parameters["local_axes"]), np.asarray(point.parameters["fields"]),
            np.asarray(point.parameters["commuting_couplings"]), float(temperature),
        )
        return state, {
            "solver": "exact_classical_product_basis_gibbs_mixture",
            "energy": state.energy,
            "energy_variance": state.energy_variance,
            "max_block_rdm_frobenius_deviation": 0.0,
            "minimum_rdm_eigenvalue": 0.0,
            "random_vectors": 0,
            "lanczos_dimension": 0,
            "uncertainty_blocks": 0,
            "converged": True,
            "constructive_proof": "explicit_convex_mixture_of_local_axis_product_eigenstates",
        }
    return None


def point_metadata(point: HamiltonianPoint) -> dict[str, Any]:
    return {
        "family": point.family,
        "N": point.N,
        "boundary": "OBC",
        "spin_convention": "S=sigma/2 for XXZ/J1-J2; historical D_H uses project Pauli convention",
        "parameters": point.parameters,
        "point_seed": point.point_seed,
        "separability_proof": point.separability_proof,
        "term_count": len(point.terms),
    }

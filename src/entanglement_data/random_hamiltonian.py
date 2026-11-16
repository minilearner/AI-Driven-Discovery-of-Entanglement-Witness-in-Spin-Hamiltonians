from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .states.mps import PurifiedMPS


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

AXES = ("Sx", "Sy", "Sz")
# ``SpinHalfSite(conserve=None)`` uses the project basis ``up, down`` so the
# project Pauli matrices are exactly two times TeNPy's Sx/Sy/Sz matrices.
PAULI_TO_TENPY = np.asarray([2.0, 2.0, 2.0])


@dataclass(frozen=True)
class LocalHamiltonianTemplate:
    local_axes: np.ndarray
    fields: np.ndarray
    commuting_couplings: np.ndarray
    noncommuting_couplings: np.ndarray
    profile_type: str
    lambda_initial: float
    lambda_bin: str
    template_seed: int

    @property
    def N(self) -> int:
        return int(self.local_axes.shape[0])


@dataclass(frozen=True)
class GroundStateDiagnostics:
    solver: str
    converged: bool
    energy: float
    relative_energy_change: float
    entropy_change: float
    sweeps: int
    chi_max_requested: int
    final_max_bond_dimension: int
    average_bond_dimension: float
    truncation_error: float
    norm_error: float
    runtime_seconds: float


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("cannot normalize a near-zero vector")
    return values / norms


def _sample_profile(rng: np.random.Generator) -> str:
    value = float(rng.random())
    if value < 0.4:
        return "disordered"
    if value < 0.7:
        return "weakly_homogeneous"
    return "smooth"


def _sample_lambda(rng: np.random.Generator) -> tuple[float, str]:
    value = float(rng.random())
    if value < 0.3:
        return float(rng.uniform(0.10, 0.30)), "weak"
    if value < 0.8:
        return float(rng.uniform(0.30, 0.80)), "moderate"
    return float(rng.uniform(0.80, 1.50)), "strong"


def _profile_axes(N: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    if profile == "disordered":
        return _normalize_rows(rng.normal(size=(N, 3)))
    base = _normalize_rows(rng.normal(size=(1, 3)))[0]
    if profile == "weakly_homogeneous":
        return _normalize_rows(base[None, :] + 0.18 * rng.normal(size=(N, 3)))
    direction = rng.normal(size=3)
    direction -= direction.dot(base) * base
    direction /= max(np.linalg.norm(direction), 1e-12)
    frequency = int(rng.integers(1, 3))
    phase = float(rng.uniform(0, 2 * np.pi))
    coordinate = np.arange(N, dtype=float)
    modulation = np.sin(2 * np.pi * frequency * coordinate / max(N - 1, 1) + phase)
    return _normalize_rows(base[None, :] + 0.45 * modulation[:, None] * direction[None, :])


def _profile_commuting_couplings(N: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    if profile == "disordered":
        return rng.uniform(-0.15, 0.15, size=N - 1)
    base = float(rng.uniform(-0.10, 0.10))
    if profile == "weakly_homogeneous":
        return np.clip(base + rng.normal(scale=0.025, size=N - 1), -0.15, 0.15)
    frequency = int(rng.integers(1, 3))
    phase = float(rng.uniform(0, 2 * np.pi))
    coordinate = np.arange(N - 1, dtype=float) + 0.5
    return np.clip(base + 0.06 * np.sin(2 * np.pi * frequency * coordinate / N + phase), -0.15, 0.15)


def _profile_matrices(N: int, profile: str, rng: np.random.Generator) -> np.ndarray:
    if profile == "disordered":
        return rng.normal(size=(N - 1, 3, 3))
    base = rng.normal(size=(3, 3))
    if profile == "weakly_homogeneous":
        return base[None, :, :] + 0.18 * rng.normal(size=(N - 1, 3, 3))
    second = rng.normal(size=(3, 3))
    frequency = int(rng.integers(1, 3))
    phase = float(rng.uniform(0, 2 * np.pi))
    coordinate = np.arange(N - 1, dtype=float) + 0.5
    modulation = np.sin(2 * np.pi * frequency * coordinate / N + phase)
    return base[None, :, :] + 0.55 * modulation[:, None, None] * second[None, :, :]


def sample_local_hamiltonian_template(N: int, *, seed: int) -> LocalHamiltonianTemplate:
    if N < 2:
        raise ValueError("N must be at least two")
    rng = np.random.default_rng(int(seed))
    profile = _sample_profile(rng)
    axes = _profile_axes(N, profile, rng)
    commuting = _profile_commuting_couplings(N, profile, rng)
    fields = rng.uniform(1.0, 1.5, size=N)
    neighbor_bound = np.zeros(N, dtype=float)
    neighbor_bound[:-1] += np.abs(commuting)
    neighbor_bound[1:] += np.abs(commuting)
    safety_margin = 0.25
    fields = np.maximum(fields, neighbor_bound + safety_margin)

    matrices = _profile_matrices(N, profile, rng)
    projected = np.empty_like(matrices)
    for i, matrix in enumerate(matrices):
        commuting_component = float(axes[i] @ matrix @ axes[i + 1])
        matrix = matrix - commuting_component * np.outer(axes[i], axes[i + 1])
        norm = float(np.linalg.norm(matrix))
        if norm < 1e-10:
            matrix = np.eye(3) - np.outer(axes[i], axes[i + 1])
            norm = float(np.linalg.norm(matrix))
        projected[i] = matrix / norm
        if abs(float(axes[i] @ projected[i] @ axes[i + 1])) > 1e-12:
            raise RuntimeError("commuting interaction component removal failed")
    lambda_initial, lambda_bin = _sample_lambda(rng)
    return LocalHamiltonianTemplate(
        local_axes=axes,
        fields=fields,
        commuting_couplings=commuting,
        noncommuting_couplings=projected,
        profile_type=profile,
        lambda_initial=lambda_initial,
        lambda_bin=lambda_bin,
        template_seed=int(seed),
    )


def total_bond_couplings(template: LocalHamiltonianTemplate, lambda_ent: float) -> np.ndarray:
    parent = -template.commuting_couplings[:, None, None] * np.einsum(
        "ia,ib->iab", template.local_axes[:-1], template.local_axes[1:]
    )
    return parent + float(lambda_ent) * template.noncommuting_couplings


def product_ground_state(template: LocalHamiltonianTemplate) -> PurifiedMPS:
    tensors: list[torch.Tensor] = []
    for nx, ny, nz in template.local_axes:
        theta = math.acos(float(np.clip(nz, -1.0, 1.0)))
        phi = math.atan2(float(ny), float(nx))
        local = torch.tensor(
            [math.cos(theta / 2), np.exp(1j * phi) * math.sin(theta / 2)], dtype=torch.complex128
        ).reshape(1, 2, 1, 1)
        tensors.append(local)
    return PurifiedMPS(tensors).normalize()


def analytic_parent_energy(template: LocalHamiltonianTemplate) -> float:
    return -float(template.fields.sum() + template.commuting_couplings.sum())


def random_local_tenpy_model(template: LocalHamiltonianTemplate, lambda_ent: float):
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import SpinHalfSite

    class _RandomLocalHamiltonian(CouplingMPOModel):
        default_lattice = Chain
        force_default_lattice = True

        def init_sites(self, model_params):
            return SpinHalfSite(conserve=None)

        def init_terms(self, model_params):
            axes = np.asarray(model_params["local_axes"], dtype=float)
            fields = np.asarray(model_params["fields"], dtype=float)
            bonds = np.asarray(model_params["bond_couplings"], dtype=float)
            for i in range(len(fields)):
                for a, op in enumerate(AXES):
                    self.add_onsite_term(-fields[i] * axes[i, a] * PAULI_TO_TENPY[a], i, op)
            for i in range(len(bonds)):
                for a, left_op in enumerate(AXES):
                    for b, right_op in enumerate(AXES):
                        strength = bonds[i, a, b] * PAULI_TO_TENPY[a] * PAULI_TO_TENPY[b]
                        if abs(strength) > 1e-15:
                            self.add_coupling_term(strength, i, i + 1, left_op, right_op)

    return _RandomLocalHamiltonian(
        {
            "L": template.N,
            "local_axes": template.local_axes,
            "fields": template.fields,
            "bond_couplings": total_bond_couplings(template, lambda_ent),
            "bc_MPS": "finite",
            "bc_x": "open",
        }
    )


def solve_entangled_ground_state(
    template: LocalHamiltonianTemplate,
    lambda_ent: float,
    *,
    chi_max: int = 128,
    svd_min: float = 1e-10,
    max_sweeps: int = 30,
    energy_tol: float = 1e-10,
) -> tuple[PurifiedMPS, GroundStateDiagnostics]:
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS

    from .xy.model import _tenpy_pure_to_torch

    started = time.perf_counter()
    model = random_local_tenpy_model(template, lambda_ent)
    local_vectors = [tensor[0, :, 0, 0].detach().cpu().numpy() for tensor in product_ground_state(template).tensors]
    psi = MPS.from_product_state(model.lat.mps_sites(), local_vectors, bc="finite", dtype=np.complex128)
    result = dmrg.run(
        psi,
        model,
        {
            "mixer": True,
            "max_sweeps": int(max_sweeps),
            "max_E_err": float(energy_tol),
            "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
            "combine": True,
        },
    )
    state = _tenpy_pure_to_torch(psi)
    sweep_stats = result.get("sweep_statistics", {})
    energies = sweep_stats.get("E", [])
    sweeps = len(energies)
    delta_e = abs(float(sweep_stats.get("Delta_E", [math.inf])[-1])) if sweeps else math.inf
    delta_s = abs(float(sweep_stats.get("Delta_S", [math.inf])[-1])) if sweeps else math.inf
    energy = float(result.get("E", math.nan))
    relative_delta_e = delta_e / max(abs(energy), 1.0)
    errors = result.get("bond_statistics", {}).get("err", [])
    truncation = float(sum(float(getattr(error, "eps", 0.0)) for error in errors))
    bond_dimensions = np.asarray(state.bond_dimensions[1:-1], dtype=float)
    diagnostics = GroundStateDiagnostics(
        solver="tenpy_two_site_dmrg_random_local_obc",
        converged=bool(np.isfinite(energy) and relative_delta_e < energy_tol and delta_s < 1e-5),
        energy=energy,
        relative_energy_change=float(relative_delta_e),
        entropy_change=float(delta_s),
        sweeps=int(sweeps),
        chi_max_requested=int(chi_max),
        final_max_bond_dimension=int(max(state.bond_dimensions)),
        average_bond_dimension=float(bond_dimensions.mean()),
        truncation_error=truncation,
        norm_error=abs(float(state.norm().item()) - 1.0),
        runtime_seconds=time.perf_counter() - started,
    )
    return state, diagnostics


def schmidt_entropy_profile(state: PurifiedMPS, threshold: float = 1e-15) -> np.ndarray:
    values: list[float] = []
    for cut in range(1, state.N):
        singular = state.schmidt_values(cut).detach().cpu().numpy()
        probabilities = singular**2
        probabilities = probabilities[probabilities > threshold]
        values.append(float(-(probabilities * np.log(probabilities)).sum()))
    return np.asarray(values, dtype=np.float64)


def dense_hamiltonian(template: LocalHamiltonianTemplate, lambda_ent: float) -> np.ndarray:
    if template.N > 8:
        raise ValueError("dense Hamiltonian is restricted to N <= 8")
    pauli = [
        np.asarray([[0, 1], [1, 0]], dtype=complex),
        np.asarray([[0, -1j], [1j, 0]], dtype=complex),
        np.asarray([[1, 0], [0, -1]], dtype=complex),
    ]
    identity = np.eye(2, dtype=complex)

    def operator(terms: dict[int, np.ndarray]) -> np.ndarray:
        out = np.asarray([[1.0]], dtype=complex)
        for site in range(template.N):
            out = np.kron(out, terms.get(site, identity))
        return out

    hamiltonian = np.zeros((2**template.N, 2**template.N), dtype=complex)
    for i in range(template.N):
        local = sum(template.local_axes[i, a] * pauli[a] for a in range(3))
        hamiltonian -= template.fields[i] * operator({i: local})
    bonds = total_bond_couplings(template, lambda_ent)
    for i in range(template.N - 1):
        for a in range(3):
            for b in range(3):
                hamiltonian += bonds[i, a, b] * operator({i: pauli[a], i + 1: pauli[b]})
    return (hamiltonian + hamiltonian.conj().T) / 2

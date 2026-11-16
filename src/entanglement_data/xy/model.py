from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..states.mps import PurifiedMPS


# NumPy/SciPy used by TeNPy and PyTorch can load different Intel OpenMP DLLs on
# Windows.  This must be set before importing TeNPy in the mixed process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class SolverMetadata:
    solver: str
    solver_converged: bool
    chi_max_requested: int
    final_max_bond_dimension: int
    imaginary_time_step: float
    number_of_steps: int
    total_truncation_error: float
    discarded_weight: float
    norm_error: float
    runtime_state_generation: float
    energy_generation_hamiltonian: float = math.nan


def xy_tenpy_model(N: int, gamma: float, h: float, *, epsilon_x: float = 0.0, pbc: bool = True):
    """Build the finite TeNPy MPO in this project's Pauli convention.

    TeNPy's spin-1/2 basis is ``down, up`` while this project uses Z=diag(1,-1).
    Thus Z_project=-2*Sz_TeNPy, whereas X_project=2*Sx_TeNPy.  The mapped
    parameters below exactly implement the requested Pauli Hamiltonian.
    """
    from tenpy.models.spins import SpinModel

    return SpinModel(
        {
            "L": int(N),
            "S": 0.5,
            "Jx": -2.0 * (1.0 + float(gamma)),
            "Jy": -2.0 * (1.0 - float(gamma)),
            "Jz": 0.0,
            "hx": 2.0 * float(epsilon_x),
            "hy": 0.0,
            "hz": -2.0 * float(h),
            "conserve": None,
            "bc_MPS": "finite",
            "bc_x": "periodic" if pbc else "open",
        }
    )


def purified_xy_tenpy_model(N: int, gamma: float, h: float, *, pbc: bool = True):
    """MPO on a four-level (physical x ancilla) site, acting only physically."""
    from tenpy.linalg import np_conserved as npc
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import Site

    class _PurifiedXYModel(CouplingMPOModel):
        default_lattice = Chain
        force_default_lattice = True

        def init_sites(self, model_params):
            leg = npc.LegCharge.from_trivial(4)
            I = np.eye(2, dtype=complex)
            X = np.asarray([[0, 1], [1, 0]], dtype=complex)
            Y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
            Z = np.asarray([[1, 0], [0, -1]], dtype=complex)
            return Site(
                leg,
                ["00", "01", "10", "11"],
                sort_charge=False,
                Xp=np.kron(X, I),
                Yp=np.kron(Y, I),
                Zp=np.kron(Z, I),
            )

        def init_terms(self, model_params):
            g = model_params.get("gamma", 0.0, "real")
            field = model_params.get("h", 0.0, "real")
            self.add_onsite(-field, 0, "Zp")
            for u1, u2, dx in self.lat.pairs["nearest_neighbors"]:
                self.add_coupling(-(1.0 + g) / 2.0, u1, "Xp", u2, "Xp", dx)
                self.add_coupling(-(1.0 - g) / 2.0, u1, "Yp", u2, "Yp", dx)

    return _PurifiedXYModel(
        {
            "L": int(N),
            "gamma": float(gamma),
            "h": float(h),
            "bc_MPS": "finite",
            "bc_x": "periodic" if pbc else "open",
        }
    )


def _tenpy_purification_to_torch(psi) -> PurifiedMPS:
    tensors: list[torch.Tensor] = []
    for i in range(psi.L):
        B = psi.get_B(i, form="B")
        labels = B.get_leg_labels()
        wanted = ["vL", "p", "q", "vR"]
        if labels != wanted:
            B = B.transpose(wanted)
        tensors.append(torch.as_tensor(np.asarray(B.to_ndarray()), dtype=torch.complex128))
    return PurifiedMPS(tensors).normalize()


def _tenpy_pure_to_torch(psi) -> PurifiedMPS:
    tensors: list[torch.Tensor] = []
    for i in range(psi.L):
        B = psi.get_B(i, form="B")
        labels = B.get_leg_labels()
        wanted = ["vL", "p", "vR"]
        if labels != wanted:
            B = B.transpose(wanted)
        a = np.asarray(B.to_ndarray())[:, :, None, :]
        tensors.append(torch.as_tensor(a, dtype=torch.complex128))
    return PurifiedMPS(tensors).normalize()


def _tenpy_combined_purification_to_torch(psi) -> PurifiedMPS:
    tensors: list[torch.Tensor] = []
    for i in range(psi.L):
        B = psi.get_B(i, form="B")
        labels = B.get_leg_labels()
        wanted = ["vL", "p", "vR"]
        if labels != wanted:
            B = B.transpose(wanted)
        a = np.asarray(B.to_ndarray()).reshape(B.shape[0], 2, 2, B.shape[2])
        tensors.append(torch.as_tensor(a, dtype=torch.complex128))
    return PurifiedMPS(tensors).normalize()


def exact_factorized_product(N: int, gamma: float, h: float, *, branch: int = 1) -> PurifiedMPS:
    if abs(gamma * gamma + h * h - 1.0) > 1e-10:
        raise ValueError("factorized anchor must lie on gamma^2+h^2=1")
    if branch not in (-1, 1):
        raise ValueError("factorized branch must be +1 or -1")
    argument = float(np.clip(h / (1.0 + gamma), -1.0, 1.0))
    phi = 0.5 * math.acos(argument)
    local = torch.tensor([math.cos(phi), branch * math.sin(phi)], dtype=torch.complex128).reshape(1, 2, 1, 1)
    return PurifiedMPS([local.clone() for _ in range(N)])


def exact_polarized_product(N: int) -> PurifiedMPS:
    local = torch.tensor([1.0, 0.0], dtype=torch.complex128).reshape(1, 2, 1, 1)
    return PurifiedMPS([local.clone() for _ in range(N)])


def generate_thermal_state(
    N: int,
    gamma: float,
    h: float,
    T: float,
    *,
    chi_max: int = 64,
    dt: float = 0.25,
    svd_min: float = 1e-10,
    max_sweeps: int = 2,
    pbc: bool = True,
) -> tuple[PurifiedMPS, SolverMetadata]:
    """Finite-N thermal purification using two-site TDVP Krylov evolution.

    Physical and ancilla indices are fused into a four-level TeNPy site.  The
    finite PBC Hamiltonian MPO acts as H_XY tensor I_ancilla, so no dense Gibbs
    matrix is constructed at any stage.
    """
    if T <= 0:
        raise ValueError("thermal generation requires T>0")
    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    from tenpy.networks.mps import MPS

    started = time.perf_counter()
    model = purified_xy_tenpy_model(N, gamma, h, pbc=pbc)
    bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    psi = MPS.from_product_state(model.lat.mps_sites(), [bell] * N, bc="finite")
    target_tau = 0.5 / float(T)
    steps = max(1, int(math.ceil(target_tau / dt)))
    actual_dt = target_tau / steps
    engine = TwoSiteTDVPEngine(
        psi,
        model,
        {
            "trunc_params": {"chi_max": int(chi_max), "svd_min": float(svd_min)},
            "lanczos_params": {"P_tol": min(1e-13, float(svd_min) * 0.1)},
            "max_dt": None,
            "preserve_norm": False,
        },
    )
    converged = True
    try:
        engine.run_evolution(steps, -1j * actual_dt)
    except Exception:
        converged = False
        raise
    truncation = float(getattr(engine.trunc_err, "eps", 0.0))
    state = _tenpy_combined_purification_to_torch(psi)
    norm_error = abs(float(state.norm().item()) - 1.0)
    meta = SolverMetadata(
        solver="tenpy_two_site_tdvp_krylov_thermal_purification",
        solver_converged=converged and np.isfinite(truncation),
        chi_max_requested=int(chi_max),
        final_max_bond_dimension=max(state.bond_dimensions),
        imaginary_time_step=float(actual_dt),
        number_of_steps=steps,
        total_truncation_error=truncation,
        discarded_weight=truncation,
        norm_error=norm_error,
        runtime_state_generation=time.perf_counter() - started,
    )
    return state, meta


def generate_ground_state(
    N: int,
    gamma: float,
    h: float,
    *,
    epsilon_x: float,
    chi_max: int = 128,
    svd_min: float = 1e-12,
    max_sweeps: int = 30,
    energy_tol: float = 1e-11,
    pbc: bool = True,
) -> tuple[PurifiedMPS, SolverMetadata]:
    """Finite-N PBC DMRG ground state, optionally selecting the +x branch."""
    from tenpy.algorithms import dmrg
    from tenpy.networks.mps import MPS

    started = time.perf_counter()
    model = xy_tenpy_model(N, gamma, h, epsilon_x=epsilon_x, pbc=pbc)
    plus_x = np.asarray([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
    psi = MPS.from_product_state(model.lat.mps_sites(), [plus_x] * N, bc="finite")
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
    sweeps = len(sweep_stats.get("E", []))
    delta_E = abs(float(sweep_stats.get("Delta_E", [math.inf])[-1])) if sweeps else math.inf
    delta_S = abs(float(sweep_stats.get("Delta_S", [math.inf])[-1])) if sweeps else math.inf
    converged = delta_E / max(abs(float(result.get("E", 0.0))), 1.0) < energy_tol and delta_S < 1e-5
    errors = result.get("bond_statistics", {}).get("err", [])
    truncation = float(sum(float(getattr(err, "eps", 0.0)) for err in errors))
    meta = SolverMetadata(
        solver="tenpy_two_site_dmrg",
        solver_converged=converged,
        chi_max_requested=int(chi_max),
        final_max_bond_dimension=max(state.bond_dimensions),
        imaginary_time_step=math.nan,
        number_of_steps=sweeps,
        total_truncation_error=truncation,
        discarded_weight=truncation,
        norm_error=abs(float(state.norm().item()) - 1.0),
        runtime_state_generation=time.perf_counter() - started,
        energy_generation_hamiltonian=float(result.get("E", math.nan)),
    )
    return state, meta

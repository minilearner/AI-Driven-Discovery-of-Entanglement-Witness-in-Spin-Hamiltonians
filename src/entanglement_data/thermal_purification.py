from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .ood_hamiltonian_n16 import CachedMomentState, HamiltonianPoint
from .states.mps import PurifiedMPS
from .xy.model import _tenpy_combined_purification_to_torch


PAULI_NAMES = ("Xp", "Yp", "Zp")


@dataclass(frozen=True)
class PurificationDiagnostics:
    solver: str
    temperature: float
    beta: float
    chi_max_requested: int
    final_max_bond_dimension: int
    imaginary_time_step_max: float
    number_of_steps: int
    truncation_error: float
    norm_error: float
    runtime_seconds: float


def as_cached_moment_state(state: PurifiedMPS) -> CachedMomentState:
    """Expose PyTorch MPS RDMs through the NumPy certificate interface."""
    return CachedMomentState(
        state.N,
        lambda sites: state.reduced_density_matrix(sites).detach().cpu().numpy(),
    )


def purified_point_tenpy_model(point: HamiltonianPoint):
    """Build H(point) tensor I_ancilla on four-level purification sites."""
    from tenpy.linalg import np_conserved as npc
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.site import Site

    class _PurifiedPointModel(CouplingMPOModel):
        default_lattice = Chain
        force_default_lattice = True

        def init_sites(self, model_params):
            leg = npc.LegCharge.from_trivial(4)
            identity = np.eye(2, dtype=np.complex128)
            x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
            y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
            z = np.asarray([[1, 0], [0, -1]], dtype=np.complex128)
            return Site(
                leg,
                ["00", "01", "10", "11"],
                sort_charge=False,
                Xp=np.kron(x, identity),
                Yp=np.kron(y, identity),
                Zp=np.kron(z, identity),
            )

        def init_terms(self, model_params):
            for sites, axes, coefficient in model_params["terms"]:
                if len(sites) == 1:
                    self.add_onsite_term(
                        float(coefficient), int(sites[0]), PAULI_NAMES[int(axes[0])]
                    )
                else:
                    self.add_coupling_term(
                        float(coefficient),
                        int(sites[0]),
                        int(sites[1]),
                        PAULI_NAMES[int(axes[0])],
                        PAULI_NAMES[int(axes[1])],
                    )

    return _PurifiedPointModel(
        {
            "L": int(point.N),
            "terms": [
                (term.sites, term.axes, term.coefficient) for term in point.terms
            ],
            "bc_MPS": "finite",
            "bc_x": "open",
        }
    )


def generate_thermal_path(
    point: HamiltonianPoint,
    temperatures: Sequence[float],
    *,
    chi_max: int = 128,
    dt: float = 0.05,
    svd_min: float = 1.0e-11,
) -> tuple[dict[float, PurifiedMPS], dict[float, dict[str, Any]]]:
    """Generate Gibbs purifications along one path by monotone imaginary time.

    Temperatures are traversed from high to low so each state continues from the
    previous inverse temperature.  The physical state is
    Tr_anc |exp(-beta H/2)><exp(-beta H/2)|.
    """
    values = tuple(sorted({float(value) for value in temperatures}, reverse=True))
    if not values or any(value <= 0 for value in values):
        raise ValueError("all temperatures must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
    from tenpy.networks.mps import MPS

    started = time.perf_counter()
    model = purified_point_tenpy_model(point)
    bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    psi = MPS.from_product_state(
        model.lat.mps_sites(), [bell] * point.N, bc="finite", dtype=np.complex128
    )

    states: dict[float, PurifiedMPS] = {}
    diagnostics: dict[float, dict[str, Any]] = {}
    tau = 0.0
    cumulative_steps = 0
    cumulative_truncation = 0.0
    for temperature in values:
        target_tau = 0.5 / temperature
        delta_tau = target_tau - tau
        steps = max(1, int(math.ceil(delta_tau / dt)))
        actual_dt = delta_tau / steps
        engine = TwoSiteTDVPEngine(
            psi,
            model,
            {
                "trunc_params": {
                    "chi_max": int(chi_max),
                    "svd_min": float(svd_min),
                },
                "lanczos_params": {"P_tol": min(1.0e-13, float(svd_min) * 0.1)},
                "max_dt": None,
                "preserve_norm": False,
            },
        )
        engine.run_evolution(steps, -1j * actual_dt)
        cumulative_steps += steps
        cumulative_truncation += float(getattr(engine.trunc_err, "eps", 0.0))
        tau = target_tau
        state = _tenpy_combined_purification_to_torch(psi)
        states[temperature] = state
        diag = PurificationDiagnostics(
            solver="tenpy_two_site_tdvp_thermal_purification_obc",
            temperature=temperature,
            beta=1.0 / temperature,
            chi_max_requested=int(chi_max),
            final_max_bond_dimension=int(max(state.bond_dimensions)),
            imaginary_time_step_max=float(dt),
            number_of_steps=int(cumulative_steps),
            truncation_error=float(cumulative_truncation),
            norm_error=abs(float(state.norm()) - 1.0),
            runtime_seconds=time.perf_counter() - started,
        )
        diagnostics[temperature] = diag.__dict__
    return states, diagnostics

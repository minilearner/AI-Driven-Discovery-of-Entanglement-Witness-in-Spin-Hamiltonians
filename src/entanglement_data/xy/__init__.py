"""Finite-size XY-chain state generation and benchmark utilities."""

from .free_fermion_exact import XYFreeFermionExactBackend, appendix_a_thermodynamics

from .model import (
    SolverMetadata,
    exact_factorized_product,
    exact_polarized_product,
    generate_ground_state,
    generate_thermal_state,
    xy_tenpy_model,
)

__all__ = [
    "XYFreeFermionExactBackend",
    "appendix_a_thermodynamics",
    "SolverMetadata",
    "exact_factorized_product",
    "exact_polarized_product",
    "generate_ground_state",
    "generate_thermal_state",
    "xy_tenpy_model",
]

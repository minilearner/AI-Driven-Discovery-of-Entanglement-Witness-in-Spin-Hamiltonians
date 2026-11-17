"""Two physical N=16 examples: product spins and a nearest-neighbor Bell pair.

The full route is state -> Pauli correlations -> LU-invariant tokens -> EntNet.
The output is an empirical classifier score, not a certificate or calibrated
probability. The Bell pair is independently NPT-certified on sites (0, 1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entanglement_data.certification.ppt import certify_two_site_npt  # noqa: E402
from entanglement_data.final_entnet import predict_final_ensemble, tokens_from_pauli_features  # noqa: E402
from entanglement_data.observables.engine import compute_features  # noqa: E402
from entanglement_data.observables.feature_spec import FeatureSpec  # noqa: E402
from entanglement_data.states.mps import PurifiedMPS  # noqa: E402
from entanglement_data.states.separable import SeparableMixture  # noqa: E402


def bell_pair_chain(n: int = 16) -> PurifiedMPS:
    if n < 2:
        raise ValueError("a Bell pair needs at least two sites")
    dtype = torch.complex128
    first = torch.zeros((1, 2, 1, 2), dtype=dtype)
    first[0, 0, 0, 0] = 2 ** -0.5
    first[0, 1, 0, 1] = 2 ** -0.5
    second = torch.zeros((2, 2, 1, 1), dtype=dtype)
    second[0, 0, 0, 0] = 1
    second[1, 1, 0, 0] = 1
    rest = []
    for _ in range(n - 2):
        tensor = torch.zeros((1, 2, 1, 1), dtype=dtype)
        tensor[0, 0, 0, 0] = 1
        rest.append(tensor)
    return PurifiedMPS([first, second, *rest])


def product_chain(n: int = 16) -> SeparableMixture:
    bloch = torch.zeros((1, n, 3), dtype=torch.float64)
    bloch[:, :, 2] = 1.0
    return SeparableMixture(torch.ones(1, dtype=torch.float64), bloch)


def main() -> None:
    torch.set_num_threads(4)
    n = 16
    spec = FeatureSpec(n)
    product = product_chain(n)
    bell = bell_pair_chain(n)
    certificate = certify_two_site_npt(bell, max_distance=1)
    if not certificate.certified or certificate.pair != (0, 1):
        raise RuntimeError("the Bell-pair example lost its independent NPT certificate")
    raw = np.stack([compute_features(product, spec).values,
                    compute_features(bell, spec).values])
    tokens = tokens_from_pauli_features(raw, N=n)
    scores = predict_final_ensemble(tokens, ROOT)
    print(f"N={n}; raw Pauli correlations per state={raw.shape[1]}; "
          f"full LU-invariant channels per site={tokens.shape[2]}")
    print(f"Product |0>^N: constructively separable; EntNet score={scores[0]:.8f}")
    print(f"Bell pair on sites (0,1): NPT min eigenvalue={certificate.min_eigenvalue:.6f}; "
          f"EntNet score={scores[1]:.8f}")


if __name__ == "__main__":
    main()

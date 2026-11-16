"""Numerical check for the separate appendix-control checkpoint route."""

from pathlib import Path
import sys

import numpy as np

from entanglement_data.control_entnet import predict_control_ensemble


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_r2support_interventions import intervention_scores  # noqa: E402


def test_r2support_control_matches_frozen_j1j2_scores() -> None:
    fixture = np.load(ROOT / "tests/fixtures/r2support_control_9rows.npz")
    predicted = predict_control_ensemble(fixture["selected_tokens"], ROOT)
    np.testing.assert_allclose(
        predicted,
        fixture["reference_probability"],
        rtol=0.0,
        atol=5.0e-6,
    )


def test_mask_aware_interventions_match_saved_scores() -> None:
    fixture = np.load(ROOT / "tests/fixtures/r2support_control_9rows.npz")
    scores = intervention_scores(fixture["selected_tokens"])
    np.testing.assert_allclose(scores["p_no_high"], fixture["reference_no_high"], rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(scores["p_no_pair"], fixture["reference_no_pair"], rtol=0.0, atol=5e-6)

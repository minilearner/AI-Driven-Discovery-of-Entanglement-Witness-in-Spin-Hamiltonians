from pathlib import Path

import numpy as np

from entanglement_data.final_entnet import predict_final_ensemble, tokens_from_pauli_features


ROOT = Path(__file__).resolve().parents[1]


def test_final_ensemble_matches_frozen_xy_scores() -> None:
    fixture = np.load(ROOT / "tests" / "fixtures" / "xy_reference_9rows.npz")
    actual = predict_final_ensemble(fixture["full_tokens"], ROOT)
    # Original scores were produced on GPU; this smoke test runs on CPU.
    np.testing.assert_allclose(actual, fixture["ensemble_probability"], rtol=0, atol=5e-6)


def test_final_model_rejects_wrong_channel_width() -> None:
    try:
        predict_final_ensemble(np.zeros((1, 16, 21), np.float32), ROOT)
    except ValueError as exc:
        assert "[B,N,122]" in str(exc)
    else:
        raise AssertionError("input schema was not checked")


def test_raw_pauli_preprocessing_matches_frozen_tokens() -> None:
    fixture = np.load(ROOT / "tests" / "fixtures" / "xy_reference_9rows.npz")
    actual = tokens_from_pauli_features(fixture["raw_features"], N=16)
    np.testing.assert_allclose(actual, fixture["full_tokens"], rtol=0, atol=2e-6)

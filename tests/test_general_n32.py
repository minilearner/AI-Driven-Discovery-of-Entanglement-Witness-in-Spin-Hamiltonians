from __future__ import annotations

from collections import Counter

import numpy as np

from entanglement_data.general_n32 import (
    CLASS_FAMILIES,
    PPT_TOLERANCE,
    LocalRawSpec,
    connected_from_raw,
    density_from_state,
    generate_family_state,
    hierarchical_certificates,
    local_masks,
    local_raw,
    partial_trace,
    production_manifest,
)


def test_partial_trace_bell_state_is_maximally_mixed() -> None:
    bell = np.asarray([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2)
    rho = density_from_state(bell)
    assert np.allclose(partial_trace(rho, (0,)), np.eye(2) / 2, atol=1e-12)
    assert np.allclose(partial_trace(rho, (1,)), np.eye(2) / 2, atol=1e-12)


def test_production_manifest_has_frozen_counts_and_is_deterministic() -> None:
    first = production_manifest()
    second = production_manifest()
    assert first == second
    assert len(first) == 2000
    assert Counter(row["class_name"] for row in first) == Counter(
        {name: sum(count for _, count in families) for name, families in CLASS_FAMILIES.items()}
    )
    assert Counter(row["family"] for row in first) == Counter(
        {family: count for families in CLASS_FAMILIES.values() for family, count in families}
    )
    assert len({row["seed"] for row in first}) == 2000


def test_every_family_hits_its_hierarchical_target() -> None:
    for class_name, families in CLASS_FAMILIES.items():
        for ordinal, (family, _) in enumerate(families):
            state, metadata = generate_family_state(family, 1234567 + ordinal)
            result = hierarchical_certificates(state, strict_separable=class_name == "strict_separable")
            assert result["class_name"] == class_name, family
            assert metadata["block_size"] <= 4
            if class_name in {"E3", "E4"}:
                assert np.min(result["pairs"].min_pt) >= PPT_TOLERANCE
            if class_name == "E4":
                assert result["triples"] is not None and not result["triples"].certified


def test_local_raw_schema_masks_bounds_and_connected_consistency() -> None:
    state, _ = generate_family_state("dephased_multipartite_E4", 99123)
    spec = LocalRawSpec(32, 4)
    raw = local_raw(state, spec)
    assert raw["T1"].shape == (32, 3)
    assert raw["T2"].shape == (32, 4, 3, 3)
    assert raw["T3"].shape == (32, 3, 3, 3)
    assert raw["T4"].shape == (32, 3, 3, 3, 3)
    assert raw["K2"].shape == raw["T2"].shape
    assert raw["K3"].shape == raw["T3"].shape
    assert max(np.max(np.abs(raw[key])) for key in ("T1", "T2", "T3", "T4")) <= 1 + 1e-12
    pair, triple, four = local_masks(spec)
    assert pair.sum() == 118
    assert triple.sum() == 30
    assert four.sum() == 29
    k2, k3 = connected_from_raw(raw["T1"], raw["T2"], raw["T3"], spec)
    assert np.allclose(k2, raw["K2"], atol=1e-12)
    assert np.allclose(k3, raw["K3"], atol=1e-12)


def test_constructive_separable_mixtures_have_no_false_certificate() -> None:
    for family in ("random_product", "classical_correlated_separable_mixture", "noisy_separable_mixture"):
        state, metadata = generate_family_state(family, 82731)
        result = hierarchical_certificates(state, strict_separable=True)
        assert metadata["constructive_proof"].startswith("convex_mixture_of_")
        assert result["pairs"].npt_margin <= 1e-6
        assert result["triples"] is not None and not result["triples"].certified
        assert result["fours"] is not None and not result["fours"].certified

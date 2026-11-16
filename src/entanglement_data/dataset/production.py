from __future__ import annotations

import json
import math
import copy
import shutil
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import h5py

from ..certification.ppt import (
    NPTCertificate,
    apply_channel_to_two_site_rdms,
    certify_two_site_npt,
    certify_two_site_rdms,
    mps_two_site_density_matrices,
)
from ..observables.engine import compute_features
from ..observables.feature_spec import FeatureSpec
from ..states.mps import PurifiedMPS, random_mps
from ..states.noise import apply_local_noise, channel_kraus, haar_local_unitaries
from ..states.separable import SeparableMixture
from .diagnostics import validate_hdf5
from .generator import resolve_device
from .writer import write_dataset


def numeric_summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    q = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q05": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "q95": float(q[4]),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _largest_remainder_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    norm = sum(proportions.values())
    raw = {key: total * value / norm for key, value in proportions.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _schedule(total: int, proportions: dict[str, float], seed: int) -> tuple[list[str], dict[str, int]]:
    counts = _largest_remainder_counts(total, proportions)
    values = [key for key, count in counts.items() for _ in range(count)]
    random.Random(seed).shuffle(values)
    return values, counts


def _balanced_schedule(total: int, names: list[str], seed: int) -> list[str]:
    proportions = {name: 1.0 for name in names}
    return _schedule(total, proportions, seed)[0]


def _split_schedule(total: int, seed: int, proportions: dict[str, float]) -> list[str]:
    return _schedule(total, proportions, seed)[0]


def _feature_diagnostics(values: np.ndarray, N: int) -> dict[str, float]:
    two_start = 3 * N
    two_count = 9 * N * (N - 1) // 2
    two = values[two_start : two_start + two_count]
    nearest: list[float] = []
    cursor = two_start
    for i in range(N):
        for j in range(i + 1, N):
            block = values[cursor : cursor + 9]
            if j == i + 1:
                nearest.append(float(np.linalg.norm(block)))
            cursor += 9
    return {
        "correlation_mean": float(values.mean()),
        "correlation_std": float(values.std()),
        "correlation_l2_norm": float(np.linalg.norm(values)),
        "correlation_rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        "correlation_abs_max": float(np.abs(values).max()),
        "mean_abs_two_body_correlation": float(np.mean(np.abs(two))),
        "nearest_neighbor_correlation_norm": float(np.mean(nearest)),
    }


def _mixture_diagnostics(weights: torch.Tensor) -> tuple[float, float]:
    w = weights.detach().cpu().double().numpy()
    k_eff = float(1.0 / np.square(w).sum())
    entropy = float(-(w[w > 0] * np.log(w[w > 0])).sum())
    return k_eff, entropy


def _draw_dirichlet(K: int, alpha: float, seed: int, device: torch.device) -> torch.Tensor:
    values = np.random.default_rng(seed).dirichlet(np.full(K, alpha, dtype=np.float64))
    return torch.tensor(values, dtype=torch.float64, device=device)


def _metric_separation(label0: np.ndarray, label1: np.ndarray) -> dict[str, float]:
    x0 = np.asarray(label0, dtype=np.float64)
    x1 = np.asarray(label1, dtype=np.float64)
    x0, x1 = x0[np.isfinite(x0)], x1[np.isfinite(x1)]
    if not len(x0) or not len(x1):
        return {"auc_separation": 1.0, "histogram_overlap": 0.0}
    combined = np.concatenate((x0, x1))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    auc = (ranks[len(x0) :].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x0) * len(x1))
    auc_separation = float(max(auc, 1.0 - auc))
    low, high = float(combined.min()), float(combined.max())
    if high == low:
        overlap = 1.0
    else:
        bins = np.linspace(low, high, 51)
        h0 = np.histogram(x0, bins=bins)[0] / len(x0)
        h1 = np.histogram(x1, bins=bins)[0] / len(x1)
        overlap = float(np.minimum(h0, h1).sum())
    return {"auc_separation": auc_separation, "histogram_overlap": overlap}


def shortcut_diagnostics(metadata: list[dict[str, Any]], auc_limit: float = 0.90) -> dict[str, Any]:
    metrics = [
        "purity",
        "mean_local_bloch_norm",
        "correlation_rms",
        "mean_abs_two_body_correlation",
        "nearest_neighbor_correlation_norm",
    ]
    result: dict[str, Any] = {}
    failures: list[str] = []
    for metric in metrics:
        zero = np.asarray([row[metric] for row in metadata if row["label"] == 0], dtype=np.float64)
        one = np.asarray([row[metric] for row in metadata if row["label"] == 1], dtype=np.float64)
        separation = _metric_separation(zero, one)
        separation["label_0_summary"] = numeric_summary(zero)
        separation["label_1_summary"] = numeric_summary(one)
        result[metric] = separation
        if separation["auc_separation"] > auc_limit:
            failures.append(
                f"{metric}: single-threshold AUC separation {separation['auc_separation']:.3f} exceeds {auc_limit:.3f}"
            )
    result["auc_limit"] = auc_limit
    result["passed"] = not failures
    result["failures"] = failures
    return result


def _metadata_report(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {
        str(label): [row for row in metadata if row["label"] == label]
        for label in (0, 1)
    }
    def summary_by_label(key: str) -> dict[str, Any]:
        return {label: numeric_summary([row[key] for row in rows]) for label, rows in by_label.items()}

    return {
        "chi_histogram": dict(sorted(Counter(str(row["requested_chi_max"]) for row in by_label["1"]).items())),
        "K_histogram": dict(sorted(Counter(str(row["K"]) for row in by_label["0"]).items(), key=lambda x: int(x[0]))),
        "K_eff_summary": numeric_summary([row["K_eff"] for row in by_label["0"]]),
        "dirichlet_alpha_summary": numeric_summary([row["dirichlet_alpha"] for row in by_label["0"]]),
        "mixture_weight_entropy_summary": numeric_summary([row["mixture_weight_entropy"] for row in by_label["0"]]),
        "generator_type_counts": dict(Counter(row["generator_type"] for row in metadata)),
        "noise_type_counts": dict(Counter(row["noise_type"] for row in by_label["1"])),
        "noise_regime_counts": dict(Counter(row["noise_regime"] for row in by_label["1"])),
        "noise_strength_summary": numeric_summary([row["noise_strength"] for row in by_label["1"]]),
        "purity_summary_by_label": summary_by_label("purity"),
        "mean_local_bloch_norm_summary_by_label": summary_by_label("mean_local_bloch_norm"),
        "correlation_norm_summary_by_label": summary_by_label("correlation_l2_norm"),
        "correlation_rms_summary_by_label": summary_by_label("correlation_rms"),
        "mean_abs_two_body_summary_by_label": summary_by_label("mean_abs_two_body_correlation"),
        "nearest_neighbor_correlation_norm_summary_by_label": summary_by_label("nearest_neighbor_correlation_norm"),
        "npt_min_eigenvalue_summary": numeric_summary([row["npt_min_eigenvalue"] for row in by_label["1"]]),
        "split_counts": dict(Counter(f"label_{row['label']}:{row['split']}" for row in metadata)),
    }


def _noise_at_boundary_fraction(
    base: PurifiedMPS,
    noise_type: str,
    regime: str,
    rng: random.Random,
    *,
    max_distance: int,
    tolerance: float,
    bisection_steps: int,
    regime_ranges: dict[str, list[float]],
) -> tuple[PurifiedMPS, NPTCertificate, float, float, int] | None:
    base_rdms = mps_two_site_density_matrices(base, max_distance=max_distance)
    evaluations = 1
    base_certificate = certify_two_site_rdms(base_rdms, tolerance=tolerance)
    if not base_certificate.certified:
        return None
    def certificate_at(strength: float) -> NPTCertificate:
        kraus = channel_kraus(noise_type, strength, device=base.device, dtype=base.dtype)
        noisy_rdms = apply_channel_to_two_site_rdms(base_rdms, kraus)
        return certify_two_site_rdms(noisy_rdms, tolerance=tolerance)

    low, high = 0.0, 1.0
    for _ in range(bisection_steps):
        mid = (low + high) / 2
        certificate = certificate_at(mid)
        evaluations += 1
        if certificate.certified:
            low = mid
        else:
            high = mid
    lower_fraction, upper_fraction = regime_ranges[regime]
    strength = max(1e-6, low * rng.uniform(float(lower_fraction), float(upper_fraction)))
    state = apply_local_noise(base, noise_type, strength)
    certificate = certificate_at(strength)
    evaluations += 1
    if not certificate.certified:
        return None
    # One full purification-MPS cross-check guards the RDM channel shortcut.
    full_certificate = certify_two_site_npt(state, max_distance=max_distance, tolerance=tolerance)
    if not full_certificate.certified or abs(full_certificate.min_eigenvalue - certificate.min_eigenvalue) > 1e-8:
        raise RuntimeError("RDM-channel NPT certificate disagrees with full purification MPS")
    return state, full_certificate, strength, low, evaluations


def generate_production_dataset(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    N = int(config["system"]["N"])
    seed0 = int(config["master_seed"])
    device = resolve_device(config["runtime"].get("device", "cpu"))
    if device.type != "cpu":
        raise ValueError("production v1 is explicitly configured for CPU")
    num_sep = int(config["dataset"]["num_separable"])
    num_ent = int(config["dataset"]["num_entangled"])
    total_samples = num_sep + num_ent
    spec = FeatureSpec(N)
    feature_names = spec.names()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    progress_every = int(config["runtime"].get("progress_every", 100))

    temp_features = output / ".features.float32.tmp"
    X = np.memmap(temp_features, mode="w+", dtype=np.float32, shape=(total_samples, spec.num_features))
    y = np.empty(total_samples, dtype=np.int8)
    metadata: list[dict[str, Any]] = []
    row_index = 0
    generation_total = 0.0
    feature_total = 0.0

    split_cfg = config["splits"]
    sep_splits = _split_schedule(num_sep, seed0 + 501, split_cfg)
    ent_splits = _split_schedule(num_ent, seed0 + 502, split_cfg)
    sep_schedule, requested_sep_counts = _schedule(num_sep, config["separable"]["type_proportions"], seed0 + 601)
    k_low, k_high = map(int, config["separable"]["K_integer_range"])
    alpha_log_low, alpha_log_high = map(float, config["separable"]["log10_alpha_range"])

    for index, generator_type in enumerate(sep_schedule):
        sample_seed = seed0 + 1_000_003 * (index + 1)
        rng = random.Random(sample_seed)
        tg = torch.Generator(device=device).manual_seed(sample_seed)
        t0 = time.perf_counter()
        if generator_type == "product_state":
            state = SeparableMixture.random(N, 1, generator=tg, device=device)
            K, alpha = 1, float("nan")
        elif generator_type == "classical_ghz_mixture":
            state = SeparableMixture.classical_ghz_mixture(N, generator=tg, device=device)
            K, alpha = 2, float("nan")
        elif generator_type == "separable_convex_mixture":
            K = rng.randint(k_low, k_high)
            alpha = 10 ** rng.uniform(alpha_log_low, alpha_log_high)
            state = SeparableMixture.random(N, K, generator=tg, device=device)
            weights = _draw_dirichlet(K, alpha, sample_seed + 17, device)
            state = SeparableMixture(weights, state.bloch)
        else:
            raise ValueError(f"unknown separable generator {generator_type}")
        unitaries, rotations = haar_local_unitaries(N, generator=tg, device=device)
        del unitaries
        state = state.apply_rotations(rotations)
        generation_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        feature_result = compute_features(state, spec)
        feature_seconds = time.perf_counter() - t1
        values = feature_result.values
        local = values[: 3 * N].reshape(N, 3)
        k_eff, entropy = _mixture_diagnostics(state.weights)
        diagnostics = _feature_diagnostics(values, N)
        state_id = f"sep-{index:08d}"
        row = {
            "state_id": state_id,
            "parent_state_id": state_id,
            "N": N,
            "label": 0,
            "split": sep_splits[index],
            "certificate_type": "constructive_fully_separable",
            "generator_type": generator_type,
            "seed": sample_seed,
            "requested_chi_max": 1,
            "bond_dimension": 1,
            "actual_bond_dimensions": ",".join(["1"] * (N + 1)),
            "K": K,
            "dirichlet_alpha": alpha,
            "K_eff": k_eff,
            "mixture_weight_entropy": entropy,
            "noise_type": "none",
            "noise_regime": "none",
            "noise_strength": 0.0,
            "estimated_npt_boundary_strength": float("nan"),
            "local_unitary_applied": True,
            "purity": state.purity(),
            "mean_local_bloch_norm": float(np.linalg.norm(local, axis=1).mean()),
            "npt_min_eigenvalue": float("nan"),
            "npt_pair": "",
            "generation_seconds": generation_seconds,
            "feature_seconds": feature_seconds,
            **diagnostics,
        }
        X[row_index] = values
        y[row_index] = 0
        metadata.append(row)
        row_index += 1
        generation_total += generation_seconds
        feature_total += feature_seconds
        peak_rss = max(peak_rss, process.memory_info().rss)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"separable: {index + 1}/{num_sep}", flush=True)

    ent_cfg = config["entangled"]
    chi_low, chi_high = map(int, ent_cfg["chi_integer_range"])
    noise_schedule = _balanced_schedule(num_ent, list(ent_cfg["noise_types"]), seed0 + 701)
    regime_schedule, requested_regime_counts = _schedule(num_ent, ent_cfg["noise_regime_proportions"], seed0 + 702)
    cert_cfg = config["certification"]
    accepted = attempts = probe_evaluations = 0
    while accepted < num_ent:
        attempts += 1
        if attempts > int(ent_cfg["max_candidate_multiplier"] * num_ent):
            raise RuntimeError(f"only accepted {accepted}/{num_ent} after {attempts} candidates")
        sample_seed = seed0 + 10_000_019 * attempts + 97
        rng = random.Random(sample_seed)
        tg = torch.Generator(device=device).manual_seed(sample_seed)
        requested_chi = rng.randint(chi_low, chi_high)
        noise_type = noise_schedule[accepted]
        regime = regime_schedule[accepted]
        t0 = time.perf_counter()
        base = random_mps(N, requested_chi, generator=tg, device=device)
        actual_bonds = base.bond_dimensions
        unitaries, _ = haar_local_unitaries(N, generator=tg, device=device)
        base = base.apply_local_unitaries(unitaries)
        result = _noise_at_boundary_fraction(
            base, noise_type, regime, rng,
            max_distance=int(cert_cfg["max_distance"]),
            tolerance=float(cert_cfg["tolerance"]),
            bisection_steps=int(ent_cfg["boundary_bisection_steps"]),
            regime_ranges=ent_cfg["noise_regime_boundary_fraction_ranges"],
        )
        if result is None:
            generation_total += time.perf_counter() - t0
            continue
        state, certificate, strength, boundary, evaluations = result
        probe_evaluations += evaluations
        purity = state.purity()
        generation_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        feature_result = compute_features(state, spec)
        feature_seconds = time.perf_counter() - t1
        values = feature_result.values
        local = values[: 3 * N].reshape(N, 3)
        diagnostics = _feature_diagnostics(values, N)
        state_id = f"ent-{accepted:08d}"
        row = {
            "state_id": state_id,
            "parent_state_id": state_id,
            "N": N,
            "label": 1,
            "split": ent_splits[accepted],
            "certificate_type": "two_site_npt",
            "generator_type": "noisy_random_mps",
            "seed": sample_seed,
            "requested_chi_max": requested_chi,
            "bond_dimension": requested_chi,
            "actual_bond_dimensions": ",".join(map(str, actual_bonds)),
            "K": 0,
            "dirichlet_alpha": float("nan"),
            "K_eff": float("nan"),
            "mixture_weight_entropy": float("nan"),
            "noise_type": noise_type,
            "noise_regime": regime,
            "noise_strength": strength,
            "estimated_npt_boundary_strength": boundary,
            "local_unitary_applied": True,
            "purity": purity,
            "mean_local_bloch_norm": float(np.linalg.norm(local, axis=1).mean()),
            "npt_min_eigenvalue": certificate.min_eigenvalue,
            "npt_pair": f"{certificate.pair[0]},{certificate.pair[1]}",
            "generation_seconds": generation_seconds,
            "feature_seconds": feature_seconds,
            **diagnostics,
        }
        X[row_index] = values
        y[row_index] = 1
        metadata.append(row)
        row_index += 1
        accepted += 1
        generation_total += generation_seconds
        feature_total += feature_seconds
        peak_rss = max(peak_rss, process.memory_info().rss)
        if progress_every and accepted % progress_every == 0:
            print(f"entangled: {accepted}/{num_ent} (candidates={attempts})", flush=True)

    X.flush()
    h5_path = write_dataset(output, X, y, metadata, feature_names, config)
    validation = validate_hdf5(
        h5_path,
        spec.num_features,
        expected_samples=total_samples,
        expected_per_label=num_sep,
        feature_names_path=output / "feature_names.json",
        strict_certificates=True,
    ) if num_sep == num_ent else validate_hdf5(
        h5_path,
        spec.num_features,
        expected_samples=total_samples,
        feature_names_path=output / "feature_names.json",
        strict_certificates=True,
    )
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "N": N,
        "num_features": spec.num_features,
        "num_separable": num_sep,
        "num_entangled": num_ent,
        "total_accepted_samples": total_samples,
        "total_candidate_entangled_states": attempts,
        "rejected_unknown_states": attempts - accepted,
        "npt_acceptance_rate": accepted / attempts,
        "npt_probe_evaluations": probe_evaluations,
        "total_runtime_seconds": elapsed,
        "generation_seconds_per_candidate": generation_total / (num_sep + attempts),
        "feature_extraction_seconds_per_accepted_state": feature_total / total_samples,
        "peak_rss_bytes": peak_rss,
        "dataset_file_size_bytes": h5_path.stat().st_size,
        "master_seed": seed0,
        "requested_separable_type_counts": requested_sep_counts,
        "requested_noise_regime_counts": requested_regime_counts,
        "validation": validation,
        **_metadata_report(metadata),
    }
    report["shortcut_diagnostics"] = shortcut_diagnostics(
        metadata, auc_limit=float(config["pilot_gate"]["max_single_metric_auc_separation"])
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    del X
    temp_features.unlink(missing_ok=True)
    return report


def _production_shard_worker(payload: tuple[int, dict[str, Any], str]) -> tuple[int, dict[str, Any]]:
    shard_id, config, output = payload
    torch.set_num_threads(1)
    report = generate_production_dataset(config, output)
    return shard_id, report


def _read_shard_metadata(h5: h5py.File) -> list[dict[str, Any]]:
    group = h5["metadata"]
    arrays: dict[str, np.ndarray] = {}
    for key, dataset in group.items():
        arrays[key] = dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) is not None else dataset[:]
    rows: list[dict[str, Any]] = []
    for index in range(h5["labels/y"].shape[0]):
        row: dict[str, Any] = {}
        for key, array in arrays.items():
            value = array[index]
            row[key] = value.item() if isinstance(value, np.generic) else value
        rows.append(row)
    return rows


def generate_parallel_production_dataset(
    config: dict[str, Any], output_dir: str | Path, *, workers: int = 2
) -> dict[str, Any]:
    """Generate deterministic shards in parallel, then merge and globally re-split."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return generate_production_dataset(config, output_dir)
    started = time.perf_counter()
    output = Path(output_dir)
    shard_root = output / "_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    total_sep = int(config["dataset"]["num_separable"])
    total_ent = int(config["dataset"]["num_entangled"])
    sep_counts = [total_sep // workers + (i < total_sep % workers) for i in range(workers)]
    ent_counts = [total_ent // workers + (i < total_ent % workers) for i in range(workers)]
    master_seed = int(config["master_seed"])
    payloads: list[tuple[int, dict[str, Any], str]] = []
    shard_seeds: list[int] = []
    for shard_id in range(workers):
        shard_config = copy.deepcopy(config)
        shard_seed = master_seed + (shard_id + 1) * 1_000_000_007
        shard_seeds.append(shard_seed)
        shard_config["master_seed"] = shard_seed
        shard_config["dataset"]["num_separable"] = sep_counts[shard_id]
        shard_config["dataset"]["num_entangled"] = ent_counts[shard_id]
        shard_config["runtime"]["progress_every"] = max(100, min(sep_counts[shard_id], ent_counts[shard_id]) // 10)
        payloads.append((shard_id, shard_config, str(shard_root / f"shard_{shard_id:02d}")))
    reports: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_production_shard_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            shard_id, report = future.result()
            reports[shard_id] = report
            print(f"production shard {shard_id + 1}/{workers} completed", flush=True)

    N = int(config["system"]["N"])
    spec = FeatureSpec(N)
    total = total_sep + total_ent
    temp_features = output / ".features.merge.float32.tmp"
    X = np.memmap(temp_features, mode="w+", dtype=np.float32, shape=(total, spec.num_features))
    y = np.empty(total, dtype=np.int8)
    metadata: list[dict[str, Any]] = []
    cursor = 0
    for shard_id in range(workers):
        shard_file = shard_root / f"shard_{shard_id:02d}" / "dataset.h5"
        with h5py.File(shard_file, "r") as h5:
            count = h5["labels/y"].shape[0]
            for start in range(0, count, 256):
                end = min(count, start + 256)
                X[cursor + start : cursor + end] = h5["features/X"][start:end]
            y[cursor : cursor + count] = h5["labels/y"][:]
            rows = _read_shard_metadata(h5)
        for row in rows:
            row["state_id"] = f"shard{shard_id:02d}-{row['state_id']}"
            row["parent_state_id"] = f"shard{shard_id:02d}-{row['parent_state_id']}"
            row["shard_id"] = shard_id
        metadata.extend(rows)
        cursor += count
    if cursor != total:
        raise RuntimeError(f"merged {cursor} rows, expected {total}")

    # Reassign exact global stratified splits after shard merge.
    label_positions = {label: [i for i, row in enumerate(metadata) if row["label"] == label] for label in (0, 1)}
    for label in (0, 1):
        split_values = _split_schedule(len(label_positions[label]), master_seed + 900 + label, config["splits"])
        for position, split in zip(label_positions[label], split_values):
            metadata[position]["split"] = split

    X.flush()
    h5_path = write_dataset(output, X, y, metadata, spec.names(), config)
    validation = validate_hdf5(
        h5_path,
        spec.num_features,
        expected_samples=total,
        expected_per_label=total_sep,
        feature_names_path=output / "feature_names.json",
        strict_certificates=True,
    ) if total_sep == total_ent else validate_hdf5(
        h5_path,
        spec.num_features,
        expected_samples=total,
        feature_names_path=output / "feature_names.json",
        strict_certificates=True,
    )
    total_candidates = sum(report["total_candidate_entangled_states"] for report in reports.values())
    generation_seconds = sum(
        report["generation_seconds_per_candidate"]
        * (report["num_separable"] + report["total_candidate_entangled_states"])
        for report in reports.values()
    )
    feature_seconds = sum(
        report["feature_extraction_seconds_per_accepted_state"] * report["total_accepted_samples"]
        for report in reports.values()
    )
    report: dict[str, Any] = {
        "N": N,
        "num_features": spec.num_features,
        "num_separable": total_sep,
        "num_entangled": total_ent,
        "total_accepted_samples": total,
        "total_candidate_entangled_states": total_candidates,
        "rejected_unknown_states": total_candidates - total_ent,
        "npt_acceptance_rate": total_ent / total_candidates,
        "npt_probe_evaluations": sum(r["npt_probe_evaluations"] for r in reports.values()),
        "total_runtime_seconds": time.perf_counter() - started,
        "generation_seconds_per_candidate": generation_seconds / (total_sep + total_candidates),
        "feature_extraction_seconds_per_accepted_state": feature_seconds / total,
        "peak_rss_bytes": sum(r["peak_rss_bytes"] for r in reports.values()),
        "peak_rss_note": "sum of per-worker peak RSS; conservative concurrent aggregate estimate",
        "dataset_file_size_bytes": h5_path.stat().st_size,
        "master_seed": master_seed,
        "shard_master_seeds": shard_seeds,
        "parallel_workers": workers,
        "validation": validation,
        **_metadata_report(metadata),
    }
    report["shortcut_diagnostics"] = shortcut_diagnostics(
        metadata, auc_limit=float(config["pilot_gate"]["max_single_metric_auc_separation"])
    )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    del X
    temp_features.unlink(missing_ok=True)
    if validation["valid"] and not bool(config["runtime"].get("keep_shards", False)):
        shutil.rmtree(shard_root)
    return report

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from ..certification.ppt import certify_two_site_npt
from ..observables.engine import compute_features
from ..observables.feature_spec import FeatureSpec
from ..states.mps import random_mps
from ..states.noise import apply_local_noise, haar_local_unitaries
from ..states.separable import SeparableMixture
from .diagnostics import feature_statistics, validate_hdf5
from .writer import write_dataset


@dataclass
class GenerationReport:
    output_file: str
    num_samples: int
    separable_samples: int
    entangled_samples: int
    entangled_candidates: int
    npt_pass_rate: float
    total_seconds: float
    generation_seconds_per_state: float
    feature_seconds_per_state: float
    peak_rss_bytes: int
    file_size_bytes: int
    validation: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _torch_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _random_choice(rng: random.Random, choices: list[Any]) -> Any:
    return choices[rng.randrange(len(choices))]


def _sample_metadata(
    *,
    state_id: str,
    N: int,
    label: int,
    certificate_type: str,
    generator_type: str,
    seed: int,
    bond_dimension: int,
    K: int,
    noise_type: str,
    noise_strength: float,
    local_unitary_applied: bool,
    purity: float,
    mean_local_bloch_norm: float,
    features: np.ndarray,
    npt_min_eigenvalue: float,
    npt_pair: str,
    generation_seconds: float,
    feature_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "state_id": state_id,
        "N": N,
        "label": label,
        "certificate_type": certificate_type,
        "generator_type": generator_type,
        "seed": seed,
        "bond_dimension": bond_dimension,
        "K": K,
        "noise_type": noise_type,
        "noise_strength": noise_strength,
        "local_unitary_applied": local_unitary_applied,
        "purity": purity,
        "mean_local_bloch_norm": mean_local_bloch_norm,
        "npt_min_eigenvalue": npt_min_eigenvalue,
        "npt_pair": npt_pair,
        "generation_seconds": generation_seconds,
        "feature_seconds": feature_seconds,
    }
    row.update(feature_statistics(features))
    return row


def generate_dataset(config: dict[str, Any], output_dir: str | Path | None = None) -> GenerationReport:
    started = time.perf_counter()
    N = int(config["system"]["N"])
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    seed0 = int(config.get("seed", 20260809))
    rng = random.Random(seed0)
    num_sep = int(config["dataset"]["num_separable"])
    num_ent = int(config["dataset"]["num_entangled"])
    output_dir = Path(output_dir or config["dataset"]["output_dir"])
    spec = FeatureSpec(N=N)
    connected = bool(config.get("features", {}).get("connected", False))
    progress_every = int(config.get("runtime", {}).get("progress_every", 0))
    local_u_probability = float(config.get("augmentation", {}).get("local_unitary_probability", 0.5))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    features: list[np.ndarray] = []
    connected_features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    names: list[str] | None = None
    connected_names: list[str] | None = None
    generation_total = 0.0
    feature_total = 0.0

    sep_cfg = config["separable"]
    k_choices = [int(k) for k in sep_cfg["K_choices"]]
    product_probability = float(sep_cfg.get("product_probability", 0.25))
    classical_ghz_probability = float(sep_cfg.get("classical_ghz_probability", 0.15))
    if product_probability + classical_ghz_probability > 1.0:
        raise ValueError("separable generator probabilities sum to more than one")
    for index in range(num_sep):
        sample_seed = seed0 + 1_000_003 * (index + 1)
        tg = _torch_generator(sample_seed, device)
        t0 = time.perf_counter()
        kind_draw = rng.random()
        if kind_draw < product_probability:
            K = 1
            generator_type = "product_state"
            state = SeparableMixture.random(N, K, generator=tg, device=device)
        elif kind_draw < product_probability + classical_ghz_probability:
            K = 2
            generator_type = "classical_ghz_mixture"
            state = SeparableMixture.classical_ghz_mixture(N, generator=tg, device=device)
        else:
            K = _random_choice(rng, k_choices)
            generator_type = "separable_convex_mixture"
            state = SeparableMixture.random(N, K, generator=tg, device=device)
        local_u = rng.random() < local_u_probability
        if local_u:
            _, rotations = haar_local_unitaries(N, generator=tg, device=device)
            state = state.apply_rotations(rotations)
        generation_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        result = compute_features(state, spec, connected=connected)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        feature_seconds = time.perf_counter() - t1
        names = result.names
        connected_names = result.connected_names
        features.append(result.values)
        if result.connected_values is not None:
            connected_features.append(result.connected_values)
        labels.append(0)
        metadata.append(
            _sample_metadata(
                state_id=f"sep-{index:08d}", N=N, label=0, certificate_type="constructive_fully_separable",
                generator_type=generator_type, seed=sample_seed,
                bond_dimension=1, K=K, noise_type="none", noise_strength=0.0,
                local_unitary_applied=local_u, purity=state.purity(),
                mean_local_bloch_norm=state.mean_local_bloch_norm(), features=result.values,
                npt_min_eigenvalue=float("nan"), npt_pair="", generation_seconds=generation_seconds,
                feature_seconds=feature_seconds,
            )
        )
        generation_total += generation_seconds
        feature_total += feature_seconds
        peak_rss = max(peak_rss, process.memory_info().rss)
        if progress_every and (index + 1) % progress_every == 0:
            print(f"separable: {index + 1}/{num_sep}", flush=True)

    ent_cfg = config["entangled"]
    chi_choices = [int(v) for v in ent_cfg["chi_choices"]]
    noise_types = list(ent_cfg["noise_types"])
    noise_probability = float(ent_cfg.get("noise_probability", 0.75))
    noise_min, noise_max = map(float, ent_cfg.get("noise_strength_range", [0.0, 0.15]))
    max_attempts = int(ent_cfg.get("max_attempts", max(100, 20 * num_ent)))
    cert_distance = int(config["certification"].get("max_distance", 1))
    cert_tolerance = float(config["certification"].get("tolerance", 1e-9))
    accepted = 0
    attempts = 0
    while accepted < num_ent and attempts < max_attempts:
        attempts += 1
        sample_seed = seed0 + 10_000_019 * attempts + 97
        tg = _torch_generator(sample_seed, device)
        chi = _random_choice(rng, chi_choices)
        t0 = time.perf_counter()
        state = random_mps(N, chi, generator=tg, device=device)
        noise_type = "none"
        noise_strength = 0.0
        if rng.random() < noise_probability:
            noise_type = _random_choice(rng, noise_types)
            noise_strength = rng.uniform(noise_min, noise_max)
            state = apply_local_noise(state, noise_type, noise_strength)
        local_u = rng.random() < local_u_probability
        if local_u:
            unitaries, _ = haar_local_unitaries(N, generator=tg, device=device)
            state = state.apply_local_unitaries(unitaries)
        certificate = certify_two_site_npt(state, max_distance=cert_distance, tolerance=cert_tolerance)
        generation_seconds = time.perf_counter() - t0
        generation_total += generation_seconds
        if not certificate.certified:
            del state
            continue
        t1 = time.perf_counter()
        result = compute_features(state, spec, connected=connected)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        feature_seconds = time.perf_counter() - t1
        feature_total += feature_seconds
        names = result.names
        connected_names = result.connected_names
        features.append(result.values)
        if result.connected_values is not None:
            connected_features.append(result.connected_values)
        labels.append(1)
        local_bloch = result.values[: 3 * N].reshape(N, 3)
        metadata.append(
            _sample_metadata(
                state_id=f"ent-{accepted:08d}", N=N, label=1, certificate_type="two_site_npt",
                generator_type="random_mps" if noise_type == "none" else "noisy_random_mps",
                seed=sample_seed, bond_dimension=chi, K=0, noise_type=noise_type,
                noise_strength=noise_strength, local_unitary_applied=local_u, purity=float("nan"),
                mean_local_bloch_norm=float(np.linalg.norm(local_bloch, axis=1).mean()),
                features=result.values, npt_min_eigenvalue=certificate.min_eigenvalue,
                npt_pair=f"{certificate.pair[0]},{certificate.pair[1]}" if certificate.pair else "",
                generation_seconds=generation_seconds, feature_seconds=feature_seconds,
            )
        )
        accepted += 1
        peak_rss = max(peak_rss, process.memory_info().rss)
        del state
        if progress_every and accepted % progress_every == 0:
            print(f"entangled: {accepted}/{num_ent} (candidates={attempts})", flush=True)
    if accepted < num_ent:
        raise RuntimeError(
            f"only obtained {accepted}/{num_ent} NPT-certified states after {attempts} candidates; "
            "increase max_attempts or reduce the noise range"
        )

    assert names is not None
    X = np.stack(features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int8)
    connected_X = np.stack(connected_features).astype(np.float32, copy=False) if connected_features else None
    h5_path = write_dataset(
        output_dir, X, y, metadata, names, config, connected_X=connected_X, connected_names=connected_names
    )
    validation = validate_hdf5(h5_path, spec.num_features)
    total = time.perf_counter() - started
    report = GenerationReport(
        output_file=str(h5_path.resolve()), num_samples=num_sep + num_ent, separable_samples=num_sep,
        entangled_samples=num_ent, entangled_candidates=attempts,
        npt_pass_rate=(accepted / attempts if attempts else 0.0), total_seconds=total,
        generation_seconds_per_state=generation_total / max(1, num_sep + attempts),
        feature_seconds_per_state=feature_total / max(1, num_sep + num_ent), peak_rss_bytes=peak_rss,
        file_size_bytes=h5_path.stat().st_size, validation=validation,
    )
    (output_dir / "report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report

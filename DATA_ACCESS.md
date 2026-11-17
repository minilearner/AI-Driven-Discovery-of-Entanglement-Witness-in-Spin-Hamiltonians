# Benchmark data and access

The six HDF5 files below are the exact final test sets used for the paper's
system-size, synthetic-state, separable-control, and unseen-Hamiltonian
results. They are **not in Git history**. The single archive is published as
[GitHub Release benchmarks-v1](https://github.com/minilearner/AI-Driven-Discovery-of-Entanglement-Witness-in-Spin-Hamiltonians/releases/tag/benchmarks-v1).
The real public asset URL is recorded in `data/benchmark_manifest.json`.

| Benchmark | Path after installation | Rows | Size (MB) | Classes (Sep/E2/E3/E4/Unknown) |
| --- | --- | ---: | ---: | --- |
| XXZ adaptive Gibbs | `xxz/dataset.h5` | 343 | 0.45 | 79/103/52/30/79 |
| J1–J2 adaptive Gibbs | `j1j2/dataset.h5` | 463 | 0.72 | 30/155/80/30/168 |
| Random local Hamiltonian | `random_local_h/dataset.h5` | 626 | 2.16 | 39/238/87/40/222 |
| Matched N=32 XY | `n32_xy/dataset.h5` | 500 | 3.03 | 20/312/80/40/48 |
| Independent N=32 synthetic | `n32_synthetic/dataset.h5` | 2,000 | 45.31 | 400/400/600/600/0 |
| General-Sep control | `general_sep/dataset.h5` | 2,000 | 64.64 | 2000/0/0/0/0 |

The machine-readable manifest specifies each file's byte count, SHA-256,
schema, generator, paper experiment, and reference metrics. The source HDF5
files and their labels were checked against the private preserved originals;
the package contains no unrelated historical OOD variants. A scan of HDF5
string values and attributes found no credentials, email addresses, or
personal paths. The package contains 5,932 rows; its compressed size is
108,648,418 bytes and SHA-256 is
`dc8bf42f7ac56b97f1f477a86e0300bd7f35ca14727522211cd4bad9a96027ca`.

## Install and verify

Download the public archive, verify it, and install all six datasets:

```bash
python scripts/download_benchmarks.py
```

If the verified archive is already available locally, use:

```bash
python scripts/download_benchmarks.py --archive-file /path/to/entnet-benchmarks-v1.tar.gz
```

The downloader checks free space, reports progress, verifies the archive and
every HDF5 SHA-256, skips files already verified, and extracts to
`data/benchmarks/`. A bad or incomplete download raises an error rather than
silently entering evaluation. The archive preparation source is
`scripts/prepare_benchmark_release.py`; it is for release maintainers with
access to the preserved originals, not for routine users.

Then recompute all six benchmarks with one public entry point:

```bash
python scripts/reproduce_benchmarks.py --data-root data/benchmarks --output outputs/benchmarks
```

This loads the Dv1-only baseline checkpoints, verifies input hashes and
class counts, applies the correct HDF5 schema adapters, and fixes the
classification threshold at 0.5. Unknown states are excluded from labeled
metrics. The output contains per-row scores and a summary with deviations
from manuscript values. No private preservation manifest or prior score CSV
is needed.

## Rebuilding data from source

The full HDF5 regeneration code is retained for scientific inspection, but
several routes require substantial compute and intermediate physical states:

| Dataset | Generator | Main resource/constraint |
| --- | --- | --- |
| N=16 primary Dv1 | `scripts/generate_production.py`, `configs/production_n16_v1.yaml` | 50,000 certified states; pilot gate before production |
| Adaptive XXZ/J1–J2/random-H | `scripts/run_ood_hamiltonian_n16.py`, then `scripts/generate_adaptive_gibbs_ood.py` | Thermal calculations, 312-row base sets, adaptive certificate scan; optional TeNPy for some families |
| N=32 matched XY | `scripts/generate_xy_n32_size_transfer.py`, `configs/xy_n32_pbc_matched_exact500_v1.json` | Fixed source N=16 sample set and exact XY correlations |
| N=32 synthetic | `scripts/generate_dpi_n32_general.py`, `configs/dpi_n32_general_v1.json` | Independent state construction and certificates |
| General-Sep | `scripts/generate_general_separable_ood.py` | Constructive separability control |

Generator configs and the final benchmark HDF5 files serve different purposes:
the packaged files reproduce reported inference directly, while complete
data regeneration is a separate high-cost calculation. The 50,000-row Dv1
training HDF5 and large intermediate caches are not part of the six-benchmark
package. A 24-row genuine XY fixture in `tests/fixtures/` supports the quick
inference and training checks without these large inputs.

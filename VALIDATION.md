# Reproducibility verification

Verification on Windows with Python 3.10.20 (CPU) is recorded here. The
independent branch was cloned from the public GitHub URL into a new directory,
and a fresh virtual environment was created. The preserved private archive
was read to assemble the benchmark package; it was not edited. The package
was then downloaded anonymously from the public GitHub Release in a separate
check using the clean virtual environment, and verified before a second
six-benchmark inference run. All routine
inference, download, plotting, and smoke training entry points use
release-relative paths. An Ubuntu GitHub Actions run for the implementation
passed installation, 60 tests, model inference, the physical example,
the small HDF5 metric check, and the final-route training smoke:
[run 44020776648](https://github.com/minilearner/AI-Driven-Discovery-of-Entanglement-Witness-in-Spin-Hamiltonians/actions/runs/44020776648).

## Results checked

| Experiment | Public source | Model inference | Figure | Result |
| --- | --- | --- | --- | --- |
| XXZ adaptive Gibbs, 343 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | Aggregate metrics match |
| J1–J2 adaptive Gibbs, 463 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | Aggregate metrics match |
| Random local Hamiltonian, 626 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | Aggregate metrics match |
| Matched N=32 XY, 500 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | Aggregate metrics match |
| Independent N=32 synthetic, 2,000 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | Aggregate metrics match |
| General-Sep, 2,000 rows | Release HDF5 | Baseline ensemble verified | Fig. 2 | TNR matches |
| XY temperature traces and certificates | Committed compact data | Frozen baseline scores | Fig. 3(a,b) | Plotted; independent certificates preserved |
| XX thermal decision map and fit | Committed compact data | Frozen augmented predictions | Fig. 3(c,d) | Plotted; fit matches |

The six benchmark HDF5 files were checked by SHA-256 and class counts before
inference. From the fresh GitHub clone they were extracted by the public
downloader from the prepared local release archive, and the six benchmarks
were recomputed into ignored `outputs/benchmarks/`. Relative to the paper's rounded
reference numbers, the maximum absolute metric difference was `1.61e-5`
(J1–J2 E2 recall); all other differences were at most `8.74e-6`. The counts
of unknown rows were 79, 168, 222, 48, 0, and 0 in the table's order.
Unknown rows were excluded. The six-file archive was built, verified, and
installed using `scripts/download_benchmarks.py --archive-file ...`. It was
also downloaded anonymously in the separate public-release check with the default
`python scripts/download_benchmarks.py --data-root outputs/public_download_check`
command, SHA-256 verified and installed, then used for all six fresh benchmark
reports under `outputs/public_benchmarks/`. Running the downloader again
correctly skipped every verified file.

Both `plot_fig2.py` and `plot_fig3.py` completed in the fresh clone with only
compact committed sources and wrote PDFs to `outputs/figures/`. Fig. 3(c,d) uses 4,941 base-grid
points and 48,632 directly evaluated visible-window points. The 19-point
activated fit is `C_*=0.0020259765`, `R²(log10 T_*)=0.9994261`, and
`RMSE(log10 T_*)=0.0061410`. The thermal inference grid itself was obtained
earlier from exact XY correlations and the augmented ensemble; this release
test redrew the committed direct predictions without expensive recomputation.

## Quick start, training and automated checks

- `python -m pytest -q tests`: 60 tests passed in the fresh environment,
  covering physical states,
  certificates, exact XY correlations, LU features, and checkpoint inference.
- `python scripts/physical_example.py`: product-state score `0.00000389`;
  independently NPT-certified Bell-pair score `0.99998659`.
- `python scripts/evaluate_final.py --dataset tests/fixtures/physical_smoke_n16.h5
  --output outputs/smoke_evaluation.json`: 24 genuine XY rows; AUROC, E2
  recall, and TNR all `1.0` at fixed `p=0.5`. This tiny fixture is a workflow
  check, not a paper benchmark.
- `python scripts/smoke_train_dv1.py`: the actual
  final training entry point completed one CPU epoch using 16 training and
  eight validation rows, wrote a smoke-only checkpoint, and reloaded it.
  This is an integration test, not full-model retraining.
- The current CI workflow also runs the frozen model example, physical
  example, HDF5 metric check, and one-epoch final-route smoke on Ubuntu.

## Limits

Full 50,000-row primary-data generation, all selector training,
and full three-seed retraining were not repeated. The original training-time
Git commit remains unknown; the public implementation is supported by strict
checkpoint loading and saved-score compatibility, not byte-identical code
provenance. Some HDF5 score vectors differ slightly from older saved tables
while aggregate decisions and paper metrics agree. See `PROVENANCE.md`.

On this Windows machine, `pip install -e '.[test]'` failed with `WinError 206`
because the workspace and pip's temporary wheel path were unusually deep.
Direct installation of the same runtime and test dependencies in the fresh
venv succeeded, after which the entire quick-start, test, download, six-set
inference, figure, and smoke-training sequence passed. This is a Windows path
limit, not an unavailable Python package; the README gives a short-clone-path
recommendation and the tested dependency-only fallback. Ubuntu CI passed
the ordinary editable-install command.

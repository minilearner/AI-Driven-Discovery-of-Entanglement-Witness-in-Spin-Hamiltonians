# Reproducibility verification

Verification on Windows with Python 3.10.20 (CPU) is recorded here. A fresh
clone test and Linux CI status are reported separately after publication.
The preserved private archive was read to assemble the benchmark package;
it was not edited. All routine inference, download, plotting, and smoke
training entry points use release-relative paths.

## Results checked

| Experiment | Model inference from HDF5 | Figure redrawn from compact data | Result |
| --- | --- | --- | --- |
| XXZ adaptive Gibbs, 343 rows | Yes, baseline ensemble | Fig. 2 | Aggregate metrics match |
| J1–J2 adaptive Gibbs, 463 rows | Yes, baseline ensemble | Fig. 2 | Aggregate metrics match |
| Random local Hamiltonian, 626 rows | Yes, baseline ensemble | Fig. 2 | Aggregate metrics match |
| Matched N=32 XY, 500 rows | Yes, baseline ensemble | Fig. 2 | Aggregate metrics match |
| Independent N=32 synthetic, 2,000 rows | Yes, baseline ensemble | Fig. 2 | Aggregate metrics match |
| General-Sep, 2,000 rows | Yes, baseline ensemble | Fig. 2 | TNR matches |
| XY temperature traces and certificates | Previously frozen baseline inputs | Fig. 3(a,b) | Plotted; independent certificates preserved |
| XX thermal decision map and fit | Direct augmented-ensemble grids committed | Fig. 3(c,d) | Plotted; fit matches |

The six benchmark HDF5 files were checked by SHA-256 and class counts before
inference. Recomputed results are in the ignored local
`outputs/benchmark_reproduction/` directory. Relative to the paper's rounded
reference numbers, the maximum absolute metric difference was `1.61e-5`
(J1–J2 E2 recall); all other differences were at most `8.74e-6`. The counts
of unknown rows were 79, 168, 222, 48, 0, and 0 in the table's order.
Unknown rows were excluded. The six-file archive was built, verified, and
installed using `scripts/download_benchmarks.py --archive-file ...`; the
public network-download path awaits the external asset URL.

Both `plot_fig2.py` and `plot_fig3.py` completed with only compact committed
sources and wrote PDFs to `outputs/figures/`. Fig. 3(c,d) uses 4,941 base-grid
points and 48,632 directly evaluated visible-window points. The 19-point
activated fit is `C_*=0.0020259765`, `R²(log10 T_*)=0.9994261`, and
`RMSE(log10 T_*)=0.0061410`. The thermal inference grid itself was obtained
earlier from exact XY correlations and the augmented ensemble; this release
test redrew the committed direct predictions without expensive recomputation.

## Quick start, training and automated checks

- `python -m pytest -q tests`: 60 tests passed, covering physical states,
  certificates, exact XY correlations, LU features, and checkpoint inference.
- `python scripts/physical_example.py`: product-state score `0.00000389`;
  independently NPT-certified Bell-pair score `0.99998659`.
- `python scripts/evaluate_final.py --dataset tests/fixtures/physical_smoke_n16.h5
  --output outputs/smoke_evaluation.json`: 24 genuine XY rows; AUROC, E2
  recall, and TNR all `1.0` at fixed `p=0.5`. This tiny fixture is a workflow
  check, not a paper benchmark.
- `python scripts/smoke_train_dv1.py --output outputs/route_smoke`: the actual
  final training entry point completed one CPU epoch using 16 training and
  eight validation rows, wrote a smoke-only checkpoint, and reloaded it.
  This is an integration test, not full-model retraining.
- The current CI workflow also runs the frozen model example, physical
  example, HDF5 metric check, and one-epoch final-route smoke on Ubuntu.

## Limits

The external benchmark asset is not yet live, so a clean clone cannot perform
the network-download step until the Release is uploaded and the manifest URL
is updated. Full 50,000-row primary-data generation, all selector training,
and full three-seed retraining were not repeated. The original training-time
Git commit remains unknown; the public implementation is supported by strict
checkpoint loading and saved-score compatibility, not byte-identical code
provenance. Some HDF5 score vectors differ slightly from older saved tables
while aggregate decisions and paper metrics agree. See `PROVENANCE.md`.

# Verification record

This record distinguishes regenerated plots, frozen-model inference, and full
training. The source archive was not rewritten to prepare this repository.

## Figure data and plotting

- The Fig. 2 and appendix generators ran from the retained compact numerical
  inputs. This validates plotting and numerical-table consistency, not a
  complete retraining of every control model.
- Fig. 3(a,b) uses the Dv1-only thermal trajectory table and independent
  PPT/CCNR classifications. Fig. 3(c,d) uses the newer Dv1+R2Support
  three-seed ensemble. Its base grid has 4,941 direct predictions; the two
  visible broken-axis windows contain another 48,632 direct predictions with
  no probability interpolation. The 19-point activated fit gives
  `C_*=0.0020259765`, `R2(log10 T_*)=0.9994261`, and
  `RMSE(log10 T_*)=0.0061410`.

## Frozen-model checks

- All three Dv1-only Joint-K16 checkpoints and normalizers load with fixed
  channel order. Nine saved XY ensemble scores agree within `3.34e-6`;
  three raw-Pauli-to-token rows agree within `1.49e-8`.
- The complete 912-row N=16 XY benchmark was reevaluated. Of these, 864
  certified rows entered metrics and 48 unknown rows were excluded. AUROC,
  E2 recall, E>2 recall, and TNR are all `1.0` for this benchmark.
- Six transfer sets were reevaluated from locally preserved HDF5 files:
  adaptive XXZ (343 rows), J1-J2 (463), random-local-H (626), matched N=32
  XY (500), independently constructed N=32 synthetic (2,000), and
  General-Sep (2,000). Aggregate metrics match the manuscript. Dataset
  identities and hashes are recorded in `DATA_ACCESS.md` and
  `reports/phase3_benchmarks/benchmark_reproduction.json`.
- Stable-ID comparisons against saved N=32 and General-Sep scores had
  maximum probability deviations of `1.18e-4`, `8.27e-4`, and `2.13e-4`
  for the three corresponding comparisons. Aggregate decisions and metrics
  agree; the individual score vectors are not bitwise identical.
- The Dv1+R2Support checkpoints reproduce all 463 saved J1-J2 intervention
  predictions within `5.4e-7` after matching row IDs.

## Environment and remaining limits

A clean Python 3.10 CPU environment installed from `pyproject.toml`; the
inference example and representative figure generators ran, and 60 tests
passed. A separate 128-training-row/64-validation-row smoke run exercised
data loading, token construction, normalization, one optimizer epoch, and
checkpoint reload. Its metric is not a manuscript result.

Full HDF5 regeneration and three-seed retraining were not repeated in this
public package. The exact training-time Git commit for the saved final weights
is unknown. The current implementation is supported by inference agreement,
but byte identity to the historical training code is not claimed. Optional
TeNPy-dependent generation and large raw HDF5 files require additional data
and compute. See `PROVENANCE.md` and `DATA_ACCESS.md`.

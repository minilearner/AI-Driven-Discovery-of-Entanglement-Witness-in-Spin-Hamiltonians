# Scientific provenance

This document distinguishes the released weights, reference numerical data,
and regenerated results. The exact Git commit used for the original final
training runs is **unknown**. The current source is an inference-compatible
implementation, not a claim of byte identity with that training-time code.

## Baseline EntNet

The frozen three-seed Dv1-only ensemble uses `configs/joint_K16.json` in its
original channel order. The manifest's byte SHA-256 is
`fdcb0124a01b7445569f1b24dd6b4f6a370683edc286d37fa147d420f4e5fe86`.
Each checkpoint has its own saved mean, standard deviation, and model config.

| Seed | `best.pt` SHA-256 |
| --- | --- |
| 20260829 | `9ad3d0631a3257f989eb19b1992e33f6a72504e9dc446f89ecf13ffa62a7df69` |
| 20260830 | `0e2a51dbefd0a9da5aa8097bc28e9f2e5e411443a9eed95c1484963ae4e3dce5` |
| 20260831 | `480b428e8f4619dc673923e1abf9d0491903a3940925d1f8e9ad5f26da8f9758` |

The saved route audit reports 35,000 training, 7,500 validation, and 7,500
test rows, with no parent IDs crossing splits. The final classifier's training
positives are Dv1 two-site-NPT (`E2`) rows. Higher-order `E3/E4` classes enter
evaluation, not final training. The five-selector channel ranking used Dv1
plus two auxiliary selection collections; the fixed 16-channel classifier
was subsequently trained on **Dv1 only**. These are separate stages.

The model loads all three original state dictionaries strictly. On nine
frozen XY rows, CPU ensemble scores differed from saved reference values by
at most `3.34e-6`; three canonical Pauli rows converted to full invariant
tokens differed from the saved token cache by at most `1.49e-8`. These are
floating-point compatibility checks, not a reconstruction of the unknown
training-time commit.

## Distance-two-augmented EntNet

Fig. 3(c,d) and the appendix distance-two intervention use a separate
three-seed ensemble trained on Dv1 plus R2Support, with zero Dpair or DH
training rows. It uses the same selected 16 channels and five masks, with
separate saved normalization arrays. Its checkpoint hashes are:

| Seed | `best.pt` SHA-256 |
| --- | --- |
| 20260829 | `0cc9f419dc4ff5cd2502969daa4aa2f5ec999a7ff909f37c7a1411927dc95882` |
| 20260830 | `edc4ba2f6469480c1a8a04e000cd8f1c5014fb3af7496651836dc93288e83416` |
| 20260831 | `019ad9253d67817e0c9b68d6febc251095636ec79d967a228de76fd138079b00` |

A full 463-row J1–J2 intervention rerun matched the saved full-input and
block-replaced probabilities within `5.4e-7` after matching row IDs.
`tests/fixtures/r2support_control_9rows.npz` retains a small independent
reference. The final XX thermal map was computed from exact N=16 periodic XY
correlations through the augmented ensemble, not from an interpolated score
field. `reports/phase3_thermal/thermal_report.json` records the model,
checkpoint hashes, Hamiltonian, grid, threshold, and activated-fit diagnostics:
`C_* = 0.0020259765`, `R²(log10 T_*) = 0.9994261`,
`RMSE(log10 T_*) = 0.0061410` on 19 points.

## Data and figures

`data/benchmark_manifest.json` identifies the six final HDF5 files by
SHA-256, byte size, class counts, input schema, generator, and paper result.
Their package is outside Git history and available from the public
[`benchmarks-v1` Release](https://github.com/minilearner/AI-Driven-Discovery-of-Entanglement-Witness-in-Spin-Hamiltonians/releases/tag/benchmarks-v1).
The base Dv1-only ensemble was independently rerun on all six HDF5 files
downloaded from that Release; aggregate paper metrics match. The public
`scripts/reproduce_benchmarks.py` uses only those files, the manifest, and
released checkpoints, with no private prior-score or preservation-manifest
dependency. Per-row values in some older saved score tables are not bitwise
identical to the fresh rerun, so the independent result is reported as metric
agreement, not exact vector agreement.

Fig. 2 and Fig. 3 can be redrawn from compact data using the two scripts in
`paper_reproduction/figures/`. Fig. 3(a,b) uses baseline-model temperature
trajectories and independent PPT/CCNR certificates. Fig. 3(c,d) uses the
augmented ensemble and its directly evaluated thermal grids. Fig. 1 consists
of author-supplied schematics outside this code repository.

The 24-row `tests/fixtures/physical_smoke_n16.h5` is a genuine subset of a
certified N=16 XY dataset, with provenance and exact source row numbers in
the adjacent JSON. It exists only to exercise clean-clone inference and a
one-epoch final-route training check. Its AUROC is not a paper result.

Full 50,000-row primary-data generation, all selector training, and three
full final-model retrainings were not repeated for this repository update.
Some data-generation routes need TeNPy or substantial CPU/GPU resources.
See `DATA_ACCESS.md` and `VALIDATION.md` for the executable and tested scope.

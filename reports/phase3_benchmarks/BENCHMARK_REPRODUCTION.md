# Frozen-model benchmark reproduction (2026-11-16)

`scripts/reproduce_benchmarks.py` recomputed scores from the **final three
Dv1-only Joint-K16 checkpoints**, with their saved input normalization and the
fixed `p >= 0.5` decision rule. These are model inference results, not plots
made from old score tables. The complete per-row results and source hashes are
in `benchmark_reproduction.json` and the six `*_scores.csv` files.

| Benchmark | Source rows (Sep/E2/E3/E4/unknown) | AUROC | E2 recall | E3 recall | E4 recall | TNR | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XXZ | 343 (79/103/52/30/79) | 1.0000 | 0.9903 | 1.0000 | 1.0000 | 1.0000 | full model rerun |
| J1-J2 | 463 (30/155/80/30/168) | 1.0000 | 0.8645 | 1.0000 | 1.0000 | 1.0000 | full model rerun |
| Random local H | 626 (39/238/87/40/222) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | full model rerun |
| N=32 matched XY | 500 (20/312/80/40/48) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | full model rerun |
| N=32 independent synthetic | 2000 (400/400/600/600/0) | 0.99984375 | 0.9875 | 0.9667 | 1.0000 | 1.0000 | full model rerun |
| General-Sep | 2000 (2000/0/0/0/0) | n/a | n/a | n/a | n/a | 0.9970 | full model rerun |

For a direct manuscript comparison, each cell below lists
`AUROC / E2 recall / E3 recall / E4 recall / TNR` in that order. An em dash
denotes a metric that is undefined for an all-separable control. The final
column is the largest absolute difference across defined metrics, computed
against the paper's stored numerical values before display rounding.

| Benchmark | Model | Paper result | Recomputed result | Max absolute difference | Status |
| --- | --- | --- | --- | ---: | --- |
| XXZ | Dv1-only Joint-K16 | 1 / .9903 / 1 / 1 / 1 | 1 / .990291 / 1 / 1 / 1 | 0.00000874 | full model rerun |
| J1-J2 | Dv1-only Joint-K16 | 1 / .8645 / 1 / 1 / 1 | 1 / .864516 / 1 / 1 / 1 | 0.00001613 | full model rerun |
| Random local H | Dv1-only Joint-K16 | 1 / 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 / 1 | 0 | full model rerun |
| N=32 matched XY | Dv1-only Joint-K16 | 1 / 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 / 1 | 0 | full model rerun |
| N=32 independent synthetic | Dv1-only Joint-K16 | .99984375 / .9875 / .966667 / 1 / 1 | .99984375 / .9875 / .966667 / 1 / 1 | <1e-15 | full model rerun |
| General-Sep | Dv1-only Joint-K16 | — / — / — / — / .997 | — / — / — / — / .997 | 0 | full model rerun |

The XXZ/J1-J2/random-local data are the **adaptive 343/463/626-row Gibbs
benchmarks** used in the current appendix. The older 312-row HDF5 files in the
private archive are not used here. The matched N=32 XY benchmark and the
independently constructed N=32 synthetic benchmark are separate datasets and
were evaluated separately. Unknown rows are excluded from AUROC and recall;
they are never negative examples. E3 and E4 are mutually exclusive certificate
classes in these files. General-Sep contains only constructively separable
rows, so only TNR is defined.

The five HDF5 files preserved from the remote server matched their SHA-256
records in the 2026-11-06 preservation manifests. The N=32 matched XY file
was already generated locally and has no server-preservation manifest entry;
the current file hash is recorded in `benchmark_reproduction.json`. Source
sample IDs were checked for uniqueness and aligned with historical per-row
scores where available.

The recomputed **aggregate metrics match the current manuscript**. Against
the previously saved N=32 and General-Sep score CSVs, the maximum absolute
per-row probability differences are `1.18e-4` (N=32 XY), `8.27e-4` (N=32
synthetic), and `2.13e-4` (General-Sep), respectively. This is not a bitwise
prediction reproduction. The previously saved LU-control comparison
copied its Joint-K16 scores from earlier per-seed CSVs rather than running the
checkpoint again, which is why that comparison is not independent inference
evidence. Current model inference reproduces all reported decisions and
metrics, but the per-row numerical discrepancy remains documented.

No J1-J2 R2Support predictions are substituted for the Dv1-only zero-shot
result in this table. The separate enhanced-route intervention is reported
under `reports/phase3_thermal` and in the appendix-control verification.

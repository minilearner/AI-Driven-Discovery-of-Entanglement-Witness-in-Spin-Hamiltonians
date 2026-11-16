# Provenance and scientific boundaries

This file records what can be traced to the private research archive as of
2026-11-17. It does not assign a previously unknown Git commit to any model.

## Final ensemble

The three `best.pt` files were copied byte-for-byte from
`models/joint_k16_training_distribution_ablation_v1/` in the private archive.
Their SHA-256 values, in seed order, are:

| Seed | `best.pt` SHA-256 |
| --- | --- |
| 20260829 | `9ad3d0631a3257f989eb19b1992e33f6a72504e9dc446f89ecf13ffa62a7df69` |
| 20260830 | `0e2a51dbefd0a9da5aa8097bc28e9f2e5e411443a9eed95c1484963ae4e3dce5` |
| 20260831 | `480b428e8f4619dc673923e1abf9d0491903a3940925d1f8e9ad5f26da8f9758` |

Each seed also has its original `input_mean.npy`, `input_std.npy`, and
`model_config.json`. The only edited metadata field is `selection_manifest`:
its absolute archive path became the portable relative path
`configs/joint_K16.json`. The manifest's byte SHA-256 remains
`fdcb0124a01b7445569f1b24dd6b4f6a370683edc286d37fa147d420f4e5fe86`
and is checked by the loader. The checkpoint metadata records the Dv1-only
route and distinct training and validation hashes. It records
`implementation_git_commit: unknown`; this remains unresolved.

The saved training-set audit reports 35,000 rows (17,500 constructively
separable and 17,500 two-site-NPT positives), 7,500 validation rows, and 7,500
test rows, with zero parent IDs crossing splits. All final training rows came
from Dv1. The E3/E4 classes were used in evaluation, not training. The
five-selector consensus that selected the channels predates this final
three-seed training and used a broader selection pool. The frozen channel
order is retained; the release does not imply it was selected on Dv1-only.

## Reference comparisons

`tests/fixtures/xy_reference_9rows.npz` contains nine frozen XY token rows,
the corresponding canonical Pauli rows, and saved seed/ensemble probabilities.
Source hashes are in the adjacent JSON file. On the current CPU implementation,
the maximum ensemble difference from the saved GPU reference was
`3.34e-6`; the test uses an absolute tolerance of `5e-6`. Converting three
canonical Pauli rows into 122-channel local tokens differed from the saved
token cache by at most `1.49e-8`.

The inference implementation in `src/entanglement_data/ml/invariant_transformer.py`
loads the three original state dictionaries strictly and reproduces these
saved outputs within the stated float tolerance. This is direct evidence of
evaluation compatibility, **not** proof that the file is byte-identical to the
unknown training-time implementation.

## Source correspondence and remaining uncertainty

- The final-training candidate source
  `scripts/run_joint_k16_training_distribution_ablation.py` came from the
  private archive's `.worktrees/pair-range-r8` tree (tree HEAD
  `34a1dcbc38b5b23a59d0ac19513d29f229d20cdd`). That tree also supplied
  required training helpers. This tree hash is a source-copy pointer, **not**
  the checkpoint implementation commit.
- The candidate extracts the spatial-invariant layout and canonical route
  loader from exploratory CNN/Siamese scripts into
  `src/entanglement_data/ml/spatial_invariants.py` and
  `scripts/training_routes.py`. The final-route wrapper fixes Dv1-only and
  defaults to the release directory as data root; these are release
  integration edits, not recovered training-time provenance.
- `src/entanglement_data/` and the other `scripts/` were copied from the
  private archive working tree. Some source files were untracked there but
  retained because the final paper uses them. No private Git history was
  rewritten.
- Compact figure data and figure scripts came from the staged
  `reports/github_release_20261106/repo` materials and later corrected
  manuscript figure sources, with the current paper checked after a fresh
  build on 2026-11-16.
- Some retained audit JSON files contain original `/root/autodl-tmp/...`
  source paths. They are historical provenance fields, not runtime settings;
  the referenced large files are not bundled. Runtime checkpoint and manifest
  configuration uses release-relative paths.
- Fig. 3(a,b) uses the Dv1-only ensemble. The author-adopted Fig. 3(c,d) and
  Appendix Fig. 4(e,f) use the newer Dv1+R2Support control. The earlier
  Fig. 3(c,d) used a different fixed-K16 model. Its weights were recovered from the
  research server on 2026-11-16 into the **private** preservation archive,
  outside this public repository. The old phase-grid report's
  `training_route` fields are null, but the recovered training script and
  route-hash-matched audit identify its actual training sources as
  Dv1+Dpair+DH+R2Support. Historical scripts and grids are excluded because
  they are not part of the adopted figure.
- Figure 1 consists of author-supplied schematics. Its source assets are not
  included in this code repository.

The large HDF5 training and benchmark datasets, most non-final weights, and
training caches are not packaged. The necessary generators and compact
results are kept where they could be verified; remaining limits are listed
in `VALIDATION.md`.

## Dv1+R2Support intervention and thermal ensemble

The adopted appendix intervention and Fig. 3(c,d) calculations use three
checkpoints copied
byte-for-byte from
`models/joint_k16_dv1_r2support_control_v1/JointK16_Dv1_R2Support_Control_seed*`.
The copied `best.pt` SHA-256 values in seed order are:

| Seed | `best.pt` SHA-256 |
| --- | --- |
| 20260829 | `0cc9f419dc4ff5cd2502969daa4aa2f5ec999a7ff909f37c7a1411927dc95882` |
| 20260830 | `edc4ba2f6469480c1a8a04e000cd8f1c5014fb3af7496651836dc93288e83416` |
| 20260831 | `019ad9253d67817e0c9b68d6febc251095636ec79d967a228de76fd138079b00` |

The reports explicitly list `D_v1` and `R2Support` as training sources and
exclude `D_pair` and `D_H`. A 463-row J1–J2 intervention rerun with these
weights matches the saved per-row full-input, higher-order-replaced, and
pair-replaced probabilities to a maximum absolute difference of `5.4e-7`
after joining on row ID. Nine reference rows are retained in
`tests/fixtures/r2support_control_9rows.npz`.

The Dv1-only r=4 ablation's two non-baseline channel manifests were rebuilt
from the saved arm/channel lists and the unchanged 122-channel dictionary by
`scripts/prepare_r4_ablation_manifests.py`. The generated files are in
`configs/r4_ablation/`. They preserve the paper's fixed channel sets (remove
44 and 51; replace them by 85 and 112 in the alternative arm), but the nine
ablation training checkpoints are not included.

The historical, superseded thermal Fig. 3(c) grid records model directories named
`k24_r2support_domain_generalization_v1/Fixed_K16_R2Support_seed*`. Its saved
report calls this a Dv1+R2Support route, but its per-seed `training_route`
fields are null. The three old checkpoints and normalizers have now been
recovered privately. Their actual route inherited 35,000 Dv1, 1,412 Dpair,
and 1,400 DH training rows before adding 1,400 R2Support rows. The newer
control used 35,000 Dv1 and 1,400 R2Support rows, with zero Dpair/DH rows.
The R2Support dataset hash and all 21 selected input channels match, as do
the recorded architecture and optimizer settings. On the same 81 exact XY
thermal inputs at gamma=0, h/J=1.2, the recovered old checkpoints reproduce
the original grid within `5.99e-7` in probability. The old and new outputs
are genuinely different learned-model results. See the private
`reports/remote_preservation_20261106/historical_fixed_k16_r2support/RECOVERY_AUDIT.md`.
The old weights are **not included in this repository** and should not be
silently substituted by the newer route.

## Independently recomputed adopted thermal figure

The author adopted the newer Dv1+R2Support three-seed ensemble above for
Fig. 3(c,d) in Overleaf. The figure sources in this repository come from
`scripts/recompute_xy_thermal_r2support.py`, which builds
fresh N=16 periodic XY thermal correlations, constructs the same frozen 16
physical channels and five masks, applies each new seed's saved normalization,
and averages sigmoid outputs. It does not recycle the old probability grid.
The full 4,941-point base scores, 48,632 direct visible-window scores,
per-seed values, refined boundaries, checkpoint hashes, and old/new comparison
are in `reports/phase3_thermal/`. The final
activated-fit constant is `0.0020259765`, versus `0.0010410980` for the old
figure. This is a **new-model scientific result**, not a bitwise reproduction
of the old checkpoint. Fig. 3(a,b) remains the Dv1-only ensemble. The
machine-readable parameters, fit diagnostics, and checkpoint hashes are in
`reports/phase3_thermal/thermal_report.json`.

The final Dv1-only checkpoint was also rerun on the six current transfer
benchmarks from their actual HDF5 inputs. All aggregate paper metrics match.
The three older OOD HDF5 files with 312 rows each were excluded; the current
adaptive Gibbs datasets have 343/463/626 rows. The saved N=32 and General-Sep
per-row scores differ from the fresh predictions by at most `8.27e-4`, despite
identical aggregate decisions and metrics. The historical LU-control figure
assembly copied its Joint-K16 scores from saved per-seed CSVs. Thus the
current independent inference validates the reported metrics, while exact
per-row equivalence to those older CSVs remains unproven. Dataset hashes,
stable-ID comparisons, and score files are in `reports/phase3_benchmarks/`.

The small Dv1-only CPU training smoke run confirms the candidate path from
primary HDF5 rows through canonical token extraction, training-only
normalization, one optimizer epoch, validation, and checkpoint reload.
It does **not** establish the unknown original training-time Git commit or
full three-seed retrain equivalence.

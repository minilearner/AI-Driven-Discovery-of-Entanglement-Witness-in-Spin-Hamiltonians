# EntNet: entanglement detection from spin correlations

This repository contains the code, six frozen model checkpoints, selected
local-unitary (LU) invariant features, and compact numerical data used for the
manuscript **AI-Driven Discovery of Entanglement Witness in Spin Hamiltonians**.
The main model is a three-seed, Dv1-only Joint-K16 RoPE Transformer. A separate
three-seed Dv1+R2Support model is used for the distance-two intervention and
the final thermal decision map in Fig. 3(c,d).

EntNet's output is an empirical classifier score. It is **not** a rigorous
entanglement witness, an entanglement measure, or a calibrated probability.
PPT/CCNR and constructive certificates supply the labels. Uncertified states
are excluded from classification metrics; they are not treated as separable.

## What is included

- Three final Dv1-only `best.pt` checkpoints with per-seed normalization and
  architecture metadata in `checkpoints/final_joint_k16/`.
- Three distinct Dv1+R2Support checkpoints and normalizers in
  `checkpoints/dv1_r2support_control/`.
- The frozen 16 physical LU-invariant channels and five geometry masks in
  `configs/joint_K16.json`, with model, feature, and certificate code under
  `src/entanglement_data/`.
- Figure generators, compact source tables, direct thermal prediction grids,
  benchmark score tables, training and dataset-generation entry points.
- Reference-input tests and checkpoint hashes. `PROVENANCE.md`,
  `DATA_ACCESS.md`, `VALIDATION.md`, and `SOURCE_MAP.csv` state what has been
  reproduced and what remains dependent on the original research archive.

The large HDF5 datasets and non-final training checkpoints are not included.
The included final weights are sufficient for the inference example and for
evaluating a compatible benchmark file.

## Installation

Use Python 3.10 or later. In a clean environment, from the repository root:

```bash
python -m venv .venv
# Activate .venv using your shell's command.
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q tests
```

PyTorch may need a platform-specific CPU or CUDA wheel. `pyproject.toml` is the
dependency source; `requirements.txt` is a convenience entry point. TeNPy is
an optional, substantial dependency for regenerating certain Gibbs datasets.

## Quick start: final model inference

```bash
python scripts/infer_example.py
```

The example loads all three **Dv1-only** final checkpoints and scores nine
saved XY reference rows. The programmatic API is:

```python
from pathlib import Path
from entanglement_data.final_entnet import predict_final_ensemble

# full_tokens: [batch, sites, 122] in the canonical channel order.
scores = predict_final_ensemble(full_tokens, Path('.'))
```

For raw canonical Pauli-feature rows, use
`entanglement_data.final_entnet.tokens_from_pauli_features(features, N=16)`
first. It constructs the 117 physical candidate invariants and five masks.
The frozen configuration then selects 16 physical channels plus all five
masks. Each seed applies its own saved mean and standard deviation. The
reported ensemble score is the **mean of three sigmoid probabilities**.

The five-selector channel-ranking procedure and the final three-model
ensemble are different stages. The ranking used Dv1 plus two auxiliary
selection collections. The final Joint-K16 classifier was retrained on the
35,000 Dv1-only training rows after the channel set was frozen. Its positive
training labels are two-site PPT-certified (`E2`); `E3` and `E4` labels appear
in evaluation, not final-model training.

## Reproducing the results

### Level 1: redraw figures from compact numerical data

These commands do not train or evaluate a model:

```bash
python paper_reproduction/figures/plot_fig2.py
python paper_reproduction/figures/plot_fig3.py
```

`plot_fig2.py` redraws the selection, budget, transfer, and appendix panels.
`plot_fig3.py` redraws the final thermal figure. It takes Fig. 3(a,b) from the
frozen Dv1-only trajectory and certificate table, and Fig. 3(c,d) from the
**newer Dv1+R2Support** prediction grids in `reports/phase3_thermal/`. The
broken-axis windows contain 48,632 directly evaluated points, not
interpolated old-model scores. The fitted lower boundary uses 19 points at
`1.05 <= h/J <= 1.50` and gives `C_* = 0.0020259765`,
`R2(log10 T_*) = 0.9994261`, and `RMSE(log10 T_*) = 0.0061410`.

The author-supplied Fig. 1 schematic is not distributed or regenerated here.
Figure PDFs produced by these commands are outputs; the compact CSV, JSON, and
NPZ files are the source data.

### Level 2: reevaluate frozen models

For a compatible HDF5 benchmark with canonical `features/X` and certificate
labels:

```bash
python scripts/evaluate_final.py \
  --dataset /path/to/benchmark.h5 --N 16 --output evaluation.json
```

The script fixes the three Dv1-only checkpoints and the `p=0.5` decision
threshold. Unknown labels do not enter recall, AUROC, or TNR calculations.
The six paper transfer benchmarks can be rerun after placing their HDF5 files
at the documented relative paths under an archive root:

```bash
python scripts/reproduce_benchmarks.py --archive-root /path/to/archive
```

For the separate Dv1+R2Support intervention model, use
`scripts/evaluate_r2support_interventions.py` with the J1-J2 HDF5 benchmark.
The final thermal map can be recomputed from exact XY correlations by running
`scripts/recompute_xy_thermal_r2support.py` and then
`scripts/recompute_xy_thermal_dense.py --workers 4`. These are model
evaluations and can take substantial CPU time. `DATA_ACCESS.md` identifies
the exact benchmark files and their checksums.

### Level 3: regenerate data, select features, and retrain

The necessary physical-state, Pauli-observable, PPT/CCNR-certificate,
LU-invariant, selector, and Transformer implementations are retained in
`src/` and `scripts/`. The primary Dv1 generation begins with
`scripts/generate_production.py`; the final training entry point is
`scripts/train_final_joint_k16.py`. The latter fixes the Dv1-only route and
requires the full Dv1 HDF5 and invariant cache. Other generators cover XY,
adaptive XXZ/J1-J2/random-local-H, N=32 matched XY, independently constructed
N=32 synthetic, and General-Sep benchmarks. Full regeneration and three-seed
training require substantial compute. A small data-to-training smoke command
is available:

```bash
python scripts/smoke_train_dv1.py \
  --dataset /path/to/data/n16_production_v1/dataset.h5
```

That smoke run is not a reproduction of the final model weights. The exact
training-time Git commit of the saved checkpoints is unknown; the current
source reproduces their inference within the documented floating-point
tolerance. `PROVENANCE.md` explains this boundary.

## Paper-to-code map

| Paper component | Code and compact inputs | Model route |
| --- | --- | --- |
| Fig. 2(a,b): channel ranking and K budget | `paper_reproduction/figures/plot_fig2.py`, `paper_reproduction/data/joint_channel_selection_summary.json`, `paper_reproduction/data/joint_k_budget_balanced_metrics.csv` | Five-selector ranking; separate Dv1-only K models |
| Fig. 2(c): unseen Hamiltonians | `plot_fig2.py`, `paper_reproduction/figures/combined_structural_lu_controls_source.csv`, `scripts/reproduce_benchmarks.py` | Final Dv1-only Joint-K16 |
| Fig. 2(d-g): raw, soft LU, exact LU inputs | `plot_fig2.py`, `paper_reproduction/data/lu_controls/`, `scripts/run_dv1_raw_t4_lu_control.py` | Controlled Dv1 training |
| Fig. 3(a,b): temperature traces and certificates | `plot_fig3.py`, `reports/xy_temperature_sweep_dv1_v1/` | Dv1-only scores; independent certificates |
| Fig. 3(c,d): XX thermal map and activated fit | `plot_fig3.py`, `reports/phase3_thermal/`, `scripts/recompute_xy_thermal_r2support.py`, `scripts/recompute_xy_thermal_dense.py` | New Dv1+R2Support ensemble |
| Appendix: r=4 ablation | `plot_fig2.py`, `paper_reproduction/data/r4_pair_ablation_dv1_only_summary.json` | Dv1-only ablation models |
| Appendix: distance-two support interventions | `plot_fig2.py`, `scripts/evaluate_r2support_interventions.py`, `paper_reproduction/data/joint_k16_dv1_r2support_interventions_source.csv` | Dv1-only and Dv1+R2Support |

N=32 matched XY and N=32 independently constructed synthetic states are
different benchmarks and have distinct generators and score tables.

## Citation and license

Please cite the associated manuscript when its public bibliographic record is
available. The repository title is currently the stable citation identifier;
author, journal, and DOI metadata will be added when confirmed. **No
open-source license has been granted.** Public availability does not imply
permission to redistribute or modify this code or the included weights;
contact the authors for permissions.

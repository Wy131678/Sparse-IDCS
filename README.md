# Sparse-IDCS: manuscript experiment source

This repository contains runnable source for the synthetic experiments in
*Sparse indirect data conditional simulation method based on crosshole GPR
data*. The proposed method represents the straight-ray traveltime operator in
CSR/CSC sparse formats and updates candidate likelihoods by sparse columns.
It is a **sparse incremental** method, not a strictly matrix-free operator.

The deleted field-validation section is not included. No manuscript result
tables, reconstructed images, or figure data are preloaded. Every experiment
regenerates its own outputs under the ignored `run_outputs/` directory.

## Installation

For the manuscript's numerical environment, use Python 3.8.10:

```bash
python -m venv .venv
```

Activate `.venv` (`.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux), then install and test:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-paper.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest
```

On Windows, use a reasonably short repository path, such as
`C:\work\Sparse-IDCS`. Deep directory trees can exceed the legacy Windows path
limit when experiment output filenames are created. Alternatively, pass a
short absolute directory with `--output` to an individual experiment script.

As a first paper-scale run, reproduce Case V of the 50 x 100 comparison:

```bash
python experiments/ablation_50x100/run.py --cases V
```

A newer Python and numerical-library stack can run the code, but can
select a different realization even with the same seed. The generated
`summary.json` records software versions and key method settings.
Runtime comparisons depend on hardware, BLAS, process contention, and version.

## Experiments

Each `run.py` contains its own losslessly compressed training image and
reference crop as Python string constants. No input file paths, downloads, or
separate `data/` directory are needed. The strings are packed binary pixels
from the same 2500 x 2500 `channels.npy` source field, decoded and checksum
verified in memory. Class 1 corresponds to 0.06 m/ns and class 0 to 0.08 m/ns.
The upper-left 500 x 500 training image does not overlap the reference crops.

| Directory | Manuscript material | Generated files | Command |
| --- | --- | --- | --- |
| `experiments/reference_geometry/` | Figures 2 and 3 | `figure_2_reference_images.png`, `figure_3_small_rays.png`, `figure_3_large_rays.png` | `python experiments/reference_geometry/run.py` |
| `experiments/ablation_50x100/` | Table 1 and Figures 4 and 5; Cases I-V | `comparison.csv`, `figure_4_metrics.png`, `figure_5_images.png` | `python experiments/ablation_50x100/run.py` |
| `experiments/penetration_sweep/` | Figures 6 and 7 and Appendix Figures A.1 and A.2; widths 5, 7, 10, 13, 15, and 20 m | `comparison.csv`, `figure_6_scale_metrics.png`, `figure_7_sparse_images.png`, `figure_A1_baseline_scale_images.png`, `figure_A2_baseline_scale_runtime.png` | `python experiments/penetration_sweep/run.py` |
| `experiments/large_seed_stability/` | Table 2 and Figure 8; 150 x 300 grid | `table_2_generated.csv`, `figure_8_seed_stability.png` | `python experiments/large_seed_stability/run.py` |
| `experiments/forward_data/` | Appendix Figures B.1 and B.2 | `figure_B1_forward_fit.png`, `figure_B2_ray_matrices.png`, `forward_fit.json` | `python experiments/forward_data/run.py` |
| `experiments/likelihood_cost/` | Candidate-likelihood cost decomposition | `cost.csv`, `candidate_cost.png` | `python experiments/likelihood_cost/run.py` |

Figure 1 is a conceptual workflow diagram rather than the output of a numerical
experiment. Its steps are implemented in `src/sparse_idcs/` and exercised by
the scripts above.

`python run_all.py --list` shows the groups. `python run_all.py --all` runs
them serially. **The complete original-IDCS penetration sweep can take many
hours or days**; select a group or pass `--widths`, `--methods`, or
`--cases` to its script for a smaller run. All computational scripts except
the reference-geometry display offer `--smoke`; smoke outputs are code checks,
not paper results. Use `--help` on any script for exact options.

## Historical versus paired observations

The historical original-IDCS wrapper draws Gaussian noise for every candidate
ray before the 50-degree filter. The historical Sparse-IDCS wrapper draws noise
only for retained rays. Both use seed 0, but **the retained observations are
not identical**. The default commands preserve this historical ordering to
reproduce the manuscript's recorded runs. For a separate paired-observation
comparison, use `--shared-observations` with the ablation or penetration
script; this makes Sparse-IDCS use the original pre-filter noise sequence and
writes into a separate `shared/` output folder. That mode is an additional
fairness check, **not** the original manuscript experiment. The claim that the
historical methods used identical noisy observations needs correction in the
manuscript; this code does not hide the discrepancy.

## Outputs and code boundaries

Each run produces a `summary.json`, `reconstruction.npy`, and
`observed_traveltime_ns.npy`. An experiment also generates its own figures
and CSV tables. These are produced at runtime and ignored by Git. The original
IDCS source modules are vendored under `vendor/idcs_original/`; our wrapper
uses only their linear `qsCat` path, avoiding the optional pygimli backend.
The Sparse-IDCS solver lives in `src/sparse_idcs/`.

## Attribution and license

See [NOTICE](NOTICE), [LICENSE](LICENSE), and [CITATION.cff](CITATION.cff).
The original IDCS implementation is from Levy et al. (2024),
*Computers & Geosciences* 187, 105581,
https://doi.org/10.1016/j.cageo.2024.105581. The binary-channel source image
is attributed to Zahner et al. (2016). 

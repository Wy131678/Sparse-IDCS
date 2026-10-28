# Experiment map

Each subdirectory has one independent `run.py` with its own input pixels
embedded as compressed Python constants. Its `--help` output lists case
selection and smoke-check options. All generated artifacts go to
`run_outputs/<experiment>/` unless `--output` is supplied. The folder
contains no precomputed manuscript results.

The scripts reproduce the numerical workflow; the generated figures are
functional renderings of regenerated data, not the journal's final typography.
The penetration sweep produces manuscript Figures 6 and 7 and Appendix
Figures A.1 and A.2. The original-IDCS scale cases may take many hours on one
CPU.

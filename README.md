# Markowitz meets Talmud

Replication of Jun Tu and Guofu Zhou (2011). We examine how combining an equal-weight portfolio with estimated portfolio rules affects risk-adjusted returns. [Journal of Financial Economics article](https://doi.org/10.1016/j.jfineco.2010.08.013).

| Part | Materials |
| --- | --- |
| 1. Paper and presentation | [Slides](defense/slides.pdf) · [Speaker outline (Russian)](defense/outline.md) |
| 2. Methods and simulations | [Code and run instructions](RUN_part2.md) · [Status](docs/part2_status.md) · [Main run](results/sim_full/tables1_3_comparison.csv) · [Footnotes 8–9](results/footnotes89/footnote89_paired.csv) |
| 3. Empirical replication | [Part 3 slides](defense/part3_slides.pdf) · [Speaker notes (Russian)](defense/part3_talk.md) · [Table 6](results/replication/table6.csv) · [Data and discrepancies](docs/data.md) |
| 4. Extensions and figures | [Run instructions, results, and audit](EXTENSIONS.md) · [Figures and captions](defense/figure_captions.md) · [Conclusions](defense/extension_conclusions.md) |
| Internal team notes | [Work plan and role assignments (Russian)](docs/team/README.md) |

[Plain-language article walkthrough (Russian)](docs/article_walkthrough.md) · [Codebase Q&A (Russian)](QA.md)

## Reproducing Table 6

Run these commands from the repository root with Python 3.12.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part3.lock
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

python -m replication.data --download
python -m pytest -q
python -m replication.backtest --mode smoke --out runs/smoke/empirical
python -m replication.backtest --mode full --out results/replication
python -m replication.empirical_diagnostics --download
python -m replication.presentation
```

The full run compares all 140 Table 6 cells with the published values. The match tolerance is 0.005 percentage points. Monthly returns, weights, and processed panels are rebuilt locally. The [manifest](results/replication/empirical_manifest.json) records archive versions, hashes, and date ranges.

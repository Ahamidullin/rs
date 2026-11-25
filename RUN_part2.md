# Running Part 2

Use Python 3.12 with NumPy, SciPy, pandas, and pytest. Run these commands from the repository root.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/test_methods.py
# Full run: 10,000 replications, six T values, A0 (gamma=3 and 1 on the same X), and Aalpha. Resumes from completed chunks.
python -m replication.simulate --mode full --reps 10000 --workers 8 --out results/sim_full
python -m replication.aggregate results/sim_full references/tables1_3.csv results/sim_full/results
# Footnotes 8–9 sensitivity checks: N=50 and no common factor.
python -m replication.simulate --mode full --reps 10000 --workers 8 --scenarios Aa_N50,NF --out results/sim_fn8
```

On one core, a replication over all six T values takes roughly 1.1 seconds for A0 with both gamma settings and 1.0 second for Aalpha. This is about 5.8 CPU hours in total, or about 45 minutes on eight cores. The N=50 scenario for footnote 8 costs roughly three to four times as much.

The outputs are `tables1_3_comparison.csv` (published, computed, mcse, delta_pp, status), `sim_summary.csv` (utility, standard deviation, Monte Carlo standard error, Sharpe ratio, mean and standard deviation of delta, and clipping rates), `footnote4_ew_cash.csv`, and `footnote10_dispersion.csv`.

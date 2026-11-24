# Часть 02: запуск

Python 3.12, numpy, scipy, pandas, pytest. Из корня пакета:

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q tests/test_methods.py
# полный прогон: 10 000 повторов, 6 T, A0 (gamma=3 и 1 на общих X) и Aalpha; возобновляется по готовым чанкам
python -m replication.simulate --mode full --reps 10000 --workers 8 --out results/sim_full
python -m replication.aggregate results/sim_full references/tables1_3.csv results/sim_full/results
# сноски 8-9 (sensitivity): N=50 и no-factor
python -m replication.simulate --mode full --reps 10000 --workers 8 --scenarios Aa_N50,NF --out results/sim_fn8
```

Оценка времени: около 1.1 с на повтор (все T, оба gamma) для A0 и около 1.0 с для Aalpha на одно ядро,
то есть около 5.8 ч CPU всего; на 8 ядрах около 45 мин. Сноска 8 с N=50 примерно в 3-4 раза дороже.
Выходы: tables1_3_comparison.csv (published, computed, mcse, delta_pp, status), sim_summary.csv
(U, SD, MCSE, SR, среднее/SD delta, доли отсечения), footnote4_ew_cash.csv, footnote10_dispersion.csv.

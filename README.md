# Markowitz meets Talmud

Репликация статьи Jun Tu и Guofu Zhou (2011). Проверяем, как смеси равновзвешенного портфеля с оценочными правилами меняют риск-скорректированную доходность. [Статья в Journal of Financial Economics](https://doi.org/10.1016/j.jfineco.2010.08.013).

| Часть | Материалы |
| --- | --- |
| 1. Статья и защита | [Слайды](defense/slides.pdf) · [сценарий](defense/outline.md) |
| 2. Методы и симуляции | [Код и запуск](RUN_part2.md) · [статус](docs/part2_status.md) · [основной прогон](results/sim_full/tables1_3_comparison.csv) · [сноски 8–9](results/footnotes89/footnote89_paired.csv) |
| 3. Эмпирическая репликация | [Слайды части 3](defense/part3_slides.pdf) · [текст выступления](defense/part3_talk.md) · [таблица 6](results/replication/table6.csv) · [данные и расхождения](docs/data.md) |
| Командные материалы | [План проекта и форматы](docs/team/README.md) |

## Воспроизведение таблицы 6

Из корня репозитория, Python 3.12.

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

Полный прогон сравнивает 140 ячеек таблицы 6 с публикацией. Совпадения отмечены с допуском 0,005 п.п. Помесячные доходности, веса и подготовленные панели восстанавливаются локально. В [манифесте](results/replication/empirical_manifest.json) указаны версии архивов, хеши и интервалы.

# Ответ на ревью части 3

Проверены исходный `main` на `bb92bc3` и материалы расширений на `origin/andrew` `721384b`. Формулы сверены со страницами 206–207 журнальной версии. [Статья Tu и Zhou](https://down.aefweb.net/WorkingPapers/w715.pdf).

## 1. Комплект таблицы 6 · исправлено

В исходном `main` таблица содержала `run_id=78841781ffa7d107` и SHA‑256 `a31eabbd6d18b2009819d290215b31ecc0842d849bd3c2ad3cfc1a263167304d`. Оба манифеста расширений и манифест графиков в `andrew` ссылаются на `e63b28306638e6fe` и SHA‑256 `07415a84710b7aedcbc25764b432eb25141a29d983677d4a487b4deffa6ce4f0` для `runs/part3_rebuilt/table6.csv`. Сам временный каталог отсутствует в Git. Поздние снимки сверены с оригинальными манифестами. [Оборот](https://github.com/Ahamidullin/rs/blob/721384b/results/extensions/turnover/manifest.json) · [Устойчивость](https://github.com/Ahamidullin/rs/blob/721384b/results/extensions/stability/manifest.json) · [Графики](https://github.com/Ahamidullin/rs/blob/721384b/results/figures/manifest.json).

Основной комплект теперь расположен в `results/replication`. Режим CKZ — `eq13_scaled`, `run_id=07ded1e559fe42d2`. Панели для него лежат в `data/processed`. [Манифест с хешами семи панелей и версиями пакетов](../results/replication/empirical_manifest.json).

| Файл в `results/replication` | SHA‑256 |
| --- | --- |
| `table6.csv` | `e6983aecd80d9ad238a784bde5420b8112dcb9e8a48a577c717314b7814286cf` |
| `monthly_returns.parquet` | `a6d2d49feccdcab26a3dc97e5f1d412315355d83c6f68f7fd46473fb7b5e9479` |
| `weights.parquet` | `853069d97a91f9967d0335925fa6fccbb495f8249629e082175ead01b19f1095` |
| `empirical_checks.csv` | `ed24bcea58721d9b922d7e518c5aef02ac9e0bf0df95a759887c39130f18a79f` |
| `empirical_manifest.json` | `acaed7869cc903fed807cc707e2720e788d3e8ca1e396cf513748a7fb7543b21` |

SHA‑256 кода `methods.py` — `0627713e3cd34cc61fa31538e9c6e4d4101932c40b072a2bcfae0cf6c6388887`, `backtest.py` — `dd870915ccf65994c8e11b1d6ba629630b785ecc9246e0725364d4952b4b167f`. Остальные хеши кода и входов записаны в манифесте.

В сверке проверены хеши, идентификаторы, 140 уникальных ключей, даты `fit_end` и следующего месяца, покрытие `T−M` для обоих окон, cash и сумма весов. Из 58 660 строк доходностей и 1 311 660 весов пересобраны доходности и все OOS CER. ML-IS пересчитан отдельно на каждой полной панели. Пересборка с другим путём к панелям и изменённым временем скачивания сохранила `run_id` и хеши четырёх выходов. Подмена `table6.csv` отклоняется по хешу. Smoke-прогон требует отдельного каталога.

Для части 4 проверены обе программы из `andrew` с новым кодом частей 2–3 в отдельной локальной копии. Обе сверили 28 из 28 ячеек baseline по CER, датам, месячным доходностям и весам. Сборка графиков отклонила сочетание новой таблицы со старыми манифестами и приняла его после пересчёта обоих расширений на новом baseline. Опубликованные ранее результаты части 4 принадлежат старому запуску. Для следующего прогона из ветки с кодом расширений и новой формулой используйте один вход для обеих команд и сборки графиков.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m replication.data --download --out data/processed
python -m pytest -q
python -m replication.backtest --mode full --ckz-mode eq13_scaled --out results/replication
python -m replication.empirical_diagnostics --verify-only --out results/replication

python -m replication.ext_turnover --baseline results/replication --out runs/part4/extensions/turnover
python -m replication.ext_stability --baseline results/replication --out runs/part4/extensions/stability
python -m replication.figures --replication results/replication --extensions runs/part4/extensions --out runs/part4/figures
```

Точные хеши опубликованного запуска относятся к указанным версиям Python и пакетов. На другой платформе пересоберите комплект целиком и используйте его собственный манифест.

## 2. Формула CKZ · исправлено

В (13), §2.3, с. 207, перекрёстный член использует `m′ Σ̃⁻¹ q`. Масштабирование определено в (4), с. 206. При `Σ̂ = S0`, где `S0` делится на `M`, это даёт `(M−N−2)/M × m′ S0⁻¹ q`. Для диагностики из ревью воспроизведён и прямой `m′ S0⁻¹ q`. Его ошибка ниже на выбранной исторической панели, а основная формула выбрана по обозначению в PDF. Для всех трёх вариантов одинаковы `q`, KZ, члены (9) и (14), ограничение коэффициента и данные.

| Режим | CKZ MAE, годовые п.п. | Industry, M=120 при опубликованных 3,02 | Совпало ячеек таблицы 6 |
| --- | ---: | ---: | ---: |
| Прежний `legacy_adjusted` | 5,77476 | −0,5940 | 32/140 |
| Диагностика `unscaled_plugin` | 1,40737 | 1,1941 | 32/140 |
| Основной `eq13_scaled` | 2,09775 | 1,0205 | 32/140 |

Все 126 ячеек остальных правил совпали между тремя расчётами в допуске (10⁻⁹) п.п. На первом окне Industry `1963-07…1973-06` с `M=120`, `N=11`, `γ=3` и `c1=1,113404` получены следующие значения.

| Режим | Перекрёстный член | π̂₁₃ | δ до и после ограничения |
| --- | ---: | ---: | ---: |
| `legacy_adjusted` | 0,089977 | 0,007119 | 0,999647 |
| `unscaled_plugin` | 0,166555 | −0,000523 | 0,674956 |
| `eq13_scaled` | 0,148512 | 0,001277 | 0,706588 |

Для этого окна отдельно вычислен обратный ход через масштабированную ковариацию. В регрессионном тесте проверены π̂₁₃, сырой и ограниченный δ, итоговые веса CKZ и совпадение KZ между режимами. Архивные симуляции используют `legacy_adjusted`. Правка части 2 ограничена поддержкой формулы, регрессионным тестом и записью P2‑06. [14 парных CER](../results/replication/ckz_comparison.csv) · [Входы, q и полные веса фиксированного окна](../results/replication/ckz_fixed_window.json).

```sh
python -m replication.backtest --mode full --ckz-mode legacy_adjusted --out runs/review/legacy_adjusted
python -m replication.backtest --mode full --ckz-mode unscaled_plugin --out runs/review/unscaled_plugin
python -m replication.empirical_diagnostics --out results/replication \
  --ckz-controls runs/review/legacy_adjusted runs/review/unscaled_plugin
```

## 3. Данные и CMP

- **FF‑4 · отклонено изменение входа.** DGU ZIP совпал с ожидаемым SHA‑256, повторная сборка дала те же панели. На 497 месяцах `1963-07…2004-11` базовый rolling ML независимо даёт −129,0261 и −38,8408. При тех же делителях подстановка MOM из FIZ даёт −129,7917 и −38,4586, а из CIZ — −128,9716 и −37,5126. Ни одна из 12 комбинаций версии MOM и делителей не воспроизводит обе опубликованные ячейки. CSV сохраняет каждый результат, хеш входной панели и хеш заменившего MOM. [Диагностика FF‑4](../results/replication/ff24_diagnostics.csv).
- **FF25 и Indu49 · проверка точного винтажа заблокирована.** Для расчёта подтверждены FIZ 202412, месячный value-weighted блок, вычитание RF один раз, три фактора и рабочие интервалы `1963-07…2007-08` и `1969-07…2007-08`. Ранние месяцы с пропусками для 49 отраслей исключены. Перепроверены CIZ, равновзвешенные портфели, альтернативное окончание и исторический FF3 200807. Исходные версии двух портфельных файлов и даты авторов для таблицы 6 не установлены. Каждая альтернатива привязана к собственным хешам ZIP. [Диагностика French](../results/replication/french_sensitivity.csv) · [Описание данных](data.md).
- **CMP · уже исправлено, различие перепроверено.** Текст перед §3.3 на с. 211 указывает 50/50 для MP. В таблице используется `cmp50`, вариант jackknife учитывается как `cmp`. Среднее абсолютное отклонение строки в статье — 0,020 п.п. для 50/50 и 0,968 п.п. для jackknife. На одинаковых датах CMP50 улучшает MP в 14/14 случаях, а jackknife — в 12/14. [Парные проверки](../results/replication/empirical_checks.csv) · [Исходный текст статьи](https://down.aefweb.net/WorkingPapers/w715.pdf#page=8).

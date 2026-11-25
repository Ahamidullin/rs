# Data and backtest for Table 6

The replication follows Tu and Zhou, §3.3, pp. 211–214, Table 6, and equation (22). It uses risk aversion `γ=3` and estimation windows `M=120` and `M=240`. [Article](https://doi.org/10.1016/j.jfineco.2010.08.013).

| ID | Source and assets | First and last months | T | OOS 120 / 240 |
| --- | --- | --- | ---: | ---: |
| industry11 | DGU `Industry.txt`, 10 industries and the market factor | 1963-07 – 2004-11 | 497 | 377 / 257 |
| international9 | DGU `international.txt`, 8 countries in USD and World | 1970-01 – 2001-07 | 379 | 259 / 139 |
| factors3 | DGU `F-F_Research_Data_Factors.txt`, SMB, HML, Mkt-RF | 1963-07 – 2004-11 | 497 | 377 / 257 |
| ff21 | DGU `25_Portfolios_5x5_1Factor.txt`, 20 portfolios and Mkt-RF | 1963-07 – 2004-11 | 497 | 377 / 257 |
| ff24 | DGU `25_Portfolios_5x5_MOM.txt`, the same 20 portfolios plus SMB, HML, MOM, and Mkt-RF | 1963-07 – 2004-11 | 497 | 377 / 257 |
| ff28 | French `25_Portfolios_5x5_CSV.zip` and FF3, 25 size/book-to-market portfolios and three factors | 1963-07 – 2007-08 | 530 | 410 / 290 |
| industry52 | French `49_Industry_Portfolios_CSV.zip` and FF3, 49 industries and three factors | 1969-07 – 2007-08 | 458 | 338 / 218 |

- The DGU archive matched the SHA-256 in the team plan. Its asset series are monthly excess returns in decimal units. RF is stored separately. For `international9`, the source calls RF `T-bill(cont.comp)`. The series is used as supplied, following the DGU `LOG=0` convention. For `ff21`, the delimiter immediately before the market factor can be whitespace. For `ff24`, the missing early MOM observation predates the selected sample. [DGU author’s data page](https://sites.google.com/view/victor-demiguel).
- The two additional samples in the main run use monthly value-weighted FIZ returns from the CRSP 202412 vintage. Subtract RF from the portfolio total returns. Mkt-RF, SMB, and HML already have the required interpretation. Convert the factors and RF from percentages to decimals and join the inputs one-to-one by month. Treat `-99.99` and `-999` as missing before conversion. [FIZ archive](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/data_library_202412_archive.html).
- Section 3.3 lists the composition of the additional samples. The ranges starting in July 1963 or July 1969 and ending in August 2007 are working reconstructions. Both data vintages have at least one missing industry in `industry52` before July 1969. The main end date of August 2007 comes from §3.1, where the authors use it to calibrate FF3. Relative to an end date of November 2004, it gives 1/N CERs that are closer to the published benchmarks for both windows. The end-date choice alone does not reproduce in-sample ML. [Sensitivity to dates, weights, and vintage](../results/replication/french_sensitivity.csv).
- The checks cover CIZ 202609 over the same dates, both end dates, equal weighting within the source portfolios, and a hybrid of FIZ with official July 2008 FF3. The paper’s original vintages for the 25 portfolios and 49 industries were unavailable. For example, `industry52` in-sample ML is 54.01 with FIZ, 54.07 with CIZ, and 55.69 with FIZ plus July 2008 FF3. The paper reports 57.67. [Input revisions](../results/replication/french_input_revisions.csv) · [Historical FF3](https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/f-f_factors_archive.html).

## Calculation

For realized month `i`, the code fits weights using `X[i-M:i]`. The outputs record the last estimation month and the following realization month. For `industry11` with `M=120`, the first estimation window runs from July 1963 to June 1973, first realization is July 1973, and `n=377`. The first equal-weight excess return is `0.04552727`, with weight `1/11` per series and zero cash. For `M=240`, the first realization is July 1983. The two window lengths share an end month and cover different out-of-sample periods.

Annualized CER in percentage points is `1200 × (mean(r) − γ/2 × var(r, ddof=1))`, where `r` contains monthly excess returns. Rolling ML uses a covariance denominator of `M−1`. Retrospective ML-IS uses a covariance denominator of `T` and fits and applies the weights to the entire panel. Table 6 includes this same benchmark in both panels.

The main table uses `CMP50` for the CMP row, following the interpretation suggested near the end of §3.2 (p. 211). We also report the mixture with the jackknife coefficient from equation (21). Across 14 cells, the mean absolute difference from the published row is 0.020 percentage points for CMP50 and 0.968 for the jackknife. CMP50 improves on MP in 14/14 cases and beats 1/N in 0/14. The jackknife improves on MP in 12/14. For footnote 4, equal weighting with cash improves on 1/N in 1/14 cases and has a median difference of −0.057 percentage points. PJ50 improves on the implemented CPJ in 9/14 cases. KZ50 and ML50 improve on the implemented CKZ and CML in 12/14 and 4/14 cases, respectively. [Paired checks](../results/replication/empirical_checks.csv).

## Validation and discrepancies

- Of 140 cells, 32 fall within ±0.005 percentage points. On the five DGU panels, 1/N matches in 10/10 cells, rolling ML in 8/10, and ML-IS in 8/10. [Cell-by-cell comparison](../results/replication/table6.csv).
- For `ff24`, rolling ML gives −129.0261 and −38.8408 versus −128.59 and −38.63 in the paper, and ML-IS gives 54.5640 versus 54.55. Varying the covariance and CER degrees of freedom and replacing only MOM with FIZ or CIZ did not recover both windows. The closest value in panel A is still 0.294 percentage points below the published figure, while the closest in panel B is 0.020 above. The main run keeps the complete DGU series. [MOM and normalization checks](../results/replication/ff24_diagnostics.csv).
- CPJ and CKZ use the implementations from Part 2. Some DGU mixtures diverge where 1/N and rolling ML match. For example, CKZ for `industry11` with `M=120` is −0.59 versus 3.02 in the paper. This remains an open method discrepancy. CKZ beats 1/N in 10/14 replication cases and 11/14 published cases. [Part 2 decisions](decisions_part2.csv).
- Independent validation checks keys, dates, RF, and every return by multiplying the weights by the following month’s returns. It also checks weight sums including cash and `gross_risky`. A test changes the realization-month row and subsequent rows and verifies the existing weights. [First- and last-month checks](../results/replication/empirical_endpoint_checks.csv).

A rerun downloaded the sources into a separate directory. All 140 values, paired CERs, monthly returns, and weights agreed with the first run. The comparison excluded run IDs, which depend on download metadata.

## Interface for Part 4

`rolling_backtest(panel, M, gamma, policy)` accepts a canonical panel with `month,rf,asset_*`. The call `policy(history, base_fit, state)` receives a copy of the history through the estimation endpoint, a dictionary of weights and diagnostics for the base rules, and a mutable state dictionary. The policy returns a `fit_rules`-style dictionary with `weights` and optional `delta`, `delta_raw`, and `diagnostics`. The backtest returns two tables, monthly returns and weights.

`monthly_returns.parquet` includes only out-of-sample strategies, including CMP50 and the jackknife. `weights.parquet` contains all target weights, including CASH. The README commands rebuild both intermediate tables and the seven input panels. [Run manifest](../results/replication/empirical_manifest.json) · [Sources for additional checks](../results/replication/empirical_diagnostics_sources.json).

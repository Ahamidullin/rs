# Part 4. Trading costs, stability, and figures

Two experiments extend the Table 6 backtest on seven datasets with γ = 3 and M = 120 or 240. The Part 2 implementation of CKZ serves as the baseline. [Paper walkthrough](docs/article_walkthrough.md) · [Defense conclusions](defense/extension_conclusions.md) · [Figure captions](defense/figure_captions.md) · [Project Q&A](QA.md).

| Comparison | Median Δ CER, annual pp | Wins / complete pairs | Conclusion |
| --- | ---: | ---: | --- |
| Turnover-aware CKZ vs. CKZ at 25 bps, net | +5.70 | 9/12 out of 14 | Preliminary support. Median paired change in turnover was −4.04 times NAV. |
| Stability-adjusted CKZ vs. CKZ at 0 bps, gross | +0.69 | 9/14 | The comparison meets some of the criteria. |
| Stability-adjusted CKZ vs. CKZ50 at 0 bps, gross | −3.27 | 5/14 | The stability hypothesis did not meet the full set of criteria. |

The comparisons are conditional on the project's CKZ implementation, the archived datasets, and the linear cost model.

## Data and reproduction

The original folder contained PDFs, Parts 2–3 result CSVs, manifests, and an eight-month fixture. The raw ZIP archives and Parquet files were recovered. The SHA‑256 hashes of all four ZIP archives match the Part 3 manifest. DGU supplies five datasets and French FIZ 202412 supplies two. The team reconstructed the date ranges for the French datasets. The exact historical portfolio vintages used in the paper are unknown. [Sources and date ranges](docs/data.md).

Run the commands from rs-main. Dependencies are listed in requirements-part3.lock. Verification used Python 3.13.1 with the NumPy, SciPy, pandas, PyArrow, and Matplotlib versions from the lock file.

~~~sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m replication.data --download
python -m pytest -q

python -m replication.backtest --mode full --out runs/part3_rebuilt
python -m replication.ext_turnover --baseline runs/part3_rebuilt --out results/extensions/turnover
python -m replication.ext_stability --baseline runs/part3_rebuilt --out results/extensions/stability
python -m replication.figures --replication runs/part3_rebuilt --out results/figures
~~~

With the saved Table 6 results, the full backtest can be skipped. Run the three Part 4 commands with results/replication as the argument to --baseline and --replication. This checks CKZ and 1/N dates and CER for all 28 baseline cells. When Parquet files are available in the selected directory, the commands also verify file hashes, monthly returns, and monthly weights. Each extension fits the main rules once per month and reuses those fits across cost settings. Neither extension requires the MP fit or jackknife. The full Part 3 backtest took about 72 seconds and each extension took several seconds on the verification machine. The figures use the saved Monte Carlo results.

The rebuilt backtest is stored in runs/part3_rebuilt. Its 140 CER values match the supplied table within a maximum absolute difference of 1.48 × 10⁻¹¹ pp. The new run ID incorporates the software environment and data sources.

## Protocol

### Turnover-aware CKZ

Write w(δ) = e + δd, where e is 1/N and d = vKZ − e. Estimate the curvature K = γd′S₁d with sample covariance S₁ from the historical window. Evaluate 101 values on [0, 1] in steps of 0.01, plus the baseline δ̂, using the monthly objective ½K(δ−δ̂)² + k‖w(δ)−p‖₁. Break ties by choosing the candidate closest to δ̂, then the smaller candidate. When k = 0, use the exact baseline CKZ weights. At positive costs, both the turnover-aware rule and every baseline pay to trade. Costs of 0, 10, 25, and 50 bps were specified in advance, with 25 bps as the primary setting.

Before the first forecast, p = e. Thereafter, each rule carries forward its own weights after the market move. Given monthly excess returns X and a risk-free rate RF, the gross portfolio growth factor is G = 1 + RF + w′X, and the drifted weight is p′ᵢ = wᵢ(1 + RF + Xᵢ)/G. Drifted cash is cash × (1 + RF)/G. The cost k × Σ|wᵢ−pᵢ| is applied to beginning-of-month NAV and withdrawn proportionally at month-end. Net excess return is w′X less the cost. Factors are treated as synthetic excess-return positions whose collateral earns RF. The international9 series uses the supplied RF.

For a manual check, let p = (0.5, 0.5) and w = (1.1, 0.5). Turnover is 0.60 times NAV and the cash weight is −0.60. At 25 bps, the cost is 0.0015 times NAV. If X = (0.02, −0.01), gross excess return is 0.017 and net excess return is 0.0155. At RF = 0.003, G = 1.020 and next month's pre-trade weight of the first position is 1.1 × 1.023 / 1.020.

### Stability-adjusted CKZ

Fit the original KZ estimator on two adjacent halves of the current window to obtain v₁ and v₂. The score s divides the risk of v₁−v₂ by the risk of the full-window direction vKZ−e plus 10⁻¹⁶. Both risks use the full-window covariance. The mixing coefficient becomes δ̂/(1+s). If the full-window direction is zero, set the coefficient to zero. If v₁ = v₂ and the direction is nonzero, s = 0 so the coefficient stays at δ̂. The final KZ vector comes from the full window. Evaluate costs of 0 and 25 bps. The additional CKZ50 baseline uses the same full-window vector.

M/2 is 60 or 120, greater than N+4 even for N = 52. The covariance condition number is stored for each half. A fitting error identifies the dataset, window, and date. The largest condition number was about 4.16 million. Every half-window fit succeeded. After the first 12 months, the high/low labels compare the current s with the median of past scores. Return differences and CER within each group are conditional on its assigned months.

### Evaluation

- Both paths in each pair use the same dates. A nonpositive gross or net NAV triggers a stop. Subsequent dates remain in the output with a status flag. Full-period CER is blank. The prefix columns report statistics and separate intervals for the common period before the stop.
- Use 2,000 paired circular bootstrap resamples with 12-month blocks and seed 201104. Both paths use the same resampled indices and CER is recalculated for each. The 2.5th–97.5th percentile intervals are conditional on the realized series and fixed rules. The rules are not refit. Win counts across overlapping datasets are descriptive.
- To aggregate across both windows, first average the complete pairs within each dataset, then give equal weight to each available dataset. The median and win rate use individual complete pairs. The turnover criterion requires a positive median net CER difference, at least 8 positive differences out of 14 pairs, and a negative median paired change in turnover. Incomplete pairs make the result preliminary. The stability criterion requires a positive median gross CER difference and at least 8 wins against CKZ, plus a nonnegative median difference against CKZ50.

## Output files

| File | Contents |
| --- | --- |
| summary.csv | One row per dataset, window, cost, and baseline combination. Gross and net CER, both Sharpe measures ×100, turnover, mean δ and gross exposure, differences and intervals, coverage, and stops. |
| monthly.parquet | Each OOS date by rule and cost. Target, pre-trade, and drifted risky weights in manifest asset order, cash, costs, returns, diagnostics, and status. |
| aggregate.csv | Equal-weight dataset mean, median paired difference, wins, turnover, and coverage. |
| baseline_checks.csv | CKZ and 1/N checks for 7 datasets × 2 windows, including dates. With Parquet files, also reports the maximum absolute monthly differences in returns and weights. |
| turnover/break_even.csv | Adjacent costs where Δ net CER goes from nonpositive to positive and gross CER at the higher cost is below the baseline. Multiple crossings are possible. |
| stability/groups.csv | High, low, and unclassified months. Count, mean and sum of excess-return differences, and subset CER. |
| manifest.json | Run ID, versions, parameters, and hashes for code, archives, tables, and outputs. |

[Turnover comparisons](results/extensions/turnover/summary.csv) · [Stability comparisons](results/extensions/stability/summary.csv) · [Figures and captions](defense/figure_captions.md).

## Work completed before Part 4

- **Part 1** covered the paper and produced the initial slides and presentation notes. [Slides](defense/slides.pdf) · [Speaker notes](defense/outline.md).
- **Part 2** implemented eight estimated rules and 1/N, the data generators, Monte Carlo simulation, aggregation, Tables 1–3 comparisons, and footnote checks. MP uses profile optimization. CMP leave-one-out reuses sufficient statistics and batches Newton solves. The saved full run includes 300 comparisons, with 133 marked open under the team's threshold. The team stopped before running full simulations for Tables 2B, 4, and 5. [Rule implementations](src/replication/methods.py) · [Part 2 status](docs/part2_status.md).
- **Part 3** implemented data acquisition and validation, rolling OOS and in-sample ML backtests, 140 comparisons, monthly weights and returns, French vintage and MOM diagnostics, and presentation materials. Table 6 uses CMP50 following §3.2, with jackknife CMP checked separately. Thirty-two of the 140 cells match the publication within ±0.005 pp. [Output table](results/replication/table6.csv) · [Data preparation](docs/data.md).

## Inherited code audit

1. **The CKZ implementation differs from equation (13) on p. 207.** The final term in the PDF uses the direct plug-in m′S₀⁻¹q, where q = η̂m + (1−η̂)ĝ·1. The code uses an identity evaluated with separately adjusted estimates of θ² and ψ². The identity holds for the population quantities. The two plug-in calculations give different results. A diagnostic replaced this term with m @ solve(S0, q), keeping KZ, π̂₁, π̂₃, and coefficient clipping fixed. The mean absolute discrepancy across 14 CKZ CERs fell from **5.77 to 1.41 pp**. For Industry at M = 120, the project baseline was −0.59 and the diagnostic returned 1.19, against 3.02 in the paper. For FF‑1 at M = 120, the figures were 20.79 and 29.29, against 28.97. These differences warrant a separate check of equation (13). Using the accepted control in the extensions maintains comparability with Part 3. [CKZ implementation](src/replication/methods.py#L86) · [Decision log](docs/decisions_part2.csv).
2. **Simulation resume checks whether a chunk already exists.** The filename encodes the scenario, T, and replicate numbers. The cache check omits formula versions and seeds. Rerunning in the same directory after a code change can mix old and new outputs. Use a new directory for each model specification. [Simulator](src/replication/simulate.py#L30) · [Aggregator](src/replication/aggregate.py).
3. **Data differences account for some discrepancies.** The later FIZ vintage and reconstructed dates affect industry52 and ff28. For ff24, a discrepancy remains even for rolling ML. CKZ and CPJ also differ on DGU datasets where EW and ML agree. The equation (13) diagnostic shows that the implementation also matters. [Data checks](docs/data.md).

## Further checks

1. Repeat both extensions with the direct plug-in from equation (13) as a prespecified alternative baseline. This tests how much of the gain depends on the original implementation's turnover and formulas.
2. Compare the stability rule with a constant δ fitted on a separate earlier period and use a common OOS period for both windows. This tests the incremental effect of the dynamic score at comparable average shrinkage over the same market months.
3. Test the reserve specification with diagonal covariance shrinkage for the KZ direction. Keep δ̂ and the estimated means fixed, and replace S₀ in the weight solve with (1−λ)S₀ + λ·diag(S₀). Use λ = 0.25 as the main setting, λ = 0 as the control, and 0.10 and 0.50 for sensitivity. Compare CER and leverage, particularly for ff28 and industry52. The plan reserves this experiment until the primary results are accepted.
4. After verifying the formulas, test sensitivity to French data vintages, the treatment of international9 RF, and different trading costs across risky positions.

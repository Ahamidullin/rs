# Part 2 status as of 18 November

## Completed work

1. **MP maximum likelihood estimation (Appendix C).** Profile likelihood reduces to a problem in the unit direction u. The objective is F(u) = (N−1) ln(tr U − u′Uu) + ln(u′Uu − (u′m)²), where U = S0 + mm′. Stationary points satisfy (mm′ + t S0)u = nu·u, with t = 1 − s2/c in [0,1). The boundary a = 0 corresponds to t = 0. The estimates are mu_MP = (u′m)u, c = u′S0u, s2 = (tr U − u′Uu)/(N−1), and w_MP = mu_MP/(gamma c). The result agrees with BFGS on the full (mu, s2, a) parameterization. Starting from this solution, Nelder–Mead returns the same negative log likelihood to within 4e−10. The profile formulation bypasses the ambiguity over an extra “−1” in equation (35).
2. **Exact leave-one-out (LOO) for CMP.** A batched Newton iteration over (u, nu, t) fits all T deletions, starting from the full-sample fit. The residual tolerance is 1e−13. Nonconverged fits and cases with t < 0 are globally refitted and counted in n_refit. The difference from directly refitting every deletion was 2.3e−15. At T = 6,000, the runtime fell from 39.4 to 0.38 seconds per replication. All six T values take around 0.82 seconds per replication, or about 2.3 hours per scenario on one core. Since w_MP is proportional to 1/gamma, the two gamma settings share the same LOO fits.
3. **A0/Aalpha data-generating process (DGP).** The adopted DGU design treats the factor as an investable asset, giving N = 25 from 24 assets plus the factor. With 25 assets excluding the factor, the Table 1A True benchmark is around 4.01 and reached 4.17 in none of 50,000 sigma draws. With the factor, True = 1200 × (muf²/varf)/(2 gamma), yielding 4.1667 and 12.50 regardless of the sigma vector. The selected sigma vector comes from `default_rng(3267).uniform(0.10,0.30,24)`, the first draw that simultaneously reproduced the six True/EW benchmarks in 1A, 1B, and 2A at published precision. Alternative seeds include 4776, 6561, and 8516. An analogous F3alpha design would have 22 assets and three FF3 factors. Its Table 2B benchmarks would be True = 14.60 and EW = 3.85.
4. **CMP mixture weight.** Equation (18) simplifies to delta = (e−w*)′Sigma(e − E[v]) / E[(e−v)′Sigma(e−v)], with w* entering the expanded expression only through mu′E[v]/gamma. The initial plan mixed a plug-in w* = S0⁻¹m/gamma in C and B with an adjusted pi1 in A. That produced an estimated norm near 18 versus a true norm near 1 and drove delta to 0 or 1. The adopted calculation uses jackknife estimates of E[v] and E[v′S0v], with Sigma estimated by S0 and mu by m.

## Preliminary CMP check for Table 1A

This check used R = 200 for T ≤ 960 and R = 20 for T ≥ 3,000.

| T | Computed | ±MCSE | Published |
|---|---|---|---|
| 120 | 2.31 | 0.23 | 2.19 |
| 240 | 3.09 | 0.09 | 3.05 |
| 480 | 3.54 | 0.05 | 3.48 |
| 960 | 3.70 | 0.02 | 3.67 |
| 3000 | 3.83 | 0.03 | 3.80 |
| 6000 | 3.87 | 0.01 | 3.83 |

At T = 120 and R = 2,000, the Table 1A estimates (published benchmarks in parentheses) were ML −86.99 ± 0.88 (−85.72), MP 2.10 ± 0.06 (2.11), KZ −2.38 ± 0.11 (−2.15), CML 1.84 ± 0.09 (1.68), and PJ −13.55 ± 0.27 (−12.85).

## Team decision on 18 November

At that stage the team used working reconstructions of CKZ and CPJ and deferred the F3alpha simulations for Tables 2B, 4, and 5. [Part 2 decision log, P2-06 and P2-07](decisions_part2.csv) · [Later CKZ audit](../EXTENSIONS.md#inherited-code-audit).

## Preliminary run with 1,000 replications per cell

Of 300 cells across Tables 1A, 1B, 2A, 3A, and 3B, 77 matched, 131 were Monte Carlo compatible, and 92 remained open. [Preliminary results](../results/prelim_R1000/tables1_3_comparison.csv).

- ML, PJ, KZ, and CML were generally close. For example, Table 1A ML at T = 120 was −86.92 versus −85.72, with an MCSE of about 0.9.
- MP ran 0.05–0.20 higher at large T. The limiting Table 3A Sharpe ratio was 13.99 versus 13.89. The MP limit depends on the authors’ unknown sigma vector.
- CPJ was close in Table 1A at T ≤ 960. The paper has nonmonotone values, including 7.46 in 1B at T = 3,000 and 3.92 in 2A at T = 960, whereas the replication results were smooth.
- CMP was close in Tables 1A, 2A, 3A, and 3B. In 1B at T = 120, it was 5.48 versus 6.57.
- CKZ was lower at T = 120, with 1.26 versus 3.71 in Table 1A. The values draw closer at T ≥ 960. See P2-06.

## Full run with 10,000 replications per cell

The 18 November run took 8,344 seconds on two CPUs. The first 1,000 replications in each cell share the preliminary seeds indexed by scenario, T, and replication, and use matching chunk names. Of 300 cells, 88 matched, 79 were Monte Carlo compatible, and 133 remained open. The compatibility threshold is 0.005 + 3 MCSE, which is narrow at R = 10,000. [Full results](../results/sim_full/tables1_3_comparison.csv).

- True and EW matched in 60/60 cells. ML was within Monte Carlo tolerance in 29/30, KZ in 27/30, and PJ in 24/30.
- CML, MP, and CMP showed systematic differences of 0.03–0.25 percentage points. At large T this reflects the unknown sigma vector used by the authors. The limiting Table 3A MP Sharpe ratio was 13.99 versus 13.89.
- CPJ was close at T ≤ 960 in Tables 1A, 1B, and 2A. The replication did not recover the paper’s nonmonotone large-T results.
- CKZ was lower at T = 120, with 1.38 versus 3.71 in Table 1A. The values draw closer at T ≥ 960. See P2-06.

## Footnotes 8–9 sensitivity checks with 1,000 replications per cell

The authors’ DGPs for these cases are unpublished. Both specifications are project choices, documented in P2-08, with gamma = 3. [Paired sensitivity results](../results/footnotes89/footnote89_paired.csv).

- **Aa_N50.** There are 49 assets plus the factor, with alpha as in Table 2A. True utility is 7.76 and EW is 4.03. The four mixtures improve on their components at every T except CMP versus MP at T ≥ 240. They beat 1/N starting at T = 480 for CML, CPJ, and CKZ, and T = 3,000 for CMP. This supports the qualitative conclusion in footnote 8.
- **NF.** There is no common factor, mu = alpha + 0.08/12, and Sigma = D. True versus EW utility is 99.4 versus 7.7, and the corresponding Sharpe figures are 70.5 versus 53.6. CML, CPJ, and CKZ beat 1/N from T = 120. CKZ is close to KZ, with delta near 1. MP and CMP perform poorly because the MP model is misspecified in this scenario. The NF Sharpe results for footnote 9 are in `footnote89_summary.csv`.

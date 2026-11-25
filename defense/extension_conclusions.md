# Extension conclusions for the defense

The two experiments use seven historical datasets and 120- or 240-month estimation windows. CER differences are annualized and reported in percentage points. The CKZ baseline uses the code from the accepted empirical replication. [Data, protocol, and audit](../EXTENSIONS.md).

## 1. Turnover-aware CKZ

At **25 bps**, turnover-aware CKZ beats baseline CKZ in **9 of 12 complete pairs**. The median Δ net CER is **+5.70 pp**, with a range of **−5.16 to +36.89 pp**. The median paired change in monthly turnover is **−4.04 times NAV**. Median turnover is 4.06 times NAV for baseline CKZ and 0.025 for the turnover-aware rule. The median Δ gross CER is −2.43 pp. The savings on trades outweighed the typical shortfall before costs. In three of the 12 pairs, the 95% block bootstrap interval for the net CER difference lies entirely above zero.

At M = 240, CKZ had nonpositive NAV after costs in March 2000 for FF‑4 and in April 2000 for FF25. Full-period CER is unavailable for these pairs. On the common prefixes of 200 and 201 months, Δ net CER is **−12.84 pp** and **+16.71 pp**, respectively. The result under the prespecified criterion is therefore preliminary.

At 10 bps, the median Δ net CER is +6.24 pp and 11/14 pairs are positive. At 50 bps, the median is +15.93 pp and 11/12 pairs are positive. For Indu49 at M = 240, Δ net CER becomes positive between 10 and 25 bps. At 25 bps its gross CER is 5.90 pp below CKZ. For FF‑1 at M = 240, the sign changes more than once across the grid. At 25 bps, Δ net CER is −5.16 pp, and at 50 bps it is positive. [Sign changes across the cost grid](../results/extensions/turnover/break_even.csv).

At 25 bps, the turnover-aware rule chooses δ = 0 in about **77%** of out-of-sample months across all paths. It has 2 wins, 9 CER ties, and 3 losses against 1/N, with a median difference of zero. Against KZ, it wins 9 of 12 complete pairs and has a median net CER gain of +8.63 pp. Its low turnover follows from the strong tilt toward 1/N. The size of the gain against CKZ is sensitive to baseline turnover and portfolio weights. [Paired results](../results/extensions/turnover/summary.csv) · [Aggregates](../results/extensions/turnover/aggregate.csv).

## 2. Stability-adjusted CKZ

At **0 bps**, the median Δ gross CER against CKZ is **+0.69 pp** and the rule wins **9 of 14** complete pairs. The range is −5.62 to +14.20 pp. In two pairs, the 95% block bootstrap interval lies entirely above zero. Against CKZ50, the median is **−3.27 pp** and the rule wins **5 of 14** pairs. Six intervals lie entirely below zero. The result **fails to meet the full set of criteria** for the stability hypothesis. The rule wins 13/14 pairs against 1/N and 11/14 against KZ.

Across all 4,190 out-of-sample months, the median applied KZ mixing coefficient falls from **0.94 to 0.22**. The time-series standard deviation of δ decreases in 8 of 14 paths. The level of δ falls more consistently than its variability. The median group-specific CER difference across the 14 pairs is **−4.13 pp** in the high-score group and **+4.76 pp** in the low-score group. For FF25 at M = 240, the differences are −10.78 pp over 125 high-score months and +17.51 pp over 153 low-score months. Use the monthly return differences in groups.csv to assess each group's contribution. Group CERs are nonadditive.

At **25 bps**, the same stops leave 12 complete pairs against CKZ. Seven have positive gross CER differences, with a median of +0.29 pp. The median net CER difference across those pairs is +7.59 pp. This setting stress-tests the common trading cost. The zero-cost setting remains the primary stability comparison. [Paired results](../results/extensions/stability/summary.csv) · [Score groups](../results/extensions/stability/groups.csv).

The 14 pairs overlap in dates and assets. The intervals measure uncertainty conditional on the realized return paths and the fixed rules. [Turnover run aad093ed9437441c](../results/extensions/turnover/manifest.json) · [Stability run ef8905b358179511](../results/extensions/stability/manifest.json).

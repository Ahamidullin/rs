# Figures and captions for the defense

All four PNGs are 1920 × 1080 pixels. Annual percentage points are obtained by multiplying monthly utility or CER in decimal units by 12 × 100. [Figure sources and hashes](../results/figures/manifest.json).

## 01. When do the blends catch up with 1/N?

[Open 01_simulation.png](../results/figures/01_simulation.png).

The horizontal axis is the number of training months T on a logarithmic scale. The panels show a one-factor model with zero α, a one-factor model with nonzero α, and a three-factor model with nonzero α. Solid colored lines represent CML and CKZ. Dashed lines represent ML and KZ. The gray and black lines represent 1/N and True. The strip below each panel shows ML on a scale that includes the extreme negative values. In Tables 1A and 2A, circles represent replicated values, crosses represent published values, and bands span ±1.96 Monte Carlo standard errors. True and 1/N match after rounding. Table 2B uses the authors' published values on p. 210 because the team did not run its Monte Carlo simulation.

- In 1A, CKZ at T = 120 is **1.38 in the replication and 3.71 in the paper**. On the plotted grid, published CKZ first exceeds 1/N at T = 3,000 and replicated CKZ first exceeds it at T = 6,000.
- In 2A, CKZ at T = 240 is **3.08 in the replication and 3.95 in the paper**. Published CKZ exceeds 1/N at 240 months and replicated CKZ at 960 months. ML at T = 120 is **−84.57 in the replication and −84.75 in the paper**.
- In 2B at T = 120, the paper reports **5.09 for CKZ**, **3.85 for 1/N**, and **14.60 for True**. The comparison shows how much the gain from combining depends on the gap between 1/N and the optimum.

[Saved Part 2 Monte Carlo results](../results/sim_full/tables1_3_comparison.csv) · [Published Table 2B values](../references/table2b_figure.csv).

## 02. Where does the replication differ from Table 6?

[Open 02_empirical_gap.png](../results/figures/02_empirical_gap.png).

Each cell shows replicated minus published CER for a method, dataset, and window length. Blue represents negative differences and red represents positive differences. A symmetric logarithmic scale makes both small and large gaps visible. Diamonds after dataset names identify the two French FIZ datasets. DGU supplies the other five. The side panel shows CKZ CER minus 1/N CER for the same data. Crosses represent published values and circles represent replicated values.

- At M = 120 on **Indu49**, the gaps are **+142.28 pp for ML** and **−17.44 pp for CKZ**.
- At M = 240, the gaps are **+42.55 pp for CPJ on FF‑1** and **−13.00 pp for CKZ on FF25**.
- On **Industry at M = 120**, CKZ minus 1/N is **−4.25 pp** in the replication and **−0.64 pp** in the paper. CKZ beats 1/N in 11/14 published pairs and 10/14 replicated pairs.

Industry, FF3, FF‑1, and FF‑4 cover July 1963 to November 2004. Intl covers January 1970 to July 2001. FIZ FF25 covers July 1963 to August 2007, and FIZ Indu49 covers July 1969 to August 2007. The out-of-sample period begins after the first M months. [Comparison table](../results/replication/table6.csv) · [Dataset inventory](../docs/data.md).

## 03. Does reducing CKZ turnover pay off?

[Open 03_turnover.png](../results/figures/03_turnover.png).

The horizontal axis shows the cost per unit of two-sided turnover in risky assets. The vertical axis shows Δ net CER between turnover-aware CKZ and baseline CKZ at the same trading cost. Thin lines represent individual datasets. The thick line is the median of complete pairs at each M. Crosses mark incomplete pairs. The text above each panel gives median turnover at 25 bps among complete pairs.

At M = 120 and 25 bps, summary.csv gives **+17.37 pp for Industry**, **+28.22 pp for FF25**, and **−0.38 pp for FF3**. At M = 240, the full CER is unavailable for FF‑4 and FF25 because CKZ stopped. At 0 bps, the rules agree exactly across all 14 pairs. Table 6 in the paper reports gross CER. The zero-cost setting connects to that protocol, and the positive-cost settings are part of the extension. [Turnover results](../results/extensions/turnover/summary.csv).

## 04. Does additional shrinkage improve CKZ?

[Open 04_stability.png](../results/figures/04_stability.png).

Points show Δ gross CER at 0 bps for each dataset and M. Teal circles compare with CKZ, and blue diamonds compare with CKZ50. Lines show 95% paired circular block bootstrap intervals. The side tables show Δ CER versus CKZ in high- and low-score months, with month counts in parentheses. The first 12 months of each path are unclassified.

At M = 120, summary.csv and groups.csv give the following differences from CKZ in pp: **Industry +4.43 [0.02, 8.63]**, **FF‑1 −0.13 [−15.68, 14.45]**, and **Indu49 +14.20 [2.28, 26.80]**. For Indu49, the high group gives **+11.80 (161)** and the low group gives **+14.37 (165)**. Table 5 in the paper compares true and estimated mixing coefficients in simulations. This extension uses historical data and tests stability across the two halves of each estimation window. [Stability results](../results/extensions/stability/summary.csv) · [Score groups](../results/extensions/stability/groups.csv).

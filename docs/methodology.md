# Statistical conventions and data provenance

## Portfolio problem

Monthly asset returns are in fractions. For risky NAV weights w, excess portfolio return is w'R and residual cash is 1−sum(w). Shorting and borrowing are allowed; risky weights are not normalized to sum to one. The objective is w'μ−γ/2·w'Σw, with γ=3 in the historical study. CER is 1200·(mean excess return−γ/2·sample variance), with variance ddof=1. This is a mean–variance evaluation score rather than compound wealth growth.

The main rules are EW, estimated Markowitz (ML), Jorion (PJ), MacKinlay–Pastor (MP), Kan–Zhou (KZ) and the four EW mixtures CML/CPJ/CMP/CKZ. CML's finite-sample-scaled component differs from standalone historical ML. Historical ML uses S1; the formula derivations use S0 and the stated finite-sample corrections. The fixed 50/50 MP mixture in historical Table 6 is labeled CMP50 in code; jackknife CMP remains separate.

## Main-paper simulation

Scenarios A0, Aa and F3a use N=25, six sample sizes and 10,000 replications. A0 has a single market factor and zero alphas; Aa adds nonzero alphas; F3a uses three factors and nonzero alphas. Population μ and Σ score the fitted weights. Those population quantities are not given to the feasible estimation rules.

Following Section 3.1 of the article, the one-factor market has assumed annual expected excess return 8% and standard deviation 16%. Monthly mean is 0.08/12 and variance 0.16²/12. These are generator inputs, not estimates of the realized synthetic sample. Betas span 0.5–1.5; residual annual volatility spans 10–30%. Nonzero annual alphas span −2% to +2%.

The reconstruction includes investable factors within the 25 assets: 24 test assets plus one factor, or 22 test assets plus three factors. Three-factor moments use 530 months, July 1963–August 2007, from the July 2008 French archive. Residual-volatility seed 3267 and three-factor loading-pairing seed 1971 are reconstruction choices, not identified author seeds. The former was inherited from earlier matching of deterministic calibration anchors; the reconstruction should not be described as a blind recovery of the authors' population.

Table 4 distinguishes across-replication dispersion from the MCSE of a simulated mean. Table 5 uses oracle and estimated combination coefficients. Oracle comparisons are explanatory population benchmarks, not deployable allocation strategies. The simulation checkpoint specification hashes prevent mixing code or configuration revisions.

## Historical data

Five panels use the DGU replication archive. FF28 and Industry52 use archived French FIZ data and reconstructed date endpoints. Main-panel samples end between 2001 and 2007. Exact authors' versions are not fully identified. The published values are transcribed references, not calibration targets for portfolio weights.

Weights for month i use only X[i−M:i]. Each dataset/window has T−M realized months. Windows 120 and 240 have different OOS starts in the main historical study; comparing them confounds history length and calendar. In-sample ML uses the whole panel and is not an investable OOS comparator.

## Retained extensions

At each decision, estimate the raw-history Ledoit–Wolf intensity λ. Replace a model-specific risk matrix A by (1−λ)A+λ·tr(A)/N·I. Preserve that component's signal q=A·v and its original portfolio-mixture coefficient. PJ uses its adjusted covariance and MP its model-implied covariance. Transporting raw-history λ to those derived matrices is a defined experimental modification, not a proof of their optimal shrinkage intensity.

All three studies retain 19 distinct rules: 10 original OOS rules, including EW and separate CMP/CMP50, plus 9 regularized counterparts. Historical regularization uses the original 14 dataset/window calendars. The dimension/window and market studies share 2012–2024 OOS; training history begins in 1992. Industry subsets are fixed, nested and chosen from labels/seeds without selecting on outcomes. The full 49-industry universe appears once per window.

The newer U.S. sources use the French December 2024 FIZ archive. International size/value series use separately pinned later snapshots truncated to 2024-12, with common U.S. RF. These are revised historical data, not point-in-time feeds. Local source pins reject changes to already downloaded archives. A future fresh download of an international file may have a different vintage and need not reproduce the saved results exactly.

Rebalancing costs use absolute risky turnover relative to drifted holdings. The primary rate is 25 bps; zero-cost results are also stored. A nonpositive gross or net NAV stops the affected path. Full-period CER and complete-pair effects are missing for failed paths. Bootstrap intervals use circular 12-month blocks and 2,000 resamples. Subperiods are descriptive and do not reset holdings. Financing spreads, short-borrow costs and market impact are omitted.

Prior unsuccessful experiments informed the retained design. These findings are exploratory, use dependent cases and have no multiple-testing adjustment. Neither conditional intervals nor favorable medians establish universal superiority over EW.

## Sources

- Tu and Zhou (2011), [journal DOI](https://doi.org/10.1016/j.jfineco.2010.08.013), Sections 2–3 and Appendices A–C.
- [DGU replication materials](https://sites.google.com/view/victor-demiguel).
- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
- Exact source and output hashes, environment records and calibration moments are retained in the committed execution manifests.

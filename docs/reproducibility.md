# Reproducibility and publication artifacts

## Environments

Use Python >=3.12. The pinned base environment is `requirements-part3.lock`. The retained studies additionally need `requirements-ml.lock`, which includes the base lock. `requirements-gpu.lock` adds the CUDA backend to the base environment and is optional. To run the studies on CUDA, install both the ML and GPU lock files in the same environment.

The editable install (`python -m pip install -e . --no-deps`) registers `src/replication`; it does not select a CPU/GPU backend or start any calculation. Pytest also reads the source path from `pyproject.toml`. No IDE configuration or machine-specific absolute path is required.

## Separate stages

```sh
python -m replication.pipeline --stage check --device cuda
python -m replication.pipeline --stage smoke --device cuda --out runs/smoke_new
python -m replication.pipeline --stage simulations --device cuda --out runs/article_new
python -m replication.pipeline --stage empirical --device cuda --out runs/article_new
python -m replication.pipeline --stage figures --device cuda --out runs/article_new
```

`all` combines checks, full simulations, empirical evaluation and figures. The additional footnote-8/9 scenarios have a separate `--stage footnotes`. They are not part of the three retained extensions. The published compact snapshot does not substitute old preliminary footnote runs for a newly verified execution.

Individual retained studies use `--experiment regularization`, `dimension_window`, or `markets`. `all` runs those three only. A full main-paper historical baseline with `eq13_scaled` is required for regularization. It includes the weight/return Parquet files and source panels produced by the empirical stage. The optional `--retained` argument cross-checks an older successful CKZ output when present; it is not required for a fresh run.

The source protocol, checkpoint identity, code hashes, model settings and realized dates are written to manifests. Do not reuse a completed study directory for a new execution. Simulation resume is only valid for matching specifications and code hashes. A smoke run, unit tests or GPU parity samples do not establish Monte Carlo replication.

## Compact committed results

`results/article/simulation` contains the completed 10,000-replication comparison tables for Tables 1вЂ“5, auxiliary summaries and the original simulation execution manifest. `results/article/empirical` contains Table 6, empirical checks and the original baseline execution manifest. `results/extensions/<study>` contains all per-case comparison summaries, model scores, case definitions and the original execution manifest.

The original execution manifests are copied byte for byte. They deliberately record the full original outputs, including large local Parquet files and source hashes that are not distributed in the compact snapshot. They are provenance records, not claims that every listed original file is included in Git. Refactoring shared helpers and packaging after the run changes current code hashes; saved execution hashes are not rewritten to disguise that difference. Core estimator formulas were preserved during publication cleanup.

`results/publication_manifest.json` hashes every copied compact artifact and records its original run path. `results/summary.csv` is derived from full-period, 25-bps per-case rows and covers all nine regularized models across all three studies. It records complete/expected pairs, wins against the original, median CER changes, positive conditional intervals, and separate EW denominators. Failed pairs are excluded from medians and numerical win counts, not turned into successes.

```sh
# Verify the committed snapshot without research inputs or downloads
python scripts/verify_results.py

# Rebuild all six publication figures from the committed CSVs
python scripts/build_figures.py

# Export a new completed local run after independently executing it
python scripts/export_results.py --article runs/article_new --studies runs/studies_new
```

The export validates full source-run hashes, including locally held paths, before copying compact artifacts. It does not refit portfolios or rerun simulations. The verifier checks the shipped subset and recomputes the summary from shipped CSVs. A new research run may differ because of backend details, an updated source vintage or unresolved reconstruction choices.

## Publication scope

The public implementation contains the article replication and three retained studies. Retired trial entry points and their obsolete tests are preserved in an ignored local archive. Personal launch helpers, IDE files, preparation notes, speech translations, slide drafts and old result trees remain local and are ignored. Public reproduction instructions are platform-neutral and separate from those personal materials.

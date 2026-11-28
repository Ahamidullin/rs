"""Verify the compact publication snapshot without data downloads or model fitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    manifest = json.loads((RESULTS / "publication_manifest.json").read_text())
    for name, record in manifest["files"].items():
        if sha(RESULTS / name) != record["sha256"]:
            raise ValueError(f"Publication checksum mismatch: {name}")
    if sha(RESULTS / "summary.csv") != manifest["summary_sha256"]:
        raise ValueError("Summary checksum mismatch")
    summary = pd.read_csv(RESULTS / "summary.csv")
    if len(summary) != 27 or summary.duplicated(["study", "model"]).any():
        raise ValueError("Require nine unique models for each retained study")
    for study, expected in (("regularization", 14), ("dimension_window", 33), ("markets", 4)):
        frame = pd.read_csv(RESULTS / "extensions" / study / "summary.csv")
        frame = frame.loc[frame.period.eq("full") & frame.cost_bps.eq(25)]
        for row in summary.loc[summary.study.eq(study)].itertuples():
            pairs = frame.loc[frame.comparison.eq("regularization_effect") & frame.control.eq(row.model)]
            complete = pairs.loc[pairs.complete]
            ew = frame.loc[frame.comparison.eq("versus_ew") & frame.variant.eq("lw_" + row.model) & frame.complete]
            actual = [len(pairs), len(complete), int(complete.delta_net_cer_pp.gt(0).sum()),
                      int(complete.net_ci_lo_pp.gt(0).sum()), len(ew), int(ew.delta_net_cer_pp.gt(0).sum())]
            reported = [row.expected_pairs, row.complete_pairs, row.wins_vs_original,
                        row.positive_95_intervals, row.complete_vs_ew, row.wins_vs_ew]
            if len(pairs) != expected or actual != reported:
                raise ValueError(f"Coverage/count mismatch: {study}/{row.model}")
            np.testing.assert_allclose(complete.delta_net_cer_pp.median(), row.median_delta_net_cer_pp,
                                       rtol=1e-12, atol=1e-12)
    figures = json.loads((RESULTS / "figures/manifest.json").read_text())
    for name, expected in figures["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Figure input checksum mismatch: {name}")
    for name, expected in figures["outputs_sha256"].items():
        if sha(RESULTS / "figures" / name) != expected:
            raise ValueError(f"Figure checksum mismatch: {name}")
    simulation = pd.read_csv(RESULTS / "article/simulation/tables1_5_comparison.csv")
    empirical = pd.read_csv(RESULTS / "article/empirical/table6.csv")
    if len(simulation) != 552 or len(empirical) != 140:
        raise ValueError("Incomplete main-paper comparison tables")
    print("Verified all compact artifact hashes, 552 simulation cells, 140 historical cells,")
    print("27 all-model summary rows and publication figure provenance. No research was run.")


if __name__ == "__main__":
    verify()

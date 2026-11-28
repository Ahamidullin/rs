"""Covariance regularization and verified historical baseline inputs."""

import json
from pathlib import Path

import numpy as np

from . import backend
from .data import checksum
from .extensions import DATASETS


def baseline_panel_path(source, dataset):
    """Read relative Windows or POSIX manifest paths on either platform."""
    return Path(source["processed_path"].replace("\\", "/")) / f"{dataset}.parquet"


def inverse_only_weights(history, original_kz, experiment="shrinkage", device="cpu"):
    """Replace only inverse covariance; freeze the original KZ signal and CKZ delta."""
    if experiment != "shrinkage":
        raise ValueError("Only covariance shrinkage is retained")
    from sklearn.covariance import LedoitWolf

    centered = history - history.mean(axis=0)
    sample = centered.T @ centered / len(history)
    fitted = LedoitWolf(assume_centered=False).fit(history)
    covariance = fitted.covariance_
    weights = backend.solve(covariance, sample @ original_kz, device)
    if not np.isfinite(weights).all():
        raise ValueError("Non-finite modified portfolio")
    return (
        weights,
        dict(
            original_condition=float(np.linalg.cond(sample)),
            new_condition=float(np.linalg.cond(covariance)),
            shrinkage_intensity=float(fitted.shrinkage_),
            target="trace(S0)/N times identity",
        ),
    )


def verify_baseline(baseline):
    source = json.loads((baseline / "empirical_manifest.json").read_text())
    if source["mode"] != "full" or source["ckz_mode"] != "eq13_scaled":
        raise ValueError("Require a completed full eq13_scaled empirical baseline")
    for name, expected in source["output_sha256"].items():
        if checksum(baseline / name) != expected:
            raise ValueError(f"Baseline output checksum mismatch: {name}")
    for dataset in DATASETS:
        panel = baseline_panel_path(source, dataset)
        if checksum(panel) != source["datasets"][dataset]["parquet_sha256"]:
            raise ValueError(f"Panel checksum differs from baseline: {dataset}")
    return source

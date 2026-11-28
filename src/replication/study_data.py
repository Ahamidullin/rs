"""Pinned data, fixed subsets and path diagnostics for the retained studies."""

import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from .data import FRENCH_ROOT, checksum, json_write, parse_french, read_member


SPEC = {
    "history_start": "1992-01", "test_start": "2012-01", "test_end": "2024-12",
    "gamma": 3.0, "windows": [120, 240], "grid_windows": [60, 120, 240],
    "grid_dimensions": [10, 25, 49], "subset_seeds": [201104, 201105, 201106, 201107, 201108],
    "cost_bps": [0, 25], "primary_cost_bps": 25,
    "bootstrap_replicates": 2000, "bootstrap_block": 12, "bootstrap_seed": 201104,
    "periods": [["2012-2015", "2012-01", "2015-12"],
                ["2016-2019", "2016-01", "2019-12"],
                ["2020-2024", "2020-01", "2024-12"]],
    "ckz_mode": "eq13_scaled", "weighting": "value", "currency": "USD",
}
SOURCES = {
    "ff3_fiz": ("ftp_202412", "F-F_Research_Data_Factors_CSV.zip", None),
    "industry49": ("ftp_202412", "49_Industry_Portfolios_CSV.zip", 49),
    "developed6": ("ftp", "Developed_6_Portfolios_ME_BE-ME_CSV.zip", 6),
    "emerging6": ("ftp", "Emerging_Markets_6_Portfolios_ME_BE-ME_CSV.zip", 6),
}


def parse_portfolios(contents, expected_n):
    """Read the monthly value-weighted block, including unlabelled global files."""
    try:
        frame, header = parse_french(contents, "value")
    except StopIteration:
        # International files can start directly with an unlabelled CSV table.
        # Do not mistake an explicitly labelled equal-weighted table for value.
        if re.search(r"average\s+.*weighted returns\s*--\s*monthly", contents, re.I):
            raise ValueError("No identifiable monthly value-weighted table")
        lines = contents.splitlines()
        first = next((i for i, line in enumerate(lines)
                      if re.match(r"^\s*\d{6}\s*,", line)), None)
        if first is None:
            raise ValueError("No monthly portfolio returns")
        previous = first - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0 or not lines[previous].lstrip().startswith(","):
            raise ValueError("Missing international portfolio header")
        header = [v.strip() for v in next(csv.reader([lines[previous]]))][1:]
        rows, dates = [], []
        for line in lines[first:]:
            if not re.match(r"^\s*\d{6}\s*,", line):
                break
            date, *fields = next(csv.reader([line]))
            date = date.strip()
            dates.append(date[:4]+"-"+date[4:])
            if len(fields) != len(header):
                raise ValueError("International row width differs from header")
            rows.append([float(x) if x.strip() else np.nan for x in fields])
        frame = pd.DataFrame(rows, index=dates, columns=header)
        frame = frame.replace([-99.99, -999.0], np.nan)/100
    if (len(frame.columns) != expected_n or not frame.columns.is_unique
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
        raise ValueError(f"Expected {expected_n} unique portfolio series and ordered unique months")
    return frame, header


def load_source(key, raw, download=False):
    directory, filename, expected_n = SOURCES[key]
    target = raw / directory / filename
    pin = target.with_suffix(".source.json")
    # Existing U.S. FIZ archives contain the full history, even though article
    # panels selected only early dates. Copy into a separate pinned study cache.
    legacy = Path("data/raw/french/fiz") / filename
    if directory == "ftp_202412" and not target.exists() and legacy.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy, target)
    expected_hash = json.loads(pin.read_text())["sha256"] if pin.exists() else None
    if expected_hash and target.exists() and checksum(target) != expected_hash:
        raise ValueError(f"Pinned source changed: {target}. Do not mix data revisions.")
    contents, info = read_member(target, f"{FRENCH_ROOT}/{directory}/{filename}",
                                 download=download, expected_hash=expected_hash)
    if not pin.exists():
        json_write(pin, info)
    if expected_n is None:
        frame, header = parse_french(contents)
        if "RF" not in frame:
            raise ValueError("Missing U.S. risk-free rate")
    else:
        frame, header = parse_portfolios(contents, expected_n)
    return frame, dict(info, header=header, source_key=key,
        vintage="FIZ 202412" if directory == "ftp_202412" else "current international snapshot at first download",
        first_month=frame.index[0], last_month=frame.index[-1], pin_sha256=checksum(pin))


def make_panel(portfolios, factors, start="1992-01", end="2024-12"):
    """Total USD portfolio returns minus the same USD RF exactly once."""
    expected = pd.period_range(start, end, freq="M").astype(str)
    for name, frame in (("portfolios", portfolios), ("risk-free", factors)):
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Duplicate or unordered {name} months")
        if not expected.isin(frame.index).all():
            raise ValueError(f"Missing {name} months: {expected.difference(frame.index).tolist()}")
    selected = portfolios.loc[expected]
    rf = factors.loc[expected, "RF"]
    if not np.isfinite(selected.to_numpy()).all() or not np.isfinite(rf).all():
        raise ValueError("Incomplete required history; no filling or asset dropping is allowed")
    if not selected.columns.is_unique or selected.columns.str.startswith("asset_").any():
        raise ValueError("Invalid portfolio labels")
    panel = selected.sub(rf, axis=0).rename(columns=lambda name: "asset_"+name)
    panel.insert(0, "rf", rf)
    return panel.rename_axis("month").reset_index()


def subset_plan(assets, dimensions=None, seeds=None):
    """Nested subsets use names/seed only; full universe appears exactly once."""
    dimensions = SPEC["grid_dimensions"] if dimensions is None else dimensions
    seeds = SPEC["subset_seeds"] if seeds is None else seeds
    assets = list(assets)
    if len(set(assets)) != len(assets) or any(n < 1 or n > len(assets) for n in dimensions):
        raise ValueError("Invalid dimensions or duplicate asset names")
    output = []
    for seed in seeds:
        ordering = np.random.default_rng(seed).permutation(assets).tolist()
        for n in dimensions:
            if n < len(assets):
                # Preserve source column order inside the selected set.
                selected = set(ordering[:n])
                output.append((n, seed, [a for a in assets if a in selected]))
    if len(assets) in dimensions:
        output.append((len(assets), -1, assets))
    return output


def active_prefix(path):
    stops = np.flatnonzero(path.status.ne("active"))
    return int(stops[0]) if len(stops) else len(path)


def risk_metrics(path):
    observed = path.loc[path.status.ne("after_stop")]
    complete = bool(path.status.eq("active").all())
    if not len(observed):
        return dict(complete=False, observed_months=0, expected_months=len(path),
                    max_drawdown=np.nan, worst_net_excess=np.nan, historical_es90_monthly=np.nan,
                    mean_turnover=np.nan, mean_gross_exposure=np.nan, annual_volatility=np.nan,
                    stop_reason="stopped_before_period")
    nav = np.r_[1., np.cumprod(observed.net_nav_factor.to_numpy())]
    losses = np.sort(-observed.net_excess.to_numpy())[::-1]
    mass = .1*len(losses)
    whole = int(np.floor(mass))
    tail = losses[:whole].sum()+(mass-whole)*losses[whole]
    return dict(complete=complete, observed_months=len(observed), expected_months=len(path),
        max_drawdown=float(np.min(nav/np.maximum.accumulate(nav)-1)),
        worst_net_excess=float(observed.net_excess.min()), historical_es90_monthly=float(tail/mass),
        mean_turnover=float(observed.turnover.mean()), mean_gross_exposure=float(observed.gross_exposure.mean()),
        annual_volatility=float(observed.net_excess.std(ddof=1)*np.sqrt(12)),
        stop_reason=None if complete else observed.status.iloc[-1])

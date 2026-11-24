"""Reproduce Tu–Zhou Table 6 from canonical excess-return panels."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .data import checksum, json_write, load_config
from .methods import combine, fit_rules, suff
from .metrics import cer


BASE_RULES = ("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
PAIRS = {
    "cml": "ml", "cpj": "pj", "cmp50": "mp", "ckz": "kz",
    "cmp": "mp", "ml50": "cml", "pj50": "cpj", "kz50": "ckz", "ew_cash": "ew",
}


def ml_in_sample(X, gamma):
    """Retrospective benchmark with maximum-likelihood covariance."""
    _, _, mean, cov = suff(X)
    return X @ (np.linalg.solve(cov, mean) / gamma)


def rolling_backtest(panel, M, gamma, policy=None, *, dataset="", run_id="",
                     options=None, stop_after=None):
    """Return monthly OOS returns and long target weights, including cash.

    A policy receives a copy of the history through fit_end, the base fit and a
    mutable state dict reused through time. It returns a fit_rules-style dict
    for its own strategies. Without a policy, emit the base and controls.
    """
    cols = [name for name in panel.columns if name.startswith("asset_")]
    X = panel[cols].to_numpy(dtype=float)
    months = panel["month"].to_numpy()
    rf = panel["rf"].to_numpy(dtype=float)
    if not len(X) or not cols or not np.isfinite(X).all() or not np.isfinite(rf).all():
        raise ValueError("Panel requires finite monthly excess returns and RF")
    expected = pd.period_range(months[0], months[-1], freq="M").astype(str)
    if not np.array_equal(months, expected):
        raise ValueError("Panel months must be unique, ordered and consecutive")
    if not 0 < M < len(X) or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Require 0 < M < T and positive finite gamma")

    assets = [name.removeprefix("asset_") for name in cols] + ["CASH"]
    opts = {"rules": BASE_RULES, "cmp50": True, **(options or {})}
    monthly, weights = [], []
    state = {}
    end = len(X) if stop_after is None else min(len(X), M + stop_after)
    for i in range(M, end):
        fit_end, realized_month = months[i - 1:i + 1]
        try:
            fit = fit_rules(X[i - M:i], gamma, opts)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"{dataset} M={M}, fit_end={fit_end}: {exc}") from exc
        if policy is not None:
            fit = policy(panel.iloc[:i].copy(), fit, state)
        fitted = fit["weights"]
        delta = fit.get("delta", {})
        raw = fit.get("delta_raw", {})
        diag = fit.get("diagnostics", {})
        if policy is None:
            fitted["ew_cash"] = np.full(len(cols), 1.0 / (len(cols) + 1))
            for name in ("ml", "pj", "kz"):
                if name in fitted:
                    fitted[name + "50"] = combine(fitted["ew"], fitted[name], 0.5)
                    delta[name + "50"] = 0.5
            if "cmp50" in fitted:
                delta["cmp50"] = 0.5

        for rule, w in fitted.items():
            w = np.asarray(w, dtype=float)
            if w.shape != (len(cols),) or not np.isfinite(w).all():
                raise ValueError(f"{dataset} {fit_end} {rule}: invalid target weights")
            cash = 1.0 - w.sum()
            monthly.append((dataset, M, rule, fit_end, realized_month,
                            float(X[i] @ w), rf[i], delta.get(rule), run_id, raw.get(rule),
                            cash, float(np.abs(w).sum()), diag.get(rule + "_flag"),
                            diag.get("cmp_n_refit") if rule == "cmp" else None))
            weights.extend((dataset, M, rule, fit_end, name, weight, run_id)
                           for name, weight in zip(assets, [*w, cash], strict=True))
    return (
        pd.DataFrame(monthly, columns=["dataset", "window", "rule", "fit_end",
                     "realized_month", "ret_excess", "rf", "delta", "run_id",
                     "delta_raw", "cash", "gross_risky", "flag", "cmp_n_refit"]),
        pd.DataFrame(weights, columns=["dataset", "window", "rule", "fit_end", "asset", "weight", "run_id"]),
    )


def compare(reference, monthly, panel, info, gamma, window, run_id):
    scores = monthly.groupby("rule").ret_excess.agg(lambda r: cer(r, gamma)).to_dict()
    scores["ml_is"] = cer(ml_in_sample(panel.iloc[:, 2:].to_numpy(), gamma), gamma)
    rows = []
    vintage = "+".join(dict.fromkeys(source.get("release", "dgu") for source in info["sources"]))
    for row in reference.to_dict("records"):
        rule = "cmp50" if row["rule"] == "cmp" else row["rule"]
        computed = scores[rule]
        gap = computed - row["published"]
        in_sample = rule == "ml_is"
        first = panel["month"].iat[0 if in_sample else window]
        rows.append({
            **row, "computed_rule": rule, "computed": computed, "delta_pp": gap,
            "status": "match" if abs(gap) <= 0.005 else "different",
            "window": window, "N": info["N"], "T": len(panel),
            "observations": len(panel) if in_sample else len(panel) - window,
            "first_month": first, "last_month": panel["month"].iat[-1],
            "vintage": info["vintage"] + (f":{vintage}" if vintage != "dgu" else ""),
            "run_id": run_id,
        })
    checks = []
    for rule, benchmark in [*PAIRS.items(), ("cmp50", "cmp"),
                            *((name, "ew") for name in ("cml", "cpj", "cmp50", "ckz", "cmp"))]:
        check = {"ew_cash": "footnote_4", "cmp50": "cmp_interpretation"}.get(rule, "paired_cer")
        checks.append((reference["case"].iat[0], window, check, rule, benchmark,
                       scores[rule], scores[benchmark], scores[rule] - scores[benchmark], run_id))
    return rows, checks


def run(args):
    config = load_config(args.config)
    gamma = config["gamma"]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.processed / "empirical_data_manifest.json").read_text())
    reference = pd.read_csv(args.reference, dtype={"table": str})
    key = ["case", "rule", "panel"]
    if len(reference) != 140 or reference.duplicated(key).any():
        raise ValueError("Expected 140 distinct reference cells from Table 6")

    code = {path.name: checksum(path) for path in
            (Path(__file__), Path(__file__).with_name("data.py"), Path(__file__).with_name("methods.py"),
             Path(__file__).with_name("mp.py"), Path(__file__).with_name("metrics.py"))}
    identity = {
        "mode": args.mode,
        "gamma": gamma,
        "windows": config["windows"],
        "config_sha256": checksum(args.config),
        "reference_sha256": checksum(args.reference),
        "code_sha256": code,
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "pyarrow")},
        "python": platform.python_version(),
        "units": source["units"],
        "datasets": source["datasets"],
    }
    run_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    traces, targets, rows, checks = [], [], [], []
    for dataset, info in source["datasets"].items():
        path = args.processed / f"{dataset}.parquet"
        if checksum(path) != info["parquet_sha256"]:
            raise ValueError(f"Checksum differs from data manifest: {path}")
        panel = pd.read_parquet(path)
        for window, label in zip(config["windows"], ("A", "B"), strict=True):
            print(f"{dataset} M={window}: fitting", flush=True)
            monthly, weights = rolling_backtest(
                panel, window, gamma, dataset=dataset, run_id=run_id,
                stop_after=2 if args.mode == "smoke" else None,
            )
            traces.append(monthly)
            targets.append(weights)
            if args.mode == "full":
                cells, comparisons = compare(reference.query("case == @dataset and panel == @label"),
                                              monthly, panel, info, gamma, window, run_id)
                rows.extend(cells)
                checks.extend(comparisons)
            print(f"{dataset} M={window}: {monthly.realized_month.nunique()} OOS months", flush=True)
    returns = pd.concat(traces, ignore_index=True)
    positions = pd.concat(targets, ignore_index=True)
    trade_key = ["dataset", "window", "rule", "fit_end", "run_id"]
    if returns.duplicated(trade_key).any() or positions.duplicated(trade_key + ["asset"]).any():
        raise RuntimeError("Duplicate trade or target weight keys")
    totals = positions.groupby(trade_key).weight.sum()
    if len(totals) != len(returns) or not np.allclose(totals, 1.0):
        raise RuntimeError("Missing targets or portfolio weights do not sum to 1 with cash")
    returns.to_parquet(out / "monthly_returns.parquet", index=False)
    positions.to_parquet(out / "weights.parquet", index=False)
    if args.mode == "full":
        if len(rows) != len(reference):
            raise RuntimeError("Full run did not produce every reference cell")
        pd.DataFrame(rows).to_csv(out / "table6.csv", index=False)
        pd.DataFrame(checks, columns=["dataset", "window", "check", "rule", "benchmark",
                                     "cer_pp", "benchmark_cer_pp", "delta_pp", "run_id"]).to_csv(
            out / "empirical_checks.csv", index=False)
    manifest = {
        **identity, "run_id": run_id, "cer": "1200 * (mean(excess) - gamma/2 * var(excess, ddof=1))",
        "ml_is_cov_ddof": 0, "published_cmp_uses": "cmp50",
        "monthly_rows": len(returns), "position_rows": len(positions),
        "comparison_cells": len(rows),
        "oos_months": {f"{d}/{w}": int(group.realized_month.nunique())
                       for (d, w), group in returns.groupby(["dataset", "window"])},
        "parquet_sha256": {name: checksum(out / name)
                           for name in ("monthly_returns.parquet", "weights.parquet")},
    }
    json_write(out / "empirical_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/empirical.toml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("references/table6.csv"))
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

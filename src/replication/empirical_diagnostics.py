"""Verify Table 6 traces and measure sensitivity to French input revisions."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .backtest import ml_in_sample
from .data import (FACTORS, FRENCH_PATH, FRENCH_ROOT, french_panel, json_write,
                   load_config, month_label, read_member, select_panel)
from .methods import fit_rules
from .metrics import cer


MOM_ZIP = "F-F_Momentum_Factor_CSV.zip"
HIST_FF3 = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/Historical_Archives/08%202008%20Update/ftp/F-F_Research_Data_Factors_TXT.zip"
HIST_MEMBER = "Historical_Archives/08 2008 Update/ftp/F-F_Research_Data_Factors.txt"


def french_sensitivity(config, raw, out, reference, download):
    tests = []
    revisions = []
    sources = []
    gamma = config["gamma"]
    hist, info = read_member(raw / "french/historical/FF3_2008-07.zip", HIST_FF3,
                             HIST_MEMBER, download)
    sources.append({**info, "release": "200807"})
    ff3 = pd.DataFrame(
        [[month_label(row[0]), *(float(val) / 100 for val in row[1:])]
         for line in hist.splitlines() if re.match(r"^\d{6}\s", line)
         for row in [line.split()]],
        columns=["month", *FACTORS, "rf"],
    ).set_index("month")

    for dataset in ("ff28", "industry52"):
        spec = config["dataset"][dataset]
        anchor = reference[reference.case.eq(dataset)]
        published = [
            anchor.query("rule == 'ew' and panel == 'A'").published.item(),
            anchor.query("rule == 'ew' and panel == 'B'").published.item(),
            anchor.query("rule == 'ml_is' and panel == 'A'").published.item(),
        ]
        panels = {}
        for vintage in ("fiz", "ciz"):
            for weighting in ("value", "equal"):
                frame, metadata, coverage = french_panel(raw, spec["portfolio_zip"], vintage, weighting, download)
                sources.extend(metadata)
                panels[vintage, weighting] = frame
                frames = [(vintage, frame)]
                if vintage == "fiz" and weighting == "value":
                    merged = frame.join(ff3.add_suffix("_historical"), how="inner", validate="one_to_one")
                    portfolio_cols = frame.columns.difference(["rf", *FACTORS], sort=False)
                    merged[portfolio_cols] = merged[portfolio_cols].add(merged["rf"], axis=0).sub(
                        merged["rf_historical"], axis=0)
                    merged[["rf", *FACTORS]] = merged[[name + "_historical" for name in ["rf", *FACTORS]]].to_numpy()
                    frames.append(("fiz+ff3_200807", merged[frame.columns]))
                for origin, candidate in frames:
                    for start in sorted({"1963-07", max("1963-07", coverage["first_complete_month"])}):
                        for end in ("2004-11", "2007-08"):
                            row = {"dataset": dataset, "vintage": origin, "weighting": weighting,
                                   "start": start, "end": end, "M_A": 120, "M_B": 240}
                            try:
                                panel = select_panel(candidate, start, end, spec["n"], f"{dataset}/{origin}")
                            except ValueError as exc:
                                tests.append({**row, "status": "missing", "reason": str(exc)})
                                continue
                            X = panel.iloc[:, 2:].to_numpy()
                            for name, got, expected in zip(
                                ("ew120", "ew240", "ml_is"),
                                (cer(X[120:].mean(1), gamma), cer(X[240:].mean(1), gamma),
                                 cer(ml_in_sample(X, gamma), gamma)),
                                published, strict=True,
                            ):
                                row[name] = got
                                row[name + "_published"] = expected
                                row[name + "_delta_pp"] = got - expected
                            tests.append({**row, "status": "computed", "T": len(X)})

        left = panels["fiz", "value"].loc[spec["start"]:spec["end"]]
        right = panels["ciz", "value"].loc[spec["start"]:spec["end"]]
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            raise ValueError(f"{dataset}: FIZ and CIZ must use the same dates and series")
        delta = right - left
        for col in left:
            revisions.append({
                "dataset": dataset, "series": col, "T": len(left),
                "mean_abs_change": delta[col].abs().mean(), "max_abs_change": delta[col].abs().max(),
                "fiz_mean": left[col].mean(), "ciz_mean": right[col].mean(),
                "fiz_std": left[col].std(), "ciz_std": right[col].std(),
            })
    pd.DataFrame(tests).to_csv(out / "french_sensitivity.csv", index=False)
    pd.DataFrame(revisions).to_csv(out / "french_input_revisions.csv", index=False)
    return sources


def ff24_diagnostics(args, gamma, reference):
    panel = pd.read_parquet(args.processed / "ff24.parquet").set_index("month")
    sources = []
    variants = {"dgu": panel}
    for vintage in ("fiz", "ciz"):
        url = f"{FRENCH_ROOT}/{FRENCH_PATH[vintage]}/{MOM_ZIP}"
        body, source = read_member(args.raw / "french" / vintage / MOM_ZIP, url,
                                   download=args.download)
        release = re.search(r"(\d{6}) CRSP database", body).group(1)
        sources.append({**source, "release": release})
        mom = {}
        for line in body.splitlines():
            if re.match(r"^\d{6}\s*,", line):
                date, value = line.split(",")
                mom[month_label(date)] = float(value) / 100
        candidate = panel.copy()
        candidate["asset_MOM"] = pd.Series(mom).reindex(panel.index)
        if not np.isfinite(candidate["asset_MOM"]).all():
            raise ValueError(f"Incomplete {vintage} momentum factor")
        variants[vintage] = candidate

    rows = []
    for vintage, candidate in variants.items():
        X = candidate.iloc[:, 1:].to_numpy()
        volatility = (candidate.asset_MOM - panel.asset_MOM).abs()
        for window, label in ((120, "A"), (240, "B")):
            pub = reference.query("case == 'ff24' and panel == @label").set_index("rule").published
            for cov_ddof in (0, 1):
                r_is = X @ (np.linalg.solve(np.cov(X, rowvar=False, ddof=cov_ddof), X.mean(0)) / gamma)
                rolling = []
                for i in range(window, len(X)):
                    train = X[i - window:i]
                    w = np.linalg.solve(np.cov(train, rowvar=False, ddof=cov_ddof), train.mean(0)) / gamma
                    rolling.append(X[i] @ w)
                for cer_ddof in (0, 1):
                    for rule, returns in (("ml", rolling), ("ml_is", r_is)):
                        value = cer(returns, gamma, cer_ddof)
                        rows.append({
                            "mom_vintage": vintage, "window": window, "rule": rule,
                            "cov_ddof": cov_ddof, "cer_ddof": cer_ddof,
                            "computed": value, "published": pub[rule], "delta_pp": value - pub[rule],
                            "mom_mean_absolute_diff": volatility.mean(),
                            "mom_max_absolute_diff": volatility.max(),
                        })
    pd.DataFrame(rows).to_csv(args.out / "ff24_diagnostics.csv", index=False)
    return sources


def verify_traces(args, config):
    money_key = ["dataset", "window", "rule", "fit_end", "run_id"]
    rows = []
    for dataset, spec in config["dataset"].items():
        panel = pd.read_parquet(args.processed / f"{dataset}.parquet")
        X = panel.iloc[:, 2:].to_numpy()
        columns = [col.removeprefix("asset_") for col in panel.columns[2:]]
        months = panel.month.to_numpy()
        for window in config["windows"]:
            filters = [("dataset", "=", dataset), ("window", "=", window)]
            returns = pd.read_parquet(args.out / "monthly_returns.parquet", filters=filters)
            weights = pd.read_parquet(args.out / "weights.parquet", filters=filters)
            values = weights.pivot(index=money_key, columns="asset", values="weight")
            trace = returns.set_index(money_key).sort_index()
            pd.testing.assert_index_equal(trace.index, values.index)
            np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(values.CASH, trace.cash, rtol=0, atol=1e-10)
            np.testing.assert_allclose(values[columns].abs().sum(axis=1), trace.gross_risky, atol=1e-10)
            for _, rule in trace.groupby(level="rule"):
                np.testing.assert_array_equal(rule.index.get_level_values("fit_end"), months[window - 1:-1])
                np.testing.assert_array_equal(rule.realized_month, months[window:])
                np.testing.assert_array_equal(rule.rf, panel.rf.iloc[window:])
            idx = pd.Index(months).get_indexer(trace.realized_month)
            np.testing.assert_allclose(np.sum(values[columns].to_numpy() * X[idx], axis=1),
                                       trace.ret_excess, atol=1e-12)
            for i in (window, len(X) - 1):
                train = X[i - window:i]
                gamma = config["gamma"]
                known = {
                    "ew": np.full(spec["n"], 1 / spec["n"]),
                    "ml": np.linalg.solve(np.cov(train, rowvar=False, ddof=1), train.mean(0)) / gamma,
                    "ckz": fit_rules(train, gamma, {"rules": ["ckz"]})["weights"]["ckz"],
                }
                for name, w in known.items():
                    match = trace.xs((name, months[i - 1]), level=("rule", "fit_end"))
                    np.testing.assert_allclose(X[i] @ w, match.ret_excess.item(), atol=1e-12)
                    np.testing.assert_allclose(1 - w.sum(), match.cash.item(), atol=1e-12)
                    rows.append({"dataset": dataset, "window": window, "rule": name,
                                 "fit_start": months[i - window], "fit_end": months[i - 1],
                                 "realized_month": months[i], "ret_excess": X[i] @ w,
                                 "gross_risky": np.abs(w).sum(), "cash": 1 - w.sum()})
    pd.DataFrame(rows).to_csv(args.out / "empirical_endpoint_checks.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/empirical.toml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("references/table6.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/replication"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    reference = pd.read_csv(args.reference)
    sources = french_sensitivity(config, args.raw, args.out, reference, args.download)
    sources += ff24_diagnostics(args, config["gamma"], reference)
    verify_traces(args, config)
    json_write(args.out / "empirical_diagnostics_sources.json",
               {"verification_utc": datetime.now(timezone.utc).isoformat(), "sources": sources})
    print("Verified trace keys, monthly timing, RF, returns, weights and cash for all 14 cases.")


if __name__ == "__main__":
    main()

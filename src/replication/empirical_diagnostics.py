"""Verify Table 6 traces and measure sensitivity to French input revisions."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .backtest import BASE_RULES, ml_in_sample
from .data import (
    FACTORS,
    FRENCH_PATH,
    FRENCH_ROOT,
    checksum,
    french_panel,
    json_write,
    load_config,
    month_label,
    read_member,
    select_panel,
)
from .methods import fit_rules, kz_adjust
from .metrics import cer


MOM_ZIP = "F-F_Momentum_Factor_CSV.zip"
HIST_FF3 = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.french/Data_Library/Historical_Archives/08%202008%20Update/ftp/F-F_Research_Data_Factors_TXT.zip"
HIST_MEMBER = "Historical_Archives/08 2008 Update/ftp/F-F_Research_Data_Factors.txt"


def french_sensitivity(config, raw, out, reference, download):
    tests = []
    revisions = []
    sources = []
    gamma = config["gamma"]
    hist, info = read_member(
        raw / "french/historical/FF3_2008-07.zip", HIST_FF3, HIST_MEMBER, download
    )
    sources.append({**info, "release": "200807"})
    ff3 = pd.DataFrame(
        [
            [month_label(row[0]), *(float(val) / 100 for val in row[1:])]
            for line in hist.splitlines()
            if re.match(r"^\d{6}\s", line)
            for row in [line.split()]
        ],
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
                frame, metadata, coverage = french_panel(
                    raw, spec["portfolio_zip"], vintage, weighting, download
                )
                sources.extend(metadata)
                panels[vintage, weighting] = frame
                portfolio_hash, ff3_hash = [item["sha256"] for item in metadata]
                frames = [(vintage, frame, ff3_hash)]
                if vintage == "fiz" and weighting == "value":
                    merged = frame.join(
                        ff3.add_suffix("_historical"),
                        how="inner",
                        validate="one_to_one",
                    )
                    portfolio_cols = frame.columns.difference(
                        ["rf", *FACTORS], sort=False
                    )
                    merged[portfolio_cols] = (
                        merged[portfolio_cols]
                        .add(merged["rf"], axis=0)
                        .sub(merged["rf_historical"], axis=0)
                    )
                    merged[["rf", *FACTORS]] = merged[
                        [name + "_historical" for name in ["rf", *FACTORS]]
                    ].to_numpy()
                    frames.append(
                        ("fiz+ff3_200807", merged[frame.columns], info["sha256"])
                    )
                for origin, candidate, factor_hash in frames:
                    for start in sorted(
                        {"1963-07", max("1963-07", coverage["first_complete_month"])}
                    ):
                        for end in ("2004-11", "2007-08"):
                            row = {
                                "dataset": dataset,
                                "vintage": origin,
                                "weighting": weighting,
                                "start": start,
                                "end": end,
                                "M_A": 120,
                                "M_B": 240,
                                "portfolio_sha256": portfolio_hash,
                                "ff3_sha256": factor_hash,
                            }
                            try:
                                panel = select_panel(
                                    candidate,
                                    start,
                                    end,
                                    spec["n"],
                                    f"{dataset}/{origin}",
                                )
                            except ValueError as exc:
                                tests.append(
                                    {**row, "status": "missing", "reason": str(exc)}
                                )
                                continue
                            X = panel.iloc[:, 2:].to_numpy()
                            for name, got, expected in zip(
                                ("ew120", "ew240", "ml_is"),
                                (
                                    cer(X[120:].mean(1), gamma),
                                    cer(X[240:].mean(1), gamma),
                                    cer(ml_in_sample(X, gamma), gamma),
                                ),
                                published,
                                strict=True,
                            ):
                                row[name] = got
                                row[name + "_published"] = expected
                                row[name + "_delta_pp"] = got - expected
                            tests.append({**row, "status": "computed", "T": len(X)})

        left = panels["fiz", "value"].loc[spec["start"] : spec["end"]]
        right = panels["ciz", "value"].loc[spec["start"] : spec["end"]]
        if not left.index.equals(right.index) or not left.columns.equals(right.columns):
            raise ValueError(
                f"{dataset}: FIZ and CIZ must use the same dates and series"
            )
        delta = right - left
        for col in left:
            revisions.append(
                {
                    "dataset": dataset,
                    "series": col,
                    "T": len(left),
                    "mean_abs_change": delta[col].abs().mean(),
                    "max_abs_change": delta[col].abs().max(),
                    "fiz_mean": left[col].mean(),
                    "ciz_mean": right[col].mean(),
                    "fiz_std": left[col].std(),
                    "ciz_std": right[col].std(),
                }
            )
    pd.DataFrame(tests).to_csv(out / "french_sensitivity.csv", index=False)
    pd.DataFrame(revisions).to_csv(out / "french_input_revisions.csv", index=False)
    return sources


def ff24_diagnostics(args, gamma, reference):
    panel_file = args.processed / "ff24.parquet"
    panel_hash = checksum(panel_file)
    panel = pd.read_parquet(panel_file).set_index("month")
    sources = []
    variants = {"dgu": (panel, None)}
    for vintage in ("fiz", "ciz"):
        url = f"{FRENCH_ROOT}/{FRENCH_PATH[vintage]}/{MOM_ZIP}"
        body, source = read_member(
            args.raw / "french" / vintage / MOM_ZIP, url, download=args.download
        )
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
        variants[vintage] = candidate, source["sha256"]

    rows = []
    for vintage, (candidate, mom_hash) in variants.items():
        X = candidate.iloc[:, 1:].to_numpy()
        volatility = (candidate.asset_MOM - panel.asset_MOM).abs()
        for window, label in ((120, "A"), (240, "B")):
            pub = (
                reference.query("case == 'ff24' and panel == @label")
                .set_index("rule")
                .published
            )
            for cov_ddof in (0, 1):
                r_is = X @ (
                    np.linalg.solve(np.cov(X, rowvar=False, ddof=cov_ddof), X.mean(0))
                    / gamma
                )
                rolling = []
                for i in range(window, len(X)):
                    train = X[i - window : i]
                    w = (
                        np.linalg.solve(
                            np.cov(train, rowvar=False, ddof=cov_ddof), train.mean(0)
                        )
                        / gamma
                    )
                    rolling.append(X[i] @ w)
                for cer_ddof in (0, 1):
                    for rule, returns in (("ml", rolling), ("ml_is", r_is)):
                        value = cer(returns, gamma, cer_ddof)
                        rows.append(
                            {
                                "mom_vintage": vintage,
                                "window": window,
                                "rule": rule,
                                "dgu_panel_sha256": panel_hash,
                                "replacement_mom_sha256": mom_hash,
                                "cov_ddof": cov_ddof,
                                "cer_ddof": cer_ddof,
                                "computed": value,
                                "published": pub[rule],
                                "delta_pp": value - pub[rule],
                                "mom_mean_absolute_diff": volatility.mean(),
                                "mom_max_absolute_diff": volatility.max(),
                            }
                        )
    pd.DataFrame(rows).to_csv(args.out / "ff24_diagnostics.csv", index=False)
    return sources


def verify_traces(args, config):
    manifest = json.loads((args.out / "empirical_manifest.json").read_text())
    if (manifest["mode"], manifest["gamma"], manifest["windows"]) != (
        "full",
        config["gamma"],
        config["windows"],
    ):
        raise ValueError(
            "Expected a full baseline with the configured windows and gamma"
        )
    expected_files = {
        "table6.csv",
        "empirical_checks.csv",
        "monthly_returns.parquet",
        "weights.parquet",
    }
    hashes = manifest["output_sha256"]
    if set(hashes) != expected_files:
        raise ValueError("Incomplete baseline checksums in the empirical manifest")
    for name, digest in hashes.items():
        if checksum(args.out / name) != digest:
            raise ValueError(f"{name}: checksum differs from empirical_manifest.json")
    run_id = manifest["run_id"]
    table = pd.read_csv(args.out / "table6.csv")
    checks = pd.read_csv(args.out / "empirical_checks.csv")
    for name, frame in (("Table 6", table), ("paired checks", checks)):
        if set(frame.run_id) != {run_id}:
            raise ValueError(f"{name}: rows belong to a different run")
    expected_keys = {
        (label, dataset, rule)
        for label in ("A", "B")
        for dataset in config["dataset"]
        for rule in (*BASE_RULES, "ml_is")
    }
    actual_keys = set(
        table[["panel", "case", "rule"]].itertuples(index=False, name=None)
    )
    if (
        len(table) != 140
        or len(table) != len(actual_keys)
        or actual_keys != expected_keys
    ):
        raise ValueError(
            "Table 6 does not contain the 140 distinct expected panel/dataset/rule keys"
        )
    expected_rules = {*BASE_RULES, "cmp50", "ew_cash", "ml50", "pj50", "kz50"}
    if (manifest["published_cmp_uses"], manifest["diagnostic_cmp_uses"]) != (
        "cmp50",
        "cmp",
    ):
        raise ValueError(
            "The empirical manifest must distinguish CMP50 from jackknife CMP"
        )
    if set(checks.rule) | set(checks.benchmark) != expected_rules:
        raise ValueError("The paired checks do not cover the declared OOS strategies")

    money_key = ["dataset", "window", "rule", "fit_end", "run_id"]
    rows = []
    monthly_count = position_count = 0
    gamma = config["gamma"]
    for dataset, spec in config["dataset"].items():
        path = args.processed / f"{dataset}.parquet"
        if checksum(path) != manifest["datasets"][dataset]["parquet_sha256"]:
            raise ValueError(
                f"{dataset}: prepared panel differs from empirical_manifest.json"
            )
        panel = pd.read_parquet(path)
        X = panel.iloc[:, 2:].to_numpy()
        columns = [col.removeprefix("asset_") for col in panel.columns[2:]]
        months = panel.month.to_numpy()
        sample_cov = np.cov(X, rowvar=False, ddof=0)
        sample_weights = np.linalg.solve(sample_cov, X.mean(axis=0)) / gamma
        in_sample_returns = X @ sample_weights
        ml_is = 1200 * (
            in_sample_returns.mean() - gamma / 2 * in_sample_returns.var(ddof=1)
        )
        for window, label in zip(config["windows"], ("A", "B"), strict=True):
            filters = [("dataset", "=", dataset), ("window", "=", window)]
            returns = pd.read_parquet(
                args.out / "monthly_returns.parquet", filters=filters
            )
            weights = pd.read_parquet(args.out / "weights.parquet", filters=filters)
            n_oos = len(X) - window
            if (
                set(returns.run_id) != {run_id}
                or set(weights.run_id) != {run_id}
                or set(returns.rule) != expected_rules
                or set(weights.rule) != expected_rules
                or manifest["oos_months"][f"{dataset}/{window}"] != n_oos
            ):
                raise ValueError(
                    f"{dataset}/{window}: trace coverage or run ID mismatch"
                )
            monthly_count += len(returns)
            position_count += len(weights)
            values = weights.pivot(index=money_key, columns="asset", values="weight")
            trace = returns.set_index(money_key).sort_index()
            pd.testing.assert_index_equal(trace.index, values.index)
            np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=1e-10)
            np.testing.assert_allclose(values.CASH, trace.cash, rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                values[columns].abs().sum(axis=1), trace.gross_risky, atol=1e-10
            )
            for _, rule in trace.groupby(level="rule"):
                np.testing.assert_array_equal(
                    rule.index.get_level_values("fit_end"), months[window - 1 : -1]
                )
                np.testing.assert_array_equal(rule.realized_month, months[window:])
                np.testing.assert_array_equal(rule.rf, panel.rf.iloc[window:])
            idx = pd.Index(months).get_indexer(trace.realized_month)
            np.testing.assert_allclose(
                np.sum(values[columns].to_numpy() * X[idx], axis=1),
                trace.ret_excess,
                atol=1e-12,
            )
            scores = {}
            for rule, group in trace.groupby(level="rule"):
                r = group.ret_excess.to_numpy()
                scores[rule] = 1200 * (r.mean() - gamma / 2 * r.var(ddof=1))
            cells = table.query("case == @dataset and panel == @label")
            for cell in cells.itertuples():
                base_rule = "cmp50" if cell.rule == "cmp" else cell.rule
                if (
                    cell.computed_rule != base_rule
                    or cell.window != window
                    or cell.T != len(X)
                ):
                    raise ValueError(
                        f"{dataset}/{window}/{cell.rule}: incorrect table metadata"
                    )
                observed, start, expected = (
                    (len(X), months[0], ml_is)
                    if base_rule == "ml_is"
                    else (n_oos, months[window], scores[base_rule])
                )
                if (cell.observations, cell.first_month, cell.last_month) != (
                    observed,
                    start,
                    months[-1],
                ):
                    raise ValueError(
                        f"{dataset}/{window}/{cell.rule}: incorrect table dates"
                    )
                np.testing.assert_allclose(expected, cell.computed, rtol=0, atol=1e-9)
            pairs = checks.query("dataset == @dataset and window == @window")
            for pair in pairs.itertuples():
                a, b = scores[pair.rule], scores[pair.benchmark]
                np.testing.assert_allclose(
                    (a, b, a - b),
                    (pair.cer_pp, pair.benchmark_cer_pp, pair.delta_pp),
                    rtol=0,
                    atol=1e-9,
                )
            for i in (window, len(X) - 1):
                train = X[i - window : i]
                known = {
                    "ew": np.full(spec["n"], 1 / spec["n"]),
                    "ml": np.linalg.solve(
                        np.cov(train, rowvar=False, ddof=1), train.mean(0)
                    )
                    / gamma,
                    "ckz": fit_rules(
                        train,
                        gamma,
                        {"rules": ["ckz"], "ckz_mode": manifest["ckz_mode"]},
                    )["weights"]["ckz"],
                }
                for name, w in known.items():
                    match = trace.xs((name, months[i - 1]), level=("rule", "fit_end"))
                    np.testing.assert_allclose(
                        X[i] @ w, match.ret_excess.item(), atol=1e-12
                    )
                    np.testing.assert_allclose(
                        1 - w.sum(), match.cash.item(), atol=1e-12
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "window": window,
                            "rule": name,
                            "fit_start": months[i - window],
                            "fit_end": months[i - 1],
                            "realized_month": months[i],
                            "ret_excess": X[i] @ w,
                            "gross_risky": np.abs(w).sum(),
                            "cash": 1 - w.sum(),
                            "run_id": run_id,
                        }
                    )
    if (monthly_count, position_count) != (
        manifest["monthly_rows"],
        manifest["position_rows"],
    ):
        raise ValueError(
            "Monthly or weight row counts differ from the empirical manifest"
        )
    print(
        f"{run_id}: 140 CER cells, {monthly_count} returns and {position_count} weights verified."
    )
    return manifest, pd.DataFrame(rows)


def compare_ckz(args, manifest):
    """Compare both control formulas with the canonical run on identical panels."""
    if manifest["ckz_mode"] != "eq13_scaled":
        raise ValueError("CKZ comparison requires the Eq. (13) baseline")
    keys = ["panel", "case", "rule"]
    canonical = pd.read_csv(args.out / "table6.csv").set_index(keys).sort_index()
    modes = {"eq13_scaled": manifest["run_id"]}
    results = canonical.query("rule == 'ckz'")[["window", "published"]].copy()
    results["eq13_scaled_cer"] = canonical.loc[results.index, "computed"]
    input_hashes = {
        case: info["parquet_sha256"] for case, info in manifest["datasets"].items()
    }

    for mode, directory in zip(
        ("legacy_adjusted", "unscaled_plugin"), args.ckz_controls, strict=True
    ):
        other = json.loads((directory / "empirical_manifest.json").read_text())
        source_hashes = {
            case: info["parquet_sha256"] for case, info in other["datasets"].items()
        }
        if (
            other["ckz_mode"] != mode
            or source_hashes != input_hashes
            or other["gamma"] != manifest["gamma"]
            or other["windows"] != manifest["windows"]
            or other["code_sha256"] != manifest["code_sha256"]
            or other["reference_sha256"] != manifest["reference_sha256"]
        ):
            raise ValueError(f"{directory}: CKZ control used a different setup")
        file = directory / "table6.csv"
        if checksum(file) != other["output_sha256"]["table6.csv"]:
            raise ValueError(f"{file}: table differs from its manifest")
        control = pd.read_csv(file).set_index(keys).sort_index()
        pd.testing.assert_index_equal(control.index, canonical.index)
        np.testing.assert_array_equal(control.published, canonical.published)
        not_ckz = canonical.index.get_level_values("rule") != "ckz"
        np.testing.assert_allclose(
            canonical.computed[not_ckz], control.computed[not_ckz], rtol=0, atol=1e-9
        )
        modes[mode] = other["run_id"]
        results[mode + "_cer"] = control.loc[results.index, "computed"]

    for mode, run_id in modes.items():
        results[mode + "_run_id"] = run_id
        results[mode + "_abs_error_pp"] = (
            results[mode + "_cer"] - results.published
        ).abs()
    results.to_csv(args.out / "ckz_comparison.csv")

    panel = pd.read_parquet(args.processed / "industry11.parquet")
    window, gamma = 120, manifest["gamma"]
    X = panel.iloc[:window, 2:].to_numpy()
    T, N = X.shape
    m, S0 = X.mean(axis=0), np.cov(X, rowvar=False, ddof=0)
    ones = np.ones(N)
    e = ones / N
    g = (ones @ np.linalg.solve(S0, m)) / (ones @ np.linalg.solve(S0, ones))
    demeaned = m - g
    theta = kz_adjust(m @ np.linalg.solve(S0, m), N, T)
    psi = kz_adjust(demeaned @ np.linalg.solve(S0, demeaned), N - 1, T)
    eta = psi / (psi + N / T)
    q = eta * m + (1 - eta) * g
    c1 = (T - 2) * (T - N - 2) / ((T - N - 1) * (T - N - 4))
    scaled_inverse = np.linalg.solve(S0 * T / (T - N - 2), q)
    kz = scaled_inverse / (gamma * c1)
    pi1 = e @ S0 @ e - 2 * e @ m / gamma + theta / gamma**2
    pi3 = theta / gamma**2 - (theta - N / T * eta) / (gamma**2 * c1)
    terms = {
        "legacy_adjusted": theta - (1 - eta) * psi,
        "unscaled_plugin": m @ np.linalg.solve(S0, q),
        "eq13_scaled": m @ scaled_inverse,
    }
    probes = {}
    for mode, cross in terms.items():
        pi13 = theta / gamma**2 - e @ m / gamma + (e @ q - cross / gamma) / (gamma * c1)
        denominator = pi1 - 2 * pi13 + pi3
        raw = (pi1 - pi13) / denominator
        delta = float(np.clip(raw, 0, 1)) if denominator > 0 else float(pi1 > pi3)
        weights = (1 - delta) * e + delta * kz
        fit = fit_rules(X, gamma, {"rules": ("kz", "ckz"), "ckz_mode": mode})
        np.testing.assert_allclose(
            fit["diagnostics"]["ckz_pi13"], pi13, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            (fit["delta_raw"]["ckz"], fit["delta"]["ckz"]),
            (raw, delta),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(fit["weights"]["kz"], kz, rtol=0, atol=1e-10)
        np.testing.assert_allclose(fit["weights"]["ckz"], weights, rtol=0, atol=1e-10)
        probes[mode] = {
            "cross": cross,
            "pi13": pi13,
            "denominator": denominator,
            "delta_raw": raw,
            "delta": delta,
            "weights": weights.tolist(),
            "cash": 1 - weights.sum(),
        }
    json_write(
        args.out / "ckz_fixed_window.json",
        {
            "dataset": "industry11",
            "window": window,
            "fit_start": panel.month.iat[0],
            "fit_end": panel.month.iat[window - 1],
            "realized_month": panel.month.iat[window],
            "input_sha256": input_hashes["industry11"],
            "assets": manifest["datasets"]["industry11"]["assets"],
            "gamma": gamma,
            "c1": c1,
            "theta2_adjusted": theta,
            "psi2_adjusted": psi,
            "eta": eta,
            "gmv_mean": g,
            "q": q.tolist(),
            "pi1": pi1,
            "pi3": pi3,
            "kz_weights": kz.tolist(),
            "modes": probes,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/empirical.toml"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--reference", type=Path, default=Path("references/table6.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/replication"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check baseline hashes, dates, weights and CER without downloading sources",
    )
    parser.add_argument(
        "--ckz-controls",
        nargs=2,
        type=Path,
        metavar=("LEGACY_DIR", "UNSCALED_DIR"),
        help="Also compare completed CKZ controls against the checked Eq. (13) run",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    manifest, endpoint_checks = verify_traces(args, config)
    if args.verify_only:
        return
    if args.ckz_controls:
        compare_ckz(args, manifest)
    reference = pd.read_csv(args.reference)
    sources = french_sensitivity(config, args.raw, args.out, reference, args.download)
    sources += ff24_diagnostics(args, config["gamma"], reference)
    endpoint_checks.to_csv(args.out / "empirical_endpoint_checks.csv", index=False)
    json_write(
        args.out / "empirical_diagnostics_sources.json",
        {
            "run_id": manifest["run_id"],
            "verification_utc": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
        },
    )


if __name__ == "__main__":
    main()

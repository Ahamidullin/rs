"""Validate Monte Carlo coverage and reproduce all 552 cells of Tables 1--5."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .methods import c1_
from .metrics import performance
from .simulate import COLS, RULES, check_chunk, chunk_filename, digest


def load(out):
    out = Path(out)
    manifest = json.loads((out / "sim_manifest.json").read_text())
    if "spec" not in manifest or manifest.get("status") != "complete":
        raise ValueError(
            "Need a completed schema-2 run; legacy or unfinished chunks cannot be aggregated"
        )
    spec = manifest["spec"]
    identity = digest(spec)
    if manifest["spec_sha256"] != identity:
        raise ValueError("Manifest specification checksum mismatch")
    frames, expected = [], set()
    for name in spec["scenarios"]:
        for T in spec["Ts"]:
            for r0 in range(0, spec["reps"], spec["chunk_size"]):
                r1 = min(spec["reps"], r0 + spec["chunk_size"])
                file = out / "chunks" / chunk_filename(name, T, r0, r1)
                expected.add(file.name)
                data = check_chunk(
                    file, identity, name, T, r0, r1, spec["populations"][name]["gamma"]
                )
                frames.append(
                    pd.DataFrame(data, columns=COLS).assign(scenario=name, T=T)
                )
    actual = {p.name for p in (out / "chunks").glob("*.npz")}
    if actual != expected:
        raise ValueError(
            f"Unexpected chunk set: extra={actual - expected}, missing={expected - actual}"
        )
    frame = pd.concat(frames, ignore_index=True)
    names = {**dict(enumerate(RULES)), 98: "scaled_ml", 99: "ew_cash"}
    frame["rule"] = frame.rule.astype(int).map(names)
    return frame


def sd_mcse(values):
    x = np.asarray(values, dtype=float)
    n, variance = len(x), x.var(ddof=1)
    if n < 4 or variance == 0:
        return 0.0
    fourth = np.mean((x - x.mean()) ** 4)
    return np.sqrt(
        max(0, (fourth - (n - 3) / (n - 1) * variance**2) / (4 * n * variance))
    )


def oracle_stats(component, e, mu, cov, gamma):
    """Eq. (25); moments are averaged before calculating one oracle delta per T."""
    z = component[["eSigmaV", "muV", "vSigmaV"]].to_numpy()
    esv, mv, vsv = z.mean(axis=0)
    A = e @ cov @ e
    em = e @ mu
    num = A - esv - em / gamma + mv / gamma
    den = A - 2 * esv + vsv
    delta = float(np.clip(num / den, 0, 1)) if den > 0 else float(den - 2 * num < 0)
    error = 0.0
    if den > 0:
        gradient = np.array(
            [(-den + 2 * num) / den**2, 1 / (gamma * den), -num / den**2]
        )
        error = np.sqrt(
            max(0, gradient @ np.cov(z, rowvar=False, ddof=1) @ gradient / len(z))
        )

    def utility(d):
        return 1200 * (
            (1 - d) * em
            + d * mv
            - gamma / 2 * ((1 - d) ** 2 * A + 2 * d * (1 - d) * esv + d * d * vsv)
        )

    return delta, error, utility


def main(out, ref, dest):
    out, dest = Path(out), Path(dest)
    data = load(out)
    manifest = json.loads((out / "sim_manifest.json").read_text())
    spec = manifest["spec"]
    groups = data.groupby(["scenario", "gamma", "T", "rule"], sort=True)
    summary = groups.agg(
        n=("U", "size"),
        U=("U", "mean"),
        U_sd=("U", "std"),
        SR=("SR", "mean"),
        SR_sd=("SR", "std"),
        delta=("delta", "mean"),
        delta_sd=("delta", "std"),
        clip0=("delta_raw", lambda x: np.mean(x < 0)),
        clip1=("delta_raw", lambda x: np.mean(x > 1)),
    ).reset_index()
    summary["U_mcse"] = summary.U_sd / np.sqrt(summary.n)
    summary["SR_mcse"] = summary.SR_sd / np.sqrt(summary.n)
    indexed = summary.set_index(["scenario", "gamma", "T", "rule"])
    rows, oracle_checks = [], []

    def add(table, panel, T, rule, metric, value, error=0.0, n=0):
        unit = (
            "coefficient_x100"
            if table == "5"
            else "monthly_x100"
            if metric.startswith("SR")
            else "annual_pp"
        )
        rows.append(
            dict(
                table=table,
                panel=panel,
                case=T,
                rule=rule,
                metric=metric,
                unit=unit,
                computed=value,
                mcse=error,
                n=n,
            )
        )

    designs = [
        ("A0", 3.0, "1", "A", "3", "A"),
        ("A0", 1.0, "1", "B", None, None),
        ("Aa", 3.0, "2", "A", "3", "B"),
        ("F3a", 3.0, "2", "B", None, None),
    ]
    for name, gamma, table, panel, sr_table, sr_panel in designs:
        if name not in spec["scenarios"]:
            continue
        pop = spec["populations"][name]
        mu, cov = np.array(pop["mu"]), np.array(pop["covariance"])
        e = np.full(len(mu), 1 / len(mu))
        true = np.linalg.solve(cov, mu) / gamma
        true_u, true_sr = performance(true, mu, cov, gamma)
        for T in spec["Ts"]:
            add(table, panel, T, "true", "U", 1200 * true_u)
            if sr_table:
                add(sr_table, sr_panel, T, "true", "SR", 100 * true_sr)
            if name == "F3a":
                add("4", "A", T, "true", "U_sd", 0.0)
                add("4", "B", T, "true", "SR_sd", 0.0)
            for rule in RULES:
                result = indexed.loc[(name, gamma, T, rule)]
                add(table, panel, T, rule, "U", result.U, result.U_mcse, result.n)
                if sr_table:
                    add(
                        sr_table,
                        sr_panel,
                        T,
                        rule,
                        "SR",
                        result.SR,
                        result.SR_mcse,
                        result.n,
                    )
                if name == "F3a":
                    sample = groups.get_group((name, gamma, T, rule))
                    add(
                        "4",
                        "A",
                        T,
                        rule,
                        "U_sd",
                        result.U_sd,
                        sd_mcse(sample.U),
                        result.n,
                    )
                    add(
                        "4",
                        "B",
                        T,
                        rule,
                        "SR_sd",
                        result.SR_sd,
                        sd_mcse(sample.SR),
                        result.n,
                    )
            if name != "F3a":
                continue
            for letter, combo, base in [
                ("A", "cml", "scaled_ml"),
                ("B", "cpj", "pj"),
                ("C", "cmp", "mp"),
                ("D", "ckz", "kz"),
            ]:
                component = groups.get_group((name, gamma, T, base))
                delta, error, utility = oracle_stats(component, e, mu, cov, gamma)
                if combo == "cml":
                    wstar = np.linalg.solve(cov, mu) / gamma
                    pi1 = (e - wstar) @ cov @ (e - wstar)
                    pi2 = (
                        (c1_(T, len(mu)) - 1) * (mu @ np.linalg.solve(cov, mu))
                        + c1_(T, len(mu)) * len(mu) / T
                    ) / gamma**2
                    delta, error = pi1 / (pi1 + pi2), 0.0
                result = indexed.loc[(name, gamma, T, combo)]
                sample = groups.get_group((name, gamma, T, combo))
                add(
                    "5",
                    letter,
                    T,
                    combo,
                    "delta_oracle",
                    100 * delta,
                    100 * error,
                    result.n,
                )
                add(
                    "5",
                    letter,
                    T,
                    combo,
                    "delta_mean",
                    100 * result.delta,
                    100 * result.delta_sd / np.sqrt(result.n),
                    result.n,
                )
                add(
                    "5",
                    letter,
                    T,
                    combo,
                    "delta_sd",
                    100 * result.delta_sd,
                    100 * sd_mcse(sample.delta),
                    result.n,
                )
                oracle_checks.append(
                    dict(
                        scenario=name,
                        T=T,
                        rule=combo,
                        oracle_delta=delta,
                        U_estimated=result.U,
                        U_oracle=utility(delta),
                        U_50=utility(0.5),
                    )
                )

    calculated = pd.DataFrame(rows)
    reference = pd.read_csv(ref, dtype={"table": str, "panel": str})
    keys = ["table", "panel", "case", "rule", "metric", "unit"]
    if reference.duplicated(keys).any() or calculated.duplicated(keys).any():
        raise ValueError("Duplicate table cell keys")
    comparison = reference.merge(calculated, on=keys, how="left", validate="one_to_one")
    comparison["delta_pp"] = comparison.computed - comparison.published
    tolerance = 0.005 + 3 * comparison.mcse
    comparison["status"] = np.select(
        [
            comparison.computed.isna(),
            comparison.delta_pp.abs() <= 0.005,
            comparison.delta_pp.abs() <= tolerance,
        ],
        ["not_run", "match", "mc_compatible"],
        default="open",
    )
    comparison["spec_sha256"] = manifest["spec_sha256"]
    dest.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(dest / "tables1_5_comparison.csv", index=False)
    comparison.loc[comparison.table.isin(["1", "2", "3"])].to_csv(
        dest / "tables1_3_comparison.csv", index=False
    )
    for table in ["1", "2", "3", "4", "5"]:
        comparison.loc[comparison.table.eq(table)].to_csv(
            dest / f"table{table}.csv", index=False
        )
    summary.to_csv(dest / "sim_summary.csv", index=False)
    pd.DataFrame(oracle_checks).to_csv(dest / "oracle_vs_estimated.csv", index=False)
    if "A0" in spec["scenarios"]:
        summary.query("scenario=='A0' and gamma==3 and rule in ['ew','ew_cash']").pivot(
            index="T", columns="rule", values="U"
        ).to_csv(dest / "footnote4_ew_cash.csv")
    summary.query("rule not in ['ew_cash','scaled_ml']")[
        ["scenario", "gamma", "T", "rule", "U_sd", "SR_sd"]
    ].to_csv(dest / "footnote10_dispersion.csv", index=False)
    print(comparison.groupby(["table", "status"]).size().to_string())
    return comparison, summary


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Footnotes 8-9 (sensitivity): paired comparisons combination vs component and vs EW."""

import sys, os, json, numpy as np, pandas as pd
from pathlib import Path
from replication.aggregate import load
from replication.simulate import scenario
from replication.dgp import true_U, ew_U

PAIRS = [("cml", "ml"), ("cpj", "pj"), ("cmp", "mp"), ("ckz", "kz")]


def main(out, dest):
    d = load(out)
    d = d[~d.rule.isin(["ew_cash", "scaled_ml"])]
    populations = json.loads((Path(out) / "sim_manifest.json").read_text())["spec"][
        "populations"
    ]
    os.makedirs(dest, exist_ok=True)
    s = (
        d.groupby(["scenario", "T", "rule"])
        .agg(
            n=("U", "size"),
            U=("U", "mean"),
            U_sd=("U", "std"),
            SR=("SR", "mean"),
            SR_sd=("SR", "std"),
            delta=("delta", "mean"),
        )
        .reset_index()
    )
    s["U_mcse"] = s.U_sd / np.sqrt(s.n)
    s["SR_mcse"] = s.SR_sd / np.sqrt(s.n)
    tr = []
    for sc in s.scenario.unique():
        mu = np.asarray(populations[sc]["mu"])
        S = np.asarray(populations[sc]["covariance"])
        tu = true_U(mu, S, 3.0)
        tr.append(
            dict(
                scenario=sc,
                N=len(mu),
                true_U=tu,
                true_SR=np.sqrt(2 * 3 * tu / 1200) * 100,
                ew_U=ew_U(mu, S, 3.0),
            )
        )
    pd.DataFrame(tr).to_csv(
        os.path.join(dest, "footnote89_population.csv"), index=False
    )
    s.to_csv(os.path.join(dest, "footnote89_summary.csv"), index=False)
    w = d.pivot_table(
        index=["scenario", "T", "rep"], columns="rule", values="U"
    ).reset_index()
    rows = []
    for (sc, T), g in w.groupby(["scenario", "T"]):
        for c, b in PAIRS + [(c, "ew") for c, _ in PAIRS]:
            x = g[c] - g[b]
            rows.append(
                dict(
                    scenario=sc,
                    T=T,
                    combo=c,
                    vs=b,
                    n=len(x),
                    dU=x.mean(),
                    mcse=x.std() / np.sqrt(len(x)),
                    share_better=(x > 0).mean(),
                )
            )
    p = pd.DataFrame(rows)
    p["sign"] = np.where(
        p.dU - 2 * p.mcse > 0, "+", np.where(p.dU + 2 * p.mcse < 0, "-", "0")
    )
    p.to_csv(os.path.join(dest, "footnote89_paired.csv"), index=False)
    return s, p, pd.DataFrame(tr)


if __name__ == "__main__":
    s, p, tr = main(sys.argv[1], sys.argv[2])
    print(tr.to_string(index=False))
    print(
        s.pivot_table(index=["scenario", "rule"], columns="T", values="U")
        .round(2)
        .to_string()
    )
    print(
        s[s.scenario == "NF"]
        .pivot_table(index="rule", columns="T", values="SR")
        .round(2)
        .to_string()
    )
    print(
        p.pivot_table(
            index=["scenario", "combo", "vs"],
            columns="T",
            values="sign",
            aggfunc="first",
        ).to_string()
    )

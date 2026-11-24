"""Aggregate simulation chunks -> results tables vs published (Tables 1-3) + footnote checks."""
import glob, os, sys, json
import numpy as np, pandas as pd
RULES = ("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
NAME = {"ew": "ew", "ml": "ml", "pj": "pj", "mp": "mp", "kz": "kz", "cml": "cml", "cpj": "cpj", "cmp": "cmp", "ckz": "ckz"}
def load(out):
    fr = []
    for p in glob.glob(os.path.join(out, "chunks", "*.npz")):
        b = os.path.basename(p).split("_T")
        name = b[0]; T = int(b[1].split("_")[0])
        a = np.load(p)["data"]
        d = pd.DataFrame(a, columns=["gamma", "rep", "rule", "U", "SR", "delta_raw", "delta"])
        d["scenario"] = name; d["T"] = T; fr.append(d)
    d = pd.concat(fr)
    d["rule"] = d["rule"].astype(int).map(lambda i: "ew_cash" if i == 99 else RULES[i])
    return d
def main(out, ref, dest):
    d = load(out)
    from replication.dgp import true_U
    from replication.simulate import scenario
    g = d.groupby(["scenario", "gamma", "T", "rule"])
    s = g.agg(n=("U", "size"), U=("U", "mean"), U_sd=("U", "std"), SR=("SR", "mean"), SR_sd=("SR", "std"),
              delta=("delta", "mean"), delta_sd=("delta", "std"),
              clip0=("delta_raw", lambda x: np.mean(x < 0)), clip1=("delta_raw", lambda x: np.mean(x > 1))).reset_index()
    s["U_mcse"] = s.U_sd / np.sqrt(s.n); s["SR_mcse"] = s.SR_sd / np.sqrt(s.n)
    rows = []
    for (sc, gm, panel_u, panel_sr) in [("A0", 3.0, ("1", "A"), ("3", "A")), ("A0", 1.0, ("1", "B"), None), ("Aa", 3.0, ("2", "A"), ("3", "B"))]:
        mu, S, _ = scenario(sc)
        for _, r in s[(s.scenario == sc) & (s.gamma == gm)].iterrows():
            if r.rule == "ew_cash": continue
            rows.append((panel_u[0], panel_u[1], int(r["T"]), r.rule, "U", "annual_pp", r.U, r.U_mcse, r.n))
            if panel_sr: rows.append((panel_sr[0], panel_sr[1], int(r["T"]), r.rule, "SR", "monthly_x100", r.SR, r.SR_mcse, r.n))
        tu = true_U(mu, S, gm); tsr = np.sqrt(2 * gm * tu / 1200) * 100
        for T in sorted(s["T"].unique()):
            rows.append((panel_u[0], panel_u[1], int(T), "true", "U", "annual_pp", tu, 0.0, 0))
            if panel_sr: rows.append((panel_sr[0], panel_sr[1], int(T), "true", "SR", "monthly_x100", tsr, 0.0, 0))
    c = pd.DataFrame(rows, columns=["table", "panel", "case", "rule", "metric", "unit", "computed", "mcse", "n"])
    R = pd.read_csv(ref, dtype={"table": str, "panel": str})
    R["case"] = R["case"].astype(int)
    c = c.merge(R, on=["table", "panel", "case", "rule"], how="left")
    c["delta_pp"] = c.computed - c.published
    tol = 0.005 + 3 * c.mcse
    c["status"] = np.where(c.delta_pp.abs() <= 0.005, "match", np.where(c.delta_pp.abs() <= tol, "mc_compatible", "open"))
    os.makedirs(dest, exist_ok=True)
    c.to_csv(os.path.join(dest, "tables1_3_comparison.csv"), index=False)
    s.to_csv(os.path.join(dest, "sim_summary.csv"), index=False)
    # footnote 4: EW vs EW+cash (A0, gamma=3)
    f4 = s[(s.scenario == "A0") & (s.gamma == 3.0) & s.rule.isin(["ew", "ew_cash"])].pivot(index="T", columns="rule", values="U")
    f4.to_csv(os.path.join(dest, "footnote4_ew_cash.csv"))
    # footnote 10: dispersion of U and SR for 1A/1B/2A
    s[s.rule != "ew_cash"][["scenario", "gamma", "T", "rule", "U_sd", "SR_sd"]].to_csv(os.path.join(dest, "footnote10_dispersion.csv"), index=False)
    return c, s
if __name__ == "__main__":
    c, s = main(sys.argv[1], sys.argv[2], sys.argv[3])
    print(c.groupby(["table", "panel", "status"]).size())

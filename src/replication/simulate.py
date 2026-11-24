"""Simulation CLI for Tables 1-3 (A0, Aalpha) and footnote checks.
python -m replication.simulate --mode smoke|full --reps R --workers W --out DIR [--scenarios A0,Aa]"""
import argparse, json, os, time, hashlib
import numpy as np
from multiprocessing import Pool
from .dgp import one_factor, true_U, ew_U
from .methods import fit_rules, combine
from .metrics import performance

SIGMA_SEED = 3267          # see docs/part2_status.md: matches 6 deterministic anchors
TS = (120, 240, 480, 960, 3000, 6000)
RULES = ("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
ROOT_SEED = 20111

def scenario(name):
    if name in ("A0", "Aa"):
        sig = np.random.default_rng(SIGMA_SEED).uniform(0.10, 0.30, 24)
        mu, S = one_factor(sig, alpha=(name == "Aa"))
        return mu, S, ((3.0, 1.0) if name == "A0" else (3.0,))
    if name in ("A0_N50", "Aa_N50"):   # footnote 8 (sensitivity: sigma draw not published)
        sig = np.random.default_rng(SIGMA_SEED).uniform(0.10, 0.30, 49)
        mu, S = one_factor(sig, N=50, alpha=(name == "Aa_N50"))
        return mu, S, (3.0,)
    if name == "NF":                     # footnote 8/9: no common factor, alpha + D (sensitivity)
        sig = np.random.default_rng(SIGMA_SEED).uniform(0.10, 0.30, 25)
        mu = np.linspace(-0.02, 0.02, 25) / 12 + 0.08 / 12
        return mu, np.diag(sig ** 2 / 12), (3.0,)
    raise ValueError(name)

def run_chunk(args):
    name, T, r0, r1, out = args
    path = os.path.join(out, "chunks", f"{name}_T{T}_{r0:05d}_{r1:05d}.npz")
    if os.path.exists(path):
        return path, 0.0
    mu, S, gammas = scenario(name)
    N = len(mu); L = np.linalg.cholesky(S); e = np.ones(N) / N
    rows = []
    t0 = time.time()
    for r in range(r0, r1):
        ss = np.random.SeedSequence([ROOT_SEED, hash_name(name), T, r])
        X = mu + np.random.default_rng(ss).standard_normal((T, N)) @ L.T
        cache = {}
        for g in gammas:
            f = fit_rules(X, g, dict(cache=cache))
            for k, w in f["weights"].items():
                u, sr = performance(w, mu, S, g)
                rows.append((g, r, RULES.index(k), u * 1200, sr * 100,
                             f["delta_raw"].get(k, np.nan), f["delta"].get(k, np.nan)))
            if g == 3.0:   # footnote 4: EW + cash, 1/(N+1) in each risky asset
                u, sr = performance(np.ones(N) / (N + 1), mu, S, g)
                rows.append((g, r, 99, u * 1200, sr * 100, np.nan, np.nan))
    a = np.array(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path + ".tmp.npz", data=a, cols=np.array(["gamma", "rep", "rule", "U", "SR", "delta_raw", "delta"]))
    os.replace(path + ".tmp.npz", path)
    return path, time.time() - t0

def hash_name(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke"); ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="A0,Aa"); ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--Ts", default=",".join(map(str, TS)))
    a = ap.parse_args()
    R = a.reps or (10000 if a.mode == "full" else 20)
    Ts = [int(x) for x in a.Ts.split(",")]
    jobs = []
    for T in sorted(Ts, reverse=True):          # long windows first
        for name in a.scenarios.split(","):
            for r0 in range(0, R, a.chunk):
                jobs.append((name, T, r0, min(R, r0 + a.chunk), a.out))
    os.makedirs(a.out, exist_ok=True)
    manifest = dict(reps=R, Ts=Ts, scenarios=a.scenarios.split(","), sigma_seed=SIGMA_SEED, root_seed=ROOT_SEED,
                    rules=RULES, started=time.strftime("%Y-%m-%dT%H:%M:%S"), workers=a.workers)
    json.dump(manifest, open(os.path.join(a.out, "sim_manifest.json"), "w"), indent=1)
    t0 = time.time(); done = 0
    with Pool(a.workers) as p:
        for path, dt in p.imap_unordered(run_chunk, jobs):
            done += 1
            print(f"[{time.time()-t0:8.0f}s] {done}/{len(jobs)} {os.path.basename(path)} {dt:.1f}s", flush=True)

if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()

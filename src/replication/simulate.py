"""Reproducible CPU Monte Carlo for Tu--Zhou Tables 1--5."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from multiprocessing import Pool
import os
from pathlib import Path
import platform
import time
from tempfile import NamedTemporaryFile

import numpy as np
from .dgp import one_factor, three_factor
from .methods import CKZ_MODES, fit_rules
from .metrics import performance
from . import backend

SIGMA_SEED, ROOT_SEED = 3267, 20111
TS = (120, 240, 480, 960, 3000, 6000)
RULES = ("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz")
COLS = (
    "gamma",
    "rep",
    "rule",
    "U",
    "SR",
    "delta_raw",
    "delta",
    "eSigmaV",
    "muV",
    "vSigmaV",
)


def digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def hash_name(name):
    return int(sha256(name.encode()).hexdigest()[:8], 16)


def scenario(name, calibration=None, sigma_seed=SIGMA_SEED, loading_seed=1971):
    if name in ("A0", "Aa", "A0_N50", "Aa_N50"):
        N = 50 if name.endswith("N50") else 25
        sig = np.random.default_rng(sigma_seed).uniform(0.10, 0.30, N - 1)
        mu, cov = one_factor(sig, N=N, alpha=name.startswith("Aa"))
        return mu, cov, (3.0, 1.0) if name == "A0" else (3.0,)
    if name == "F3a":
        if calibration is None:
            raise ValueError("Run replication.calibrate before F3a")
        rng = np.random.default_rng(loading_seed)
        load = np.column_stack(
            (
                np.linspace(0.9, 1.2, 22),
                rng.permutation(np.linspace(-0.3, 1.4, 22)),
                rng.permutation(np.linspace(-0.5, 0.9, 22)),
            )
        )
        sig = np.random.default_rng(sigma_seed).uniform(0.10, 0.30, 22)
        mu, cov = three_factor(
            np.array(calibration["mean"]),
            np.array(calibration["covariance"]),
            sig,
            load,
        )
        return mu, cov, (3.0,)
    if name == "NF":
        sig = np.random.default_rng(sigma_seed).uniform(0.10, 0.30, 25)
        return (
            np.linspace(-0.02, 0.02, 25) / 12 + 0.08 / 12,
            np.diag(sig**2 / 12),
            (3.0,),
        )
    raise ValueError(f"Unknown scenario {name}")


def make_spec(names, Ts, reps, chunk_size, calibration=None, **options):
    options.setdefault("device", "cpu")
    populations = {}
    for name in names:
        mu, cov, gammas = scenario(
            name, calibration, options["sigma_seed"], options["loading_seed"]
        )
        populations[name] = {
            "mu": mu.tolist(),
            "covariance": cov.tolist(),
            "gamma": gammas,
            "N": len(mu),
        }
    sources = (
        "simulate.py",
        "dgp.py",
        "methods.py",
        "mp.py",
        "metrics.py",
        "backend.py",
    )
    return {
        "schema": 2,
        "reps": reps,
        "Ts": list(Ts),
        "scenarios": list(names),
        "chunk_size": chunk_size,
        "rules": RULES,
        "options": options,
        "calibration": calibration,
        "populations": populations,
        "python": platform.python_version(),
        "backend": backend.environment(options["device"]),
        "packages": {n: version(n) for n in ("numpy", "scipy")},
        "code_sha256": {
            n: sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in sources
        },
        "protocol": "fixed population; iid monthly Gaussian; analytic utility/SR on true moments",
        "dgp_note": "N includes investable factors following DGU; author sigma/loading seeds unknown. "
        "Sigma seed 3267 was fitted to one-factor anchors; F3 pairing is a reconstruction.",
    }


def chunk_filename(name, T, r0, r1):
    return f"{name}_T{T}_{r0:05d}_{r1:05d}.npz"


def check_chunk(path, identity, name, T, r0, r1, gammas):
    with np.load(path, allow_pickle=False) as saved:
        expected = {
            "spec_sha256": identity,
            "scenario": name,
            "T": T,
            "r0": r0,
            "r1": r1,
        }
        if any(saved[k].item() != v for k, v in expected.items()):
            raise ValueError(f"Chunk specification mismatch: {path}")
        if tuple(saved["cols"].tolist()) != COLS:
            raise ValueError(f"Unexpected chunk columns: {path}")
        rows = saved["data"].copy()
    count = sum((len(RULES) + 1 + int(g == 3)) * (r1 - r0) for g in gammas)
    if (
        rows.shape != (count, len(COLS))
        or not np.isfinite(rows[:, [0, 1, 2, 3, 4, 7, 8, 9]]).all()
    ):
        raise ValueError(f"Invalid or incomplete chunk: {path}")
    if len(np.unique(rows[:, :3], axis=0)) != len(rows):
        raise ValueError(f"Duplicate replicate/rule keys: {path}")
    for gamma in gammas:
        for code in [*range(len(RULES)), 98, *([99] if gamma == 3 else [])]:
            reps = rows[(rows[:, 0] == gamma) & (rows[:, 2] == code), 1]
            if not np.array_equal(np.sort(reps), np.arange(r0, r1)):
                raise ValueError(f"Replicate coverage mismatch: {path}")
    return rows


def run_chunk(args):
    name, T, r0, r1, out, spec = args
    identity = digest(spec)
    path = Path(out) / "chunks" / chunk_filename(name, T, r0, r1)
    pop = spec["populations"][name]
    if path.exists():
        check_chunk(path, identity, name, T, r0, r1, pop["gamma"])
        return str(path), 0.0
    mu, cov = np.array(pop["mu"]), np.array(pop["covariance"])
    N = len(mu)
    L, e = np.linalg.cholesky(cov), np.full(N, 1 / N)
    rows = []
    started = time.perf_counter()
    options = spec["options"]
    for rep in range(r0, r1):
        seed = np.random.SeedSequence([options["root_seed"], hash_name(name), T, rep])
        X = mu + np.random.default_rng(seed).standard_normal((T, N)) @ L.T
        cache = {}
        for gamma in pop["gamma"]:
            fit = fit_rules(
                X,
                gamma,
                {
                    "cache": cache,
                    "ckz_mode": options["ckz_mode"],
                    "cpj_moments": options["cpj_moments"],
                    "device": options["device"],
                },
            )
            targets = [
                (RULES.index(rule), rule, w) for rule, w in fit["weights"].items()
            ]
            targets.append((98, "scaled_ml", fit["diagnostics"]["scaled_ml"]))
            if gamma == 3:
                targets.append((99, "ew_cash", np.full(N, 1 / (N + 1))))
            for code, rule, w in targets:
                u, sr = performance(w, mu, cov, gamma)
                rows.append(
                    (
                        gamma,
                        rep,
                        code,
                        1200 * u,
                        100 * sr,
                        fit["delta_raw"].get(rule, np.nan),
                        fit["delta"].get(rule, np.nan),
                        e @ cov @ w,
                        mu @ w,
                        w @ cov @ w,
                    )
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(
            stream,
            data=np.array(rows),
            cols=np.array(COLS),
            spec_sha256=identity,
            scenario=name,
            T=T,
            r0=r0,
            r1=r1,
        )
    try:
        check_chunk(temporary, identity, name, T, r0, r1, pop["gamma"])
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path), time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--reps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", default="A0,Aa,F3a")
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--Ts", default=",".join(map(str, TS)))
    parser.add_argument(
        "--calibration", type=Path, default=Path("data/processed/ff3_calibration.json")
    )
    parser.add_argument("--ckz-mode", choices=CKZ_MODES, default="eq13_scaled")
    parser.add_argument("--cpj-moments", choices=("adj", "unb"), default="adj")
    parser.add_argument("--sigma-seed", type=int, default=SIGMA_SEED)
    parser.add_argument("--loading-seed", type=int, default=1971)
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    args = parser.parse_args()
    if args.workers is None:
        args.workers = 1 if args.device == "cuda" else min(8, os.cpu_count() or 1)
    if args.device == "cuda" and args.workers != 1:
        parser.error("Use --workers 1 on one CUDA GPU")
    names, Ts = args.scenarios.split(","), [int(T) for T in args.Ts.split(",")]
    reps = (
        args.reps if args.reps is not None else (10000 if args.mode == "full" else 20)
    )
    if (
        reps < 2
        or args.chunk < 1
        or args.workers < 1
        or len(set(names)) != len(names)
        or len(set(Ts)) != len(Ts)
    ):
        parser.error(
            "Require reps>=2, positive workers/chunk and distinct scenarios/Ts"
        )
    calibration = json.loads(args.calibration.read_text()) if "F3a" in names else None
    options = {
        key: getattr(args, key)
        for key in (
            "sigma_seed",
            "loading_seed",
            "root_seed",
            "ckz_mode",
            "cpj_moments",
            "device",
        )
    }
    spec = make_spec(names, Ts, reps, args.chunk, calibration, **options)
    if any(T <= p["N"] + 4 for T in Ts for p in spec["populations"].values()):
        parser.error("All windows must satisfy T>N+4")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "sim_manifest.json"
    identity = digest(spec)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()).get("spec_sha256") != identity:
            raise ValueError(
                "Different simulation specification; choose a new output directory"
            )
    elif (args.out / "chunks").exists() and any((args.out / "chunks").iterdir()):
        raise ValueError("Cannot resume chunks without their matching manifest")
    manifest = {
        "spec": spec,
        "spec_sha256": identity,
        "mode": args.mode,
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    jobs = [
        (name, T, r0, min(reps, r0 + args.chunk), str(args.out), spec)
        for T in sorted(Ts, reverse=True)
        for name in names
        for r0 in range(0, reps, args.chunk)
    ]
    started = time.perf_counter()
    with Pool(args.workers) as pool:
        for count, (path, elapsed) in enumerate(
            pool.imap_unordered(run_chunk, jobs), 1
        ):
            print(
                f"[{time.perf_counter() - started:.1f}s] {count}/{len(jobs)} {Path(path).name} {elapsed:.2f}s",
                flush=True,
            )
    manifest.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        completed_chunks=len(jobs),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

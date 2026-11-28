"""Explicit stages for the main-paper numerical replication."""

import argparse
from pathlib import Path
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "check",
            "smoke",
            "simulations",
            "empirical",
            "figures",
            "footnotes",
            "all",
        ),
        default="check",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("runs/article_gpu_v1"))
    args = parser.parse_args()
    if args.reps < 2 or (args.workers is not None and args.workers < 1):
        parser.error("Require reps >= 2 and positive workers")
    if args.device == "cuda" and args.workers not in (None, 1):
        parser.error("Use one worker for one CUDA GPU")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(root / "src"),
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        CUPY_CACHE_DIR=str(root / ".cache/cupy"),
        MPLBACKEND="Agg",
    )
    workers = (
        args.workers if args.workers is not None else 1 if args.device == "cuda" else 4
    )

    def run(module, *values):
        command = [sys.executable, "-m", module, *map(str, values)]
        print("\n" + subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)

    if args.stage in ("check", "all"):
        run("pip", "check")
        run("pytest", "-q")
        if args.device == "cuda":
            run("replication.gpu_check")
    if args.stage in ("smoke", "simulations", "all"):
        run("replication.calibrate", "--download")
        sim = args.out / ("smoke_simulation" if args.stage == "smoke" else "simulation")
        run(
            "replication.simulate",
            "--mode",
            "smoke" if args.stage == "smoke" else "full",
            "--reps",
            20 if args.stage == "smoke" else args.reps,
            "--workers",
            workers,
            "--device",
            args.device,
            "--out",
            sim,
        )
        run("replication.aggregate", sim, "references/tables1_5.csv", sim / "tables")
    if args.stage in ("smoke", "empirical", "all"):
        run("replication.data", "--download")
        empirical = args.out / (
            "smoke_replication" if args.stage == "smoke" else "replication"
        )
        run(
            "replication.backtest",
            "--mode",
            "smoke" if args.stage == "smoke" else "full",
            "--device",
            args.device,
            "--out",
            empirical,
        )
        if args.stage != "smoke":
            run(
                "replication.empirical_diagnostics", "--verify-only", "--out", empirical
            )
    if args.stage in ("figures", "all"):
        run(
            "replication.figures",
            "--replication",
            args.out / "replication",
            "--simulation",
            args.out / "simulation/tables/tables1_3_comparison.csv",
            "--out",
            args.out / "figures",
        )
    if args.stage == "footnotes":
        run(
            "replication.simulate",
            "--mode",
            "full",
            "--reps",
            args.reps,
            "--workers",
            workers,
            "--device",
            args.device,
            "--scenarios",
            "Aa_N50,NF",
            "--out",
            args.out / "footnotes89",
        )
        run(
            "replication.footnotes",
            args.out / "footnotes89",
            args.out / "footnotes89/tables",
        )


if __name__ == "__main__":
    main()

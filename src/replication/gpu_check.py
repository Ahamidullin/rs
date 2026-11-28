"""Short CUDA and CPU parity diagnostic. This does not run research experiments."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import backend
from .methods import fit_rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/gpu_check.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    details = backend.environment("cuda")
    rng = np.random.default_rng(91)
    load = np.linspace(0.5, 1.5, 6)
    samples = {
        "isotropic": np.random.default_rng(41).normal(0.007, 0.04, (80, 6)),
        "factor_300": 0.01 * load
        + rng.normal(size=(300, 1)) * 0.04 * load
        + rng.normal(size=(300, 6)) * 0.005,
    }
    checks = {}
    for label, X in samples.items():
        print(f"Checking {label}: T={len(X)}, N={X.shape[1]}", flush=True)
        cpu = fit_rules(X, 3, {"device": "cpu"})
        gpu = fit_rules(X, 3, {"device": "cuda"})
        gaps = {}
        for name in cpu["weights"]:
            left, right = cpu["weights"][name], gpu["weights"][name]
            np.testing.assert_allclose(
                right,
                left,
                rtol=1e-7,
                atol=1e-9,
                err_msg=f"CPU/CUDA parity: {label}/{name}",
            )
            gaps[name] = float(np.abs(left - right).max())
        checks[label] = {
            "T": len(X),
            "N": X.shape[1],
            "max_weight_gaps": gaps,
            "mp_boundary": bool(gpu["diagnostics"]["mp_boundary"]),
            "cmp_n_refit": int(gpu["diagnostics"]["cmp_n_refit"]),
        }
    if checks["factor_300"]["mp_boundary"]:
        raise AssertionError(
            "Factor diagnostic must exercise interior GPU Newton batches"
        )
    result = {
        **details,
        "checks": checks,
        "rtol": 1e-7,
        "atol": 1e-9,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "passed",
        "scope": "two fixed numerical samples, not a Monte Carlo replication",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

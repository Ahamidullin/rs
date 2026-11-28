"""Explicit NumPy/CUDA linear algebra with float64 and no silent CPU fallback."""

from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np


# SciPy can probe CuPy while importing array API support, before cuda() is called.
_runtime = Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_runtime"
if os.name == "nt" and _runtime.is_dir():
    os.environ.setdefault("CUDA_PATH", str(_runtime))


@lru_cache(maxsize=1)
def cuda():
    try:
        # Keep compiled kernels in the writable project, including IDE launches.
        cache = Path(
            os.environ.setdefault(
                "CUPY_CACHE_DIR",
                str(Path(__file__).resolve().parents[2] / ".cache/cupy"),
            )
        )
        cache.mkdir(parents=True, exist_ok=True)
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device")
        cp.cuda.Device(0).use()
        return cp
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "CUDA is unavailable. Install requirements-gpu.lock in the project .venv"
        ) from exc


def check_device(device):
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unknown device: {device}")
    if device == "cuda":
        cuda()


def solve(a, b, device="cpu"):
    if device == "cpu":
        return np.linalg.solve(a, b)
    cp = cuda()
    import cupyx

    with cupyx.errstate(linalg="raise"):
        result = cp.linalg.solve(
            cp.asarray(a, dtype=cp.float64), cp.asarray(b, dtype=cp.float64)
        )
    return cp.asnumpy(result)


def eigh(a, device="cpu"):
    if device == "cpu":
        return np.linalg.eigh(a)
    cp = cuda()
    import cupyx

    with cupyx.errstate(linalg="raise"):
        values, vectors = cp.linalg.eigh(cp.asarray(a, dtype=cp.float64))
    return cp.asnumpy(values), cp.asnumpy(vectors)


def moments(X, device="cpu"):
    if device == "cpu":
        mean = X.mean(axis=0)
        centered = X - mean
        return mean, centered.T @ centered / len(X)
    cp = cuda()
    values = cp.asarray(X, dtype=cp.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    return cp.asnumpy(mean), cp.asnumpy(centered.T @ centered / len(X))


def environment(device):
    if device == "cpu":
        return {"device": "cpu", "dtype": "float64"}
    cp = cuda()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "device": "cuda",
        "dtype": "float64",
        "cupy": cp.__version__,
        "gpu": name,
        "memory_bytes": int(properties["totalGlobalMem"]),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
    }

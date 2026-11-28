"""Shared accounting and paired inference for the retained covariance study."""

import numpy as np
import pandas as pd
from .metrics import cer


DATASETS = (
    "industry11",
    "international9",
    "factors3",
    "ff21",
    "ff24",
    "ff28",
    "industry52",
)


def replay(targets, excess, rf, cost_bps, *, base_delta=None):
    """Replay one portfolio; retain the stop month and placeholders afterwards."""
    rate = cost_bps / 10000
    equal = np.full(targets.shape[1], 1.0 / targets.shape[1])
    pretrade = equal.copy()
    stopped = False
    rows = []
    for i, (target, X, cash_rate) in enumerate(zip(targets, excess, rf, strict=True)):
        if stopped:
            rows.append({"status": "after_stop"})
            continue
        delta = np.nan if base_delta is None else base_delta[i]
        turnover = float(np.abs(target - pretrade).sum())
        gross = float(target @ X)
        cost = rate * turnover
        net = gross - cost
        gross_nav = 1.0 + cash_rate + gross
        net_nav = 1.0 + cash_rate + net
        status = "active"
        if not np.isfinite([gross_nav, net_nav]).all():
            status = "nonfinite_nav"
        elif gross_nav <= 0:
            status = "nonpositive_gross_nav"
        elif net_nav <= 0:
            status = "nonpositive_net_nav"
        drift = (
            None if status != "active" else target * (1.0 + cash_rate + X) / gross_nav
        )
        cash = 1.0 - float(target.sum())
        rows.append(
            dict(
                status=status,
                delta=delta,
                turnover=turnover,
                cost=cost,
                cash=cash,
                gross_exposure=float(np.abs(target).sum()),
                gross_excess=gross,
                net_excess=net,
                gross_nav_factor=gross_nav,
                net_nav_factor=net_nav,
                target_risky=target,
                pretrade_risky=pretrade,
                pretrade_cash=1.0 - float(pretrade.sum()),
                drift_risky=drift,
                drift_cash=None
                if drift is None
                else cash * (1.0 + cash_rate) / gross_nav,
            )
        )
        stopped = status != "active"
        pretrade = drift
    return pd.DataFrame.from_records(rows)


def circular_indices(n, reps, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(reps, (n + block - 1) // block))
    return ((starts[..., None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def _paired_cer(lhs, rhs, gamma, indices):
    diff = cer(lhs, gamma) - cer(rhs, gamma)
    a, b = (lhs[indices], rhs[indices])
    sampled = 1200 * (
        a.mean(axis=1)
        - b.mean(axis=1)
        - gamma / 2 * (a.var(axis=1, ddof=1) - b.var(axis=1, ddof=1))
    )
    return (diff, *np.quantile(sampled, [0.025, 0.975]))


def _path_metrics(path, gamma):

    def sharpe(series):
        sd = series.std(ddof=1)
        return 100 * series.mean() / sd if sd > 0 else np.nan

    return dict(
        cer_gross_pp=cer(path.gross_excess, gamma) if len(path) >= 2 else np.nan,
        cer_net_pp=cer(path.net_excess, gamma) if len(path) >= 2 else np.nan,
        sr_monthly_x100=sharpe(path.gross_excess),
        sr_net_monthly_x100=sharpe(path.net_excess),
        turnover=path.turnover.mean(),
        delta=path.delta.mean(),
        gross_exposure=path.gross_exposure.mean(),
    )


def paired_summary(variant, control, gamma, indices, *, primary="net"):
    if not variant.month.equals(control.month):
        raise ValueError("Variant and control OOS dates differ")
    expected = len(variant)
    stops = []
    for path in (variant, control):
        failures = np.flatnonzero(path.status.ne("active"))
        stops.append(int(failures[0]) if len(failures) else expected)
    n = min(stops)
    full = n == expected
    row = {
        "n": n,
        "n_expected": expected,
        "complete": full,
        "prefix_first_month": variant.month.iloc[0] if n else None,
        "prefix_last_month": variant.month.iloc[n - 1] if n else None,
    }
    for side, stop, path in zip(
        ("variant", "control"), stops, (variant, control), strict=True
    ):
        row[side + "_stop_month"] = path.month.iloc[stop] if stop < expected else None
        row[side + "_stop_reason"] = path.status.iloc[stop] if stop < expected else None
    for prefix, path in (("", variant), ("control_", control)):
        for name, value in _path_metrics(path.iloc[:n], gamma).items():
            row["prefix_" + prefix + name] = value
            row[prefix + name] = value if full else np.nan
    for kind in ("gross", "net"):
        name = f"delta_{kind}_cer_pp"
        if n >= 2:
            a, b = (
                path[kind + "_excess"].to_numpy()[:n] for path in (variant, control)
            )
            diff, lo, hi = _paired_cer(a, b, gamma, indices)
        else:
            diff = lo = hi = np.nan
        row.update(
            {
                f"prefix_{name}": diff,
                name: diff if full else np.nan,
                f"prefix_{kind}_ci_lo_pp": lo,
                f"prefix_{kind}_ci_hi_pp": hi,
                f"{kind}_ci_lo_pp": lo if full else np.nan,
                f"{kind}_ci_hi_pp": hi if full else np.nan,
            }
        )
    row.update(
        delta_cer_pp=row[f"delta_{primary}_cer_pp"],
        ci_lo_pp=row[f"{primary}_ci_lo_pp"],
        ci_hi_pp=row[f"{primary}_ci_hi_pp"],
        delta_turnover=row["turnover"] - row["control_turnover"],
    )
    return row

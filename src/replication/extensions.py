"""Shared rolling fits, portfolio accounting and paired inference for Part 4."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .backtest import rolling_backtest
from .data import checksum, json_write, load_config
from .methods import combine, fit_rules, suff
from .metrics import cer


BASE_RULES = ("ew", "kz", "ckz")
DATASETS = ("industry11", "international9", "factors3", "ff21", "ff24", "ff28", "industry52")
DELTA_GRID = np.linspace(0, 1, 101)


def turnover_delta(e, direction, pretrade, base, curvature, cost, grid=DELTA_GRID):
    if cost == 0:
        return base
    candidates = np.append(grid, base)
    distance = np.abs(candidates - base)
    turnover = np.abs(e - pretrade + candidates[:, None] * direction).sum(axis=1)
    loss = 0.5 * curvature * distance**2 + cost * turnover
    best = np.flatnonzero(loss == loss.min())
    return float(candidates[best[np.lexsort((candidates[best], distance[best]))[0]]])


def stability_delta(base, direction, covariance, difference):
    signal = float(direction @ covariance @ direction)
    disagreement = float(difference @ covariance @ difference)
    if signal <= 0:
        return 0.0, disagreement, signal, np.nan
    score = disagreement / (signal + 1e-16)
    return base / (1.0 + score), disagreement, signal, score


def fit_targets(panel, window, gamma, experiment, dataset):
    """Fit the common targets once per window and keep the policy diagnostics."""
    columns = [name for name in panel if name.startswith("asset_")]
    assets = [name.removeprefix("asset_") for name in columns]
    records = []
    if len(set(assets)) != len(assets) or "CASH" in assets:
        raise ValueError(f"{dataset}: risky asset names must be unique and distinct from CASH")
    if experiment == "stability" and (window % 2 or window // 2 <= len(assets) + 4):
        raise ValueError(f"{dataset}: stability requires even M and M/2 > N+4")

    def policy(history, fit, state):
        full = history[columns].iloc[-window:].to_numpy(dtype=float)
        end = history.month.iat[-1]
        _, _, _, cov0 = suff(full)
        cov = cov0 * window / (window - 1)
        weights = fit["weights"]
        e, kz = weights["ew"], weights["kz"]
        direction = kz - e
        base = fit["delta"]["ckz"]
        row = {"fit_end": end, "delta_base": base,
               "ckz_flag": fit["diagnostics"]["ckz_flag"]}
        if experiment == "turnover":
            row["curvature"] = float(gamma * (direction @ cov @ direction))
        else:
            halves = np.split(full, 2)
            try:
                first, second = (fit_rules(half, gamma, {"rules": ("kz",)})["weights"]["kz"]
                                 for half in halves)
                conditions = [float(np.linalg.cond(suff(half)[3])) for half in halves]
                if not np.isfinite([first, second]).all() or not np.isfinite(conditions).all():
                    raise ArithmeticError("non-finite KZ weights or condition number")
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(f"{dataset} M={window}, fit_end={end}, half-window KZ: {exc}") from exc
            delta, disagreement, signal, score = stability_delta(base, direction, cov, first - second)
            past = state.setdefault("scores", [])
            finite_past = [value for value in past if np.isfinite(value)]
            threshold = float(np.median(finite_past)) if len(past) >= 12 and finite_past else np.nan
            group = None
            if np.isfinite(score) and np.isfinite(threshold):
                group = "high" if score > threshold else "low"
            past.append(score)
            weights["stability"] = combine(e, kz, delta)
            weights["ckz50"] = combine(e, kz, 0.5)
            fit["delta"].update(stability=delta, ckz50=0.5)
            row.update(disagreement=disagreement, signal=signal, s=score,
                       delta_new=delta, group=group, s_threshold=threshold,
                       condition_first=conditions[0], condition_second=conditions[1],
                       half_size=window // 2)
        records.append(row)
        return fit

    monthly, weights = rolling_backtest(panel, window, gamma, policy, dataset=dataset,
                                        options={"rules": BASE_RULES, "cmp50": False})
    fit_dates = panel.month.iloc[window - 1:-1].tolist()
    targets = {
        rule: group.pivot(index="fit_end", columns="asset", values="weight")
                   .loc[fit_dates, assets].to_numpy()
        for rule, group in weights.groupby("rule", sort=False)
    }
    diag = pd.DataFrame.from_records(records).set_index("fit_end").loc[fit_dates]
    return monthly, targets, diag


def replay(targets, excess, rf, cost_bps, *, base_delta=None, directions=None, curvature=None):
    """Replay one portfolio; retain the stop month and placeholders afterwards."""
    rate = cost_bps / 10_000
    equal = np.full(targets.shape[1], 1.0 / targets.shape[1])
    pretrade = equal.copy()
    stopped = False
    rows = []
    for i, (target, X, cash_rate) in enumerate(zip(targets, excess, rf, strict=True)):
        if stopped:
            rows.append({"status": "after_stop"})
            continue
        delta = np.nan if base_delta is None else base_delta[i]
        if directions is not None and rate != 0:
            delta = turnover_delta(equal, directions[i], pretrade, delta, curvature[i], rate)
            if delta != base_delta[i]:
                target = equal + delta * directions[i]
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
        drift = None if status != "active" else target * (1.0 + cash_rate + X) / gross_nav
        cash = 1.0 - float(target.sum())
        rows.append(dict(
            status=status, delta=delta, turnover=turnover, cost=cost, cash=cash,
            gross_exposure=float(np.abs(target).sum()), gross_excess=gross, net_excess=net,
            gross_nav_factor=gross_nav, net_nav_factor=net_nav, target_risky=target,
            pretrade_risky=pretrade, pretrade_cash=1.0 - float(pretrade.sum()),
            drift_risky=drift, drift_cash=None if drift is None else cash * (1.0 + cash_rate) / gross_nav,
        ))
        stopped = status != "active"
        pretrade = drift
    return pd.DataFrame.from_records(rows)


def circular_indices(n, reps, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(reps, (n + block - 1) // block))
    return ((starts[..., None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def _paired_cer(lhs, rhs, gamma, indices):
    diff = cer(lhs, gamma) - cer(rhs, gamma)
    a, b = lhs[indices], rhs[indices]
    sampled = 1200 * ((a.mean(axis=1) - b.mean(axis=1))
                      - gamma / 2 * (a.var(axis=1, ddof=1) - b.var(axis=1, ddof=1)))
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
        turnover=path.turnover.mean(), delta=path.delta.mean(),
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
    row = {"n": n, "n_expected": expected, "complete": full,
           "prefix_first_month": variant.month.iloc[0] if n else None,
           "prefix_last_month": variant.month.iloc[n - 1] if n else None}
    for side, stop, path in zip(("variant", "control"), stops, (variant, control), strict=True):
        row[side + "_stop_month"] = path.month.iloc[stop] if stop < expected else None
        row[side + "_stop_reason"] = path.status.iloc[stop] if stop < expected else None
    for prefix, path in (("", variant), ("control_", control)):
        for name, value in _path_metrics(path.iloc[:n], gamma).items():
            row["prefix_" + prefix + name] = value
            row[prefix + name] = value if full else np.nan

    for kind in ("gross", "net"):
        name = f"delta_{kind}_cer_pp"
        if n >= 2:
            a, b = (path[kind + "_excess"].to_numpy()[:n] for path in (variant, control))
            diff, lo, hi = _paired_cer(a, b, gamma, indices)
        else:
            diff = lo = hi = np.nan
        row.update({f"prefix_{name}": diff, name: diff if full else np.nan,
                    f"prefix_{kind}_ci_lo_pp": lo, f"prefix_{kind}_ci_hi_pp": hi,
                    f"{kind}_ci_lo_pp": lo if full else np.nan,
                    f"{kind}_ci_hi_pp": hi if full else np.nan})
    row.update(delta_cer_pp=row[f"delta_{primary}_cer_pp"],
               ci_lo_pp=row[f"{primary}_ci_lo_pp"], ci_hi_pp=row[f"{primary}_ci_hi_pp"],
               delta_turnover=row["turnover"] - row["control_turnover"])
    return row


def group_diagnostics(variant, control, n, gamma):
    """Outcome conditional on a lagged threshold, computed on the common prefix."""
    a, b = variant.iloc[:n], control.iloc[:n]
    rows = []
    for group in ("high", "low", "unlabelled"):
        mask = a.group.isna() if group == "unlabelled" else a.group.eq(group)
        left, right = a.loc[mask], b.loc[mask]
        row = {"group": group, "n": len(left), "share": len(left) / n if n else np.nan}
        for kind in ("gross", "net"):
            r, s = left[kind + "_excess"], right[kind + "_excess"]
            row[f"{kind}_mean_delta_monthly"] = (r - s).mean()
            row[f"{kind}_sum_delta"] = (r - s).sum()
            row[f"delta_{kind}_cer_pp"] = cer(r, gamma) - cer(s, gamma) if len(r) >= 2 else np.nan
        rows.append(row)
    return rows


def aggregate_pairs(summary, experiment):
    primary = "net" if experiment == "turnover" else "gross"
    rows = []
    for label, source in [("all", summary), *summary.groupby("window")]:
        for (cost, control), pairs in source.groupby(["cost_bps", "control"]):
            observed = pairs.loc[pairs.complete]
            delta = observed[f"delta_{primary}_cer_pp"]
            wins = int((delta > 0).sum())
            dataset_means = observed.groupby("dataset")[f"delta_{primary}_cer_pp"].mean()
            rows.append(dict(window=label, cost_bps=cost, control=control,
                             n_pairs_expected=len(pairs), n_pairs=len(delta), wins=wins,
                             win_share=wins / len(delta) if len(delta) else np.nan,
                             mean_delta_cer_pp=dataset_means.mean(), median_delta_cer_pp=delta.median(),
                             median_delta_turnover=observed.delta_turnover.median(),
                             median_variant_turnover=observed.turnover.median(),
                             median_control_turnover=observed.control_turnover.median(),
                             median_delta_gross_pp=observed.delta_gross_cer_pp.median()))
    return pd.DataFrame(rows)


def _verify_baseline(rolling, table, trace, targets, baseline_weights, panel, gamma, dataset, window):
    checks = []
    for rule in ("ew", "ckz"):
        fresh = rolling.loc[rolling.rule.eq(rule)]
        official = table.query("case == @dataset and window == @window and computed_rule == @rule").iloc[0]
        value = cer(fresh.ret_excess, gamma)
        if (len(fresh), fresh.realized_month.iat[0], fresh.realized_month.iat[-1]) != (
            official.observations, official.first_month, official.last_month
        ):
            raise ValueError(f"{dataset}/{window}/{rule}: OOS coverage differs from accepted Table 6")
        np.testing.assert_allclose(value, official.computed, rtol=0, atol=1e-9,
                                   err_msg=f"{dataset}/{window}/{rule}: accepted Table 6 CER")
        ret_error = weight_error = None
        if trace is not None:
            accepted = trace.query("dataset == @dataset and window == @window and rule == @rule")
            for key in ("fit_end", "realized_month"):
                np.testing.assert_array_equal(fresh[key], accepted[key])
            error = fresh.ret_excess.to_numpy() - accepted.ret_excess.to_numpy()
            ret_error = float(np.abs(error).max())
            np.testing.assert_allclose(error, 0, rtol=0, atol=1e-12)
        if baseline_weights is not None:
            accepted = baseline_weights.query("dataset == @dataset and window == @window and rule == @rule")
            assets = [name.removeprefix("asset_") for name in panel if name.startswith("asset_")]
            accepted = accepted.pivot(index="fit_end", columns="asset", values="weight")
            np.testing.assert_array_equal(accepted.index, fresh.fit_end)
            reconstructed = np.column_stack([targets[rule], 1.0 - targets[rule].sum(axis=1)])
            error = reconstructed - accepted[assets + ["CASH"]].to_numpy()
            weight_error = float(np.abs(error).max())
            np.testing.assert_allclose(error, 0, rtol=0, atol=1e-10)
        checks.append(dict(dataset=dataset, window=window, rule=rule, n=len(fresh),
                           computed=value, accepted=official.computed, gap_pp=value - official.computed,
                           monthly_max_abs_gap=ret_error, weight_max_abs_gap=weight_error))
    return checks


def run(experiment, args):
    config = load_config(args.config)
    empirical = load_config(args.empirical_config)
    gamma, windows = empirical["gamma"], empirical["windows"]
    settings = config[experiment]
    costs = settings["cost_bps"]
    boot = config["bootstrap"]
    if (not costs or len(set(costs)) != len(costs) or any(not np.isfinite(c) or c < 0 for c in costs)
            or (boot["replicates"] < 1 or boot["block_months"] < 1)):
        raise ValueError("Require distinct nonnegative costs and positive bootstrap size and block length")
    processed = args.processed
    panels = json.loads((processed / "empirical_data_manifest.json").read_text())["datasets"]
    if set(panels) != set(DATASETS) or set(empirical["dataset"]) != set(DATASETS):
        raise ValueError("Expected the seven Table 6 datasets")
    baseline_files = {name: args.baseline / name for name in
                      ("empirical_manifest.json", "table6.csv", "monthly_returns.parquet", "weights.parquet")}
    baseline = json.loads(baseline_files["empirical_manifest.json"].read_text())
    if baseline["mode"] != "full" or baseline["gamma"] != gamma or baseline["windows"] != windows:
        raise ValueError("Baseline must be a full empirical run for the same gamma and windows")
    for dataset, info in panels.items():
        accepted = baseline["datasets"][dataset]
        for key in ("vintage", "weighting", "start", "end", "N", "T", "assets"):
            if info[key] != accepted[key]:
                raise ValueError(f"{dataset}: {key} differs from the baseline")
        archive_keys = ("archive", "member", "sha256")
        sources = [tuple(s[key] for key in archive_keys) for s in info["sources"]]
        expected = [tuple(s[key] for key in archive_keys) for s in accepted["sources"]]
        if sources != expected:
            raise ValueError(f"{dataset}: source files differ from the baseline")
    code = ["extensions", "backtest", "data", "methods", "metrics", "ext_" + experiment]
    baseline_hashes = {name: checksum(path) for name, path in baseline_files.items() if path.is_file()}
    for name, digest in baseline.get("parquet_sha256", {}).items():
        if name in baseline_hashes and baseline_hashes[name] != digest:
            raise ValueError(f"{name}: checksum differs from the baseline manifest")
    identity = dict(experiment=experiment, config=config, gamma=gamma, windows=windows, datasets=panels,
                    baseline_run_id=baseline["run_id"], baseline_sha256=baseline_hashes,
                    python=sys.version.split()[0],
                    packages={name: version(name) for name in ("numpy", "scipy", "pandas", "pyarrow")},
                    code_sha256={name + ".py": checksum(Path(__file__).with_name(name + ".py")) for name in code})
    run_id = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    table = pd.read_csv(baseline_files["table6.csv"])
    if set(table.run_id) != {baseline["run_id"]}:
        raise ValueError("Table 6 run ID differs from the baseline manifest")
    trace, baseline_weights = (
        pd.read_parquet(baseline_files[name], filters=[("rule", "in", ["ckz", "ew"])])
        if name in baseline_hashes else None
        for name in ("monthly_returns.parquet", "weights.parquet")
    )
    controls = ("ckz", "kz", "ew", "ckz50") if experiment == "stability" else ("ckz", "kz", "ew")
    primary = "gross" if experiment == "stability" else "net"
    summary, groups, checks, monthly = [], [], [], []
    indices_by_n = {}

    for dataset in DATASETS:
        file = processed / f"{dataset}.parquet"
        if checksum(file) != panels[dataset]["parquet_sha256"]:
            raise ValueError(f"{file}: checksum differs from the processed data manifest")
        panel = pd.read_parquet(file)
        X = panel.filter(regex="^asset_").to_numpy(dtype=float)
        for window in windows:
            fitted, targets, diag = fit_targets(panel, window, gamma, experiment, dataset)
            checks.extend(_verify_baseline(fitted, table, trace, targets, baseline_weights,
                                           panel, gamma, dataset, window))
            dates = panel.month.iloc[window:].to_numpy()
            rf = panel.rf.to_numpy()[window:]
            delta = {"ckz": diag.delta_base.to_numpy(), "turnover": diag.delta_base.to_numpy(),
                     "ew": np.zeros(len(dates)), "kz": np.ones(len(dates))}
            if experiment == "stability":
                delta.update(stability=diag.delta_new.to_numpy(), ckz50=np.full(len(dates), 0.5))
            metadata = diag.reset_index(drop=True)
            for cost in costs:
                paths = {}
                for rule in (experiment, *controls):
                    target = targets["ckz"] if rule == "turnover" else targets[rule]
                    kwargs = {}
                    if rule == "turnover":
                        kwargs = dict(directions=targets["kz"] - targets["ew"], curvature=diag.curvature.to_numpy())
                    paths[rule] = replay(target, X[window:], rf, cost, base_delta=delta[rule], **kwargs).assign(
                        dataset=dataset, window=window, variant=rule, cost_bps=cost, rf=rf,
                        month=dates, fit_end=diag.index.to_numpy(), run_id=run_id,
                    ).join(metadata)
                    monthly.append(paths[rule])
                a = paths[experiment]
                if cost == 0 and experiment == "turnover":
                    pd.testing.assert_frame_equal(a.drop(columns="variant"), paths["ckz"].drop(columns="variant"))
                for control in controls:
                    b = paths[control]
                    n = int((a.status.eq("active") & b.status.eq("active")).sum())
                    if n >= 2 and n not in indices_by_n:
                        indices_by_n[n] = circular_indices(n, boot["replicates"], boot["block_months"], boot["seed"])
                    keys = dict(dataset=dataset, window=window, cost_bps=cost, control=control,
                                variant=experiment, run_id=run_id)
                    row = paired_summary(a, b, gamma, indices_by_n.get(n), primary=primary)
                    summary.append({**keys, **row})
                    if experiment == "stability":
                        groups.extend({**keys, "complete": row["complete"], **g}
                                      for g in group_diagnostics(a, b, row["n"], gamma))
            print(f"{experiment}: {dataset} M={window}, {len(dates)} OOS months", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    products = ["monthly.parquet", "summary.csv", "aggregate.csv", "baseline_checks.csv"]
    pd.concat(monthly, ignore_index=True).to_parquet(args.out / "monthly.parquet", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    aggregate_pairs(summary, experiment).assign(run_id=run_id).to_csv(args.out / "aggregate.csv", index=False)
    pd.DataFrame(checks).assign(run_id=run_id).to_csv(args.out / "baseline_checks.csv", index=False)
    if groups:
        pd.DataFrame(groups).to_csv(args.out / "groups.csv", index=False)
        products.append("groups.csv")
    if experiment == "turnover":
        bracket = []
        for (dataset, window, control), pair in summary.groupby(["dataset", "window", "control"]):
            rows = list(pair.sort_values("cost_bps").itertuples())
            for left, right in zip(rows, rows[1:]):
                if (left.complete and right.complete and left.delta_net_cer_pp <= 0
                        and right.delta_net_cer_pp > 0 and right.delta_gross_cer_pp < 0):
                    bracket.append(dict(dataset=dataset, window=window, control=control,
                                        lower_bps=left.cost_bps, upper_bps=right.cost_bps, run_id=run_id))
        columns = ["dataset", "window", "control", "lower_bps", "upper_bps", "run_id"]
        pd.DataFrame(bracket, columns=columns).to_csv(args.out / "break_even.csv", index=False)
        products.append("break_even.csv")
    json_write(args.out / "manifest.json", {
        **identity, "run_id": run_id, "command": ["python", *sys.orig_argv[1:]],
        "policy": ("101-point grid plus baseline; nearest baseline then smaller delta"
                   if experiment == "turnover" else "contiguous halves; delta_base/(1+s); no fallback"),
        "accounting": "risky L1 turnover; EW initial holdings; RF as supplied including international9; "
                      "synthetic excess sleeves with RF collateral; fees on start NAV, withheld "
                      "proportionally at month end; stop on nonpositive gross or net NAV",
        "cer": "Eq. (22), annual percentage points, monthly excess fractions, ddof=1",
        "bootstrap": "paired circular blocks; realized series; quantiles 0.025 and 0.975",
        "pair_rows": len(summary), "complete_pair_rows": int(summary.complete.sum()),
        "output_sha256": {name: checksum(args.out / name) for name in products},
    })
    print(f"{run_id}: {int(summary.complete.sum())}/{len(summary)} complete pair rows", flush=True)


def main(experiment):
    parser = argparse.ArgumentParser(description=f"Run the Tu–Zhou {experiment} extension")
    parser.add_argument("--config", type=Path, default=Path(f"configs/{experiment}.toml"))
    parser.add_argument("--empirical-config", type=Path, default=Path("configs/empirical.toml"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline", type=Path, default=Path("results/replication"))
    parser.add_argument("--out", type=Path, required=True)
    run(experiment, parser.parse_args())

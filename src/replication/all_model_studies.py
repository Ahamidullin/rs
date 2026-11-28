"""Retained studies for every implemented out-of-sample Tu-Zhou rule.

Historical regularization, dimension/window and Developed/Emerging studies.
Never runs an experiment on import. Original article artifacts are read-only.
"""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from . import backend
from .data import checksum, json_write
from .extensions import circular_indices, paired_summary, replay
from .methods import fit_rules
from .study_support import baseline_panel_path, verify_baseline
from . import study_data as scope


EXPERIMENTS = ("regularization", "dimension_window", "markets")
ORIGINALS = ("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz", "cmp50")
REGULARIZED = tuple(r for r in ORIGINALS if r != "ew")
ALL_RULES = ORIGINALS + tuple("lw_"+r for r in REGULARIZED)
SPEC = dict(scope.SPEC,
    original_rules=list(ORIGINALS), regularized_rules=list(REGULARIZED),
    cmp_mode="jk", cmp50=0.5,
    primary_comparison="each regularized rule minus its own unmodified rule at 25 bps",
    secondary_comparison="every rule minus EW",
    covariance_protocol="freeze component signal and original combination delta; shrink model-specific risk matrix",
)


def pj_covariance(history):
    """The actual Jorion risk matrix from methods.fit_rules, before replacement."""
    T, N = history.shape
    mean = history.mean(0)
    centered = history-mean
    sample = centered.T@centered/T
    st = T/(T-N-2)*sample
    one = np.ones(N)
    solved = np.linalg.solve(st, np.column_stack([mean, one]))
    A = float(one@solved[:, 1])
    g = float(one@solved[:, 0]/A)
    d = mean-g*one
    spread = max(0., float(d@np.linalg.solve(st, d)))
    inverse_lambda = spread/(N+2)
    dd = 1+inverse_lambda/(T*inverse_lambda+1)
    factor = one*np.sqrt(1/(T*((T+1)*inverse_lambda+1)*A))
    return dd*st+np.outer(factor, factor)


def mp_covariance(history, weights, gamma, boundary=None):
    """Recover the MP model covariance from its saved MLE weights and moments.

    mu_MP is parallel to w_MP. Interior: Sigma=s2*I+(c-s2)*u*u'.
    Boundary a=0: Sigma=trace(S0)/N*I. This avoids recomputing historical
    leave-one-out fits merely to transform already-saved MP/CMP portfolios.
    """
    T, N = history.shape
    mean = history.mean(0)
    centered = history-mean
    sample = centered.T@centered/T
    s2_boundary = float(np.trace(sample)/N)
    if s2_boundary <= 0:
        raise ValueError("MP covariance requires positive sample variance")
    boundary_weights = mean/(gamma*s2_boundary)
    if boundary is None:
        boundary = bool(np.allclose(weights, boundary_weights, rtol=1e-7, atol=1e-10))
    if boundary:
        np.testing.assert_allclose(weights, boundary_weights, rtol=1e-6, atol=1e-9)
        return s2_boundary*np.eye(N), True
    norm = np.linalg.norm(weights)
    if norm <= 0:
        raise ValueError("Cannot recover nonboundary MP covariance from zero weights")
    u = weights/norm
    c = float(u@sample@u)
    second_moment = sample+np.outer(mean, mean)
    s2 = float((np.trace(second_moment)-u@second_moment@u)/(N-1))
    if min(c, s2) <= 0:
        raise ValueError("Invalid recovered MP covariance")
    np.testing.assert_allclose(weights, (u@mean)/(gamma*c)*u, rtol=1e-6, atol=1e-9)
    return s2*np.eye(N)+(c-s2)*np.outer(u, u), False


def shrink_model_matrix(matrix, alpha):
    """Same historical LW intensity; preserve each model's covariance scale."""
    if not 0 <= alpha <= 1:
        raise ValueError("Shrinkage intensity must be in [0,1]")
    return (1-alpha)*matrix+alpha*np.trace(matrix)/len(matrix)*np.eye(len(matrix))


def regularize_fit(history, original, gamma, device="cpu", alpha_override=None):
    """Freeze original component signals/deltas and shrink their own risk matrices.

    CML uses its scaled Markowitz component, not the unscaled ML portfolio.
    PJ/CPJ use Jorion's risk matrix; MP/CMP use the MP model risk matrix.
    The original analytical/jackknife coefficients are NEVER refitted here.
    """
    history = np.asarray(history, float)
    T, N = history.shape
    if T <= N+4 or not np.isfinite(history).all():
        raise ValueError("Require finite history and T>N+4")
    weights, delta = original["weights"], original["delta"]
    if not set(ORIGINALS).issubset(weights):
        raise ValueError("Missing an original model; partial suites are not accepted")
    mean = history.mean(0)
    centered = history-mean
    sample = centered.T@centered/T
    fitted = LedoitWolf().fit(history)
    alpha = float(fitted.shrinkage_) if alpha_override is None else float(alpha_override)
    lw = shrink_model_matrix(sample, alpha)
    scaled_ml = backend.solve(sample, mean, device)*(T-N-2)/(T*gamma)
    components = backend.solve(lw, sample@np.column_stack([
        weights["ml"], scaled_ml, weights["kz"]]), device)
    pj_risk = pj_covariance(history)
    pj_new = backend.solve(shrink_model_matrix(pj_risk, alpha), pj_risk@weights["pj"], device)
    mp_risk, boundary = mp_covariance(history, weights["mp"], gamma,
        original.get("diagnostics", {}).get("mp_boundary"))
    mp_new = backend.solve(shrink_model_matrix(mp_risk, alpha), mp_risk@weights["mp"], device)
    e = weights["ew"]
    new = dict(ml=components[:, 0], kz=components[:, 2], pj=pj_new, mp=mp_new)
    for name, component in (("cml", components[:, 1]), ("ckz", components[:, 2]),
                            ("cpj", pj_new), ("cmp", mp_new), ("cmp50", mp_new)):
        coefficient = 0.5 if name == "cmp50" else float(delta[name])
        if not 0 <= coefficient <= 1:
            raise ValueError(f"Invalid original delta: {name}")
        new[name] = e+coefficient*(component-e)
    # Check the distinct CML component against the original definition.
    np.testing.assert_allclose(weights["cml"], e+delta["cml"]*(scaled_ml-e),
                               rtol=1e-7, atol=1e-9)
    output = {r: np.asarray(weights[r]).copy() for r in ORIGINALS}
    output.update({"lw_"+r: new[r] for r in REGULARIZED})
    if any(w.shape != (N,) or not np.isfinite(w).all() for w in output.values()):
        raise ArithmeticError("Non-finite or misshaped portfolio in all-model suite")
    return output, dict(shrinkage_intensity=alpha, sample_condition=float(np.linalg.cond(sample)),
        regularized_condition=float(np.linalg.cond(lw)), mp_boundary=boundary,
        cmp_n_refit=original.get("diagnostics", {}).get("cmp_n_refit"),
        **{f"delta_{r}": (0.5 if r == "cmp50" else delta[r])
           for r in ("cml", "cpj", "cmp", "ckz", "cmp50")})


def decision_weights(history, device="cpu"):
    fit = fit_rules(history, SPEC["gamma"], dict(rules=ORIGINALS[:-1], cmp50=True,
        cmp_mode="jk", ckz_mode=SPEC["ckz_mode"], device=device))
    return regularize_fit(history, fit, SPEC["gamma"], device)


def load_original_case(args, source, case, panel):
    """Use verified full historical weights instead of rerunning the article."""
    window, dataset = case["window"], case["dataset"]
    dates = panel.month.to_numpy()[window:]
    fit_dates = panel.month.to_numpy()[window-1:-1]
    filters = [("dataset", "==", dataset), ("window", "==", window)]
    saved_w = pd.read_parquet(args.baseline/"weights.parquet", filters=filters)
    saved_r = pd.read_parquet(args.baseline/"monthly_returns.parquet", filters=filters)
    weights, deltas = {}, {}
    assets = [a.removeprefix("asset_") for a in case["assets"]]
    x = panel[case["assets"]].to_numpy()[window:]
    for rule in ORIGINALS:
        wide = saved_w.loc[saved_w.rule.eq(rule)].pivot(index="fit_end", columns="asset", values="weight")
        if not np.array_equal(wide.index, fit_dates):
            raise ValueError(f"Missing/misaligned saved weights: {dataset}/{window}/{rule}")
        weights[rule] = wide.loc[fit_dates, assets].to_numpy()
        returns = saved_r.loc[saved_r.rule.eq(rule)].set_index("realized_month").loc[dates]
        np.testing.assert_allclose(np.einsum("ij,ij->i", weights[rule], x), returns.ret_excess,
                                   rtol=0, atol=1e-10)
        if rule in ("cml", "cpj", "cmp", "ckz", "cmp50"):
            deltas[rule] = returns.delta.to_numpy()
    if args.retained is None:
        return weights, deltas, None
    reference = pd.read_parquet(args.retained/"monthly.parquet", filters=[
        *filters, ("rule", "==", "shrinkage"), ("cost_bps", "==", 0)]).reset_index(drop=True)
    if (not np.array_equal(reference.month, dates) or
            not np.array_equal(reference.fit_end, fit_dates) or not reference.status.eq("active").all()):
        raise ValueError("Retained CKZ regularization is incomplete or dates differ")
    return weights, deltas, np.stack(reference.target_risky)


def prepare_cases(args, experiment, out):
    processed = out/"processed"
    processed.mkdir()
    if experiment == "regularization":
        source = verify_baseline(args.baseline)
        if source["gamma"] != SPEC["gamma"]:
            raise ValueError("Historical baseline gamma differs from fixed study gamma")
        retained = None
        if args.retained is not None:
            retained = json.loads((args.retained/"manifest.json").read_text())
            if (retained["status"] != "complete" or retained["experiment"] != "shrinkage" or
                retained["baseline_manifest_sha256"] != checksum(args.baseline/"empirical_manifest.json")):
                raise ValueError("Require completed retained shrinkage on the same baseline")
            for name, expected in retained["output_sha256"].items():
                if checksum(args.retained/name) != expected:
                    raise ValueError(f"Retained checksum differs: {name}")
        panels, cases = {}, []
        for dataset, metadata in source["datasets"].items():
            path = baseline_panel_path(source, dataset)
            shutil.copyfile(path, processed/path.name)
            panels[dataset] = pd.read_parquet(path)
            for window in source["windows"]:
                cases.append(dict(case_id=f"{dataset}_M{window}", dataset=dataset, window=window,
                    N=len(metadata["assets"]), subset_seed=-1,
                    assets=["asset_"+a for a in metadata["assets"]]))
        metadata = dict(baseline_manifest_sha256=checksum(args.baseline/"empirical_manifest.json"),
            retained_manifest_sha256=checksum(args.retained/"manifest.json") if retained else None,
            baseline_output_sha256=source["output_sha256"], retained_output_sha256=retained["output_sha256"] if retained else None)
        return panels, cases, metadata, source
    factors, factor_info = scope.load_source("ff3_fiz", args.raw, args.download)
    keys = ("industry49",) if experiment == "dimension_window" else ("developed6", "emerging6")
    panels, sources = {}, [factor_info]
    for key in keys:
        values, info = scope.load_source(key, args.raw, args.download)
        panels[key] = scope.make_panel(values, factors, SPEC["history_start"], SPEC["test_end"])
        panels[key].to_parquet(processed/f"{key}.parquet", index=False)
        sources.append(info)
    if experiment == "dimension_window":
        assets = [c for c in panels["industry49"] if c.startswith("asset_")]
        cases = [dict(case_id=f"industry_N{n}_seed{seed}_M{window}", dataset="industry49", N=n,
                      subset_seed=seed, assets=subset, window=window)
            for n, seed, subset in scope.subset_plan(assets, SPEC["grid_dimensions"], SPEC["subset_seeds"])
            for window in SPEC["grid_windows"]]
    else:
        cases = [dict(case_id=f"{key}_M{window}", dataset=key, N=len(panels[key].columns)-2,
                      subset_seed=-1, assets=[c for c in panels[key] if c.startswith("asset_")], window=window)
                 for key in keys for window in SPEC["windows"]]
    return panels, cases, dict(sources=sources), None


def fit_case(args, experiment, case, panel, source):
    x = panel[case["assets"]].to_numpy(float)
    dates = panel.month.to_numpy()
    expected = pd.period_range(dates[0], dates[-1], freq="M").astype(str)
    if not np.array_equal(dates, expected) or not np.isfinite(x).all() or not np.isfinite(panel.rf).all():
        raise ValueError("Invalid or incomplete panel")
    window = case["window"]
    if window <= case["N"]+4:
        raise ValueError("Require M>N+4 for original analytical rules")
    cached = None
    if experiment == "regularization":
        indices = np.arange(window, len(panel))
        cached = load_original_case(args, source, case, panel)
    else:
        indices = np.flatnonzero((dates >= SPEC["test_start"]) & (dates <= SPEC["test_end"]))
        required = pd.period_range(SPEC["test_start"], SPEC["test_end"], freq="M").astype(str)
        if not np.array_equal(dates[indices], required) or indices[0] < window:
            raise ValueError("Missing common test months or historical observations")
    targets, diagnostics = {r: [] for r in ALL_RULES}, []
    for count, i in enumerate(indices):
        history = x[i-window:i].copy()
        try:
            if cached is None:
                weights, info = decision_weights(history, args.device)
            else:
                originals, coefficients, reference = cached
                fit = dict(weights={r: v[count] for r, v in originals.items()},
                           delta={r: v[count] for r, v in coefficients.items()})
                weights, info = regularize_fit(history, fit, SPEC["gamma"], args.device)
                if reference is not None:
                    np.testing.assert_allclose(weights["lw_ckz"], reference[count], rtol=1e-8, atol=1e-8)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError, AssertionError) as exc:
            raise RuntimeError(f"{case['case_id']}, fit_end={dates[i-1]}: {exc}") from exc
        for rule in ALL_RULES:
            targets[rule].append(weights[rule])
        diagnostics.append(dict(month=dates[i], fit_start=dates[i-window], fit_end=dates[i-1], **info))
        if (count+1) % 12 == 0 or count+1 == len(indices):
            print(f"  {count+1}/{len(indices)} all-model decisions", flush=True)
    return ({r: np.stack(w) for r, w in targets.items()}, pd.DataFrame(diagnostics),
            x[indices], panel.rf.to_numpy(float)[indices])


def comparisons():
    return ([("regularization_effect", "lw_"+r, r) for r in REGULARIZED]
            + [("versus_ew", r, "ew") for r in ALL_RULES if r != "ew"])


def evaluate_case(experiment, case, targets, diagnostics, excess, rf):
    tags = {k: case[k] for k in ("case_id", "dataset", "N", "window", "subset_seed")}
    periods = [("full", diagnostics.month.iloc[0], diagnostics.month.iloc[-1])]
    if experiment != "regularization":
        periods += SPEC["periods"]
    monthly, scores, summaries = [], [], []
    for cost in SPEC["cost_bps"]:
        paths = {}
        for rule in ALL_RULES:
            base_rule = rule.removeprefix("lw_")
            delta = (np.zeros(len(diagnostics)) if base_rule == "ew" else
                     diagnostics["delta_"+base_rule].to_numpy() if "delta_"+base_rule in diagnostics
                     else np.ones(len(diagnostics)))
            paths[rule] = replay(targets[rule], excess, rf, cost, base_delta=delta).assign(
                month=diagnostics.month.to_numpy(), fit_end=diagnostics.fit_end.to_numpy(), rf=rf)
            monthly.append(paths[rule].assign(rule=rule, cost_bps=cost, **tags))
        for period, start, end in periods:
            subset = {r: p.loc[p.month.between(start, end)].reset_index(drop=True) for r, p in paths.items()}
            if not len(subset["ew"]):
                continue
            for rule, path in subset.items():
                risk = scope.risk_metrics(path)
                complete = bool(path.status.eq("active").all())
                n = scope.active_prefix(path)
                def cer(values):
                    return float(1200*(values.mean()-SPEC["gamma"]/2*values.var(ddof=1)))
                scores.append(dict(**tags, period=period, rule=rule, cost_bps=cost, **risk,
                    cer_net_pp=cer(path.net_excess) if complete and len(path)>=2 else np.nan,
                    cer_gross_pp=cer(path.gross_excess) if complete and len(path)>=2 else np.nan,
                    prefix_cer_net_pp=cer(path.net_excess.iloc[:n]) if n>=2 else np.nan))
            draw_cache = {}
            for kind, variant, control in comparisons():
                a, b = subset[variant], subset[control]
                n = min(scope.active_prefix(a), scope.active_prefix(b))
                if n not in draw_cache:
                    draw_cache[n] = circular_indices(n, SPEC["bootstrap_replicates"],
                        SPEC["bootstrap_block"], SPEC["bootstrap_seed"]) if n>=2 else None
                summaries.append(dict(**tags, period=period, comparison=kind,
                    variant=variant, control=control, cost_bps=cost,
                    **paired_summary(a, b, SPEC["gamma"], draw_cache[n])))
    return pd.concat(monthly, ignore_index=True), pd.DataFrame(scores), pd.DataFrame(summaries)


def aggregate_summary(summary, experiment):
    keys = ["comparison", "variant", "control", "period", "cost_bps"]
    if experiment == "dimension_window":
        keys += ["N", "window"]
    else:
        keys += ["dataset", "N", "window"]
    records = []
    for group_keys, group in summary.groupby(keys):
        completed = group.loc[group.complete]
        records.append(dict(zip(keys, group_keys), expected_pairs=len(group),
            complete_pairs=len(completed), wins=int(completed.delta_net_cer_pp.gt(0).sum()),
            median_delta_net_cer_pp=completed.delta_net_cer_pp.median(),
            positive_intervals=int(completed.net_ci_lo_pp.gt(0).sum())))
    return pd.DataFrame(records)


def write_report(out, experiment, summary, scores):
    lines = [f"# {experiment}: all-model study", "",
        "Original out-of-sample rules: EW, ML, Jorion (PJ), MacKinlay-Pastor (MP), Kan-Zhou (KZ), CML, CPJ, CMP jackknife, CKZ and CMP50. Each non-EW rule has a separately named lw_ version. In-sample ML is not a feasible OOS strategy and is excluded.",
        "Primary: each regularized model versus its own unmodified model at 25 bps. Secondary: every method versus EW. Units: annual CER percentage points.", "",
        "| Model | Complete / expected | Wins | Median delta net CER pp | Positive intervals |",
        "|---|---:|---:|---:|---:|"]
    primary = summary.loc[summary.comparison.eq("regularization_effect") & summary.period.eq("full") & summary.cost_bps.eq(25)]
    for model in REGULARIZED:
        group = primary.loc[primary.control.eq(model)]
        full = group.loc[group.complete]
        lines.append(f"| {model} | {len(full)}/{len(group)} | {int(full.delta_net_cer_pp.gt(0).sum())} | {full.delta_net_cer_pp.median():+.4f} | {int(full.net_ci_lo_pp.gt(0).sum())} |")
    lines += ["", "## Interpretation and limits", "",
        "Models are not pooled in the aggregate CSV. The overview above pools cases only within each model; cases overlap and differ in N/window, so medians are descriptive. Complete coverage must be checked before interpreting a pooled median.",
        "ML/KZ and their combinations replace only inverse sample covariance. CML retains its distinct scaled Markowitz component. PJ/CPJ shrink the original Jorion risk matrix; MP/CMP/CMP50 shrink the MP model-implied covariance. Every replacement uses the raw-history Ledoit-Wolf intensity and the original matrix's scaled-identity target. Original component signals and analytical/jackknife deltas stay frozen; coefficients are not reoptimized for the altered estimators. These are new hybrid rules without inherited optimality guarantees.",
        "MP covariance is spherical plus rank one in the direction of its mean. Scaled-identity shrinkage therefore changes MP exposure rather than introducing a new relative allocation direction. Its boundary covariance is already spherical, so that transformation has no effect there. Do not promise a separate allocation improvement for every family.",
        "Historical regularization uses the original verified seven panels and their original 120/240-window OOS calendars. Original weights and CMP jackknife coefficients are loaded from completed article artifacts, not refitted. The old successful CKZ regularization targets are checked at every historical decision and remain unchanged on disk.",
        "Dimension/window and market studies evaluate a common 2012-2024 calendar. Five predetermined nested industry subsets at N=10/25 and the full N=49 once, with M=60/120/240; market data use six Developed/Emerging USD size/value portfolios, M=120/240. Fixed subperiods remain descriptive; holdings are not reset or restarted at boundaries.",
        "Failures remain in monthly paths and subsequent months are after_stop. Full CER is missing for incomplete paths. Medians use completed pairs only, and tables/heatmaps display coverage. scores.csv includes absolute CER, risk, turnover and failures for every model, not only relative wins.",
        "Costs 0/25 bps use risky turnover. Financing spreads, short-borrow costs and source-portfolio implementation frictions are not included; large gross exposures remain a material limitation.",
        "Paired circular bootstrap: 2,000 draws, 12-month blocks, conditional on fitted paths, no refitting or multiplicity correction. Models, windows, subsets and related markets are dependent. These hypotheses were developed after earlier results were examined; no untouched prospective holdout or universal superiority is claimed. Original article-replication discrepancies remain unresolved.",
        "CMP jackknife includes exact leave-one-out MP fits and can be expensive. No model is silently omitted, replaced by CMP50, or changed to a cheaper approximation. Numerical fitting errors stop with an incomplete manifest; financial path failures do not disappear from comparisons.",
        "", "See summary.csv for case/model/period comparisons; aggregate.csv for model-specific cells; scores.csv for absolute outcomes; cases.json for exact assets; manifest.json for source/code/output hashes. No PDF or presentation is generated.", ""]
    (out/"report.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots(out, aggregate, experiment, panel_dir=None):
    if experiment != "dimension_window":
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pink_lavender", ["#D9A9C0", "#FFFFFF", "#BBA7D2"])
    rows = aggregate.loc[aggregate.comparison.eq("regularization_effect") & aggregate.period.eq("full") & aggregate.cost_bps.eq(25)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    finite = np.abs(rows.median_delta_net_cer_pp.dropna().to_numpy())
    bound = max(float(finite.max()), .01) if len(finite) else 1.
    for ax, model in zip(axes.flat, REGULARIZED, strict=True):
        group = rows.loc[rows.control.eq(model)]
        matrix = group.pivot(index="N", columns="window", values="median_delta_net_cer_pp")
        plotted = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), cmap=cmap, vmin=-bound, vmax=bound)
        for i, n in enumerate(matrix.index):
            for j, window in enumerate(matrix.columns):
                row = group.loc[group.N.eq(n)&group.window.eq(window)].iloc[0]
                value = matrix.iloc[i, j]
                label = f"{value:+.2f}" if np.isfinite(value) else "N/A"
                ax.text(j, i, f"{label}\n{row.complete_pairs}/{row.expected_pairs}", ha="center", va="center", fontsize=8)
        ax.set(title=f"LW {model} minus {model}", xlabel="Window (months)", ylabel="Assets N",
               xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               yticks=range(len(matrix.index)), yticklabels=matrix.index)
    fig.colorbar(plotted, ax=list(axes.flat), shrink=.7, label="Delta annual net CER pp, costs 25 bps")
    fig.suptitle("Model-specific covariance regularization; labels include completed/expected cases")
    fig.savefig(out/"all_model_heatmaps.png", dpi=150)
    if panel_dir is not None:
        panel_dir.mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        fig.set_layout_engine(None)
        renderer = fig.canvas.get_renderer()
        for ax, model in zip(axes.flat, REGULARIZED, strict=True):
            bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(panel_dir / f"dimension_window_{model}.png", dpi=180,
                        bbox_inches=bounds, pad_inches=.08)
    plt.close(fig)


def run_experiment(args, experiment):
    out = args.out/experiment
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    identity = dict(status="preparing", experiment=experiment, specification=SPEC,
        backend=backend.environment(args.device),
        packages={n: version(n) for n in ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "matplotlib")},
        code_sha256={n: checksum(Path(__file__).with_name(n)) for n in
                     ("all_model_studies.py", "study_data.py", "study_support.py", "extensions.py", "methods.py", "mp.py", "backend.py", "data.py", "metrics.py")})
    json_write(out/"manifest.json", identity)
    panels, cases, metadata, source = prepare_cases(args, experiment, out)
    json_write(out/"cases.json", cases)
    identity.update(status="running", inputs=metadata,
        processed_sha256={p.name: checksum(p) for p in (out/"processed").glob("*.parquet")})
    json_write(out/"manifest.json", identity)
    monthly, scores, summaries, diagnostics = [], [], [], []
    for number, case in enumerate(cases, 1):
        print(f"{experiment}: case {number}/{len(cases)} {case['case_id']}", flush=True)
        targets, fitted, excess, rf = fit_case(args, experiment, case, panels[case["dataset"]], source)
        paths, score, summary = evaluate_case(experiment, case, targets, fitted, excess, rf)
        monthly.append(paths)
        scores.append(score)
        summaries.append(summary)
        diagnostics.append(fitted.assign(**{k: case[k] for k in ("case_id", "dataset", "N", "window", "subset_seed")}))
        pd.concat(monthly, ignore_index=True).to_parquet(out/"monthly.parquet", index=False)
        pd.concat(scores, ignore_index=True).to_csv(out/"scores.csv", index=False)
        pd.concat(summaries, ignore_index=True).to_csv(out/"summary.csv", index=False)
        pd.concat(diagnostics, ignore_index=True).to_csv(out/"diagnostics.csv", index=False)
        identity["completed_cases"] = number
        json_write(out/"manifest.json", identity)
    summary, score = pd.concat(summaries, ignore_index=True), pd.concat(scores, ignore_index=True)
    aggregate = aggregate_summary(summary, experiment)
    aggregate.to_csv(out/"aggregate.csv", index=False)
    write_report(out, experiment, summary, score)
    make_plots(out, aggregate, experiment)
    identity.update(status="complete", elapsed_seconds=time.monotonic()-started)
    identity["output_sha256"] = {str(p.relative_to(out)).replace("\\", "/"): checksum(p)
        for p in out.rglob("*") if p.is_file() and p != out/"manifest.json"}
    json_write(out/"manifest.json", identity)
    print(f"Complete: {out}/report.md", flush=True)


def run(args):
    selected = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for name in selected:
        if (args.out/name).exists() and any((args.out/name).iterdir()):
            raise FileExistsError(f"Nonempty output: {args.out/name}; use a new --out")
    if args.device == "cuda":
        os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
    backend.check_device(args.device)
    for name in selected:
        run_experiment(args, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all",)+EXPERIMENTS, default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--raw", type=Path, default=Path("data/robustness_sources"))
    parser.add_argument("--baseline", type=Path, default=Path("runs/article_gpu_v1/replication"))
    parser.add_argument("--retained", type=Path, default=None,
        help="Optional completed historical CKZ shrinkage run for exact cross-checking")
    parser.add_argument("--out", type=Path, default=Path("runs/all_models_v1"))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[2])
    run(args)


if __name__ == "__main__":
    main()

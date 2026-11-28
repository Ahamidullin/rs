"""All-model coverage, covariance semantics and synthetic integration checks."""

from argparse import Namespace
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from replication import all_model_studies as am
from replication import mp
from replication.data import checksum, json_write
from replication.methods import fit_rules
from replication.study_support import inverse_only_weights


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(22)
    x = rng.normal(.006, .03, (36, 1))*np.array([.8, 1., 1.3])+rng.normal(0, .02, (36, 3))
    fit = fit_rules(x, 3, dict(rules=am.ORIGINALS[:-1], cmp50=True, cmp_mode="jk"))
    return x, fit


def test_every_original_and_regularized_model_is_present(fitted):
    x, original = fitted
    weights, info = am.regularize_fit(x, original, 3)
    assert set(weights) == set(am.ALL_RULES)
    assert len(weights) == 19
    assert "cmp" in weights and "cmp50" in weights
    assert 0 <= info["shrinkage_intensity"] <= 1
    for model in am.ORIGINALS:
        np.testing.assert_array_equal(weights[model], original["weights"][model])


def test_zero_regularization_recovers_every_original(fitted):
    x, original = fitted
    weights, _ = am.regularize_fit(x, original, 3, alpha_override=0)
    for model in am.REGULARIZED:
        np.testing.assert_allclose(weights["lw_"+model], weights[model], rtol=1e-8, atol=1e-9)


def test_ckz_extension_is_identical_to_retained_successful_definition(fitted):
    x, original = fitted
    weights, _ = am.regularize_fit(x, original, 3)
    component, _ = inverse_only_weights(x, original["weights"]["kz"])
    equal = original["weights"]["ew"]
    expected = equal+original["delta"]["ckz"]*(component-equal)
    np.testing.assert_allclose(weights["lw_ckz"], expected, atol=1e-11)


def test_combinations_freeze_their_own_coefficients_and_cml_scaled_component(fitted):
    x, fitted_original = fitted
    original = deepcopy(fitted_original)
    e = original["weights"]["ew"]
    T, N = x.shape
    centered = x-x.mean(0)
    sample = centered.T@centered/T
    scaled_ml = np.linalg.solve(sample, x.mean(0))*(T-N-2)/(T*3)
    for model, component in (("cml", scaled_ml), ("cpj", original["weights"]["pj"]),
                              ("cmp", original["weights"]["mp"]), ("ckz", original["weights"]["kz"])):
        original["delta"][model] = .37
        original["weights"][model] = e+.37*(component-e)
    weights, _ = am.regularize_fit(x, original, 3)
    for combined, base in (("cpj", "pj"), ("cmp", "mp"), ("ckz", "kz")):
        np.testing.assert_allclose(weights["lw_"+combined], e+.37*(weights["lw_"+base]-e))
    np.testing.assert_allclose(weights["lw_cmp50"], e+.5*(weights["lw_mp"]-e))
    # ML uses S1, CML uses scaled S0 inverse. Preserve their distinct multiplier.
    ratio = (T-N-2)/(T-1)
    np.testing.assert_allclose(weights["lw_cml"], e+.37*(ratio*weights["lw_ml"]-e))


def test_pj_matrix_keeps_original_mean_signal(fitted):
    x, original = fitted
    T, N = x.shape
    sample = np.cov(x.T, bias=True)
    st = T/(T-N-2)*sample
    one = np.ones(N)
    mean = x.mean(0)
    so = np.linalg.solve(st, one)
    g = float(so@mean/(one@so))
    spread = (mean-g*one)@np.linalg.solve(st, mean-g*one)
    rho = 1/(T*spread/(N+2)+1)
    expected_mean = (1-rho)*mean+rho*g*one
    np.testing.assert_allclose(3*am.pj_covariance(x)@original["weights"]["pj"], expected_mean, atol=1e-11)


def test_mp_model_covariance_recovery_matches_mle(fitted):
    x, original = fitted
    mean, sample = x.mean(0), np.cov(x.T, bias=True)
    weights, actual = mp.mp_weights_full(mean, sample, 3)
    recovered, boundary = am.mp_covariance(x, weights, 3)
    expected = actual["s2"]*np.eye(x.shape[1]) if actual["boundary"] else actual["Sigma"]
    assert boundary == actual["boundary"]
    np.testing.assert_allclose(recovered, expected, atol=1e-10, rtol=1e-7)
    # A spherical boundary has nothing to shrink toward the same target.
    boundary_weights = mean/(3*np.trace(sample)/x.shape[1])
    sphere, is_boundary = am.mp_covariance(x, boundary_weights, 3, True)
    assert is_boundary
    np.testing.assert_allclose(am.shrink_model_matrix(sphere, .8), sphere, atol=1e-12)


def test_mp_interior_covariance_uses_factor_direction_not_sample_covariance():
    # Same interior sample as the existing MLE validation, no leave-one-out fit.
    rng = np.random.default_rng(3)
    N = 8
    beta = np.linspace(.5, 1.5, N)
    mean = .0067*beta+rng.normal(0, .002, N)
    x = rng.multivariate_normal(mean, np.outer(beta,beta)*.046**2+np.eye(N)*.035**2, 150)
    weights, full = mp.mp_weights_full(x.mean(0), np.cov(x.T, bias=True), 3)
    assert not full["boundary"]
    actual, _ = am.mp_covariance(x, weights, 3, False)
    np.testing.assert_allclose(actual, full["Sigma"], rtol=1e-7, atol=1e-11)


@pytest.fixture
def small_protocol(monkeypatch):
    monkeypatch.setattr(am, "SPEC", dict(am.SPEC, history_start="2000-01", test_start="2003-01",
        test_end="2003-02", windows=[24], grid_windows=[24], grid_dimensions=[2,3], subset_seeds=[9],
        bootstrap_replicates=20, periods=[["early", "2003-01", "2003-01"], ["late", "2003-02", "2003-02"]]))


def synthetic_panel():
    rng = np.random.default_rng(4)
    x = rng.normal(.004, .025, (38, 3))
    return pd.DataFrame(x, columns=["asset_a", "asset_b", "asset_c"]).assign(
        rf=.001, month=pd.period_range("2000-01", periods=38, freq="M").astype(str))


def test_new_scopes_fit_every_model_without_current_outcome(small_protocol):
    panel = synthetic_panel()
    case = dict(case_id="test", dataset="test", N=3, window=24,
                subset_seed=-1, assets=["asset_a", "asset_b", "asset_c"])
    args = Namespace(device="cpu")
    before, fitted, _, _ = am.fit_case(args, "markets", case, panel, None)
    panel.loc[36:, case["assets"]] = 100
    after, _, _, _ = am.fit_case(args, "markets", case, panel, None)
    assert set(before) == set(am.ALL_RULES)
    for model in before:
        np.testing.assert_array_equal(before[model][0], after[model][0])
    assert fitted.fit_end.lt(fitted.month).all()


def test_every_family_has_own_control_and_ew_comparison():
    comparisons = am.comparisons()
    assert len(comparisons) == 27
    for model in am.REGULARIZED:
        assert ("regularization_effect", "lw_"+model, model) in comparisons
    for rule in am.ALL_RULES:
        if rule != "ew":
            assert ("versus_ew", rule, "ew") in comparisons


def test_failed_original_cannot_be_counted_as_complete_improvement(small_protocol):
    diagnostics = pd.DataFrame(dict(month=["2003-01", "2003-02"],
        fit_end=["2002-12", "2003-01"], **{f"delta_{r}": [.5,.5] for r in ("cml","cpj","cmp","ckz","cmp50")}))
    targets = {r: np.array([[.5,.5],[.5,.5]]) for r in am.ALL_RULES}
    targets["ml"] = np.array([[2.,0.],[2.,0.]])
    case = dict(case_id="test", dataset="x", N=2, window=24, subset_seed=-1)
    monthly, scores, summary = am.evaluate_case("markets", case, targets, diagnostics,
        np.array([[-.6,0],[.1,0]]), np.zeros(2))
    pair = summary.loc[summary.comparison.eq("regularization_effect")&summary.control.eq("ml")&summary.period.eq("full")]
    assert not pair.complete.any()
    assert pair.delta_net_cer_pp.isna().all()
    original = scores.loc[scores.rule.eq("ml")&scores.period.eq("full")]
    assert original.cer_net_pp.isna().all()
    assert monthly.loc[monthly.rule.eq("ml")&monthly.month.eq("2003-02"), "status"].eq("after_stop").all()


def test_existing_outputs_preflight_before_any_device_or_input_work(tmp_path, monkeypatch):
    saved = tmp_path/"markets"
    saved.mkdir()
    (saved/"manifest.json").write_text("saved")
    monkeypatch.setattr(am.backend, "check_device", lambda d: pytest.fail("Preflight first"))
    with pytest.raises(FileExistsError):
        am.run(Namespace(experiment="all", out=tmp_path, device="cpu"))
    assert not (tmp_path/"regularization").exists()


@pytest.mark.parametrize("experiment", ("dimension_window", "markets"))
def test_new_scope_end_to_end(tmp_path, monkeypatch, small_protocol, experiment):
    panel = synthetic_panel().set_index("month")
    factors = panel[["rf"]].rename(columns={"rf":"RF"})
    portfolios = panel[["asset_a","asset_b","asset_c"]].add(panel.rf,axis=0)
    portfolios.columns = ["a","b","c"]
    monkeypatch.setattr(am.scope, "load_source", lambda key, raw, download:
        (factors if key=="ff3_fiz" else portfolios, {"sha256":"synthetic", "source_key":key}))
    args = Namespace(experiment=experiment, out=tmp_path/"out", raw=tmp_path/"raw", download=False, device="cpu")
    am.run(args)
    out = args.out/experiment
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["output_sha256"].items():
        assert checksum(out/name) == expected
    monthly = pd.read_parquet(out/"monthly.parquet")
    assert set(monthly.rule) == set(am.ALL_RULES)
    summary = pd.read_csv(out/"summary.csv")
    assert set(summary.variant) == set(am.ALL_RULES)-{"ew"}
    assert set(summary.comparison) == {"regularization_effect", "versus_ew"}
    scores = pd.read_csv(out/"scores.csv")
    assert set(scores.rule) == set(am.ALL_RULES)
    if experiment == "dimension_window":
        assert (out/"all_model_heatmaps.png").is_file()


@pytest.mark.parametrize("use_reference", [True, False])
def test_cached_historical_originals_are_used_without_refitting(tmp_path, monkeypatch, fitted, use_reference):
    x, original = fitted
    # Two synthetic original OOS decisions from a 36-month historical window.
    panel = pd.DataFrame(np.vstack([x, x[:2]]), columns=["asset_a","asset_b","asset_c"]).assign(
        month=pd.period_range("2000-01", periods=38,freq="M").astype(str), rf=.001)
    case = dict(case_id="historical", dataset="synthetic", N=3, window=36,
                subset_seed=-1, assets=["asset_a","asset_b","asset_c"])
    originals = {r: [] for r in am.ORIGINALS}
    deltas = {r: [] for r in ("cml","cpj","cmp","ckz","cmp50")}
    reference = []
    for i in (36,37):
        fit = fit_rules(panel[case["assets"]].iloc[i-36:i].to_numpy(),3,
            dict(rules=am.ORIGINALS[:-1],cmp50=True,cmp_mode="jk"))
        result, _ = am.regularize_fit(panel[case["assets"]].iloc[i-36:i].to_numpy(),fit,3)
        for r in originals:
            originals[r].append(fit["weights"][r])
        for r in deltas:
            deltas[r].append(.5 if r=="cmp50" else fit["delta"][r])
        reference.append(result["lw_ckz"])
    cached = ({r: np.stack(v) for r,v in originals.items()},
              {r: np.array(v) for r,v in deltas.items()},np.stack(reference) if use_reference else None)
    monkeypatch.setattr(am, "load_original_case", lambda *a: cached)
    monkeypatch.setattr(am, "decision_weights", lambda *a: pytest.fail("Must not refit historical article"))
    weights, diagnostics, _, _ = am.fit_case(Namespace(device="cpu"),"regularization",case,panel,{})
    np.testing.assert_allclose(weights["lw_ckz"], reference)
    np.testing.assert_array_equal(weights["cmp"], cached[0]["cmp"])


def test_cpu_cuda_full_family_parity(fitted, monkeypatch):
    monkeypatch.setenv("CUPY_CACHE_IN_MEMORY","1")
    cp = pytest.importorskip("cupy")
    try:
        count=cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    if not count:
        pytest.skip("No GPU")
    x, fit = fitted
    cpu, _ = am.regularize_fit(x,fit,3,"cpu")
    gpu, _ = am.regularize_fit(x,fit,3,"cuda")
    for r in am.ALL_RULES:
        np.testing.assert_allclose(cpu[r],gpu[r],rtol=1e-8,atol=1e-9)


def test_historical_loader_does_not_require_an_intermediate_run(tmp_path, fitted):
    x, fit = fitted
    panel = pd.DataFrame(np.vstack([x, x[:2]]), columns=["asset_a", "asset_b", "asset_c"])
    panel["month"] = pd.period_range("2000-01", periods=38, freq="M").astype(str)
    panel["rf"] = .001
    weights, returns = [], []
    for i in (36, 37):
        for rule in am.ORIGINALS:
            w = fit["weights"][rule]
            for asset, value in zip(("a", "b", "c"), w, strict=True):
                weights.append(dict(dataset="fixture", window=36, rule=rule,
                    fit_end=panel.month[i-1], asset=asset, weight=value))
            returns.append(dict(dataset="fixture", window=36, rule=rule,
                realized_month=panel.month[i], ret_excess=float(x[i-36] @ w), delta=.5))
    pd.DataFrame(weights).to_parquet(tmp_path / "weights.parquet", index=False)
    pd.DataFrame(returns).to_parquet(tmp_path / "monthly_returns.parquet", index=False)
    case = dict(window=36, dataset="fixture", assets=["asset_a", "asset_b", "asset_c"])
    original, deltas, reference = am.load_original_case(
        Namespace(baseline=tmp_path, retained=None), {}, case, panel)
    assert reference is None
    assert set(original) == set(am.ORIGINALS)
    np.testing.assert_array_equal(original["ckz"][0], fit["weights"]["ckz"])
    np.testing.assert_array_equal(deltas["cmp50"], [.5, .5])

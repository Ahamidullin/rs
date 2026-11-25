import numpy as np
import pandas as pd
import pytest

from replication.extensions import (
    aggregate_pairs, circular_indices, fit_targets, paired_summary, replay, stability_delta,
    turnover_delta,
)
from replication.metrics import cer


def test_turnover_grid_includes_base_and_breaks_ties_towards_it():
    e = np.array([0.5, 0.5])
    direction = np.array([0.6, 0.0])
    assert turnover_delta(e, direction, e, 0.137, 1.0, 0) == 0.137
    assert turnover_delta(e, direction, e, 0.137, 0.0, 0.0025) == 0
    assert turnover_delta(e, direction, e + 0.137 * direction, 0.137, 1.0, 0.0025,
                          grid=np.array([0.13, 0.25])) == 0.137
    assert turnover_delta(e, np.zeros(2), e, 0.137, 0.0, 0.0025) == 0.137
    assert turnover_delta(e, direction, e, 0.125, 0.0, 0.0025,
                          grid=np.array([0.0, 0.25])) == 0
    chosen = turnover_delta(e, direction, e, 0.137, 1.0, 0.0025)
    loss = lambda delta: 0.5 * (delta - 0.137)**2 + 0.0025 * delta * 0.6
    assert loss(chosen) <= loss(0.137)


def test_stability_at_agreement_and_with_more_disagreement():
    cov = np.diag([0.01, 0.02])
    d = np.array([1.0, 0.0])
    base = 0.54
    gap = np.array([0.0, 0.2])
    assert stability_delta(base, d, cov, np.zeros(2))[0] == base
    low = stability_delta(base, d, cov, gap)[0]
    high = stability_delta(base, d, cov, 2 * gap)[0]
    assert 0 < high < low < base
    assert stability_delta(base, np.zeros(2), cov, gap)[0] == 0


def test_cost_cash_and_drift_by_hand():
    targets = np.array([[1.1, 0.5], [1.0, 0.0]])
    X = np.array([[0.02, -0.01], [0.01, 0.0]])
    trace = replay(targets, X, np.array([0.003, 0.004]), 25)
    first, second = trace.iloc[0], trace.iloc[1]
    assert first.turnover == pytest.approx(0.60)
    assert first.cash == pytest.approx(-0.60)
    assert first.net_excess == pytest.approx(0.017 - 0.0015)
    assert first.gross_nav_factor == pytest.approx(1.02)
    assert first.net_nav_factor == pytest.approx(1.0185)
    expected = targets[0] * [1.023, 0.993] / 1.02
    np.testing.assert_allclose(first.drift_risky, expected)
    np.testing.assert_allclose(second.pretrade_risky, expected)
    assert first.drift_cash == pytest.approx(-0.6 * 1.003 / 1.02)


def test_failure_retains_dates_and_reports_only_the_valid_common_prefix():
    X = np.array([[0.01, 0.02], [0.0, 0.03], [-1.0, 0.01], [0.0, 0.0]])
    w = np.tile([1.1, 0.0], (len(X), 1))
    a = replay(w, X, np.zeros(len(X)), 25)
    b = replay(np.tile([0.5, 0.5], (len(X), 1)), X, np.zeros(len(X)), 25)
    for path in (a, b):
        path["month"] = ["2000-01", "2000-02", "2000-03", "2000-04"]
    assert a.status.to_list() == ["active", "active", "nonpositive_gross_nav", "after_stop"]
    assert np.isnan(a.net_excess.iloc[-1])
    summary = paired_summary(a, b, 3, circular_indices(2, 20, 12, 10))
    assert summary["n"] == 2
    assert summary["variant_stop_month"] == "2000-03"
    assert np.isnan(summary["delta_net_cer_pp"])
    assert np.isfinite(summary["prefix_delta_net_cer_pp"])
    b.loc[0, "month"] = "1999-12"
    with pytest.raises(ValueError, match="dates"):
        paired_summary(a, b, 3, circular_indices(2, 20, 12, 10))


def test_net_failure_when_gross_nav_is_positive():
    trace = replay(np.array([[1.1, 0.5]]), np.array([[-0.999 / 1.1, 0.0]]),
                   np.zeros(1), 25)
    assert trace.status[0] == "nonpositive_net_nav"


def test_bootstrap_draws_circular_blocks_and_reuses_paired_months():
    indices = circular_indices(25, 30, 12, 15)
    assert indices.shape == (30, 25)
    np.testing.assert_array_equal(indices, circular_indices(25, 30, 12, 15))
    np.testing.assert_array_equal((indices[:, 1:12] - indices[:, :11]) % 25, 1)
    trace = replay(np.full((25, 2), 0.5), np.random.default_rng(16).normal(0, .01, (25, 2)),
                   np.zeros(25), 0).assign(month=pd.period_range("2000-01", periods=25, freq="M").astype(str))
    summary = paired_summary(trace, trace, 3, indices)
    assert summary["ci_lo_pp"] == summary["ci_hi_pp"] == 0
    other = trace.assign(net_excess=trace.net_excess * 1.5 - 0.0003)
    summary = paired_summary(trace, other, 3, indices)
    left, right = trace.net_excess.to_numpy(), other.net_excess.to_numpy()
    differences = [cer(left[index], 3) - cer(right[index], 3) for index in indices]
    np.testing.assert_allclose([summary["ci_lo_pp"], summary["ci_hi_pp"]],
                               np.quantile(differences, [0.025, 0.975]))


def test_aggregate_weighs_datasets_equally_with_one_incomplete_window():
    summary = pd.DataFrame({
        "dataset": ["a", "a", "b", "b"], "window": [120, 240, 120, 240],
        "complete": [True, True, True, False], "delta_net_cer_pp": [0, 2, 5, np.nan],
    }).assign(cost_bps=25, control="ckz", delta_turnover=0, turnover=0,
              control_turnover=0, delta_gross_cer_pp=0)
    row = aggregate_pairs(summary, "turnover").query("window == 'all'").iloc[0]
    assert row.mean_delta_cer_pp == 3
    assert row.median_delta_cer_pp == 2
    assert (row.wins, row.n_pairs, row.n_pairs_expected) == (2, 3, 4)


def test_fit_and_cost_replay_do_not_see_current_or_future_return():
    rng = np.random.default_rng(42)
    X = rng.normal([0.007, 0.013], 0.04, (55, 2))
    panel = pd.DataFrame(X, columns=["asset_A", "asset_B"]).assign(
        month=pd.period_range("2000-01", periods=len(X), freq="M").astype(str), rf=0.001,
    )
    changed = panel.copy()
    changed.loc[44:, "asset_A"] += 0.02
    for experiment in ("turnover", "stability"):
        monthly, targets, diag = fit_targets(panel, 24, 3, experiment, "fixture")
        other, other_targets, other_diag = fit_targets(changed, 24, 3, experiment, "fixture")
        assert monthly.realized_month.nunique() == 31
        end = panel.month.iat[43]
        for rule in targets:
            np.testing.assert_allclose(targets[rule][:21], other_targets[rule][:21], atol=1e-14)
        pd.testing.assert_frame_equal(diag.loc[:end], other_diag.loc[:end])
        if experiment == "turnover":
            kwargs = dict(base_delta=diag.delta_base.to_numpy(),
                          directions=targets["kz"] - targets["ew"],
                          curvature=diag.curvature.to_numpy())
            path = replay(targets["ckz"], X[24:], panel.rf.to_numpy()[24:], 25, **kwargs)
            kwargs = dict(base_delta=other_diag.delta_base.to_numpy(),
                          directions=other_targets["kz"] - other_targets["ew"],
                          curvature=other_diag.curvature.to_numpy())
            other_path = replay(other_targets["ckz"], changed[["asset_A", "asset_B"]].to_numpy()[24:],
                                panel.rf.to_numpy()[24:], 25, **kwargs)
            np.testing.assert_allclose(path.delta.iloc[:21], other_path.delta.iloc[:21], atol=1e-14)
            assert path.gross_excess.iloc[20] != other_path.gross_excess.iloc[20]
            zero = replay(targets["ckz"], X[24:], panel.rf.to_numpy()[24:], 0,
                          base_delta=diag.delta_base.to_numpy(), directions=targets["kz"] - targets["ew"],
                          curvature=diag.curvature.to_numpy())
            np.testing.assert_array_equal(np.stack(zero.target_risky), targets["ckz"])
        else:
            assert (diag.delta_new <= diag.delta_base).all()
            assert diag.group.iloc[:12].isna().all()
            pos = 12
            expected = "high" if diag.s.iloc[pos] > diag.s.iloc[:pos].median() else "low"
            assert diag.group.iloc[pos] == expected
            assert diag.half_size.eq(12).all()

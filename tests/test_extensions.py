import numpy as np
import pandas as pd
import pytest
from replication.extensions import circular_indices, paired_summary, replay
from replication.metrics import cer


def test_cost_cash_and_drift_by_hand():
    targets = np.array([[1.1, 0.5], [1.0, 0.0]])
    X = np.array([[0.02, -0.01], [0.01, 0.0]])
    trace = replay(targets, X, np.array([0.003, 0.004]), 25)
    first, second = (trace.iloc[0], trace.iloc[1])
    assert first.turnover == pytest.approx(0.6)
    assert first.cash == pytest.approx(-0.6)
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
    assert a.status.to_list() == [
        "active",
        "active",
        "nonpositive_gross_nav",
        "after_stop",
    ]
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
    trace = replay(
        np.array([[1.1, 0.5]]), np.array([[-0.999 / 1.1, 0.0]]), np.zeros(1), 25
    )
    assert trace.status[0] == "nonpositive_net_nav"


def test_bootstrap_draws_circular_blocks_and_reuses_paired_months():
    indices = circular_indices(25, 30, 12, 15)
    assert indices.shape == (30, 25)
    np.testing.assert_array_equal(indices, circular_indices(25, 30, 12, 15))
    np.testing.assert_array_equal((indices[:, 1:12] - indices[:, :11]) % 25, 1)
    trace = replay(
        np.full((25, 2), 0.5),
        np.random.default_rng(16).normal(0, 0.01, (25, 2)),
        np.zeros(25),
        0,
    ).assign(month=pd.period_range("2000-01", periods=25, freq="M").astype(str))
    summary = paired_summary(trace, trace, 3, indices)
    assert summary["ci_lo_pp"] == summary["ci_hi_pp"] == 0
    other = trace.assign(net_excess=trace.net_excess * 1.5 - 0.0003)
    summary = paired_summary(trace, other, 3, indices)
    left, right = (trace.net_excess.to_numpy(), other.net_excess.to_numpy())
    differences = [cer(left[index], 3) - cer(right[index], 3) for index in indices]
    np.testing.assert_allclose(
        [summary["ci_lo_pp"], summary["ci_hi_pp"]],
        np.quantile(differences, [0.025, 0.975]),
    )

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from replication.backtest import ml_in_sample, rolling_backtest
from replication.data import parse_dgu, parse_french, select_panel
from replication.metrics import cer


FIXTURE = Path(__file__).parent / "fixtures" / "monthly_tiny.csv"


def test_manual_ew_and_cer():
    panel = pd.read_csv(FIXTURE)
    returns, weights = rolling_backtest(panel, 7, 3, options={"rules": ["ew"], "cmp50": False})
    ew = returns.query("rule == 'ew'").iloc[0]
    assert (ew.fit_end, ew.realized_month) == ("2000-07", "2000-08")
    assert ew.rf == 0.004
    assert ew.ret_excess == pytest.approx(0.01)
    assert weights.query("rule == 'ew'").set_index("asset").weight.to_dict() == {
        "A": 0.5, "B": 0.5, "CASH": 0.0,
    }
    assert cer([0.01, -0.01], 3) == pytest.approx(-0.36)


def test_rolling_policy_does_not_receive_current_or_future_data():
    rng = np.random.default_rng(12)
    n, M = 27, 12
    panel = pd.DataFrame({
        "month": pd.period_range("2000-01", periods=n, freq="M").astype(str),
        "rf": 0.003,
        "asset_A": rng.normal(0.01, 0.03, n),
        "asset_B": rng.normal(0.01, 0.04, n),
    })
    opts = {"rules": ["ew", "ml", "ckz"], "cmp50": False}
    seen = []

    def policy(history, base_fit, state):
        seen.append((history.month.iat[-1], state.get("past", 0)))
        state["past"] = state.get("past", 0) + 1
        return base_fit

    r, w = rolling_backtest(panel, M, 3, policy, options=opts)
    assert seen == [(month, i) for i, month in enumerate(panel.month.iloc[M - 1:-1])]
    modified = panel.copy()
    modified.loc[M + 3:, "asset_A"] += 0.1
    altered_r, altered_w = rolling_backtest(modified, M, 3, policy, options=opts)
    end = panel.month.iat[M + 2]
    pd.testing.assert_frame_equal(
        w.query("fit_end <= @end").reset_index(drop=True),
        altered_w.query("fit_end <= @end").reset_index(drop=True),
    )
    month = panel.month.iat[M + 3]
    r0 = r.query("realized_month == @month and rule == 'ew'").ret_excess.item()
    r1 = altered_r.query("realized_month == @month and rule == 'ew'").ret_excess.item()
    assert r1 - r0 == pytest.approx(0.05)


def test_ml_in_sample_uses_entire_panel_and_ml_covariance():
    X = pd.read_csv(FIXTURE).iloc[:, 2:].to_numpy()
    w = np.linalg.solve(np.cov(X, rowvar=False, ddof=0), X.mean(axis=0)) / 3
    np.testing.assert_allclose(ml_in_sample(X, 3), X @ w)


def test_reject_nonconsecutive_panel():
    panel = pd.read_csv(FIXTURE).drop(index=3)
    with pytest.raises(ValueError, match="months must be unique, ordered and consecutive"):
        rolling_backtest(panel, 6, 3)


def test_dgu_dates_filtered_before_checking_data():
    sample = "%Date\tRF\tA\tMOM\n19630628\t.002\t.01\n19630728\t.003\t.02\t-.01\n"
    spec = {"member": "test", "start": "1963-07", "end": "1963-07"}
    frame, header = parse_dgu(sample, spec)
    assert header == ["%Date", "RF", "A", "MOM"]
    panel = select_panel(frame, spec["start"], spec["end"], 2, "test")
    assert panel.iloc[0].tolist() == ["1963-07", 0.003, 0.02, -0.01]


def test_french_reads_monthly_block_and_handles_missing_markers():
    sample = (
        "Average Value Weighted Returns -- Monthly\n,A,B\n196307,1.0,-99.99\n"
        "196308,-999,-2.5\n\nAverage Equal Weighted Returns -- Monthly\n,A,B\n196307,8.0,9.0\n"
    )
    frame, header = parse_french(sample, "value")
    assert header == ["A", "B"]
    assert frame.loc["1963-07", "A"] == 0.01
    assert frame.loc["1963-08", "B"] == -0.025
    assert np.isnan(frame.loc["1963-07", "B"])
    assert np.isnan(frame.loc["1963-08", "A"])

"""Pinned source parsing and fixed-subset checks on synthetic inputs."""

from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from replication import study_data as rs
from replication.data import checksum


def test_parse_unlabelled_global_monthly_returns_excludes_annual_block():
    text = ("USD total returns\n\n,Small Growth,Small Neutral,Small Value,Big Growth,Big Neutral,Big Value\n"
            "199201,1,2,3,4,5,6\n199202,-99.99,2,3,4,5,6\n\nAnnual returns\n1992,100,100,100,100,100,100\n")
    frame, header = rs.parse_portfolios(text, 6)
    assert frame.index.tolist() == ["1992-01", "1992-02"]
    assert len(header) == 6
    assert frame.iloc[0, 0] == .01
    assert np.isnan(frame.iloc[1, 0])


def test_parse_labelled_value_block_not_equal_block():
    text = ("Average Value Weighted Returns -- Monthly\n,A,B\n199201,1,2\n199202,3,4\n\n"
            "Average Equal Weighted Returns -- Monthly\n,A,B\n199201,99,99\n")
    frame, _ = rs.parse_portfolios(text, 2)
    assert frame.iloc[0].tolist() == [.01, .02]


def test_parser_rejects_wrong_width_and_equal_only_table():
    with pytest.raises(ValueError):
        rs.parse_portfolios(",A,B\n199201,1,2\n199202,2,3\n", 6)
    with pytest.raises(ValueError):
        rs.parse_portfolios("Average Equal Weighted Returns -- Monthly\n,A,B\n199201,1,2\n", 2)


def test_source_snapshot_is_pinned_and_changed_archive_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "SOURCES", {"fixture": ("ftp", "example.zip", 2)})
    folder = tmp_path/"ftp"
    folder.mkdir()
    archive = folder/"example.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr("returns.csv", ",A,B\n199201,1,2\n199202,3,4\n")
    before, info = rs.load_source("fixture", tmp_path)
    again, _ = rs.load_source("fixture", tmp_path)
    pd.testing.assert_frame_equal(before, again)
    assert info["sha256"] == checksum(archive)
    original_bytes = archive.read_bytes()
    real_reader = rs.read_member

    def restore_missing(path, url, **kwargs):
        assert kwargs["expected_hash"] == info["sha256"]
        if not path.exists():
            assert kwargs["download"]
            path.write_bytes(original_bytes)
        return real_reader(path, url, **kwargs)

    archive.unlink()
    monkeypatch.setattr(rs, "read_member", restore_missing)
    restored, _ = rs.load_source("fixture", tmp_path, download=True)
    pd.testing.assert_frame_equal(before, restored)
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Pinned source changed"):
        rs.load_source("fixture", tmp_path)


def test_panel_subtracts_same_rf_once_and_never_fills_missing_months():
    portfolios = pd.DataFrame({"A": [.02, .03], "B": [.01, .04]}, index=["2000-01", "2000-02"])
    factors = pd.DataFrame({"RF": [.001, .002]}, index=portfolios.index)
    panel = rs.make_panel(portfolios, factors, "2000-01", "2000-02")
    np.testing.assert_allclose(panel.asset_A, [.019, .028])
    np.testing.assert_allclose(panel.rf, [.001, .002])
    with pytest.raises(ValueError, match="Missing"):
        rs.make_panel(portfolios.iloc[:1], factors, "2000-01", "2000-02")
    portfolios.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Incomplete"):
        rs.make_panel(portfolios, factors, "2000-01", "2000-02")


def test_nested_subsets_are_deterministic_full_universe_not_duplicated():
    names = [f"asset_{i}" for i in range(49)]
    before = rs.subset_plan(names)
    assert before == rs.subset_plan(names)
    assert len(before) == 11
    assert sum(n == 49 for n, _, _ in before) == 1
    for seed in rs.SPEC["subset_seeds"]:
        ten = next(set(a) for n, s, a in before if n == 10 and s == seed)
        twenty_five = next(set(a) for n, s, a in before if n == 25 and s == seed)
        assert ten < twenty_five < set(names)

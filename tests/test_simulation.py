import json
import shutil

import numpy as np
import pytest

from replication.aggregate import load
from replication.simulate import digest, make_spec, run_chunk, scenario


def spec_for_test():
    return make_spec(
        ["A0"],
        [40],
        2,
        2,
        root_seed=1,
        sigma_seed=3267,
        loading_seed=1971,
        ckz_mode="eq13_scaled",
        cpj_moments="adj",
    )


def test_resume_rejects_changed_formula_or_seed(tmp_path):
    spec = spec_for_test()
    args = ("A0", 40, 0, 2, str(tmp_path), spec)
    first, elapsed = run_chunk(args)
    assert elapsed > 0
    assert run_chunk(args)[1] == 0
    changed = json.loads(json.dumps(spec))
    changed["options"]["root_seed"] = 2
    with pytest.raises(ValueError, match="specification"):
        run_chunk(("A0", 40, 0, 2, str(tmp_path), changed))


def test_aggregation_requires_exact_coverage(tmp_path):
    spec = spec_for_test()
    first, _ = run_chunk(("A0", 40, 0, 2, str(tmp_path), spec))
    (tmp_path / "sim_manifest.json").write_text(
        json.dumps({"spec": spec, "spec_sha256": digest(spec), "status": "complete"})
    )
    frame = load(tmp_path)
    assert frame.groupby(["gamma", "rule"]).rep.nunique().eq(2).all()
    shutil.copy(first, tmp_path / "chunks" / "unexpected.npz")
    with pytest.raises(ValueError, match="chunk set"):
        load(tmp_path)


def test_three_factor_population_includes_investable_factors():
    calibration = {
        "mean": [0.006, 0.002, 0.003],
        "covariance": np.diag([0.002, 0.001, 0.001]).tolist(),
    }
    mu, cov, gammas = scenario("F3a", calibration)
    assert mu.shape == (25,) and cov.shape == (25, 25) and gammas == (3.0,)
    np.testing.assert_allclose(mu[-3:], calibration["mean"])
    np.testing.assert_allclose(cov[-3:, -3:], calibration["covariance"])
    np.linalg.cholesky(cov)

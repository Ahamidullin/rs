import numpy as np, pytest
from scipy.optimize import minimize
from replication.methods import fit_rules, kz_adjust, suff, combine
from replication.mp import mp_weights_full, mp_loo, _negll
from replication.dgp import one_factor, true_U, ew_U
from replication.metrics import cer, performance

def _sample(T=150, N=8, seed=0):
    rng = np.random.default_rng(seed)
    B = np.linspace(.5, 1.5, N); mu = 0.0067 * B + rng.normal(0, .002, N)
    S = np.outer(B, B) * .16 ** 2 / 12 + np.diag(rng.uniform(.1, .3, N) ** 2 / 12)
    return rng.multivariate_normal(mu, S, T)

def test_ew_and_ml():
    X = _sample(); f = fit_rules(X, 3, dict(rules=("ew", "ml")))
    assert np.allclose(f["weights"]["ew"], 1 / 8)
    T = len(X); S1 = np.cov(X.T, ddof=1)
    assert np.allclose(f["weights"]["ml"], np.linalg.solve(S1, X.mean(0)) / 3)

def test_true_gamma_ratio_and_anchors():
    sig = np.random.default_rng(3267).uniform(.1, .3, 24)
    mu, S = one_factor(sig)
    assert np.isclose(true_U(mu, S, 1) / true_U(mu, S, 3), 3)
    assert round(true_U(mu, S, 3), 2) == 4.17 and round(ew_U(mu, S, 3), 2) == 3.89
    assert round(ew_U(mu, S, 1), 2) == 6.63
    mua, Sa = one_factor(sig, alpha=True)
    assert round(true_U(mua, Sa, 3), 2) == 6.50 and round(ew_U(mua, Sa, 3), 2) == 3.89

def test_mp_is_mle():
    X = _sample(seed=3); N = X.shape[1]; m = X.mean(0); S0 = np.cov(X.T, bias=True)
    w, f = mp_weights_full(m, S0, 3)
    x0 = np.r_[f["mu"], np.log(f["s2"]), np.log(f["a"])]
    nll = lambda p: _negll(p[:N], np.exp(p[N]) * np.eye(N) + np.exp(p[N + 1]) * np.outer(p[:N], p[:N]), m, S0)
    o = minimize(nll, x0, method="Nelder-Mead", options=dict(xatol=1e-12, fatol=1e-15, maxiter=50000, maxfev=50000))
    assert nll(x0) <= o.fun + 1e-9
    # identity c = s2 + a ||mu||^2
    assert np.isclose(f["c"], f["s2"] + f["a"] * f["mu"] @ f["mu"])

def test_mp_loo_exact():
    X = _sample(T=80, N=10, seed=4); m = X.mean(0); S0 = np.cov(X.T, bias=True)
    w, f = mp_weights_full(m, S0, 3)
    W, mi, S0i, d = mp_loo(X, f, 3)
    for i in range(len(X)):
        Xi = np.delete(X, i, 0)
        wi, _ = mp_weights_full(Xi.mean(0), np.cov(Xi.T, bias=True), 3)
        assert np.abs(wi - W[i]).max() < 1e-10

def test_kz_adjust_limits():
    T, N = 120, 25
    for s in [2.0, 3.0, 5.0]:   # large s: correction vanishes
        assert np.isclose(kz_adjust(s, N, T), ((T - N - 2) * s - N) / T, atol=1e-6)
    assert kz_adjust(1e-4, N, T) > 0

def test_combine_cash():
    e = np.ones(2) / 2; v = np.array([1.2, -0.1])
    w = combine(e, v, 0.25)
    assert np.allclose(w, [0.675, 0.35]) and np.isclose(1 - w.sum(), -0.025)

def test_cer_two_points():
    assert np.isclose(cer([0.01, -0.01], 3), -0.36)

def test_deltas_in_unit_interval():
    sig = np.random.default_rng(3267).uniform(.1, .3, 24); mu, S = one_factor(sig)
    X = np.random.default_rng(1).multivariate_normal(mu, S, 120)
    f = fit_rules(X, 3)
    for k, v in f["delta"].items():
        assert 0 <= v <= 1

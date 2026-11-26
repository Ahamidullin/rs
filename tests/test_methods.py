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


def test_ckz_eq13_uses_scaled_sample_covariance():
    X, gamma = _sample(seed=7), 3.0
    M, N = X.shape
    e = np.ones(N) / N
    m, S0 = X.mean(axis=0), np.cov(X, rowvar=False, ddof=0)
    inverse_mean, inverse_one = np.linalg.solve(S0, np.column_stack((m, np.ones(N)))).T
    g = (e @ inverse_mean) / (e @ inverse_one)
    a = m - g
    theta = kz_adjust(m @ inverse_mean, N, M)
    psi = kz_adjust(a @ np.linalg.solve(S0, a), N - 1, M)
    eta = psi / (psi + N / M)
    q = eta * m + (1 - eta) * g
    c1 = (M - 2) * (M - N - 2) / ((M - N - 1) * (M - N - 4))
    scaled_cov = S0 * M / (M - N - 2)
    scaled_q = np.linalg.solve(scaled_cov, q)
    cross = m @ scaled_q
    pi13 = theta / gamma**2 - e @ m / gamma + (e @ q - cross / gamma) / (gamma * c1)
    pi1 = e @ S0 @ e - 2 * e @ m / gamma + theta / gamma**2
    pi3 = theta / gamma**2 - (theta - N / M * eta) / (gamma**2 * c1)
    denominator = pi1 - 2 * pi13 + pi3
    raw = (pi1 - pi13) / denominator
    delta = np.clip(raw, 0, 1) if denominator > 0 else float(pi1 > pi3)
    kz = scaled_q / (gamma * c1)

    fit = fit_rules(X, gamma, {"rules": ("kz", "ckz"), "ckz_mode": "eq13_scaled"})
    diagnostic = fit_rules(X, gamma, {"rules": ("kz", "ckz"), "ckz_mode": "unscaled_plugin"})
    previous = fit_rules(X, gamma, {"rules": ("kz", "ckz")})

    assert fit["diagnostics"]["ckz_pi13"] == pytest.approx(pi13, abs=1e-14)
    assert fit["delta_raw"]["ckz"] == pytest.approx(raw, abs=1e-12)
    assert fit["delta"]["ckz"] == pytest.approx(delta, abs=1e-12)
    np.testing.assert_allclose(fit["weights"]["ckz"], (1 - delta) * e + delta * kz)
    np.testing.assert_allclose(fit["weights"]["kz"], kz)
    np.testing.assert_array_equal(fit["weights"]["kz"], previous["weights"]["kz"])
    np.testing.assert_array_equal(fit["weights"]["kz"], diagnostic["weights"]["kz"])
    assert abs(diagnostic["diagnostics"]["ckz_pi13"] - pi13) > 1e-5
    assert abs(previous["diagnostics"]["ckz_pi13"] - pi13) > 1e-5

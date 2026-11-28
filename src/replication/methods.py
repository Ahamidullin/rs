"""Tu-Zhou (2011) portfolio rules. X: (T,N) monthly excess returns (fractions).
All risky weights are NAV fractions; cash = 1 - sum(w)."""

import numpy as np
from hashlib import blake2b
from . import mp as _mp
from . import backend
from scipy.special import betainc, betaln, hyp2f1


CKZ_MODES = ("legacy_adjusted", "unscaled_plugin", "eq13_scaled")


def suff(X, device="cpu"):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite (T,N) array")
    T, N = X.shape
    m, S0 = backend.moments(X, device)
    return T, N, m, S0


def c1_(T, N):
    return (T - 2.0) * (T - N - 2.0) / ((T - N - 1.0) * (T - N - 4.0))


def kz_adjust(s, d, T, variant="scipy"):
    """Kan-Zhou (2007) adjusted estimator of a squared Sharpe-type quantity.
    s: sample quadratic form (S0 normalisation), d: dimension (N for theta^2, N-1 for psi^2).
    adjusted = ((T-d-2)s - d)/T + 2 s^(d/2) (1+s)^(-(T-2)/2) / (T * B_x(d/2,(T-d)/2)),
    x = s/(1+s), B_x the (unnormalised) incomplete beta function."""
    if not np.isfinite(s) or s < -1e-12 or d < 0 or T <= d + 2:
        raise ValueError("Require finite s >= 0, d >= 0 and T > d+2")
    if d == 0 or s <= 0:
        return 0.0
    a = d / 2.0
    b = (T - d) / 2.0
    x = s / (1.0 + s)
    base = ((T - d - 2.0) * s - d) / T
    Ix = betainc(a, b, x)
    if s < 1e-6 or Ix == 0:
        # B_x(a,b)=x**a/a * 2F1(a,1-b;a+1;x). Avoid log(0)
        # and cancellation of the two d/T terms near the origin.
        h = hyp2f1(a, 1.0 - b, a + 1.0, x)
        log_ratio = (a - (T - 2.0) / 2.0) * np.log1p(s) - np.log(h)
        return max(0.0, ((T - d - 2.0) * s + d * np.expm1(log_ratio)) / T)
    logB = np.log(Ix) + betaln(a, b)
    logterm = (
        np.log(2.0) + a * np.log(s) - (T - 2.0) / 2.0 * np.log1p(s) - np.log(T) - logB
    )
    return base + np.exp(logterm)


def combine(e, v, delta):
    return (1.0 - delta) * e + delta * v


def clip01(x):
    if not np.isfinite(x):
        raise ValueError("Nonfinite combination coefficient")
    return float(min(1.0, max(0.0, x)))


def _delta_from_ABC(A, C, B):
    if not np.isfinite([A, C, B]).all():
        raise ValueError("Nonfinite estimated combination moments")
    den = A - 2.0 * C + B
    raw = (A - C) / den if den != 0 else np.nan
    if den > 0:
        return raw, clip01(raw), ""
    # non-convex: choose endpoint with smaller loss: loss(0)=A, loss(1)=B
    return raw, (0.0 if A <= B else 1.0), "nonpos_denominator"


def fit_rules(X, gamma, options=None):
    """Returns dict(weights={rule: w}, delta_raw={}, delta={}, diagnostics={})."""
    opt = dict(
        rules=("ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz"),
        cmp50=False,
        ckz_moments="adj",
        cpj_moments="adj",
        cmp_mode="jk",
        ckz_mode="eq13_scaled",
        device="cpu",
    )
    if options:
        opt.update(options)
    rules = set(opt["rules"])
    device = opt["device"]
    backend.check_device(device)
    T, N, m, S0 = suff(X, device)
    X = np.asarray(X, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    unknown = rules - {"ew", "ml", "pj", "mp", "kz", "cml", "cpj", "cmp", "ckz"}
    if unknown:
        raise ValueError(f"Unknown rules: {sorted(unknown)}")
    if rules == {"ew"} and not opt["cmp50"]:
        return dict(
            weights={"ew": np.full(N, 1.0 / N)}, delta_raw={}, delta={}, diagnostics={}
        )
    if T <= N + 4:
        raise ValueError("need T > N+4")
    one = np.ones(N)
    e = one / N
    S1 = S0 * T / (T - 1.0)
    c1 = c1_(T, N)
    np.linalg.cholesky(S0)
    sol0 = lambda b: backend.solve(S0, b, device)
    S0inv_m = sol0(m)
    S0inv_1 = sol0(one)
    theta2_hat = m @ S0inv_m
    theta2_t = kz_adjust(theta2_hat, N, T)
    W, DR, D, diag = {}, {}, {}, {}
    W["ew"] = e.copy()
    if "ml" in rules:
        W["ml"] = backend.solve(S1, m, device) / gamma
    # ---- CML  (4)-(11)
    pi1 = e @ S0 @ e - 2.0 * (e @ m) / gamma + theta2_t / gamma**2
    if "cml" in rules:
        v = (T - N - 2.0) / T * S0inv_m / gamma
        pi2 = ((c1 - 1.0) * theta2_t + c1 * N / T) / gamma**2
        raw, delta, flag = _delta_from_ABC(pi1, 0.0, pi2)
        DR["cml"] = raw
        D["cml"] = delta
        diag["cml_flag"] = flag
        W["cml"] = combine(e, v, D["cml"])
        diag["scaled_ml"] = v
    # ---- KZ  (12)-(14)
    if rules & {"kz", "ckz", "cpj"}:
        g = (one @ S0inv_m) / (one @ S0inv_1)
        dvec = m - g * one
        psi2_hat = dvec @ sol0(dvec)
        psi2_t = kz_adjust(psi2_hat, N - 1, T)
        eta = psi2_t / (psi2_t + N / T)
        q = eta * m + (1.0 - eta) * g * one
        S0inv_q = sol0(q)
        vkz = (T - N - 2.0) / (T * c1 * gamma) * S0inv_q
        W["kz"] = vkz
        diag.update(kz_eta=eta, kz_g=g, psi2_t=psi2_t, theta2_t=theta2_t)
        if "ckz" in rules:
            if opt["ckz_moments"] == "adj":
                th, ps = theta2_t, psi2_t
            else:  # unbiased quadratic forms (sensitivity)
                th = ((T - N - 2.0) * theta2_hat - N) / T
                ps = ((T - N - 1.0) * psi2_hat - (N - 1.0)) / T
            pi1c = e @ S0 @ e - 2.0 * (e @ m) / gamma + th / gamma**2
            mode = opt["ckz_mode"]
            if mode == "legacy_adjusted":
                cross = th - (1.0 - eta) * ps
            elif mode == "unscaled_plugin":
                cross = m @ S0inv_q
            elif mode == "eq13_scaled":
                # Eq. (13) uses the inverse of the scaled covariance in Eq. (4).
                cross = (T - N - 2.0) / T * (m @ S0inv_q)
            else:
                raise ValueError(
                    f"Unknown CKZ mode: {mode!r}; expected one of {CKZ_MODES}"
                )
            pi13 = (
                th / gamma**2
                - (e @ m) / gamma
                + (e @ q) / (gamma * c1)
                - cross / (gamma**2 * c1)
            )
            pi3 = th / gamma**2 - (th - N * eta / T) / (gamma**2 * c1)  # KZ (2007) form
            raw, dl, fl = _delta_from_ABC(pi1c, pi13, pi3)
            DR["ckz"] = raw
            D["ckz"] = dl
            W["ckz"] = combine(e, vkz, dl)
            diag.update(
                ckz_pi1=pi1c,
                ckz_pi13=pi13,
                ckz_pi3=pi3,
                ckz_flag=fl,
                ckz_mode=mode,
                ckz_cross=cross,
            )
    # ---- Jorion PJ (15)-(16), App. B
    if rules & {"pj", "cpj"}:
        St = T / (T - N - 2.0) * S0
        Sti_1 = backend.solve(St, one, device)
        Sti_m = backend.solve(St, m, device)
        A = one @ Sti_1
        gpj = (one @ Sti_m) / A
        dv = m - gpj * one
        spread = max(0.0, float(dv @ backend.solve(St, dv, device)))
        # Use inverse lambda to retain the exact equal-means limit.
        inv_lam = spread / (N + 2.0)
        lam = 1.0 / inv_lam if inv_lam else np.inf
        rho = 1.0 / (T * inv_lam + 1.0)
        mpj = (1 - rho) * m + rho * gpj * one
        dd = 1.0 + inv_lam / (T * inv_lam + 1.0)
        Dv = one * np.sqrt(1.0 / (T * ((T + 1.0) * inv_lam + 1.0) * A))
        Spj = dd * St + np.outer(Dv, Dv)
        vpj = backend.solve(Spj, mpj, device) / gamma
        W["pj"] = vpj
        diag.update(pj_lambda=lam, pj_rho=rho)
        if "cpj" in rules:
            dl, raw, fl = cpj_delta(
                m,
                S0,
                St,
                lam,
                dd,
                rho,
                A,
                gpj,
                e,
                gamma,
                T,
                N,
                c1,
                theta2_hat,
                opt["cpj_moments"],
                device=device,
            )
            DR["cpj"] = raw
            D["cpj"] = dl
            diag["cpj_flag"] = fl
            W["cpj"] = combine(e, vpj, dl)
    # ---- MP and CMP (17)-(21), App. C
    if rules & {"mp", "cmp"} or opt["cmp50"]:
        wmp, fmp = _mp.mp_weights_full(m, S0, gamma, device=device)
        W["mp"] = wmp
        diag.update(mp_t=fmp["t"], mp_boundary=fmp["boundary"])
        if opt["cmp50"]:
            W["cmp50"] = combine(e, wmp, 0.5)
        if "cmp" in rules:
            raw, dl, fl, nref = cmp_delta(
                X,
                m,
                S0,
                fmp,
                wmp,
                e,
                gamma,
                T,
                N,
                pi1,
                opt["cmp_mode"],
                opt.get("cache"),
                device=device,
            )
            DR["cmp"] = raw
            D["cmp"] = dl
            diag.update(cmp_flag=fl, cmp_n_refit=nref)
            W["cmp"] = combine(e, wmp, dl)
    W = {k: v for k, v in W.items() if k in rules or k == "cmp50" and opt["cmp50"]}
    return dict(weights=W, delta_raw=DR, delta=D, diagnostics=diag)


def _delta_wfree(eSe, eSEv, em, mEv, EvSv, gamma):
    """delta = (e-w*)'Sigma(e-Ev) / E[(e-v)'Sigma(e-v)];  w* enters only via mu'Ev/gamma,
    so no estimate of w* itself is required."""
    num = eSe - eSEv - em / gamma + mEv / gamma
    den = eSe - 2.0 * eSEv + EvSv
    if not np.isfinite([num, den]).all():
        raise ValueError("Nonfinite estimated combination moments")
    raw = num / den if den != 0 else np.nan
    if den > 0:
        return raw, clip01(raw), ""
    # L(1)-L(0) = den - 2*num. This also covers a zero denominator.
    return raw, (1.0 if den - 2.0 * num < 0 else 0.0), "nonpos_denominator"


def cmp_delta(
    X, m, S0, fmp, wmp, e, gamma, T, N, pi1, mode="jk", cache=None, device="cpu"
):
    """(18)-(20): E[v] and E[v'Sigma v] by jackknife bias correction
    T*f(full) - (T-1)*mean(f(loo)); Sigma -> S0 (full sample, fixed), mu -> m."""
    identity = (
        device
        + ":"
        + blake2b(
            np.ascontiguousarray(X, dtype=float).tobytes(), digest_size=16
        ).hexdigest()
    )
    if cache is not None and cache.get("mp_loo_identity") == identity:
        Wl = cache["mp_loo_g1"] / gamma
        d = cache["mp_loo_d"]
    else:
        Wl, mi, S0i, d = _mp.mp_loo(X, fmp, gamma, device=device)
        if cache is not None:
            cache["mp_loo_g1"] = Wl * gamma
            cache["mp_loo_d"] = d
            cache["mp_loo_identity"] = identity
    Ev = T * wmp - (T - 1.0) * Wl.mean(0)
    q_full = wmp @ S0 @ wmp
    q_loo = np.einsum("ki,ij,kj->k", Wl, S0, Wl)
    Eq = T * q_full - (T - 1.0) * q_loo.mean()
    if mode == "nojk":
        Ev, Eq = wmp, q_full
    raw, dl, fl = _delta_wfree(e @ S0 @ e, e @ S0 @ Ev, e @ m, m @ Ev, Eq, gamma)
    return raw, dl, fl, d["n_refit"]


def cpj_delta(
    m,
    S0,
    St,
    lam,
    dd,
    rho,
    A,
    g,
    e,
    gamma,
    T,
    N,
    c1,
    theta2_hat,
    moments="adj",
    device="cpu",
):
    """App. B logic: S_PJ^{-1} = St^{-1}/d - B0, B0 = b0 St^{-1}11'St^{-1} frozen;
    E[St^{-1}] = Sigma^{-1}, E[St^{-1} Sigma St^{-1}] = c1 Sigma^{-1}; rho, GMV h frozen.
    Every quadratic form in mu reduces to theta^2, psi^2, A:
      (P mu)'Sigma^{-1}(P mu) = (theta^2-psi^2) + (1-rho)^2 psi^2,  1'Sigma^{-1}P mu = A mu_g,
      A mu_g^2 = theta^2 - psi^2,  mu'Sigma^{-1}P mu = theta^2 - rho psi^2.
    theta^2, psi^2 -> KZ adjusted (moments='adj') or unbiased ('unb'); linear terms in mu -> m."""
    one = np.ones(N)
    Si = backend.solve(St, np.eye(N), device)
    Si1 = Si @ one
    th_h = theta2_hat
    dv = m - g * one
    ps_h = dv @ backend.solve(S0, dv, device)
    if moments == "adj":
        th, ps = kz_adjust(th_h, N, T), kz_adjust(ps_h, N - 1, T)
    else:
        th = ((T - N - 2.0) * th_h - N) / T
        ps = ((T - N - 1.0) * ps_h - (N - 1.0)) / T
    ps = max(ps, 0.0)
    tg = max(th - ps, 0.0)  # = A mu_g^2
    inv_lam = 0.0 if np.isinf(lam) else 1.0 / lam
    s02 = 1.0 / (T * ((T + 1.0) * inv_lam + 1.0) * A)
    b0 = s02 / (dd**2 + dd * s02 * A)
    Pm = (1 - rho) * m + rho * g * one
    P = (1 - rho) * np.eye(N) + rho * np.outer(one, Si1 / A)
    S1 = S0 * T / (T - 1.0)
    eSe = e @ S1 @ e
    eSEv = ((e @ Pm) / dd - b0 * A * g) / gamma
    mEv = ((th - rho * ps) / dd - b0 * A * tg) / gamma
    Jc = b0**2 * A - 2.0 * b0 / dd
    J = c1 * Si / dd**2 + Jc * np.outer(Si1, Si1)
    EvSv = (
        np.trace(J @ P @ St @ P.T) / T
        + c1 * (tg + (1 - rho) ** 2 * ps) / dd**2
        + Jc * A * tg
    ) / gamma**2
    raw, dl, fl = _delta_wfree(eSe, eSEv, e @ m, mEv, EvSv, gamma)
    return dl, raw, fl

"""MacKinlay-Pastor (2000) MLE and exact leave-one-out refits.

Model: x_t ~ N(mu, Sigma), Sigma = s2*I + a*mu mu', a >= 0.
Reduction (verified against brute-force likelihood): with U = S0 + m m',
unit direction u, s = u'Uu, p = u'm, c = s - p^2 (variance along u),
s2 = (tr U - s)/(N-1), the profile objective is
    F(u) = (N-1) ln(tr U - s) + ln(s - p^2)
Stationary points satisfy (m m' + t S0) u = nu u with t = 1 - s2/c in [0,1)
(t = 0 <=> a = 0 boundary).  mu_MP = p u, w_MP = p u / (gamma c).
"""

import numpy as np
from . import backend


def _F(u, U, m, trU, N):
    s = np.einsum("...i,...ij,...j->...", u, U, u)
    p = np.einsum("...i,...i->...", u, m)
    return (N - 1) * np.log(trU - s) + np.log(s - p * p)


def _boundary(m, S0, N):
    # a = 0: Sigma = s2 I, mu = m, s2 = tr(S0)/N
    s2 = np.trace(S0, axis1=-2, axis2=-1) / N
    return s2


def fit_mp_full(m, S0, grid=200, device="cpu"):
    """Global MLE for one sample. Returns dict with u, t, p, c, s2, boundary flag."""
    N = m.shape[0]
    U = S0 + np.outer(m, m)
    trU = np.trace(U)
    ts = np.linspace(0.0, 1.0, grid, endpoint=False)
    Ms = np.outer(m, m)[None] + ts[:, None, None] * S0[None]
    lam, Q = backend.eigh(Ms, device)
    us = Q[:, :, -1]
    Fs = _F(us, U[None], m[None], trU, N)
    k = int(np.nanargmin(Fs))
    # refine with joint Newton
    sol = newton_mp(m[None], U[None], S0[None], us[k][None], np.array([ts[k]]))
    u, t, ok = sol["u"][0], sol["t"][0], sol["ok"][0]
    # compare with boundary a=0 by full log-likelihood
    res = _pack(u, m, S0, U, N)
    ll_int = _negll(res["mu"], res["Sigma"], m, S0) if ok and t >= 0 else np.inf
    s2b = np.trace(S0) / N
    ll_b = _negll(m, s2b * np.eye(N), m, S0)
    if ll_b < ll_int:
        norm = np.linalg.norm(m)
        direction = m / norm if norm else np.eye(N)[0]
        return dict(
            u=direction, t=0.0, mu=m.copy(), c=s2b, s2=s2b, boundary=True, ok=True
        )
    res.update(t=t, boundary=False, ok=bool(ok))
    return res


def _negll(mu, Sigma, m, S0):
    sgn, ld = np.linalg.slogdet(Sigma)
    d = m - mu
    return ld + np.trace(np.linalg.solve(Sigma, S0 + np.outer(d, d)))


def _pack(u, m, S0, U, N):
    p = u @ m
    s = u @ U @ u
    c = s - p * p
    s2 = (np.trace(U) - s) / (N - 1)
    a = (c - s2) / (p * p)
    mu = p * u
    Sigma = s2 * np.eye(N) + a * np.outer(mu, mu)
    return dict(u=u, mu=mu, c=c, s2=s2, a=a, Sigma=Sigma)


def newton_mp(m, U, S0, u0, t0, iters=30, tol=1e-13, xp=np):
    """Batched Newton on (u, nu, t): (mm'+tS0)u = nu u, u'u = 1, t = 1 - s2/c.
    Shapes: m (K,N), U,S0 (K,N,N), u0 (K,N), t0 (K,)."""
    K, N = m.shape
    u = u0.copy()
    t = t0.astype(float).copy()
    if u.shape[0] != K:
        u = xp.broadcast_to(u, (K, N)).copy()
    if t.shape[0] != K:
        t = xp.broadcast_to(t, (K,)).copy()
    mm = m[:, :, None] * m[:, None, :]
    trU = xp.trace(U, axis1=1, axis2=2)
    # align sign / start nu
    M = mm + t[:, None, None] * S0
    nu = xp.einsum("ki,kij,kj->k", u, M, u)
    I = xp.eye(N)[None]
    ok = xp.zeros(K, bool)
    res = xp.full(K, xp.inf)
    for _ in range(iters):
        M = mm + t[:, None, None] * S0
        S0u = xp.einsum("kij,kj->ki", S0, u)
        Uu = xp.einsum("kij,kj->ki", U, u)
        c = xp.einsum("ki,ki->k", u, S0u)
        s = xp.einsum("ki,ki->k", u, Uu)
        s2 = (trU - s) / (N - 1)
        r1 = xp.einsum("kij,kj->ki", M, u) - nu[:, None] * u
        r2 = 0.5 * (xp.einsum("ki,ki->k", u, u) - 1.0)
        r3 = t - 1.0 + s2 / c
        res = xp.sqrt((r1**2).sum(1) + r2**2 + r3**2)
        ok = res < tol
        if ok.all():
            break
        J = xp.zeros((K, N + 2, N + 2))
        J[:, :N, :N] = M - nu[:, None, None] * I
        J[:, :N, N] = -u
        J[:, :N, N + 1] = S0u
        J[:, N, :N] = u
        J[:, N + 1, :N] = (-2.0 * Uu / (N - 1)) / c[:, None] - (s2 / c**2)[
            :, None
        ] * 2.0 * S0u
        J[:, N + 1, N + 1] = 1.0
        r = xp.concatenate([r1, r2[:, None], r3[:, None]], 1)
        step = xp.linalg.solve(J, -r[:, :, None])[:, :, 0]
        upd = ~ok
        u[upd] += step[upd, :N]
        nu[upd] += step[upd, N]
        t[upd] += step[upd, N + 1]
    return dict(u=u, t=t, nu=nu, ok=ok, res=res)


def mp_weights_from(u, m, S0, gamma):
    """w = p u/(gamma c), batched."""
    p = np.einsum("...i,...i->...", u, m)
    c = np.einsum("...i,...ij,...j->...", u, S0, u)
    return (p / (gamma * c))[..., None] * u


def mp_loo(X, full, gamma, fallback=True, device="cpu"):
    """Exact leave-one-out MP weights for all T deletions (rows of result).
    Warm-started Newton from the full-sample optimum; non-converged or
    boundary cases are refitted globally (counted in diagnostics)."""
    T, N = X.shape
    m = X.mean(0)
    U = X.T @ X / T
    mi = (T * m[None] - X) / (T - 1)
    Ui = (T * U[None] - X[:, :, None] * X[:, None, :]) / (T - 1)
    S0i = Ui - mi[:, :, None] * mi[:, None, :]
    if device == "cuda" and not full["boundary"]:
        W, idx_refit = _cuda_loo(mi, Ui, S0i, full, gamma)
    elif full["boundary"]:
        idx_refit = np.arange(T)
        W = np.empty((T, N))
    else:
        try:
            sol = newton_mp(
                mi,
                Ui,
                S0i,
                np.broadcast_to(full["u"], (T, N)).copy(),
                np.full(T, full["t"]),
            )
            W = mp_weights_from(sol["u"], mi, S0i, gamma)
            idx_refit = np.where(~sol["ok"] | (sol["t"] < 0))[0]
        except np.linalg.LinAlgError:
            # A singular warm-start Jacobian does not invalidate the sample.
            # Refit every deletion through the full MLE entry point.
            W = np.empty((T, N))
            idx_refit = np.arange(T)
    for i in idx_refit:
        f = fit_mp_full(mi[i], S0i[i], device=device)
        W[i] = (
            f["mu"] / (gamma * f["c"])
            if not f["boundary"]
            else f["mu"] / (gamma * f["s2"])
        )
    return W, mi, S0i, dict(n_refit=len(idx_refit))


def mp_weights_full(m, S0, gamma, device="cpu"):
    f = fit_mp_full(m, S0, device=device)
    w = f["mu"] / (gamma * (f["s2"] if f["boundary"] else f["c"]))
    return w, f


def _cuda_loo(means, second, covariances, full, gamma, batch_size=256):
    """Bounded-VRAM exact Newton batches; failed deletions use the full MLE."""
    cp = backend.cuda()
    import cupyx

    count, N = means.shape
    weights = np.empty((count, N))
    refit = []
    for first in range(0, count, batch_size):
        last = min(count, first + batch_size)
        m = cp.asarray(means[first:last], dtype=cp.float64)
        U = cp.asarray(second[first:last], dtype=cp.float64)
        S = cp.asarray(covariances[first:last], dtype=cp.float64)
        initial_u = cp.broadcast_to(cp.asarray(full["u"]), (last - first, N)).copy()
        initial_t = cp.full(last - first, full["t"], dtype=cp.float64)
        try:
            with cupyx.errstate(linalg="raise"):
                fitted = newton_mp(m, U, S, initial_u, initial_t, xp=cp)
            u = fitted["u"]
            p = cp.einsum("ki,ki->k", u, m)
            c = cp.einsum("ki,kij,kj->k", u, S, u)
            w = p[:, None] * u / (gamma * c[:, None])
            valid = fitted["ok"] & (fitted["t"] >= 0) & cp.isfinite(w).all(axis=1)
            weights[first:last] = cp.asnumpy(w)
            refit.extend((first + cp.asnumpy(cp.flatnonzero(~valid))).tolist())
        except np.linalg.LinAlgError:
            refit.extend(range(first, last))
    return weights, np.array(refit, dtype=int)

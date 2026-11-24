"""DGPs. Following DGU (2009) simulation design: N = (N-K) test assets + K investable factors.
A0/Aalpha: K=1, 24 assets with beta = linspace(0.5,1.5,24), idio annual vol ~ U[0.10,0.30].
F3alpha: K=3 (FF3 moments), 22 assets."""
import numpy as np
MU_F = 0.08 / 12; VAR_F = 0.16 ** 2 / 12
def one_factor(sig_annual, N=25, alpha=False):
    n = N - 1
    beta = np.linspace(0.5, 1.5, n)
    a = np.linspace(-0.02, 0.02, n) / 12 if alpha else np.zeros(n)
    B = np.r_[beta, 1.0][:, None]
    al = np.r_[a, 0.0]
    D = np.diag(np.r_[sig_annual ** 2 / 12, 0.0])
    mu = al + B[:, 0] * MU_F
    Sig = VAR_F * B @ B.T + D
    return mu, Sig
def three_factor(mu_f, cov_f, sig_annual, load, N=25, alpha=True):
    n = N - 3
    a = np.linspace(-0.02, 0.02, n) / 12 if alpha else np.zeros(n)
    B = np.vstack([load, np.eye(3)])
    al = np.r_[a, np.zeros(3)]
    D = np.diag(np.r_[sig_annual ** 2 / 12, np.zeros(3)])
    return al + B @ mu_f, B @ cov_f @ B.T + D
def true_U(mu, Sig, gamma):
    return mu @ np.linalg.solve(Sig, mu) / (2 * gamma) * 1200
def ew_U(mu, Sig, gamma):
    e = np.ones(len(mu)) / len(mu)
    return (e @ mu - gamma / 2 * e @ Sig @ e) * 1200

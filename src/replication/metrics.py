import numpy as np


def performance(w, mu, cov, gamma):
    """Monthly utility and monthly Sharpe ratio of risky weights w (cash earns RF, excess 0)."""
    mean = w @ mu
    var = w @ cov @ w
    return mean - gamma / 2.0 * var, (mean / np.sqrt(var) if var > 0 else np.nan)


def cer(returns, gamma, ddof=1):
    r = np.asarray(returns, float)
    return 1200.0 * (r.mean() - gamma / 2.0 * r.var(ddof=ddof))

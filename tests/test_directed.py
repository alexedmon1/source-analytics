"""Known-signal tests for the MVAR / DTF directed-connectivity kernel."""

import numpy as np

from source_analytics.spectral.directed import (
    fit_mvar,
    mvar_spectral_radius,
    compute_dtf,
)


def _directed_chain(n=5, T=40000, leak=0.8, seed=0):
    """Linear directed chain 0->1->...->(n-1) plus a shared latent (collinearity)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T))
    latent = rng.standard_normal(T)
    for t in range(2, T):
        x[0, t] = 0.5 * x[0, t - 1] + rng.standard_normal()
        for i in range(1, n):
            x[i, t] = 0.5 * x[i, t - 1] + 0.5 * x[i - 1, t - 1] + rng.standard_normal()
    return x + leak * latent[None, :]


def test_dtf_recovers_direction():
    x = _directed_chain()
    n = x.shape[0]
    node_ts = {f"n{i}": x[i] for i in range(n)}
    results, names = compute_dtf(node_ts, sfreq=500.0, bands={"broad": (1, 60)})
    D = results["broad"]["dtf"]  # D[i, j] = source i -> target j
    forward = np.mean([D[i, i + 1] for i in range(n - 1)])
    backward = np.mean([D[i + 1, i] for i in range(n - 1)])
    assert forward > backward * 2, (forward, backward)
    assert names == [f"n{i}" for i in range(n)]


def test_dtf_in_unit_range_and_zero_diagonal():
    x = _directed_chain(seed=1)
    node_ts = {f"n{i}": x[i] for i in range(x.shape[0])}
    results, _ = compute_dtf(node_ts, sfreq=500.0, bands={"a": (1, 10), "b": (20, 40)})
    for band in results.values():
        D = band["dtf"]
        assert D.min() >= 0.0 and D.max() <= 1.0 + 1e-9
        assert np.allclose(np.diag(D), 0.0)


def test_ridge_keeps_collinear_mvar_stable():
    """On near-rank-deficient (collinear, leakage-like) data the ridge MVAR is
    stable and far better conditioned than the plain least-squares fit — the
    property DTF relies on for real source data (mean inter-node |corr| ≈ 0.64)."""
    rng = np.random.default_rng(2)
    k, n, T = 3, 10, 20000           # n channels driven by k<n latent AR sources
    lat = np.zeros((k, T))
    for t in range(1, T):
        lat[:, t] = 0.6 * lat[:, t - 1] + rng.standard_normal(k)
    x = rng.standard_normal((n, k)) @ lat + 0.02 * rng.standard_normal((n, T))

    A_ridge, _ = fit_mvar(x, order=8, ridge=0.05)
    assert mvar_spectral_radius(A_ridge) < 1.0       # ridge fit is stationary

    # Ridge must markedly reduce the conditioning of the AR normal equations.
    xc = x - x.mean(axis=1, keepdims=True)
    X = np.vstack([xc[:, 8 - kk:T - kk] for kk in range(1, 9)])
    G = X @ X.T
    cond_plain = np.linalg.cond(G)
    cond_ridge = np.linalg.cond(G + 0.05 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]))
    assert cond_ridge < cond_plain / 100

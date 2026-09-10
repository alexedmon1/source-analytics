"""Signature fits run with the BLAS/OpenMP pools limited to one thread.

A signature run fits a tiny model LOOCV x (1 + n_permutations) times. With the
default multithreaded BLAS the FORGE treatment logistic fits ran 60-90x slower
(thread start-up and synchronisation on 35 x 30 problems), which turned a module
that takes about an hour into one that takes days. The results are the same.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
threadpoolctl = pytest.importorskip("threadpoolctl")

from source_analytics.stats.signature import _single_threaded, run_signature  # noqa: E402


def test_thread_pools_are_limited_to_one_inside_a_fit():
    @_single_threaded
    def probe():
        return [pool["num_threads"] for pool in threadpoolctl.threadpool_info()]

    assert all(n == 1 for n in probe())


def test_run_signature_is_wrapped():
    assert hasattr(run_signature, "__wrapped__")


def test_single_threading_does_not_change_the_result():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 12))
    y = np.array([0] * 12 + [1] * 12)
    X[y == 1, :3] += 1.0
    kw = dict(classifier="logistic", cv_method="loocv", n_permutations=5, seed=42)
    limited = run_signature(X, y, **kw)
    unlimited = run_signature.__wrapped__(X, y, **kw)
    assert limited.accuracy == unlimited.accuracy
    assert limited.p_value == unlimited.p_value

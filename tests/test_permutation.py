"""Verification of the hypothesis-layer permutation adapter.

Checks (synthetic vertex maps):
  1. a pairwise contrast reproduces the base cluster_permutation_test bit-exact;
  2. a 3-group omnibus F finds the planted cluster;
  3. Freedman-Lane residualisation removes a nuisance-covariate-driven effect;
  4. a general weighted contrast runs and returns the cluster contract.

Run: uv run --no-sync pytest tests/test_permutation.py -q
"""

from __future__ import annotations

import numpy as np

from source_analytics.config import Hypothesis
from source_analytics.stats.cluster_permutation import cluster_permutation_test
from source_analytics.hypothesis.permutation import run_hypothesis_permutation

N_VERT = 40
SIGNAL = slice(10, 22)          # the planted cluster
COORDS = np.column_stack([np.arange(N_VERT), np.zeros(N_VERT), np.zeros(N_VERT)]).astype(float)
DIST = 1.5                      # connects 1-D neighbours
KW = dict(n_perms=200, threshold=2.0, distance_mm=DIST, seed=7)


def _make(groups: dict[str, tuple[int, float]], rng):
    """Build subject_maps + subject_groups. groups: name -> (n, signal_amplitude)."""
    maps, sgroups, sid = {}, {}, 0
    for gname, (n, amp) in groups.items():
        for _ in range(n):
            uid = f"s{sid:03d}"; sid += 1
            m = rng.normal(0, 1.0, N_VERT)
            m[SIGNAL] += amp
            maps[uid] = m
            sgroups[uid] = gname
    return maps, sgroups


def test_pairwise_matches_base_cluster_test():
    rng = np.random.default_rng(0)
    maps, sgroups = _make({"A": (14, 1.5), "B": (14, 0.0)}, rng)
    hyp = Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0})

    rows = run_hypothesis_permutation(hyp, maps, sgroups, COORDS, **KW)

    # Direct base call with the SAME assembly must match bit-exact.
    data_a = np.array([maps[u] for u, g in sgroups.items() if g == "A"])
    data_b = np.array([maps[u] for u, g in sgroups.items() if g == "B"])
    base = cluster_permutation_test(data_a, data_b, COORDS, n_perms=200,
                                    threshold=2.0, tail=0, distance_mm=DIST, seed=7)
    adapter_p = sorted(r["cluster_p"] for r in rows if r["cluster_id"] > 0)
    base_p = sorted(base.cluster_pvalues)
    assert len(adapter_p) == len(base_p) and base_p, "expected >=1 cluster both ways"
    assert np.allclose(adapter_p, base_p), f"{adapter_p} != {base_p}"
    assert any(r["significant"] for r in rows), "planted effect should survive"


def test_omnibus_finds_planted_cluster():
    rng = np.random.default_rng(1)
    # three groups, one clearly elevated over the planted region
    maps, sgroups = _make({"G1": (12, 0.0), "G2": (12, 0.0), "G3": (12, 1.6)}, rng)
    hyp = Hypothesis(name="omni", kind="omnibus", groups=["G1", "G2", "G3"])
    rows = run_hypothesis_permutation(hyp, maps, sgroups, COORDS, **KW)
    assert any(r["stat_type"] == "F" for r in rows)
    assert any(r["significant"] for r in rows), "omnibus should detect the group difference"


def test_freedman_lane_removes_nuisance():
    rng = np.random.default_rng(2)
    # No TRUE group effect, but a nuisance covariate drives the signal region AND
    # is imbalanced across groups -> a naive contrast sees a spurious effect.
    maps, sgroups, cov, sid = {}, {}, {}, 0
    for g, (n, cov_mean) in {"A": (16, 1.0), "B": (16, -1.0)}.items():
        for _ in range(n):
            uid = f"s{sid:03d}"; sid += 1
            x = rng.normal(cov_mean, 0.3)
            m = rng.normal(0, 1.0, N_VERT)
            m[SIGNAL] += 1.5 * x          # signal driven by the covariate, not group
            maps[uid] = m; sgroups[uid] = g; cov[uid] = np.array([x])

    hyp = Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0})
    naive = run_hypothesis_permutation(hyp, maps, sgroups, COORDS, **KW)
    adjusted = run_hypothesis_permutation(hyp, maps, sgroups, COORDS, covariates=cov, **KW)

    naive_sig = sum(r["significant"] for r in naive)
    adj_sig = sum(r["significant"] for r in adjusted)
    assert naive_sig >= 1, "naive contrast should see the spurious covariate effect"
    assert adj_sig < naive_sig, "Freedman-Lane should shrink the spurious effect"


def test_general_weighted_contrast_runs():
    rng = np.random.default_rng(3)
    maps, sgroups = _make({"A": (10, 1.4), "B": (10, 0.0), "C": (10, 0.0)}, rng)
    # avg(B,C) vs A
    hyp = Hypothesis(name="a_vs_bc", kind="contrast",
                     weights={"A": 1.0, "B": -0.5, "C": -0.5})
    rows = run_hypothesis_permutation(hyp, maps, sgroups, COORDS, **KW)
    assert rows and all(r["stat_type"] == "t" for r in rows)

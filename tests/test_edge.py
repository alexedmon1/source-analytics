"""Verification of the hypothesis-layer edge/NBS adapter.

Checks (synthetic connectivity matrices):
  1. a pairwise contrast reproduces the base nbs_permutation_test bit-exact;
  2. a 3-group omnibus F finds the planted subnetwork;
  3. Freedman-Lane residualisation removes a nuisance-covariate-driven subnetwork;
  4. a general weighted contrast runs and returns the subnetwork contract;
  5. equivalence returns a TOST summary row.

Run: uv run --no-sync pytest tests/test_edge.py -q
"""

from __future__ import annotations

import numpy as np

from source_analytics.config import Hypothesis
from source_analytics.stats.graph_metrics import nbs_permutation_test
from source_analytics.hypothesis.edge import run_hypothesis_edge

N_NODE = 14
BLOCK = slice(0, 5)             # the planted subnetwork (a clique of nodes 0..4)
KW = dict(nbs_threshold=2.0, n_perms=200, seed=7)


def _sym(rng) -> np.ndarray:
    """A symmetric, zero-diagonal baseline connectivity matrix."""
    m = rng.normal(0, 1.0, (N_NODE, N_NODE))
    m = np.triu(m, k=1)
    m = m + m.T
    return m


def _make(groups: dict[str, tuple[int, float]], rng):
    """Build subject_matrices + subject_groups. groups: name -> (n, block_amp)."""
    mats, sgroups, sid = {}, {}, 0
    for gname, (n, amp) in groups.items():
        for _ in range(n):
            uid = f"s{sid:03d}"; sid += 1
            m = _sym(rng)
            m[BLOCK, BLOCK] += amp        # elevate the clique edges
            np.fill_diagonal(m, 0.0)
            mats[uid] = m
            sgroups[uid] = gname
    return mats, sgroups


def test_pairwise_matches_base_nbs():
    rng = np.random.default_rng(0)
    mats, sgroups = _make({"A": (14, 2.0), "B": (14, 0.0)}, rng)
    hyp = Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0})

    rows = run_hypothesis_edge(hyp, mats, sgroups, **KW)

    mats_a = [mats[u] for u, g in sgroups.items() if g == "A"]
    mats_b = [mats[u] for u, g in sgroups.items() if g == "B"]
    base = nbs_permutation_test(mats_a, mats_b, nbs_threshold=2.0,
                                n_permutations=200, seed=7)
    adapter_p = sorted(r["component_p"] for r in rows if r["component_id"] > 0)
    base_p = sorted(base.component_pvalues)
    assert len(adapter_p) == len(base_p) and base_p, "expected >=1 component both ways"
    assert np.allclose(adapter_p, base_p), f"{adapter_p} != {base_p}"
    assert any(r["significant"] for r in rows), "planted subnetwork should survive"


def test_omnibus_finds_planted_subnetwork():
    rng = np.random.default_rng(1)
    mats, sgroups = _make({"G1": (12, 0.0), "G2": (12, 0.0), "G3": (12, 2.2)}, rng)
    hyp = Hypothesis(name="omni", kind="omnibus", groups=["G1", "G2", "G3"])
    rows = run_hypothesis_edge(hyp, mats, sgroups, **KW)
    assert any(r["stat_type"] == "F" for r in rows)
    assert any(r["significant"] for r in rows), "omnibus should detect the subnetwork"


def test_freedman_lane_removes_nuisance():
    rng = np.random.default_rng(2)
    mats, sgroups, cov, sid = {}, {}, {}, 0
    for g, (n, cov_mean) in {"A": (16, 1.0), "B": (16, -1.0)}.items():
        for _ in range(n):
            uid = f"s{sid:03d}"; sid += 1
            x = rng.normal(cov_mean, 0.3)
            m = _sym(rng)
            m[BLOCK, BLOCK] += 2.0 * x        # subnetwork driven by covariate, not group
            np.fill_diagonal(m, 0.0)
            mats[uid] = m; sgroups[uid] = g; cov[uid] = np.array([x])

    hyp = Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0})
    naive = run_hypothesis_edge(hyp, mats, sgroups, **KW)
    adjusted = run_hypothesis_edge(hyp, mats, sgroups, covariates=cov, **KW)

    naive_sig = sum(r["significant"] for r in naive)
    adj_sig = sum(r["significant"] for r in adjusted)
    assert naive_sig >= 1, "naive contrast should see the spurious covariate subnetwork"
    assert adj_sig < naive_sig, "Freedman-Lane should shrink the spurious subnetwork"


def test_general_weighted_contrast_runs():
    rng = np.random.default_rng(3)
    mats, sgroups = _make({"A": (10, 2.0), "B": (10, 0.0), "C": (10, 0.0)}, rng)
    hyp = Hypothesis(name="a_vs_bc", kind="contrast",
                     weights={"A": 1.0, "B": -0.5, "C": -0.5})
    rows = run_hypothesis_edge(hyp, mats, sgroups, **KW)
    assert rows and all(r["stat_type"] == "t" for r in rows)


def test_equivalence_returns_tost_summary():
    rng = np.random.default_rng(4)
    mats, sgroups = _make({"A": (16, 0.0), "B": (16, 0.0)}, rng)  # no real difference
    hyp = Hypothesis(name="a_equiv_b", kind="equivalence",
                     weights={"A": 1.0, "B": -1.0}, margin={"mode": "sd", "value": 1.0})
    rows = run_hypothesis_edge(hyp, mats, sgroups, **KW)
    assert len(rows) == 1 and rows[0]["stat_type"] == "tost"
    assert 0.0 <= rows[0]["frac_equivalent"] <= 1.0
    assert rows[0]["n_edges"] == N_NODE * (N_NODE - 1) // 2


def test_region_labels_name_subnetwork_nodes():
    # With vertex_rois provided, each subnetwork row gets a `region` naming the
    # regions of its nodes. The planted clique is nodes 0..4 -> label them Motor_R.
    rng = np.random.default_rng(0)
    mats, sgroups = _make({"A": (14, 2.0), "B": (14, 0.0)}, rng)
    hyp = Hypothesis(name="a_vs_b", kind="contrast", weights={"A": 1.0, "B": -1.0})
    vertex_rois = ["Motor_R" if i < 5 else "Visual_Parietal_L" for i in range(N_NODE)]

    rows = run_hypothesis_edge(hyp, mats, sgroups, vertex_rois=vertex_rois, **KW)

    assert all("region" in r for r in rows), "region column must be present when labeled"
    sig = [r for r in rows if r["significant"]]
    assert sig, "planted subnetwork should survive"
    # the planted clique is entirely Motor_R
    assert any("Motor_R" in (r["region"] or "") for r in sig)


def test_nbs_permutation_test_exposes_component_nodes():
    rng = np.random.default_rng(0)
    mats, sgroups = _make({"A": (14, 2.0), "B": (14, 0.0)}, rng)
    mats_a = [mats[u] for u, g in sgroups.items() if g == "A"]
    mats_b = [mats[u] for u, g in sgroups.items() if g == "B"]
    res = nbs_permutation_test(mats_a, mats_b, nbs_threshold=2.0, n_permutations=200, seed=7)
    # node membership is aligned 1:1 with component sizes/pvalues
    assert len(res.component_nodes) == len(res.component_sizes)
    assert all(isinstance(nodes, list) and len(nodes) >= 2 for nodes in res.component_nodes)


def test_nbs_significant_edges_mask_is_populated():
    """A significant component must be reflected in ``significant_edges``.

    Regression: the mask was allocated under a "Build significant edge mask"
    comment and then never filled, so ``nbs_permutation_test`` returned an
    all-False mask even for a large component at p = 0.0. Nothing in the shipped
    pipeline reads the field (``_network_base`` uses ``component_nodes`` /
    ``component_sizes``), so it was dormant — but it is a public field of the
    returned dataclass and any new consumer silently gets "no edges".

    The mask must equal the suprathreshold edges of the significant components:
    components partition the nodes, so a component's edges are the
    suprathreshold edges with both endpoints in its node set.
    """
    import numpy as np

    from source_analytics.stats.graph_metrics import nbs_permutation_test

    rng = np.random.default_rng(4)
    n, n_sub = 10, 16

    def net(shift):
        m = np.abs(rng.normal(0.5, 0.1, size=(n, n)))
        m = (m + m.T) / 2
        # a dense block of genuinely different edges between groups
        m[:5, :5] += shift
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0.0)
        return m

    A = [net(0.6) for _ in range(n_sub)]
    B = [net(0.0) for _ in range(n_sub)]
    res = nbs_permutation_test(A, B, nbs_threshold=2.0, n_permutations=500, seed=7)

    assert res.n_significant_components > 0, "fixture produced no component"
    mask = np.triu(res.significant_edges, 1)
    assert mask.sum() > 0, "significant component but an empty edge mask"

    # the mask must be a subset of the suprathreshold edges, and must exactly
    # account for the significant components' sizes
    supra = np.triu(np.abs(res.t_matrix) > 2.0, 1)
    assert not (mask & ~supra).any(), "mask contains sub-threshold edges"
    expected = sum(sz for sz, p in zip(res.component_sizes, res.component_pvalues)
                   if p < 0.05)
    assert int(mask.sum()) == expected, (
        f"mask has {int(mask.sum())} edges, significant components total {expected}"
    )


def test_subnetwork_edges_sidecar_matches_component_rows(tmp_path):
    """``write_module_hypotheses_edge(node_labels=...)`` writes
    ``<prefix>_subnetwork_edges.csv``: one row per supra-threshold edge of every
    component, labelled by ROI. Its per-component edge count must equal the
    ``n_edges`` of the matching row in ``<prefix>_hypotheses.csv``, and every
    edge of the significant component must fall inside the planted clique.
    This sidecar is the inferential source the gallery's circos draws from now
    that roi_connectivity's per-edge posthoc tables are retired.
    """
    from types import SimpleNamespace

    import pandas as pd

    from source_analytics.config import DesignSpec
    from source_analytics.hypothesis.edge import write_module_hypotheses_edge

    rng = np.random.default_rng(1)
    mats, sgroups = _make({"A": (14, 2.0), "B": (14, 0.0)}, rng)
    labels = [f"ROI{i:02d}" for i in range(N_NODE)]
    spec = DesignSpec(factor="group", reference="B", levels=["A", "B"],
                      hypotheses=[Hypothesis(name="a_vs_b", kind="contrast",
                                             weights={"A": 1.0, "B": -1.0})])
    config = SimpleNamespace(design_spec=spec)

    df = write_module_hypotheses_edge(
        {("Theta", "imag_coherence"): mats}, sgroups, config, tmp_path,
        prefix="roi_nbs", nbs_threshold=2.0, n_perms=100, seed=7, node_labels=labels)

    edges = pd.read_csv(tmp_path / "roi_nbs_subnetwork_edges.csv")
    assert list(edges.columns) == [
        "hypothesis", "band", "dv", "component_id", "component_p", "significant",
        "node_i", "node_j", "roi_i", "roi_j", "stat"]
    assert set(edges["hypothesis"]) == {"a_vs_b"}
    assert set(edges["band"]) == {"Theta"} and set(edges["dv"]) == {"imag_coherence"}

    comps = df[df["component_id"] > 0]
    per_comp = edges.groupby("component_id").size().to_dict()
    for _, row in comps.iterrows():
        assert per_comp.get(int(row["component_id"]), 0) == int(row["n_edges"])

    sig = edges[edges["significant"]]
    assert len(sig) > 0
    # Every planted clique edge is in the significant subnetwork (noise edges
    # above threshold may attach to it, so the converse is not required).
    sig_pairs = {frozenset((a, b)) for a, b in zip(sig["roi_i"], sig["roi_j"])}
    clique = [labels[i] for i in range(BLOCK.stop)]
    for i, a in enumerate(clique):
        for b in clique[i + 1:]:
            assert frozenset((a, b)) in sig_pairs
    assert (sig["stat"].abs() > 2.0).all()     # supra-threshold by construction
    assert (edges["node_i"] < edges["node_j"]).all()


def test_subnetwork_edges_sidecar_skipped_without_node_labels(tmp_path):
    """Vertex modules pass no node labels; no sidecar is written for them."""
    from types import SimpleNamespace

    from source_analytics.config import DesignSpec
    from source_analytics.hypothesis.edge import write_module_hypotheses_edge

    rng = np.random.default_rng(2)
    mats, sgroups = _make({"A": (6, 2.0), "B": (6, 0.0)}, rng)
    spec = DesignSpec(hypotheses=[Hypothesis(name="a_vs_b", kind="contrast",
                                             weights={"A": 1.0, "B": -1.0})])
    write_module_hypotheses_edge(
        {("Theta", "aec"): mats}, sgroups, SimpleNamespace(design_spec=spec), tmp_path,
        prefix="vertex_nbs", nbs_threshold=2.0, n_perms=20, seed=1)
    assert (tmp_path / "vertex_nbs_hypotheses.csv").exists()
    assert not (tmp_path / "vertex_nbs_subnetwork_edges.csv").exists()

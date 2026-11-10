"""The connectivity network layer split into graph + NBS analyses.

Verifies registration/metadata and that the split analyses inherit the
connectivity-metric + NBS config from the combined ``*_network`` block (so an
existing config drives them with no extra YAML).
"""

from source_analytics.config import StudyConfig
from source_analytics.core import ANALYSIS_REGISTRY, analysis_meta
from source_analytics.analyses.roi_network_analysis import (
    ROIGraphAnalysis,
    ROINBSAnalysis,
)


def test_split_analyses_registered():
    for name in ("roi_graph", "roi_nbs", "roi_network"):
        assert name in ANALYSIS_REGISTRY


def test_split_metadata_domain_and_supplements():
    meta = analysis_meta()
    assert meta["roi_graph"]["supplements"] == "roi_connectivity"
    assert meta["roi_nbs"]["supplements"] == "roi_connectivity"
    for n in ("roi_graph", "roi_nbs"):
        assert meta[n]["domain"] == "Connectivity"


def _config(tmp_path):
    text = """
name: "T"
groups: {WT_VEH: WT, KO_VEH: KO}
contrasts:
  - {name: disease_effect, group_a: KO_VEH, group_b: WT_VEH}
bands: {Theta: [4, 8], Alpha: [8, 13]}
paths: {results: ./r, analytics: ./a}
paradigms:
  resting:
    data_dir: ./d
    analyses:
      roi_network:
        connectivity_metrics: [imag_coherence, dwpli, pli]
        nbs_threshold: 2.5
"""
    p = tmp_path / "s.yaml"
    p.write_text(text)
    return StudyConfig.from_yaml(str(p))


def test_split_inherits_config_via_fallback(tmp_path):
    cfg = _config(tmp_path)

    g = ROIGraphAnalysis(cfg.for_paradigm_analysis("resting", "roi_graph"), tmp_path / "og")
    assert g._connectivity_metrics == ["imag_coherence", "dwpli", "pli"]
    assert g._nbs_threshold == 2.5

    n = ROINBSAnalysis(cfg.for_paradigm_analysis("resting", "roi_nbs"), tmp_path / "on")
    assert n._connectivity_metrics == ["imag_coherence", "dwpli", "pli"]
    assert n._nbs_results_filename == "roi_nbs_results.csv"


def test_nbs_figures_draw_the_components_and_each_significant_one(tmp_path):
    """roi_nbs draws its own figures (results plan 5b.4): a heatmap of the largest component
    per hypothesis and band, per measure, and a circos of each significant two-group one."""
    import json

    import numpy as np
    import pandas as pd

    p = tmp_path / "s.yaml"
    p.write_text("""
name: "T"
groups: {WT_VEH: WT, KO_VEH: KO}
design: {factor: group, reference: WT_VEH, levels: [WT_VEH, KO_VEH]}
hypotheses:
  - {name: disease_effect, kind: contrast, label: "Disease effect", weights: {KO_VEH: 1, WT_VEH: -1}}
  - {name: normal, kind: equivalence, weights: {KO_VEH: 1, WT_VEH: -1}, margin: {mode: sd, value: 0.5}}
roi_categories: {Front: [r1, r2], Back: [r3, r4]}
bands: {Theta: [4, 8], Alpha: [8, 13]}
paths: {results: ./r, analytics: ./a}
paradigms:
  resting:
    data_dir: ./d
    analyses:
      roi_network:
        connectivity_metrics: [coherence]
""")
    cfg = StudyConfig.from_yaml(str(p)).for_paradigm_analysis("resting", "roi_nbs")
    n = ROINBSAnalysis(cfg, tmp_path / "out")
    n.tbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rois = ["r1", "r2", "r3", "r4"]
    n._edges_df = pd.DataFrame([
        {"subject": f"{g}_{s}", "group": g, "band": b, "roi1": a, "roi2": c,
         "coherence": rng.uniform(0.2, 0.6)}
        for g in ("WT_VEH", "KO_VEH") for s in range(3) for b in ("Theta", "Alpha")
        for i, a in enumerate(rois) for c in rois[i + 1:]
    ])
    base = {"dv": "coherence", "measure": "coherence", "n_perms": 100, "threshold": 2.5}
    pd.DataFrame([
        {**base, "hypothesis": "disease_effect", "kind": "contrast", "band": "Theta",
         "component_id": 1, "n_edges": 3, "component_p": 0.01, "significant": True},
        {**base, "hypothesis": "disease_effect", "kind": "contrast", "band": "Alpha",
         "component_id": 1, "n_edges": 1, "component_p": 0.6, "significant": False},
        {**base, "hypothesis": "normal", "kind": "equivalence", "band": "Theta",
         "component_id": 0, "n_edges": 6, "component_p": None, "significant": None},
        {**base, "hypothesis": "group_omnibus", "kind": "omnibus", "band": "Alpha",
         "component_id": 1, "n_edges": 2, "component_p": 0.02, "significant": True},
    ]).to_csv(n.tbl_dir / "roi_nbs_hypotheses.csv", index=False)
    pd.DataFrame([
        {"hypothesis": "disease_effect", "band": "Theta", "dv": "coherence", "component_id": 1,
         "component_p": 0.01, "significant": True, "roi_i": a, "roi_j": b, "stat": 3.1}
        for a, b in (("r1", "r2"), ("r2", "r3"), ("r3", "r4"))
    ] + [
        {"hypothesis": "group_omnibus", "band": "Alpha", "dv": "coherence", "component_id": 1,
         "component_p": 0.02, "significant": True, "roi_i": a, "roi_j": b, "stat": 9.0}
        for a, b in (("r1", "r3"), ("r3", "r4"))
    ]).to_csv(n.tbl_dir / "roi_nbs_subnetwork_edges.csv", index=False)

    n._nbs_figures()

    drawn = sorted(f.name for f in n.fig_dir.glob("*.png"))
    assert drawn == ["circos__coherence__Alpha__group_omnibus.png",
                     "circos__coherence__Theta__disease_effect.png", "roi_nbs_components__coherence.png"]
    records = [json.loads(line) for line in (n.fig_dir / "figures.jsonl").read_text().splitlines()]
    by_file = {r["file"]: r for r in records}
    assert by_file["circos__coherence__Theta__disease_effect.png"] == {
        "file": "circos__coherence__Theta__disease_effect.png", "measure": "coherence",
        "contrast": "disease_effect", "facets": {"band": "Theta"}, "kind": "matrix",
        "shows": "roi_nbs_subnetwork_edges.csv"}
    assert by_file["roi_nbs_components__coherence.png"]["kind"] == "summary"
    omnibus = by_file["circos__coherence__Alpha__group_omnibus.png"]
    assert omnibus["contrast"] == "group_omnibus" and "descriptive, not a test" in omnibus["caption"]


def test_graph_and_directed_heatmaps_come_from_the_declared_tests(tmp_path):
    """roi_graph's per-contrast heatmaps and roi_directed's global heatmap read the modules'
    hypotheses tables; the stars are the tables' own `significant` (results plan 5b.4)."""
    import json

    import pandas as pd

    from source_analytics.analyses.roi_directed_analysis import ROIDirectedAnalysis

    cfg = _config(tmp_path)
    g = ROIGraphAnalysis(cfg.for_paradigm_analysis("resting", "roi_graph"), tmp_path / "og")
    g.tbl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"hypothesis": "disease_effect", "kind": "contrast", "graph_metric": "degree",
         "conn_metric": c, "spatial": roi, "band": b, "stat": 3.0 if roi == "A" else 0.5,
         "significant": roi == "A" and b == "Theta"}
        for c in ("imag_coherence", "dwpli") for roi in ("A", "B") for b in ("Theta", "Alpha")
    ]).to_csv(g.tbl_dir / "roi_graph_hypotheses.csv", index=False)
    g._graph_figures()
    assert [f.name for f in g.fig_dir.glob("*.png")] == ["roi_graph_hypotheses__degree__disease_effect.png"]
    rec = json.loads((g.fig_dir / "figures.jsonl").read_text().splitlines()[-1])
    assert (rec["measure"], rec["contrast"], rec["kind"]) == ("degree", "disease_effect", "summary")

    d = ROIDirectedAnalysis(cfg.for_paradigm_analysis("resting", "roi_directed"), tmp_path / "od")
    d.tbl_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"hypothesis": "disease_effect", "kind": "contrast", "dv": "te", "measure": "transfer_entropy",
         "band": b, "effect_size": 0.7, "significant": b == "Theta"} for b in ("Theta", "Alpha")
    ]).to_csv(d.tbl_dir / "roi_directed_global_hypotheses.csv", index=False)
    d._plot_global_hypotheses()
    assert (d.fig_dir / "roi_directed_global_hypotheses__te.png").exists()

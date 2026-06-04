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
from source_analytics.analyses.vertex_network_analysis import (
    VertexGraphAnalysis,
)


def test_split_analyses_registered():
    for name in ("roi_graph", "roi_nbs", "vertex_graph", "vertex_nbs",
                 "roi_network", "vertex_network"):
        assert name in ANALYSIS_REGISTRY


def test_split_metadata_domain_and_supplements():
    meta = analysis_meta()
    assert meta["roi_graph"]["supplements"] == "roi_connectivity"
    assert meta["roi_nbs"]["supplements"] == "roi_connectivity"
    assert meta["vertex_graph"]["supplements"] == "vertex_connectivity"
    assert meta["vertex_nbs"]["supplements"] == "vertex_connectivity"
    for n in ("roi_graph", "roi_nbs", "vertex_graph", "vertex_nbs"):
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
  vertex:
    data_dir: ./d
    analyses:
      vertex_network:
        connectivity_metrics: [imag_coherence, aec]
        nbs_threshold: 3.0
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

    vg = VertexGraphAnalysis(cfg.for_paradigm_analysis("vertex", "vertex_graph"), tmp_path / "ovg")
    assert vg._connectivity_metrics == ["imag_coherence", "aec"]
    assert vg._nbs_threshold == 3.0  # vertex default, from the vertex_network block

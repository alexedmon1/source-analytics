"""Guard the standard: figures() must be regenerable from PERSISTED data via
`--steps figures` alone — never dependent on in-memory state carried within one
process. These are structural guards (cheap, catch regressions) complementing the
end-to-end reload check in test_fcd_comparison.py.
"""

import inspect

from source_analytics.analyses.base import BaseAnalysis
from source_analytics.analyses import (
    vertex_connectivity_analysis as vc,
    vertex_directed_analysis as vd,
    vertex_cross_freq_analysis as vcf,
    fcd_comparison_analysis as fc,
)

# Map/cluster modules: figures() renders per-vertex glass brains from cluster
# results that statistics() must persist and figures() must reload.
MAP_MODULES = [
    (vc.VertexConnectivityAnalysis, "vertex_connectivity"),
    (vd.VertexDirectedAnalysis, "vertex_directed"),
    (vcf.VertexCrossFreqAnalysis, "vertex_cross_freq"),
]


def test_base_has_cluster_state_helpers():
    assert callable(getattr(BaseAnalysis, "_save_cluster_state", None))
    assert callable(getattr(BaseAnalysis, "_load_cluster_state", None))


def test_map_modules_regenerable_from_disk():
    for cls, name in MAP_MODULES:
        fig_src = inspect.getsource(cls.figures)
        stat_src = inspect.getsource(cls.statistics)
        # figures() reloads persisted cluster state when in-memory is empty
        assert "_load_cluster_state" in fig_src, \
            f"{name}.figures() must reload persisted state (not use in-memory)"
        # statistics() persists that state AND can reload its inputs from disk
        assert "_save_cluster_state" in stat_src, \
            f"{name}.statistics() must persist cluster state for figures()"
        assert "_reload_maps_from_disk" in stat_src, \
            f"{name}.statistics() must reload per-subject maps from disk"


def test_fcd_comparison_figures_reload_from_csv():
    fig_src = inspect.getsource(fc.FCDComparisonAnalysis.figures)
    assert "fcd_subject_summary.csv" in fig_src, \
        "fcd_comparison.figures() must reload its per-subject summary CSV"

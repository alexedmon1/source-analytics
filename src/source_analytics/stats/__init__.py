"""Statistical testing — R for ROI-level LMMs, Python for cluster permutation."""

from .cluster_permutation import (
    ClusterResult,
    cluster_permutation_test,
    voxelwise_ttest,
    hedges_g,
    build_adjacency,
    find_clusters,
)
from .tfce import TFCEResult, compute_tfce_scores, tfce_permutation_test
from .mvpa import MVPAResult, run_mvpa
from .graph_metrics import (
    GlobalMetrics,
    GraphMetrics,
    ROIGraphMetrics,
    NBSResult,
    AUCResult,
    GLOBAL_METRIC_NAMES,
    compute_global_metrics,
    compute_graph_metrics,
    compute_auc,
    auc_permutation_test,
    nbs_permutation_test,
)

__all__ = [
    "ClusterResult",
    "cluster_permutation_test",
    "voxelwise_ttest",
    "hedges_g",
    "build_adjacency",
    "find_clusters",
    "TFCEResult",
    "compute_tfce_scores",
    "tfce_permutation_test",
    "MVPAResult",
    "run_mvpa",
    "GlobalMetrics",
    "GraphMetrics",
    "ROIGraphMetrics",
    "NBSResult",
    "AUCResult",
    "GLOBAL_METRIC_NAMES",
    "compute_global_metrics",
    "compute_graph_metrics",
    "compute_auc",
    "auc_permutation_test",
    "nbs_permutation_test",
]

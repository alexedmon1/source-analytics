"""Threshold-Free Cluster Enhancement (TFCE) with permutation testing.

TFCE (Smith & Nichols, 2009) integrates cluster extent and height across
all possible thresholds, eliminating the need for an arbitrary initial
cluster-forming threshold. This provides better sensitivity for diffuse
effects compared to traditional cluster-based permutation testing.

TFCE(v) = sum_h { e(h)^E * h^H * dh }

where e(h) is the extent (size) of the cluster containing vertex v at
threshold h, E=0.5, H=2.0 are the default parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .cluster_permutation import build_adjacency, find_clusters, hedges_g, voxelwise_ttest

logger = logging.getLogger(__name__)


@dataclass
class TFCEResult:
    """Results from TFCE permutation testing."""

    t_map: np.ndarray  # (n_vertices,) — observed t-values
    tfce_scores: np.ndarray  # (n_vertices,) — TFCE-enhanced scores
    p_corrected: np.ndarray  # (n_vertices,) — corrected p-values
    hedges_g_map: np.ndarray  # (n_vertices,) — effect sizes
    n_permutations: int
    n_significant: int  # vertices with p < 0.05


def compute_tfce_scores(
    stat_map: np.ndarray,
    adjacency: sparse.csr_matrix,
    E: float = 0.5,
    H: float = 2.0,
    dh: float = 0.1,
) -> np.ndarray:
    """Compute TFCE scores for a statistical map.

    Parameters
    ----------
    stat_map : ndarray, shape (n_vertices,)
        Test statistic (e.g., t-values).
    adjacency : sparse.csr_matrix
        Boolean adjacency matrix.
    E : float
        Extent exponent (default 0.5).
    H : float
        Height exponent (default 2.0).
    dh : float
        Height increment for integration (default 0.1).

    Returns
    -------
    tfce : ndarray, shape (n_vertices,)
        TFCE-enhanced scores. Contains both positive and negative values
        corresponding to the sign of the original stat_map.
    """
    n = len(stat_map)
    tfce = np.zeros(n)

    # Process positive and negative tails separately
    for sign in [1, -1]:
        signed_map = sign * stat_map
        max_val = signed_map.max()
        if max_val <= 0:
            continue

        thresholds = np.arange(dh, max_val + dh, dh)
        for h in thresholds:
            # Find vertices above threshold
            above = signed_map >= h
            if not np.any(above):
                continue

            # Find connected components among supra-threshold vertices
            # Use a simple BFS approach
            visited = np.zeros(n, dtype=bool)
            for seed in range(n):
                if visited[seed] or not above[seed]:
                    continue

                # BFS to find cluster
                cluster_verts = []
                queue = [seed]
                visited[seed] = True
                while queue:
                    v = queue.pop(0)
                    cluster_verts.append(v)
                    for nb in adjacency[v].indices:
                        if not visited[nb] and above[nb]:
                            visited[nb] = True
                            queue.append(nb)

                # TFCE contribution: extent^E * height^H * dh
                extent = len(cluster_verts)
                contribution = sign * (extent ** E) * (h ** H) * dh
                for v in cluster_verts:
                    tfce[v] += contribution

    return tfce


def tfce_permutation_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    coords: np.ndarray,
    n_perms: int = 1000,
    E: float = 0.5,
    H: float = 2.0,
    dh: float = 0.1,
    distance_mm: float = 5.0,
    seed: int | None = None,
) -> TFCEResult:
    """TFCE permutation test for group comparison.

    Parameters
    ----------
    data_a : ndarray, shape (n_a, n_vertices)
        Data for group A.
    data_b : ndarray, shape (n_b, n_vertices)
        Data for group B.
    coords : ndarray, shape (n_vertices, 3)
        Source coordinates in mm.
    n_perms : int
        Number of permutations.
    E, H, dh : float
        TFCE parameters.
    distance_mm : float
        Adjacency distance threshold.
    seed : int, optional
        Random seed.

    Returns
    -------
    TFCEResult
    """
    rng = np.random.default_rng(seed)
    adjacency = build_adjacency(coords, distance_mm)

    # Observed statistics
    t_map, _ = voxelwise_ttest(data_a, data_b)
    g_map = hedges_g(data_a, data_b)
    observed_tfce = compute_tfce_scores(t_map, adjacency, E=E, H=H, dh=dh)

    logger.info(
        "Observed TFCE range: [%.2f, %.2f], running %d permutations...",
        observed_tfce.min(), observed_tfce.max(), n_perms,
    )

    # Permutation null: max|TFCE| per permutation
    combined = np.vstack([data_a, data_b])
    n_a = data_a.shape[0]
    n_total = combined.shape[0]

    null_max_tfce = np.zeros(n_perms)
    for i in range(n_perms):
        perm_idx = rng.permutation(n_total)
        perm_a = combined[perm_idx[:n_a]]
        perm_b = combined[perm_idx[n_a:]]

        perm_t, _ = voxelwise_ttest(perm_a, perm_b)
        perm_tfce = compute_tfce_scores(perm_t, adjacency, E=E, H=H, dh=dh)
        null_max_tfce[i] = np.max(np.abs(perm_tfce))

    # Corrected p-values: proportion of null max >= observed |TFCE|
    p_corrected = np.array([
        float(np.mean(null_max_tfce >= abs(score))) for score in observed_tfce
    ])

    n_sig = int(np.sum(p_corrected < 0.05))
    logger.info("TFCE: %d/%d vertices significant at p<0.05", n_sig, len(t_map))

    return TFCEResult(
        t_map=t_map,
        tfce_scores=observed_tfce,
        p_corrected=p_corrected,
        hedges_g_map=g_map,
        n_permutations=n_perms,
        n_significant=n_sig,
    )

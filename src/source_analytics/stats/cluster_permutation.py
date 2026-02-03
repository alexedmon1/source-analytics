"""Voxel-wise statistics with spatial cluster-based permutation testing.

Implements the cluster-based permutation approach (Maris & Oostenveld, 2007)
for mass-univariate testing on source-space vertex data.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse, stats

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Results from cluster-based permutation testing."""

    t_map: np.ndarray  # (n_vertices,)
    p_map: np.ndarray  # uncorrected p-values (n_vertices,)
    cluster_labels: np.ndarray  # (n_vertices,) — 0 = no cluster, 1..K = cluster ID
    cluster_stats: list[float]  # sum(t) for each cluster
    cluster_pvalues: list[float]  # corrected p-value per cluster
    n_clusters: int
    n_permutations: int


def voxelwise_ttest(
    data_a: np.ndarray,
    data_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent-samples t-test at each vertex.

    Parameters
    ----------
    data_a : ndarray, shape (n_subjects_a, n_vertices)
    data_b : ndarray, shape (n_subjects_b, n_vertices)

    Returns
    -------
    t_map : ndarray, shape (n_vertices,)
    p_map : ndarray, shape (n_vertices,)
    """
    t_map, p_map = stats.ttest_ind(data_a, data_b, axis=0, equal_var=False)
    return t_map, p_map


def hedges_g(data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray:
    """Compute Hedges' g effect size at each vertex.

    Parameters
    ----------
    data_a : ndarray, shape (n_a, n_vertices)
    data_b : ndarray, shape (n_b, n_vertices)

    Returns
    -------
    g : ndarray, shape (n_vertices,)
    """
    n_a, n_b = data_a.shape[0], data_b.shape[0]
    mean_a = data_a.mean(axis=0)
    mean_b = data_b.mean(axis=0)
    var_a = data_a.var(axis=0, ddof=1)
    var_b = data_b.var(axis=0, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    pooled_std = np.maximum(pooled_std, np.finfo(float).eps)

    d = (mean_a - mean_b) / pooled_std

    # Hedges' correction factor
    df = n_a + n_b - 2
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * correction


def build_adjacency(
    coords: np.ndarray,
    distance_mm: float = 5.0,
) -> sparse.csr_matrix:
    """Build sparse boolean adjacency matrix from source coordinates.

    Two vertices are adjacent if their Euclidean distance is <= distance_mm.

    Parameters
    ----------
    coords : ndarray, shape (n_vertices, 3)
        Source coordinates in mm.
    distance_mm : float
        Distance threshold for adjacency.

    Returns
    -------
    adjacency : sparse.csr_matrix, shape (n_vertices, n_vertices)
        Boolean adjacency matrix.
    """
    from scipy.spatial.distance import cdist

    n = coords.shape[0]
    dist_matrix = cdist(coords, coords, metric="euclidean")
    # Adjacent if within threshold (exclude self-loops via strict inequality on diagonal)
    adj = dist_matrix <= distance_mm
    np.fill_diagonal(adj, False)

    return sparse.csr_matrix(adj)


def find_clusters(
    stat_map: np.ndarray,
    adjacency: sparse.csr_matrix,
    threshold: float,
    tail: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Find connected clusters of supra-threshold vertices using BFS.

    Parameters
    ----------
    stat_map : ndarray, shape (n_vertices,)
        Test statistic (e.g., t-values).
    adjacency : sparse.csr_matrix
        Boolean adjacency matrix.
    threshold : float
        Absolute threshold for cluster formation.
    tail : int
        0 = two-tailed (|stat| > threshold), 1 = positive only,
        -1 = negative only.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Cluster labels (0 = not in a cluster).
    cluster_stats : list[float]
        Sum of stat_map values within each cluster.
    """
    n = len(stat_map)
    labels = np.zeros(n, dtype=int)

    if tail == 0:
        suprathresh = np.abs(stat_map) > threshold
    elif tail == 1:
        suprathresh = stat_map > threshold
    else:
        suprathresh = stat_map < -threshold

    cluster_id = 0
    cluster_stats = []
    visited = np.zeros(n, dtype=bool)

    for seed in range(n):
        if visited[seed] or not suprathresh[seed]:
            continue

        # BFS from seed
        cluster_id += 1
        queue = deque([seed])
        visited[seed] = True
        cluster_sum = 0.0

        while queue:
            v = queue.popleft()
            labels[v] = cluster_id
            cluster_sum += stat_map[v]

            # Get neighbors from adjacency
            neighbors = adjacency[v].indices
            for nb in neighbors:
                if not visited[nb] and suprathresh[nb]:
                    visited[nb] = True
                    queue.append(nb)

        cluster_stats.append(cluster_sum)

    return labels, cluster_stats


def cluster_permutation_test(
    data_a: np.ndarray,
    data_b: np.ndarray,
    coords: np.ndarray,
    n_perms: int = 1000,
    threshold: float = 2.0,
    tail: int = 0,
    distance_mm: float = 5.0,
    seed: int | None = None,
) -> ClusterResult:
    """Full cluster-based permutation test.

    Parameters
    ----------
    data_a : ndarray, shape (n_a, n_vertices)
        Data for group A (one row per subject).
    data_b : ndarray, shape (n_b, n_vertices)
        Data for group B.
    coords : ndarray, shape (n_vertices, 3)
        Source coordinates in mm.
    n_perms : int
        Number of permutations.
    threshold : float
        T-statistic threshold for cluster formation.
    tail : int
        0 = two-tailed, 1 = positive, -1 = negative.
    distance_mm : float
        Adjacency distance threshold in mm.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ClusterResult
        Full results including cluster labels, corrected p-values, and t-map.
    """
    rng = np.random.default_rng(seed)

    # Build adjacency once
    adjacency = build_adjacency(coords, distance_mm)

    # Observed statistics
    t_map, p_map = voxelwise_ttest(data_a, data_b)

    # Find observed clusters
    labels, cluster_stats = find_clusters(t_map, adjacency, threshold, tail)
    n_clusters = len(cluster_stats)

    if n_clusters == 0:
        logger.info("No clusters found above threshold %.1f", threshold)
        return ClusterResult(
            t_map=t_map,
            p_map=p_map,
            cluster_labels=labels,
            cluster_stats=[],
            cluster_pvalues=[],
            n_clusters=0,
            n_permutations=n_perms,
        )

    logger.info("Found %d clusters, running %d permutations...", n_clusters, n_perms)

    # Combine data for permutation
    combined = np.vstack([data_a, data_b])
    n_a = data_a.shape[0]
    n_total = combined.shape[0]

    # Null distribution: max cluster statistic per permutation
    null_max_stats = np.zeros(n_perms)

    for i in range(n_perms):
        perm_idx = rng.permutation(n_total)
        perm_a = combined[perm_idx[:n_a]]
        perm_b = combined[perm_idx[n_a:]]

        perm_t, _ = voxelwise_ttest(perm_a, perm_b)
        _, perm_cluster_stats = find_clusters(perm_t, adjacency, threshold, tail)

        if perm_cluster_stats:
            null_max_stats[i] = max(abs(s) for s in perm_cluster_stats)

    # Compute corrected p-values for each observed cluster
    cluster_pvalues = []
    for cs in cluster_stats:
        p_corr = float(np.mean(null_max_stats >= abs(cs)))
        cluster_pvalues.append(p_corr)

    logger.info(
        "Cluster p-values: %s",
        ", ".join(f"{p:.4f}" for p in cluster_pvalues),
    )

    return ClusterResult(
        t_map=t_map,
        p_map=p_map,
        cluster_labels=labels,
        cluster_stats=cluster_stats,
        cluster_pvalues=cluster_pvalues,
        n_clusters=n_clusters,
        n_permutations=n_perms,
    )

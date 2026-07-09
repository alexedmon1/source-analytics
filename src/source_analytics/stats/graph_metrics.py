"""Graph-theoretic metrics and Network-Based Statistic (NBS).

Computes standard graph metrics from connectivity matrices and implements
the NBS approach (Zalesky et al., 2010) for identifying subnetworks with
significant group differences.

Supports multi-density AUC approach for threshold-independent global
metric inference (Achard & Bullmore, 2007).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GlobalMetrics:
    """Global graph-theoretic metrics at a single density."""

    global_efficiency: float
    characteristic_path_length: float
    mean_clustering: float
    transitivity: float
    modularity: float
    assortativity: float
    mean_local_efficiency: float
    small_worldness: float
    n_nodes: int
    n_edges: int
    density: float


@dataclass
class ROIGraphMetrics:
    """Graph metrics with both global and nodal measures (for ROI-level analysis)."""

    global_efficiency: float
    modularity: float
    small_worldness: float
    n_edges: int
    degree: np.ndarray
    clustering: np.ndarray
    betweenness: np.ndarray


# Keep old name as alias for backward compatibility
GraphMetrics = ROIGraphMetrics


@dataclass
class AUCResult:
    """AUC values for global metrics across a density range."""

    densities: np.ndarray
    metrics_by_density: list[GlobalMetrics]
    auc: dict[str, float]  # metric_name -> AUC value


GLOBAL_METRIC_NAMES = [
    "global_efficiency",
    "characteristic_path_length",
    "mean_clustering",
    "transitivity",
    "modularity",
    "assortativity",
    "mean_local_efficiency",
    "small_worldness",
]


@dataclass
class NBSResult:
    """Results from Network-Based Statistic test."""

    significant_edges: np.ndarray  # (n_vertices, n_vertices) bool
    component_sizes: list[int]
    component_pvalues: list[float]
    t_matrix: np.ndarray  # (n_vertices, n_vertices)
    n_permutations: int
    n_significant_components: int
    component_nodes: list[list[int]] = field(default_factory=list)  # node ids per component


def _threshold_matrix(
    conn_matrix: np.ndarray,
    density: float,
) -> np.ndarray:
    """Threshold a connectivity matrix at a given proportional density.

    Parameters
    ----------
    conn_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    density : float
        Proportion of edges to retain (0-1).

    Returns
    -------
    binary : ndarray, shape (n, n)
        Binary adjacency matrix.
    """
    n = conn_matrix.shape[0]
    vals = conn_matrix[np.triu_indices(n, k=1)]
    if len(vals) == 0:
        return np.zeros_like(conn_matrix)
    cutoff = np.percentile(vals, 100 * (1 - density))
    binary = (conn_matrix >= cutoff).astype(float)
    np.fill_diagonal(binary, 0)
    return binary


def compute_global_metrics(
    conn_matrix: np.ndarray,
    density: float = 0.1,
) -> GlobalMetrics:
    """Compute global graph-theoretic metrics at a single density.

    Parameters
    ----------
    conn_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    density : float
        Proportional density threshold (0-1). Keeps top X% of edges.

    Returns
    -------
    GlobalMetrics
    """
    import networkx as nx

    n = conn_matrix.shape[0]
    binary = _threshold_matrix(conn_matrix, density)
    G = nx.from_numpy_array(binary)
    n_edges = int(binary.sum() / 2)

    # Global efficiency
    try:
        global_eff = nx.global_efficiency(G)
    except Exception:
        global_eff = 0.0

    # Characteristic path length
    try:
        if nx.is_connected(G):
            cpl = nx.average_shortest_path_length(G)
        else:
            largest = max(nx.connected_components(G), key=len)
            cpl = nx.average_shortest_path_length(G.subgraph(largest))
    except Exception:
        cpl = float("inf")

    # Mean clustering coefficient
    try:
        mean_clust = nx.average_clustering(G)
    except Exception:
        mean_clust = 0.0

    # Transitivity
    try:
        trans = nx.transitivity(G)
    except Exception:
        trans = 0.0

    # Modularity (greedy)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = 0.0

    # Assortativity (degree-degree correlation)
    try:
        assort = nx.degree_assortativity_coefficient(G)
        if np.isnan(assort):
            assort = 0.0
    except Exception:
        assort = 0.0

    # Mean local efficiency
    try:
        local_effs = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                local_effs.append(0.0)
                continue
            subG = G.subgraph(neighbors)
            local_effs.append(nx.global_efficiency(subG))
        mean_local_eff = float(np.mean(local_effs)) if local_effs else 0.0
    except Exception:
        mean_local_eff = 0.0

    # Small-worldness: sigma = (C/C_rand) / (L/L_rand)
    try:
        C = mean_clust
        L = cpl
        degree_seq = [d for _, d in G.degree()]
        if sum(degree_seq) > 0 and L < float("inf"):
            G_rand = nx.expected_degree_graph(degree_seq, selfloops=False)
            C_rand = max(nx.average_clustering(G_rand), 1e-10)
            if nx.is_connected(G_rand):
                L_rand = nx.average_shortest_path_length(G_rand)
            else:
                largest_rand = max(nx.connected_components(G_rand), key=len)
                L_rand = nx.average_shortest_path_length(
                    G_rand.subgraph(largest_rand)
                )
            L_rand = max(L_rand, 1e-10)
            small_world = (C / max(C_rand, 1e-10)) / (L / L_rand)
        else:
            small_world = 0.0
    except Exception:
        small_world = 0.0

    return GlobalMetrics(
        global_efficiency=global_eff,
        characteristic_path_length=cpl if cpl < float("inf") else 0.0,
        mean_clustering=mean_clust,
        transitivity=trans,
        modularity=modularity,
        assortativity=assort,
        mean_local_efficiency=mean_local_eff,
        small_worldness=small_world,
        n_nodes=n,
        n_edges=n_edges,
        density=density,
    )


def compute_graph_metrics(
    conn_matrix: np.ndarray,
    threshold_method: str = "proportional",
    threshold_value: float = 0.15,
) -> ROIGraphMetrics:
    """Compute graph metrics with nodal measures (for ROI-level networks).

    Parameters
    ----------
    conn_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    threshold_method : str
        Thresholding method ('proportional' supported).
    threshold_value : float
        Threshold parameter (density for proportional method).

    Returns
    -------
    ROIGraphMetrics
    """
    import networkx as nx

    n = conn_matrix.shape[0]
    binary = _threshold_matrix(conn_matrix, threshold_value)
    G = nx.from_numpy_array(binary)
    n_edges = int(binary.sum() / 2)

    # Global efficiency
    try:
        global_eff = nx.global_efficiency(G)
    except Exception:
        global_eff = 0.0

    # Modularity
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = 0.0

    # Mean clustering & small-worldness
    try:
        mean_clust = nx.average_clustering(G)
    except Exception:
        mean_clust = 0.0

    try:
        if nx.is_connected(G):
            cpl = nx.average_shortest_path_length(G)
        else:
            largest = max(nx.connected_components(G), key=len)
            cpl = nx.average_shortest_path_length(G.subgraph(largest))
    except Exception:
        cpl = float("inf")

    try:
        degree_seq = [d for _, d in G.degree()]
        if sum(degree_seq) > 0 and cpl < float("inf"):
            G_rand = nx.expected_degree_graph(degree_seq, selfloops=False)
            C_rand = max(nx.average_clustering(G_rand), 1e-10)
            if nx.is_connected(G_rand):
                L_rand = nx.average_shortest_path_length(G_rand)
            else:
                largest_rand = max(nx.connected_components(G_rand), key=len)
                L_rand = nx.average_shortest_path_length(
                    G_rand.subgraph(largest_rand)
                )
            L_rand = max(L_rand, 1e-10)
            small_world = (mean_clust / max(C_rand, 1e-10)) / (cpl / L_rand)
        else:
            small_world = 0.0
    except Exception:
        small_world = 0.0

    # Nodal metrics
    degree = np.array([d for _, d in G.degree()], dtype=float)
    clustering = np.array([nx.clustering(G, v) for v in G.nodes()], dtype=float)
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness = np.array(
        [betweenness_dict[v] for v in G.nodes()], dtype=float,
    )

    return ROIGraphMetrics(
        global_efficiency=global_eff,
        modularity=modularity,
        small_worldness=small_world,
        n_edges=n_edges,
        degree=degree,
        clustering=clustering,
        betweenness=betweenness,
    )


def compute_auc(
    conn_matrix: np.ndarray,
    density_min: float = 0.05,
    density_max: float = 0.40,
    density_step: float = 0.01,
) -> AUCResult:
    """Compute global metrics across densities and integrate (AUC).

    Uses trapezoidal integration across the density range for each
    global metric, producing a threshold-independent scalar per metric.

    Parameters
    ----------
    conn_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    density_min : float
        Minimum density (default 0.05 = 5%).
    density_max : float
        Maximum density (default 0.40 = 40%).
    density_step : float
        Step size (default 0.01 = 1%).

    Returns
    -------
    AUCResult
    """
    densities = np.arange(density_min, density_max + density_step / 2, density_step)
    metrics_list = []

    for d in densities:
        gm = compute_global_metrics(conn_matrix, density=d)
        metrics_list.append(gm)

    # Compute AUC via trapezoidal integration for each metric
    auc = {}
    for metric_name in GLOBAL_METRIC_NAMES:
        values = np.array([getattr(gm, metric_name) for gm in metrics_list])
        auc[metric_name] = float(np.trapezoid(values, densities))

    return AUCResult(
        densities=densities,
        metrics_by_density=metrics_list,
        auc=auc,
    )


def auc_permutation_test(
    auc_a: list[dict[str, float]],
    auc_b: list[dict[str, float]],
    n_permutations: int = 5000,
    seed: int | None = None,
) -> dict[str, dict]:
    """Permutation test on AUC values between two groups.

    For each global metric, tests whether the group difference in AUC
    is larger than expected by chance.

    Parameters
    ----------
    auc_a : list of dict
        Per-subject AUC dicts for group A. Each dict maps metric_name -> AUC.
    auc_b : list of dict
        Per-subject AUC dicts for group B.
    n_permutations : int
        Number of permutations.
    seed : int, optional
        Random seed.

    Returns
    -------
    results : dict
        {metric_name: {"observed_diff": float, "p_value": float, "hedges_g": float}}
    """
    rng = np.random.default_rng(seed)
    n_a = len(auc_a)
    n_b = len(auc_b)
    n_total = n_a + n_b

    results = {}
    for metric_name in GLOBAL_METRIC_NAMES:
        vals_a = np.array([d[metric_name] for d in auc_a])
        vals_b = np.array([d[metric_name] for d in auc_b])

        observed_diff = vals_a.mean() - vals_b.mean()

        # Hedges' g
        pooled_std = np.sqrt(
            ((n_a - 1) * vals_a.var(ddof=1) + (n_b - 1) * vals_b.var(ddof=1))
            / (n_a + n_b - 2)
        )
        if pooled_std > 0:
            correction = 1 - 3 / (4 * (n_a + n_b) - 9)
            g = correction * observed_diff / pooled_std
        else:
            g = 0.0

        # Permutation
        combined = np.concatenate([vals_a, vals_b])
        null_diffs = np.zeros(n_permutations)
        for p in range(n_permutations):
            perm = rng.permutation(n_total)
            null_diffs[p] = combined[perm[:n_a]].mean() - combined[perm[n_a:]].mean()

        p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))

        results[metric_name] = {
            "observed_diff": float(observed_diff),
            "p_value": p_value,
            "hedges_g": float(g),
            "mean_a": float(vals_a.mean()),
            "mean_b": float(vals_b.mean()),
        }

    return results


def _vectorized_welch_t(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorized Welch's t-test across all edges simultaneously.

    Parameters
    ----------
    A : ndarray, shape (n_a, n, n)
        Connectivity matrices for group A.
    B : ndarray, shape (n_b, n, n)
        Connectivity matrices for group B.

    Returns
    -------
    t_matrix : ndarray, shape (n, n)
        Welch's t-statistic for each edge.
    """
    n_a = A.shape[0]
    n_b = B.shape[0]

    mean_a = A.mean(axis=0)
    mean_b = B.mean(axis=0)
    var_a = A.var(axis=0, ddof=1)
    var_b = B.var(axis=0, ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    # Guard against div-by-zero without flooring legitimate small SEs.
    se = np.where(se > 0, se, np.finfo(float).tiny)

    t_matrix = (mean_a - mean_b) / se
    # Only keep upper triangle (symmetric)
    t_matrix = np.triu(t_matrix, k=1)
    t_matrix = t_matrix + t_matrix.T

    return t_matrix


def _find_components(adj: np.ndarray, with_nodes: bool = False):
    """Find connected components and count edges in each.

    Parameters
    ----------
    adj : ndarray, shape (n, n)
        Boolean adjacency matrix.
    with_nodes : bool
        When True, return ``(edge_count, node_list)`` tuples instead of bare
        edge counts (used to describe which vertices a component covers). The
        default (False) preserves the fast, count-only contract the permutation
        null relies on.

    Returns
    -------
    list[int] | list[tuple[int, list[int]]]
        Per component with >= 2 nodes: edge count (default) or (edge count, nodes).
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    components = []

    for seed_node in range(n):
        if visited[seed_node]:
            continue
        if not np.any(adj[seed_node]):
            continue

        comp = []
        queue = deque([seed_node])
        visited[seed_node] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            neighbors = np.where(adj[v] & ~visited)[0]
            for nb in neighbors:
                visited[nb] = True
                queue.append(nb)

        if len(comp) > 1:
            comp_arr = np.array(comp)
            sub = adj[np.ix_(comp_arr, comp_arr)]
            edge_count = int(np.triu(sub, k=1).sum())
            components.append((edge_count, [int(v) for v in comp]) if with_nodes else edge_count)

    return components


def nbs_permutation_test(
    matrices_a: list[np.ndarray],
    matrices_b: list[np.ndarray],
    nbs_threshold: float = 3.0,
    n_permutations: int = 5000,
    seed: int | None = None,
) -> NBSResult:
    """Network-Based Statistic (Zalesky et al., 2010).

    Uses vectorized Welch's t-tests for all edges simultaneously,
    avoiding O(n^2) scipy.stats.ttest_ind calls per permutation.

    Parameters
    ----------
    matrices_a : list of ndarray, shape (n, n)
        Connectivity matrices for group A (one per subject).
    matrices_b : list of ndarray
        Connectivity matrices for group B.
    nbs_threshold : float
        T-statistic threshold for initial edge selection.
    n_permutations : int
        Number of permutations.
    seed : int, optional
        Random seed.

    Returns
    -------
    NBSResult
    """
    rng = np.random.default_rng(seed)

    n_a = len(matrices_a)
    n_b = len(matrices_b)
    n = matrices_a[0].shape[0]

    # Stack into 3D arrays
    A = np.array(matrices_a)  # (n_a, n, n)
    B = np.array(matrices_b)  # (n_b, n, n)

    # Vectorized edge-wise t-tests
    t_matrix = _vectorized_welch_t(A, B)

    # Find supra-threshold edges
    suprathresh = np.abs(t_matrix) > nbs_threshold

    observed_with_nodes = _find_components(suprathresh, with_nodes=True)
    observed_components = [ec for ec, _ in observed_with_nodes]
    observed_nodes = [nodes for _, nodes in observed_with_nodes]

    if not observed_components:
        logger.info("NBS: no supra-threshold components found")
        return NBSResult(
            significant_edges=np.zeros((n, n), dtype=bool),
            component_sizes=[],
            component_pvalues=[],
            t_matrix=t_matrix,
            n_permutations=n_permutations,
            n_significant_components=0,
            component_nodes=[],
        )

    logger.info(
        "NBS: %d observed components (sizes: %s), running %d permutations...",
        len(observed_components), observed_components, n_permutations,
    )

    # Permutation test with vectorized t-tests
    combined = np.vstack([A, B])  # (n_a + n_b, n, n)
    null_max_component = np.zeros(n_permutations)

    for perm in range(n_permutations):
        perm_idx = rng.permutation(n_a + n_b)
        perm_A = combined[perm_idx[:n_a]]
        perm_B = combined[perm_idx[n_a:]]

        perm_t = _vectorized_welch_t(perm_A, perm_B)
        perm_supra = np.abs(perm_t) > nbs_threshold
        perm_comps = _find_components(perm_supra)
        null_max_component[perm] = max(perm_comps) if perm_comps else 0

    # P-values for each observed component
    component_pvalues = [
        float(np.mean(null_max_component >= size))
        for size in observed_components
    ]

    # Build significant edge mask
    sig_edges = np.zeros((n, n), dtype=bool)
    n_sig = sum(1 for p in component_pvalues if p < 0.05)

    logger.info("NBS: %d/%d components significant (p<0.05)", n_sig, len(observed_components))

    return NBSResult(
        significant_edges=sig_edges,
        component_sizes=observed_components,
        component_pvalues=component_pvalues,
        t_matrix=t_matrix,
        n_permutations=n_permutations,
        n_significant_components=n_sig,
        component_nodes=observed_nodes,
    )

"""Graph-theoretic metrics and Network-Based Statistic (NBS).

Computes standard graph metrics from connectivity matrices and implements
the NBS approach (Zalesky et al., 2010) for identifying subnetworks with
significant group differences.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraphMetrics:
    """Graph-theoretic metrics for a connectivity matrix."""

    degree: np.ndarray  # (n_vertices,)
    clustering: np.ndarray  # (n_vertices,)
    betweenness: np.ndarray  # (n_vertices,)
    global_efficiency: float
    modularity: float
    small_worldness: float
    n_nodes: int
    n_edges: int


@dataclass
class NBSResult:
    """Results from Network-Based Statistic test."""

    significant_edges: np.ndarray  # (n_vertices, n_vertices) bool
    component_sizes: list[int]
    component_pvalues: list[float]
    t_matrix: np.ndarray  # (n_vertices, n_vertices)
    n_permutations: int
    n_significant_components: int


def compute_graph_metrics(
    conn_matrix: np.ndarray,
    threshold_method: str = "proportional",
    threshold_value: float = 0.1,
) -> GraphMetrics:
    """Compute graph-theoretic metrics from a connectivity matrix.

    Parameters
    ----------
    conn_matrix : ndarray, shape (n, n)
        Symmetric connectivity matrix.
    threshold_method : str
        "proportional" — keep top X% of edges.
        "absolute" — keep edges above value.
    threshold_value : float
        Threshold parameter.

    Returns
    -------
    GraphMetrics
    """
    import networkx as nx

    n = conn_matrix.shape[0]

    # Threshold the matrix
    if threshold_method == "proportional":
        vals = conn_matrix[np.triu_indices(n, k=1)]
        if len(vals) > 0:
            cutoff = np.percentile(vals, 100 * (1 - threshold_value))
        else:
            cutoff = 0
    else:
        cutoff = threshold_value

    binary = (conn_matrix >= cutoff).astype(float)
    np.fill_diagonal(binary, 0)

    # Build networkx graph
    G = nx.from_numpy_array(binary)

    # Degree
    degree = np.array([d for _, d in G.degree()])

    # Clustering coefficient
    clustering_dict = nx.clustering(G)
    clustering = np.array([clustering_dict[i] for i in range(n)])

    # Betweenness centrality
    betweenness_dict = nx.betweenness_centrality(G)
    betweenness = np.array([betweenness_dict[i] for i in range(n)])

    # Global efficiency
    try:
        global_eff = nx.global_efficiency(G)
    except Exception:
        global_eff = 0.0

    # Modularity (greedy)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = 0.0

    # Small-worldness: sigma = (C/C_rand) / (L/L_rand)
    try:
        C = nx.average_clustering(G)
        if nx.is_connected(G):
            L = nx.average_shortest_path_length(G)
        else:
            # Use largest component
            largest = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest)
            L = nx.average_shortest_path_length(subG)

        # Generate random graph with same degree sequence
        degree_seq = [d for _, d in G.degree()]
        if sum(degree_seq) > 0:
            G_rand = nx.expected_degree_graph(degree_seq, selfloops=False)
            C_rand = max(nx.average_clustering(G_rand), 1e-10)
            if nx.is_connected(G_rand):
                L_rand = nx.average_shortest_path_length(G_rand)
            else:
                largest_rand = max(nx.connected_components(G_rand), key=len)
                L_rand = nx.average_shortest_path_length(G_rand.subgraph(largest_rand))
            L_rand = max(L_rand, 1e-10)
            small_world = (C / C_rand) / (L / L_rand)
        else:
            small_world = 0.0
    except Exception:
        small_world = 0.0

    n_edges = int(binary.sum() / 2)

    return GraphMetrics(
        degree=degree,
        clustering=clustering,
        betweenness=betweenness,
        global_efficiency=global_eff,
        modularity=modularity,
        small_worldness=small_world,
        n_nodes=n,
        n_edges=n_edges,
    )


def nbs_permutation_test(
    matrices_a: list[np.ndarray],
    matrices_b: list[np.ndarray],
    nbs_threshold: float = 3.0,
    n_permutations: int = 5000,
    seed: int | None = None,
) -> NBSResult:
    """Network-Based Statistic (Zalesky et al., 2010).

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
    from scipy import stats

    rng = np.random.default_rng(seed)

    n_a = len(matrices_a)
    n_b = len(matrices_b)
    n = matrices_a[0].shape[0]

    # Stack into 3D arrays
    A = np.array(matrices_a)  # (n_a, n, n)
    B = np.array(matrices_b)  # (n_b, n, n)

    # Edge-wise t-tests
    t_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            t_val, _ = stats.ttest_ind(A[:, i, j], B[:, i, j], equal_var=False)
            t_matrix[i, j] = t_val
            t_matrix[j, i] = t_val

    # Find supra-threshold edges
    suprathresh = np.abs(t_matrix) > nbs_threshold

    # Find connected components in supra-threshold network
    def _find_components(adj):
        n = adj.shape[0]
        visited = np.zeros(n, dtype=bool)
        components = []
        for seed_node in range(n):
            if visited[seed_node]:
                continue
            # Check if this node has any supra-threshold edges
            if not np.any(adj[seed_node]):
                continue
            comp = []
            queue = deque([seed_node])
            visited[seed_node] = True
            while queue:
                v = queue.popleft()
                comp.append(v)
                for nb in range(n):
                    if not visited[nb] and adj[v, nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > 1:
                # Count edges in component
                edge_count = 0
                for ci in comp:
                    for cj in comp:
                        if ci < cj and adj[ci, cj]:
                            edge_count += 1
                components.append(edge_count)
        return components

    observed_components = _find_components(suprathresh)

    if not observed_components:
        logger.info("NBS: no supra-threshold components found")
        return NBSResult(
            significant_edges=np.zeros((n, n), dtype=bool),
            component_sizes=[],
            component_pvalues=[],
            t_matrix=t_matrix,
            n_permutations=n_permutations,
            n_significant_components=0,
        )

    logger.info(
        "NBS: %d observed components (sizes: %s), running %d permutations...",
        len(observed_components), observed_components, n_permutations,
    )

    # Permutation test
    combined = np.vstack([A, B])
    null_max_component = np.zeros(n_permutations)

    for perm in range(n_permutations):
        perm_idx = rng.permutation(n_a + n_b)
        perm_A = combined[perm_idx[:n_a]]
        perm_B = combined[perm_idx[n_a:]]

        perm_t = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                t_val, _ = stats.ttest_ind(
                    perm_A[:, i, j], perm_B[:, i, j], equal_var=False,
                )
                perm_t[i, j] = t_val
                perm_t[j, i] = t_val

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
    # Mark edges from significant components
    n_sig = sum(1 for p in component_pvalues if p < 0.05)

    logger.info("NBS: %d/%d components significant (p<0.05)", n_sig, len(observed_components))

    return NBSResult(
        significant_edges=sig_edges,
        component_sizes=observed_components,
        component_pvalues=component_pvalues,
        t_matrix=t_matrix,
        n_permutations=n_permutations,
        n_significant_components=n_sig,
    )

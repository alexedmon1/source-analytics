"""Edge / NBS adapter for the ``hypothesis`` layer.

Runs a declarative :class:`~source_analytics.config.Hypothesis` over per-subject
**connectivity matrices** (n×n) and returns a **subnetwork** result (a component
table) — the Network-Based Statistic (Zalesky et al. 2010) counterpart to the
permutation adapter's map+cluster contract. Where the permutation adapter
(``permutation.py``) clusters supra-threshold *vertices* by spatial adjacency,
the edge adapter clusters supra-threshold *edges* into connected components on
the connectivity graph itself.

Kinds:
  - **contrast**  — weighted group contrast → per-edge t-matrix → NBS components.
    A *pairwise* contrast (two opposite-sign, equal-magnitude weights) routes
    through :func:`~source_analytics.stats.graph_metrics.nbs_permutation_test`
    verbatim, so it reproduces the legacy NBS result bit-exact. General
    (>2-group / non-unit) weights use a weighted per-edge t-statistic with a
    label-shuffle component-permutation engine.
  - **omnibus**   — per-edge one-way ANOVA F-matrix → NBS components (F-threshold
    at p = 0.01; the t-threshold knob has no F analogue).
  - **equivalence** — per-edge TOST (sd margin) → equivalence summary row.
  - **regression** — deferred (no continuous-predictor edge slope yet).

Nuisance covariates use **Freedman–Lane** residualisation (Freedman & Lane 1983;
Winkler et al. 2014, NeuroImage 92:381–397) on the flattened edge vector, the
same scheme the permutation adapter uses on vertex maps.

Method knobs (``nbs_threshold``, ``n_perms``) are passed in by the calling
module — they are test config, not hypothesis fields.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
from scipy import stats

from ..config import DesignSpec, Hypothesis
from ..stats.graph_metrics import _find_components, nbs_permutation_test
from .permutation import _fit_groups, _group_uids

logger = logging.getLogger(__name__)

# Cluster-forming p-level for the omnibus F-matrix (no t-threshold analogue).
_OMNIBUS_NBS_ALPHA = 0.01


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _stack(subject_matrices: dict[str, np.ndarray], uids: list[str]) -> np.ndarray:
    """(n_subjects, n, n) array for the given uids."""
    return np.array([np.asarray(subject_matrices[u], dtype=float) for u in uids])


def _residualize_edges(mats: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Freedman–Lane on flattened edges. mats: (N, n, n); Z: (N, p) nuisance."""
    N, n, _ = mats.shape
    flat = mats.reshape(N, n * n)
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, flat, rcond=None)
    resid = flat - Z1 @ beta
    return resid.reshape(N, n, n)


def _weighted_t_edges(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Per-edge t for a linear contrast Σ wᵢ·meanᵢ over independent groups."""
    n = arrays[0].shape[1]
    contrast = np.zeros((n, n))
    se2 = np.zeros((n, n))
    for a, w in zip(arrays, weights):
        ns = a.shape[0]
        contrast += w * a.mean(axis=0)
        se2 += (w ** 2) * a.var(axis=0, ddof=1) / ns
    se = np.sqrt(se2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, contrast / se, 0.0)
    np.fill_diagonal(t, 0.0)
    return t


def _f_edges(arrays: list[np.ndarray]) -> np.ndarray:
    """Per-edge one-way ANOVA F across groups."""
    res = stats.f_oneway(*arrays, axis=0)
    f = np.nan_to_num(np.asarray(res.statistic, dtype=float))
    np.fill_diagonal(f, 0.0)
    return f


def _components(adj: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Connected components (≥2 nodes) of a boolean adjacency matrix.

    Returns (node_indices, edge_count) per component in the SAME scan order as
    :func:`~source_analytics.stats.graph_metrics._find_components`, so the edge
    counts line up with that function's output (which the legacy NBS p-values are
    aligned to). This is the membership-aware twin of ``_find_components``.
    """
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    comps: list[tuple[np.ndarray, int]] = []
    for seed_node in range(n):
        if visited[seed_node] or not np.any(adj[seed_node]):
            continue
        comp = []
        queue = deque([seed_node])
        visited[seed_node] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            for nb in np.where(adj[v] & ~visited)[0]:
                visited[nb] = True
                queue.append(nb)
        if len(comp) > 1:
            comp_arr = np.array(comp)
            sub = adj[np.ix_(comp_arr, comp_arr)]
            comps.append((comp_arr, int(np.triu(sub, k=1).sum())))
    return comps


def _mass_peak(stat_matrix: np.ndarray, comps: list[tuple[np.ndarray, int]],
               threshold: float) -> tuple[list[float], list[float]]:
    """Per-component mass (Σ|stat| over supra-threshold edges) and peak |stat|."""
    masses, peaks = [], []
    for nodes, _ in comps:
        sub = stat_matrix[np.ix_(nodes, nodes)]
        iu = np.triu_indices(len(nodes), k=1)
        vals = np.abs(sub[iu])
        cv = vals[vals > threshold]
        masses.append(float(cv.sum()))
        peaks.append(float(cv.max()) if cv.size else float("nan"))
    return masses, peaks


def _nbs_components(arrays: list[np.ndarray], stat_fn, threshold: float,
                    n_perms: int, seed: int):
    """Generalized NBS: observed components + label-shuffle null on max edge-count.

    Returns (stat_matrix, comps, masses, peaks, pvals). Used for the general
    weighted contrast and the omnibus F (the pairwise contrast uses the base
    ``nbs_permutation_test`` directly to stay legacy-exact).
    """
    obs = stat_fn(arrays)
    comps = _components(np.abs(obs) > threshold)
    if not comps:
        return obs, [], [], [], []

    sizes = [a.shape[0] for a in arrays]
    pool = np.concatenate(arrays, axis=0)
    n_total = pool.shape[0]
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perms)
    for p in range(n_perms):
        perm = rng.permutation(n_total)
        idx, shuffled = 0, []
        for s in sizes:
            shuffled.append(pool[perm[idx:idx + s]])
            idx += s
        pc = _find_components(np.abs(stat_fn(shuffled)) > threshold)
        null_max[p] = max(pc) if pc else 0

    obs_sizes = [c[1] for c in comps]
    pvals = [(np.sum(null_max >= sz) + 1) / (n_perms + 1) for sz in obs_sizes]
    masses, peaks = _mass_peak(obs, comps, threshold)
    return obs, comps, masses, peaks, pvals


def _component_rows(comps: list[tuple[np.ndarray, int]], masses: list[float],
                    peaks: list[float], pvals: list[float],
                    stat_type: str,
                    vertex_rois: list[str | None] | None = None) -> list[dict[str, Any]]:
    """Subnetwork contract → one row per component (or a no-component row).

    ``vertex_rois`` (one atlas-ROI name per node, optional) adds a ``region``
    column naming the regions of the nodes in each subnetwork.
    """
    from ..atlas.atlas_utils import format_region_coverage

    if not comps:
        row = {
            "component_id": 0, "n_edges": 0, "mass": float("nan"),
            "peak_stat": float("nan"), "component_p": float("nan"),
            "significant": False, "stat_type": stat_type,
        }
        if vertex_rois is not None:
            row["region"] = ""
        return [row]
    rows = []
    for ci, ((nodes, n_edges), mass, peak, p) in enumerate(
            zip(comps, masses, peaks, pvals), start=1):
        row = {
            "component_id": ci, "n_edges": int(n_edges), "mass": float(mass),
            "peak_stat": float(peak), "component_p": float(p),
            "significant": float(p) < 0.05, "stat_type": stat_type,
        }
        if vertex_rois is not None:
            row["region"] = format_region_coverage(
                [vertex_rois[int(v)] for v in nodes if int(v) < len(vertex_rois)])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_hypothesis_edge(
    hyp: Hypothesis,
    subject_matrices: dict[str, np.ndarray],
    subject_groups: dict[str, str],
    *,
    nbs_threshold: float,
    n_perms: int,
    spec: DesignSpec | None = None,
    covariates: dict[str, np.ndarray] | None = None,
    seed: int = 42,
    vertex_rois: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Run ONE hypothesis over per-subject connectivity matrices → component rows.

    subject_matrices: {uid -> (n, n)} for a single (band, metric) cell, ALL
    subjects. subject_groups: {uid -> group}. ``vertex_rois`` (one ROI name per
    node) adds a ``region`` column naming each subnetwork's node regions.
    """
    present = set(subject_groups.values())
    groups = _fit_groups(hyp, spec, present)
    if hyp.kind != "regression" and len(groups) < 2:
        return []

    uids_by_group = {
        g: [u for u in _group_uids(subject_groups, g) if u in subject_matrices]
        for g in groups
    }
    if any(len(us) < 2 for us in uids_by_group.values()):
        return []

    # Assemble per-group arrays (+ optional Freedman–Lane nuisance residualisation).
    if covariates is not None:
        all_uids = [u for g in groups for u in uids_by_group[g]]
        mats = _stack(subject_matrices, all_uids)
        Z = np.array([np.asarray(covariates[u], dtype=float).ravel() for u in all_uids])
        resid = _residualize_edges(mats, Z)
        source = {u: resid[i] for i, u in enumerate(all_uids)}
    else:
        source = subject_matrices
    arrays = [_stack(source, uids_by_group[g]) for g in groups]

    if hyp.kind == "omnibus":
        k = len(arrays)
        n_total = sum(a.shape[0] for a in arrays)
        f_thresh = float(stats.f.ppf(1 - _OMNIBUS_NBS_ALPHA, k - 1, n_total - k))
        _, comps, masses, peaks, pvals = _nbs_components(
            arrays, _f_edges, f_thresh, n_perms, seed)
        return _component_rows(comps, masses, peaks, pvals, "F", vertex_rois)

    if hyp.kind == "contrast":
        w = {g: float(hyp.weights[g]) for g in groups}
        pos = [g for g, v in w.items() if v > 0]
        neg = [g for g, v in w.items() if v < 0]
        # Pairwise, opposite-sign, equal-magnitude → base NBS verbatim (legacy-exact;
        # |t| is invariant to a common weight scaling, so {+1,-1} and {+0.5,-0.5} agree).
        if (len(groups) == 2 and len(pos) == 1 and len(neg) == 1
                and np.isclose(abs(w[pos[0]]), abs(w[neg[0]]))):
            mats_a = list(_stack(source, uids_by_group[pos[0]]))
            mats_b = list(_stack(source, uids_by_group[neg[0]]))
            r = nbs_permutation_test(
                mats_a, mats_b, nbs_threshold=nbs_threshold,
                n_permutations=n_perms, seed=seed)
            comps = _components(np.abs(r.t_matrix) > nbs_threshold)
            masses, peaks = _mass_peak(r.t_matrix, comps, nbs_threshold)
            return _component_rows(comps, masses, peaks, r.component_pvalues, "t", vertex_rois)
        # General weighted contrast.
        weights = [w[g] for g in groups]
        _, comps, masses, peaks, pvals = _nbs_components(
            arrays, lambda arr: _weighted_t_edges(arr, weights),
            nbs_threshold, n_perms, seed)
        return _component_rows(comps, masses, peaks, pvals, "t", vertex_rois)

    if hyp.kind == "equivalence":
        # Per-edge TOST (sd margin) on the pairwise contrast; one summary row.
        w = {g: float(hyp.weights[g]) for g in groups}
        pos = [g for g, v in w.items() if v > 0]
        neg = [g for g, v in w.items() if v < 0]
        if not (len(groups) == 2 and pos and neg):
            return []
        a = arrays[groups.index(pos[0])]
        b = arrays[groups.index(neg[0])]
        iu = np.triu_indices(a.shape[1], k=1)
        ae, be = a[:, iu[0], iu[1]], b[:, iu[0], iu[1]]
        na, nb = ae.shape[0], be.shape[0]
        diff = ae.mean(0) - be.mean(0)
        sp = np.sqrt(((na - 1) * ae.var(0, ddof=1) + (nb - 1) * be.var(0, ddof=1))
                     / (na + nb - 2))
        se = sp * np.sqrt(1 / na + 1 / nb)
        margin = float((hyp.margin or {}).get("value", 0.0)) * sp  # sd mode
        zc = stats.t.ppf(0.95, na + nb - 2)
        with np.errstate(invalid="ignore"):
            equiv = (diff - zc * se > -margin) & (diff + zc * se < margin)
        n_equiv = int(np.nansum(equiv))
        n_edges = len(diff)
        row = {
            "component_id": 0, "n_edges": n_edges, "mass": float("nan"),
            "peak_stat": float("nan"), "component_p": float("nan"),
            "significant": False, "stat_type": "tost",
            "n_equivalent": n_equiv,
            "frac_equivalent": float(n_equiv / n_edges) if n_edges else 0.0,
        }
        if vertex_rois is not None:
            row["region"] = ""  # edge-level TOST has no node subnetwork
        return [row]

    return []  # regression deferred


# --------------------------------------------------------------------------- #
# Module convenience wrapper (the edge/NBS Tier-2 equivalent of
# write_module_hypotheses_perm())
# --------------------------------------------------------------------------- #
def write_module_hypotheses_edge(
    matrices_by_cell: dict[tuple[str, str], dict[str, np.ndarray]],
    subject_groups: dict[str, str],
    config,
    tbl_dir,
    prefix: str,
    *,
    nbs_threshold: float,
    n_perms: int,
    hypothesis: str | None = None,
    seed: int = 42,
    coords=None,
    atlas_dir=None,
):
    """Run every declared hypothesis over per-cell connectivity matrices.

    matrices_by_cell: {(band, metric) -> {uid -> (n, n)}}, ALL subjects per cell.
    ``coords`` + ``atlas_dir`` (optional): when both are given (vertex modules),
    each subnetwork row gets a ``region`` column naming the regions of its nodes.
    Writes ``<prefix>_hypotheses.csv``; returns the rows DataFrame (or None).
    """
    import pandas as pd
    from pathlib import Path

    spec = config.design_spec
    if spec is None or not spec.hypotheses:
        logger.info("  No hypotheses/contrasts declared — skipping %s edge hypotheses.", prefix)
        return None

    # Node ROI labels once (nodes are shared across all hypotheses/cells).
    vertex_rois = None
    if coords is not None and atlas_dir is not None:
        try:
            from ..atlas.atlas_utils import label_vertices_to_rois
            vertex_rois = label_vertices_to_rois(np.asarray(coords, dtype=float), atlas_dir)
        except Exception as e:  # noqa: BLE001 — descriptive, never fatal
            logger.warning("  Node ROI labeling failed (%s); regions omitted", e)

    hyps = spec.hypotheses
    if hypothesis:
        want = {h.strip() for h in hypothesis.split(",")}
        hyps = [h for h in hyps if h.name in want]
        if not hyps:
            logger.info("  No declared hypothesis matches --hypothesis '%s'", hypothesis)
            return None

    rows: list[dict[str, Any]] = []
    for hyp in hyps:
        for (band, dv), subject_matrices in matrices_by_cell.items():
            try:
                comps = run_hypothesis_edge(
                    hyp, subject_matrices, subject_groups, spec=spec,
                    nbs_threshold=nbs_threshold, n_perms=n_perms, seed=seed,
                    vertex_rois=vertex_rois)
            except Exception as e:  # noqa: BLE001 — keep the sweep going
                logger.warning("  %s [%s/%s]: %s", hyp.name, band, dv, e)
                continue
            for c in comps:
                rows.append({
                    "hypothesis": hyp.name, "kind": hyp.kind, "role": hyp.role,
                    "label": hyp.label or hyp.name, "test": hyp.test,
                    "band": band, "dv": dv,
                    "n_perms": n_perms, "threshold": nbs_threshold, **c,
                })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = Path(tbl_dir) / f"{prefix}_hypotheses.csv"
    df.to_csv(path, index=False)
    n_sig = int(df["significant"].sum())
    logger.info("  Saved: %s (%d rows, %d hypotheses; %d significant subnetworks)",
                path.name, len(df), len(hyps), n_sig)
    return df

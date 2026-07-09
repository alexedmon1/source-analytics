"""Permutation adapter for the ``hypothesis`` layer.

Runs a declarative :class:`~source_analytics.config.Hypothesis` over per-subject
**vertex maps** and returns a **map + clusters** result (the permutation contract from
DESIGN_SPEC §6.3), the counterpart to the R emmeans adapter's tabular contract.

Kinds:
  - **contrast**  — weighted group contrast → cluster-corrected t-map. A *pairwise*
    contrast (two opposite-sign groups) routes through the existing
    :func:`cluster_permutation_test` verbatim, so it reproduces the legacy vertex
    result bit-exact. General (>2-group / non-unit) weights use a weighted per-vertex
    t-statistic with the shared label-shuffle cluster engine.
  - **omnibus**   — per-vertex one-way ANOVA F-map → cluster correction (NEW; not in the
    base ``stats`` package, which assumed binary contrasts).
  - **equivalence** — per-vertex TOST (sd margin) → equivalence summary.
  - **regression** — per-vertex slope on a continuous predictor → cluster-corrected map.

Nuisance covariates are handled by **Freedman–Lane** residualisation (Freedman & Lane
1983; Winkler et al. 2014, NeuroImage 92:381–397): regress the nuisance out of the maps,
then permute the residuals.

Method knobs (n_perms, cluster-forming threshold, adjacency distance) are passed in by the
calling module — they are test config, not hypothesis fields.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from ..config import DesignSpec, Hypothesis
from ..stats.cluster_permutation import (
    build_adjacency,
    cluster_permutation_test,
    find_clusters,
)

logger = logging.getLogger(__name__)

# Cluster-forming p-level for the omnibus F-map (no t-threshold analogue).
_OMNIBUS_CLUSTER_ALPHA = 0.01


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _group_uids(subject_groups: dict[str, str], group: str) -> list[str]:
    return [u for u, g in subject_groups.items() if g == group]


def _stack(subject_maps: dict[str, np.ndarray], uids: list[str]) -> np.ndarray:
    """(n_subjects, n_vertices) array for the given uids."""
    return np.array([np.asarray(subject_maps[u], dtype=float) for u in uids])


def _residualize(maps: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Freedman–Lane: remove the nuisance fit from each vertex column.

    maps: (N, V); Z: (N, p) nuisance design (a column of ones is added). Returns the
    residuals (N, V) — the nuisance-adjusted maps to permute.
    """
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(Z1, maps, rcond=None)
    return maps - Z1 @ beta


def _weighted_t_map(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Per-vertex t for a linear contrast Σ wᵢ·meanᵢ over independent groups."""
    contrast = np.zeros(arrays[0].shape[1])
    se2 = np.zeros(arrays[0].shape[1])
    for a, w in zip(arrays, weights):
        n = a.shape[0]
        contrast += w * a.mean(axis=0)
        se2 += (w ** 2) * a.var(axis=0, ddof=1) / n
    se = np.sqrt(se2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, contrast / se, 0.0)
    return t


def _f_map(arrays: list[np.ndarray]) -> np.ndarray:
    """Per-vertex one-way ANOVA F across groups."""
    res = stats.f_oneway(*arrays, axis=0)
    return np.nan_to_num(np.asarray(res.statistic, dtype=float))


def _perm_cluster(
    arrays: list[np.ndarray],
    stat_fn,
    coords: np.ndarray,
    threshold: float,
    tail: int,
    n_perms: int,
    distance_mm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Generic label-shuffle cluster permutation for a multi-group statistic.

    Returns (stat_map, cluster_labels, cluster_masses, cluster_pvalues). Used for the
    general weighted contrast and the omnibus F (the pairwise contrast uses the base
    cluster_permutation_test directly).
    """
    adjacency = build_adjacency(coords, distance_mm)
    obs = stat_fn(arrays)
    labels, masses = find_clusters(obs, adjacency, threshold, tail)
    if not masses:
        return obs, labels, [], []

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
        _, ms = find_clusters(stat_fn(shuffled), adjacency, threshold, tail)
        null_max[p] = max((abs(m) for m in ms), default=0.0)

    cluster_p = [(np.sum(null_max >= abs(m)) + 1) / (n_perms + 1) for m in masses]
    return obs, labels, masses, cluster_p


def _format_equiv_region(equiv_mask: np.ndarray, vertex_rois: list[str | None]) -> str:
    """Anatomical coverage of the vertices judged equivalent in a TOST map."""
    from ..atlas.atlas_utils import format_region_coverage
    idx = np.where(np.asarray(equiv_mask))[0]
    return format_region_coverage([vertex_rois[i] for i in idx if i < len(vertex_rois)])


def _cluster_rows(
    stat_map: np.ndarray,
    labels: np.ndarray,
    masses: list[float],
    pvalues: list[float],
    stat_type: str,
    vertex_rois: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Map+cluster contract → one row per surviving cluster (or a no-cluster row).

    ``vertex_rois`` (one atlas-ROI name per vertex, optional) adds a ``region``
    column describing each cluster's anatomical coverage.
    """
    from ..atlas.atlas_utils import format_region_coverage

    if not masses:
        row = {
            "cluster_id": 0, "n_vertices": 0, "mass": float("nan"),
            "peak_stat": float("nan"), "peak_vertex": -1,
            "cluster_p": float("nan"), "significant": False, "stat_type": stat_type,
        }
        if vertex_rois is not None:
            row["region"] = ""
        return [row]
    rows = []
    for ci in range(1, len(masses) + 1):
        mask = labels == ci
        nv = int(mask.sum())
        if nv == 0:
            continue
        seg = stat_map[mask]
        peak_local = int(np.argmax(np.abs(seg)))
        cp = float(pvalues[ci - 1])
        row = {
            "cluster_id": ci,
            "n_vertices": nv,
            "mass": float(masses[ci - 1]),
            "peak_stat": float(seg[peak_local]),
            "peak_vertex": int(np.where(mask)[0][peak_local]),
            "cluster_p": cp,
            "significant": cp < 0.05,
            "stat_type": stat_type,
        }
        if vertex_rois is not None:
            row["region"] = format_region_coverage(
                [vertex_rois[i] for i in np.where(mask)[0] if i < len(vertex_rois)])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Per-kind adapters
# --------------------------------------------------------------------------- #
def _fit_groups(hyp: Hypothesis, spec: DesignSpec | None,
                present: set[str]) -> list[str]:
    if hyp.kind in ("omnibus", "regression"):
        wanted = hyp.groups or (spec.levels if spec and spec.levels else sorted(present))
    else:  # contrast / equivalence
        wanted = list(hyp.weights or {})
    return [g for g in wanted if g in present]


def run_hypothesis_permutation(
    hyp: Hypothesis,
    subject_maps: dict[str, np.ndarray],
    subject_groups: dict[str, str],
    coords: np.ndarray,
    *,
    n_perms: int,
    threshold: float,
    distance_mm: float,
    spec: DesignSpec | None = None,
    covariates: dict[str, np.ndarray] | None = None,
    seed: int = 42,
    vertex_rois: list[str | None] | None = None,
) -> list[dict[str, Any]]:
    """Run ONE hypothesis over per-subject vertex maps → list of cluster-row dicts.

    subject_maps: {uid -> (n_vertices,)} for a single (band, dv) cell, ALL subjects.
    subject_groups: {uid -> group}. coords: (n_vertices, 3). ``vertex_rois`` (one
    atlas-ROI name per vertex) adds a ``region`` column to each cluster row.
    """
    present = set(subject_groups.values())
    groups = _fit_groups(hyp, spec, present)
    if hyp.kind != "regression" and len(groups) < 2:
        return []

    # Assemble per-group arrays (+ optional Freedman–Lane nuisance residualisation).
    uids_by_group = {g: _group_uids(subject_groups, g) for g in groups}
    if covariates is not None:
        all_uids = [u for g in groups for u in uids_by_group[g]]
        maps = _stack(subject_maps, all_uids)
        Z = np.array([np.asarray(covariates[u], dtype=float).ravel() for u in all_uids])
        resid = _residualize(maps, Z)
        rmap = {u: resid[i] for i, u in enumerate(all_uids)}
        arrays = [_stack(rmap, uids_by_group[g]) for g in groups]
    else:
        arrays = [_stack(subject_maps, uids_by_group[g]) for g in groups]

    if hyp.kind == "omnibus":
        k = len(arrays)
        n_total = sum(a.shape[0] for a in arrays)
        f_thresh = float(stats.f.ppf(1 - _OMNIBUS_CLUSTER_ALPHA, k - 1, n_total - k))
        stat_map, labels, masses, pvals = _perm_cluster(
            arrays, _f_map, coords, f_thresh, 1, n_perms, distance_mm, seed)
        return _cluster_rows(stat_map, labels, masses, pvals, "F", vertex_rois)

    if hyp.kind == "contrast":
        w = {g: float(hyp.weights[g]) for g in groups}
        pos = [g for g, v in w.items() if v > 0]
        neg = [g for g, v in w.items() if v < 0]
        # Pairwise (two opposite-sign groups) → base test verbatim (legacy-exact).
        if len(groups) == 2 and len(pos) == 1 and len(neg) == 1:
            data_a = _stack(subject_maps if covariates is None else rmap,
                            uids_by_group[pos[0]])
            data_b = _stack(subject_maps if covariates is None else rmap,
                            uids_by_group[neg[0]])
            r = cluster_permutation_test(
                data_a, data_b, coords, n_perms=n_perms, threshold=threshold,
                tail=0, distance_mm=distance_mm, seed=seed)
            return _cluster_rows(r.t_map, r.cluster_labels, r.cluster_stats,
                                 r.cluster_pvalues, "t", vertex_rois)
        # General weighted contrast.
        weights = [w[g] for g in groups]
        stat_map, labels, masses, pvals = _perm_cluster(
            arrays, lambda arr: _weighted_t_map(arr, weights),
            coords, threshold, 0, n_perms, distance_mm, seed)
        return _cluster_rows(stat_map, labels, masses, pvals, "t", vertex_rois)

    if hyp.kind == "equivalence":
        # Per-vertex TOST (sd margin) on the pairwise contrast; summary row.
        w = {g: float(hyp.weights[g]) for g in groups}
        pos = [g for g, v in w.items() if v > 0]
        neg = [g for g, v in w.items() if v < 0]
        if not (len(groups) == 2 and pos and neg):
            return []
        a, b = arrays[groups.index(pos[0])], arrays[groups.index(neg[0])]
        diff = a.mean(0) - b.mean(0)
        sp = np.sqrt(((a.shape[0] - 1) * a.var(0, ddof=1)
                      + (b.shape[0] - 1) * b.var(0, ddof=1))
                     / (a.shape[0] + b.shape[0] - 2))
        se = sp * np.sqrt(1 / a.shape[0] + 1 / b.shape[0])
        margin_val = float((hyp.margin or {}).get("value", 0.0))
        margin = margin_val * sp  # sd mode
        df = a.shape[0] + b.shape[0] - 2
        zc = stats.t.ppf(0.95, df)
        with np.errstate(invalid="ignore"):
            equiv = (diff - zc * se > -margin) & (diff + zc * se < margin)
        n_equiv = int(np.nansum(equiv))
        row = {
            "cluster_id": 0, "n_vertices": len(diff), "mass": float("nan"),
            "peak_stat": float("nan"), "peak_vertex": -1, "cluster_p": float("nan"),
            "significant": False, "stat_type": "tost",
            "n_equivalent": n_equiv,
            "frac_equivalent": float(n_equiv / len(diff)) if len(diff) else 0.0,
        }
        if vertex_rois is not None:
            # Region of the equivalent vertices (where KO≈WT), if any.
            row["region"] = _format_equiv_region(equiv, vertex_rois)
        return [row]

    if hyp.kind == "regression":
        # Per-vertex slope on a continuous predictor + predictor-shuffle clusters.
        if covariates is None:
            return []  # predictor passed via covariates[uid] = [x]; none here
        return []  # regression on vertex maps: deferred until a predictor is wired

    return []


# --------------------------------------------------------------------------- #
# Module convenience wrapper (the permutation Tier-2 equivalent of
# write_module_hypotheses() on the R side)
# --------------------------------------------------------------------------- #
def write_module_hypotheses_perm(
    maps_by_cell: dict[tuple[str, str], dict[str, np.ndarray]],
    subject_groups: dict[str, str],
    coords: np.ndarray,
    config,
    tbl_dir,
    prefix: str,
    *,
    n_perms: int,
    threshold: float,
    distance_mm: float,
    hypothesis: str | None = None,
    seed: int = 42,
    atlas_dir=None,
):
    """Run every declared hypothesis over per-cell vertex maps; write <prefix>_hypotheses.csv.

    maps_by_cell: {(band, dv) -> {uid -> (n_vertices,)}}, ALL subjects per cell.
    ``atlas_dir`` (optional): when given, each cluster row gets a ``region``
    column naming its anatomical coverage (skip for non-source modules, e.g.
    scalp-electrode connectivity, whose coords are a montage, not brain mm).
    Returns the rows DataFrame (or None if nothing declared).
    """
    import pandas as pd
    from pathlib import Path

    spec = config.design_spec
    if spec is None or not spec.hypotheses:
        logger.info("  No hypotheses/contrasts declared — skipping %s perm hypotheses.", prefix)
        return None

    # Label vertices once (coords are shared across all hypotheses/cells).
    vertex_rois = None
    if atlas_dir is not None:
        try:
            from ..atlas.atlas_utils import label_vertices_to_rois
            vertex_rois = label_vertices_to_rois(np.asarray(coords, dtype=float), atlas_dir)
        except Exception as e:  # noqa: BLE001 — descriptive, never fatal
            logger.warning("  Vertex ROI labeling failed (%s); regions omitted", e)

    hyps = spec.hypotheses
    if hypothesis:
        want = {h.strip() for h in hypothesis.split(",")}
        hyps = [h for h in hyps if h.name in want]
        if not hyps:
            logger.info("  No declared hypothesis matches --hypothesis '%s'", hypothesis)
            return None

    rows: list[dict[str, Any]] = []
    for hyp in hyps:
        for (band, dv), subject_maps in maps_by_cell.items():
            try:
                clusters = run_hypothesis_permutation(
                    hyp, subject_maps, subject_groups, coords, spec=spec,
                    n_perms=n_perms, threshold=threshold, distance_mm=distance_mm, seed=seed,
                    vertex_rois=vertex_rois)
            except Exception as e:  # noqa: BLE001 — keep the sweep going
                logger.warning("  %s [%s/%s]: %s", hyp.name, band, dv, e)
                continue
            for c in clusters:
                rows.append({
                    "hypothesis": hyp.name, "kind": hyp.kind, "role": hyp.role,
                    "label": hyp.label or hyp.name, "test": hyp.test,
                    "band": band, "dv": dv,
                    "n_perms": n_perms, "threshold": threshold, **c,
                })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = Path(tbl_dir) / f"{prefix}_hypotheses.csv"
    df.to_csv(path, index=False)
    n_sig = int(df["significant"].sum())
    logger.info("  Saved: %s (%d rows, %d hypotheses; %d significant clusters)",
                path.name, len(df), len(hyps), n_sig)
    return df

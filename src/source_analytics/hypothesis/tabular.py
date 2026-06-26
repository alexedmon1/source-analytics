"""Tabular adapter for the ``hypothesis`` layer (Python side).

Runs a declarative :class:`~source_analytics.config.Hypothesis` over **per-subject
scalar values** laid out on a (band × spatial × facet) grid and returns a **tidy
per-cell** result — the Python counterpart of the R emmeans adapter's tabular
contract (``R/hypothesis.R``). Used by the graph-theory modules whose statistics are
computed in Python (networkx) rather than R:

  - ``roi_graph``    — per-ROI nodal metrics (degree/clustering/betweenness); the
    spatial unit is the ROI node, faceted by connectivity × graph metric.
  - ``vertex_graph`` — global multi-density AUC graph metrics; no spatial unit
    (one scalar per subject), faceted by connectivity × graph metric.

Each cell carries one scalar per subject, so every test is a **between-subjects**
contrast: a one-way model ``value ~ group`` per cell (one observation per subject —
no random term). The contrast is computed analytically from the cell means + pooled
residual SD, identical to the R ``.adapt_cell`` path (which is itself pinned to
emmeans). Multiple-comparison correction is the declarative ``fdr:`` family scope
(``hypothesis`` / ``band`` / ``spatial`` / ``none`` × BH/BY/holm/bonferroni), applied
independently within each (hypothesis × facet) family — mirroring the R side, where
each dv is a separate ``run_hypothesis`` call.

Kinds: ``omnibus`` (group F, partial ω²), ``contrast`` (weighted group contrast,
Hedges g), ``equivalence`` (TOST, sd margin). ``regression`` is not supported here
(no per-cell continuous predictor contract).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as st

from ..config import DesignSpec, Hypothesis
from .permutation import _fit_groups

logger = logging.getLogger(__name__)

_FDR_SCOPES = ("hypothesis", "band", "spatial", "none")


# --------------------------------------------------------------------------- #
# Multiple-comparison correction (mirror of R p.adjust + .apply_fdr scopes)
# --------------------------------------------------------------------------- #
def _p_adjust(p: np.ndarray, method: str) -> np.ndarray:
    """R ``p.adjust`` for BH / BY / holm / bonferroni / none, NaN-safe."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return out
    m = method.lower()
    if m in ("none", "fdr_none"):
        adj = pv
    elif m == "bonferroni":
        adj = np.minimum(1.0, pv * n)
    elif m == "holm":
        order = np.argsort(pv)
        adj_sorted = np.minimum(1.0, np.maximum.accumulate((n - np.arange(n)) * pv[order]))
        adj = np.empty(n)
        adj[order] = adj_sorted
    elif m in ("bh", "fdr"):
        order = np.argsort(pv)
        ranked = pv[order] * n / (np.arange(n) + 1)
        adj_sorted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
        adj = np.empty(n)
        adj[order] = adj_sorted
    elif m == "by":
        order = np.argsort(pv)
        cm = float(np.sum(1.0 / (np.arange(n) + 1)))
        ranked = pv[order] * n * cm / (np.arange(n) + 1)
        adj_sorted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
        adj = np.empty(n)
        adj[order] = adj_sorted
    else:
        raise ValueError(f"unknown fdr method '{method}'")
    out[ok] = adj
    return out


def _resolve_fdr(hyp: Hypothesis, spec: DesignSpec | None) -> tuple[str, str]:
    sp = (spec.fdr if spec else None) or {}
    hp = hyp.fdr or {}
    method = hp.get("method") or sp.get("method") or "BH"
    scope = hp.get("scope") or sp.get("scope") or "hypothesis"
    if scope not in _FDR_SCOPES:
        raise ValueError(f"fdr scope '{scope}' not one of {_FDR_SCOPES}")
    return method, scope


def _apply_fdr(rows: list[dict], method: str, scope: str) -> None:
    """Set ``q_value`` / ``significant`` / ``fdr_family`` on rows in place.

    ``scope`` partitions the (band × spatial) rows into the families across which
    p-values are corrected (family SIZE drives aggressiveness, not just method).
    """
    def fam_key(r: dict) -> str:
        if scope == "band":
            return f"band={r.get('band')}"
        if scope == "spatial":
            return f"spatial={r.get('spatial')}"
        if scope == "none":
            return f"row={id(r)}"
        return "all"

    families: dict[str, list[dict]] = {}
    for r in rows:
        families.setdefault(fam_key(r), []).append(r)
    for key, fam in families.items():
        p = np.array([r["p_value"] for r in fam], dtype=float)
        q = _p_adjust(p, method)
        n = int(np.sum(~np.isnan(p)))
        for r, qv in zip(fam, q):
            r["q_value"] = float(qv) if not np.isnan(qv) else float("nan")
            r["significant"] = bool(not np.isnan(qv) and qv < 0.05)
            r["fdr_family"] = (
                f"scope=none method={method}" if scope == "none"
                else f"scope={scope} method={method} family={key} n={n}"
            )


# --------------------------------------------------------------------------- #
# Per-cell kind adapter (mirror of R .adapt_cell — between-subjects one-way model)
# --------------------------------------------------------------------------- #
def _adapt_cell(by_group: dict[str, np.ndarray], hyp: Hypothesis,
                groups: list[str]) -> dict[str, Any] | None:
    arrays = []
    for g in groups:
        a = np.asarray(by_group.get(g, []), dtype=float)
        a = a[np.isfinite(a)]
        arrays.append(a)
    ns = np.array([a.size for a in arrays])
    if np.any(ns < 1):
        return None
    k = len(arrays)
    N = int(ns.sum())
    df_res = N - k
    if df_res < 1:
        return None
    means = np.array([a.mean() for a in arrays])
    sse = float(sum(((a - a.mean()) ** 2).sum() for a in arrays))
    resid_sd = float(np.sqrt(sse / df_res)) if df_res > 0 else float("nan")

    if hyp.kind == "omnibus":
        grand = float(np.concatenate(arrays).mean())
        ss_between = float(np.sum(ns * (means - grand) ** 2))
        ms_within = sse / df_res if df_res > 0 else float("nan")
        df_num = k - 1
        f = (ss_between / df_num) / ms_within if (df_num > 0 and ms_within > 0) else float("nan")
        p = float(st.f.sf(f, df_num, df_res)) if np.isfinite(f) else float("nan")
        # partial omega^2 from F (matches effectsize::F_to_omega2)
        omega2 = (df_num * (f - 1)) / (df_num * (f - 1) + N) if np.isfinite(f) else float("nan")
        omega2 = float(max(0.0, omega2)) if np.isfinite(omega2) else float("nan")
        return {
            "spatial": None, "estimate": float("nan"), "SE": float("nan"),
            "df": float(df_res), "df_num": float(df_num), "stat": float(f),
            "stat_type": "F", "p_value": p, "estimate_lcl": float("nan"),
            "estimate_ucl": float("nan"), "effect_size": omega2,
            "effect_size_type": "omega2_partial", "group_a": None, "group_b": None,
        }

    # contrast / equivalence
    weights = hyp.weights or {}
    wv = np.array([float(weights.get(g, 0.0)) for g in groups])
    est = float((wv * means).sum())
    se = float(resid_sd * np.sqrt(np.sum(wv ** 2 / ns)))
    t = est / se if se > 0 else float("nan")
    p = float(2 * st.t.sf(abs(t), df_res)) if np.isfinite(t) else float("nan")
    tcrit = float(st.t.ppf(0.975, df_res))
    pos = [g for g in groups if weights.get(g, 0.0) > 0]
    neg = [g for g in groups if weights.get(g, 0.0) < 0]
    row = {
        "spatial": None, "estimate": est, "SE": se, "df": float(df_res),
        "df_num": 1.0, "stat": float(t), "stat_type": "t", "p_value": p,
        "estimate_lcl": est - tcrit * se, "estimate_ucl": est + tcrit * se,
        "effect_size": est / resid_sd if resid_sd > 0 else float("nan"),
        "effect_size_type": "hedges_g",
        "group_a": pos[0] if len(pos) == 1 else None,
        "group_b": neg[0] if len(neg) == 1 else None,
    }
    if hyp.kind == "equivalence":
        margin = hyp.margin or {}
        mode = margin.get("mode", "sd")
        val = float(margin.get("value", 0.0))
        m = val * resid_sd if mode == "sd" else float("nan")
        zc = float(st.t.ppf(0.95, df_res))
        equivalent = bool(np.isfinite(m) and (est - zc * se > -m) and (est + zc * se < m))
        row["stat_type"] = "tost"
        row["margin_used"] = float(m)
        row["equivalent"] = equivalent
    return row


# --------------------------------------------------------------------------- #
# Module convenience wrapper (the tabular Tier-2 equivalent of
# write_module_hypotheses() / write_module_directed_edges() on the R side)
# --------------------------------------------------------------------------- #
def write_module_hypotheses_tabular(
    df,
    config,
    tbl_dir,
    prefix: str,
    *,
    value_col: str,
    spatial_col: str | None = None,
    facet_cols: tuple[str, ...] = (),
    band_col: str = "band",
    hypothesis: str | None = None,
    min_per_group: int = 3,
):
    """Run every declared hypothesis over a long scalar table; write ``<prefix>_hypotheses.csv``.

    ``df`` columns: ``subject``, the design factor (``config.design_spec.factor``),
    ``value_col``, ``band_col``, optional ``spatial_col``, and any ``facet_cols``.
    One row per (hypothesis × band × spatial × facet) cell. FDR is applied per
    (hypothesis × facet) family across the band × spatial grid, per the declared
    scope. Returns the rows DataFrame (or None if nothing was produced).
    """
    import pandas as pd

    spec = config.design_spec
    if spec is None or not spec.hypotheses:
        logger.info("  No hypotheses/contrasts declared — skipping %s tabular hypotheses.", prefix)
        return None

    factor = spec.factor
    hyps = spec.hypotheses
    if hypothesis:
        want = {h.strip() for h in hypothesis.split(",")}
        hyps = [h for h in hyps if h.name in want]
        if not hyps:
            logger.info("  No declared hypothesis matches --hypothesis '%s'", hypothesis)
            return None
    hyps = [h for h in hyps if h.kind != "regression"]

    present = set(df[factor].astype(str))
    facet_cols = tuple(facet_cols)
    spatial_vals = (
        sorted(df[spatial_col].astype(str).unique()) if spatial_col else [None]
    )
    bands = sorted(df[band_col].astype(str).unique()) if band_col in df.columns else [None]

    all_rows: list[dict] = []
    for hyp in hyps:
        groups = _fit_groups(hyp, spec, present)
        if len(groups) < 2:
            continue
        # weights may reference levels absent from the data → skip those hyps
        if hyp.kind in ("contrast", "equivalence"):
            need = [g for g, w in (hyp.weights or {}).items() if w != 0]
            if not all(g in groups for g in need):
                continue

        # Independent FDR family per facet combination (mirrors R's per-dv call).
        facet_keys = (
            df[list(facet_cols)].drop_duplicates().itertuples(index=False, name=None)
            if facet_cols else [()]
        )
        for fkey in facet_keys:
            fmask = np.ones(len(df), dtype=bool)
            for col, val in zip(facet_cols, fkey):
                fmask &= (df[col].astype(type(val)) == val) if facet_cols else fmask
            fdf = df[fmask] if facet_cols else df

            fam_rows: list[dict] = []
            for band in bands:
                bdf = fdf[fdf[band_col].astype(str) == band] if band is not None else fdf
                for sp in spatial_vals:
                    cell = bdf[bdf[spatial_col].astype(str) == sp] if spatial_col else bdf
                    by_group = {
                        g: cell.loc[cell[factor].astype(str) == g, value_col].to_numpy()
                        for g in groups
                    }
                    if any(np.isfinite(v).sum() < min_per_group for v in by_group.values()):
                        continue
                    res = _adapt_cell(by_group, hyp, groups)
                    if res is None:
                        continue
                    res["spatial"] = sp
                    res["band"] = band
                    res["hypothesis"] = hyp.name
                    res["kind"] = hyp.kind
                    res["role"] = hyp.role
                    res["label"] = hyp.label or hyp.name
                    res["test"] = hyp.test
                    for col, val in zip(facet_cols, fkey):
                        res[col] = val
                    fam_rows.append(res)

            if fam_rows:
                method, scope = _resolve_fdr(hyp, spec)
                _apply_fdr(fam_rows, method, scope)
                all_rows.extend(fam_rows)

    if not all_rows:
        logger.info("  %s: no tabular hypothesis rows produced.", prefix)
        return None

    out = pd.DataFrame(all_rows)
    # Legacy-schema aliases (lightbox / figure consumers read these names).
    out["contrast"] = out["hypothesis"]
    if "spatial" in out:
        out["roi"] = out["spatial"]
    out["t_ratio"] = np.where(out["stat_type"] == "t", out["stat"], np.nan)
    out["hedges_g"] = np.where(out["effect_size_type"] == "hedges_g", out["effect_size"], np.nan)
    out["p_fdr"] = out["q_value"]

    front = ["hypothesis", "kind", "role", "band", "spatial", *facet_cols]
    out = out[[c for c in front if c in out.columns]
              + [c for c in out.columns if c not in front]]

    path = Path(tbl_dir) / f"{prefix}_hypotheses.csv"
    out.to_csv(path, index=False)
    n_sig = int(out["significant"].sum())
    logger.info("  Saved: %s (%d rows, %d hypotheses; %d sig cells)",
                path.name, len(out), len(hyps), n_sig)
    return out

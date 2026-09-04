"""Shared base for the connectivity *network* layer (graph theory + NBS).

The network layer is really two independent analyses on the same connectivity
matrices:

* **graph theory** — nodal/global graph metrics (ROI) or the multi-density AUC
  sweep (vertex), and
* **NBS** — the Network-Based Statistic sub-network permutation test.

They share nothing but the connectivity matrices, so they are split into separate
analyses (``*_graph`` and ``*_nbs``) that can run independently or in parallel.
A combined ``*_network`` alias is kept for back-compat (same outputs as before).

This base holds the piece both halves share: parsing the connectivity-metric +
NBS config (with fall-back to the combined ``*_network`` block) and running the
NBS permutation test over ``self._conn_matrices[uid][band][metric]`` — the matrix
layout the ROI and vertex levels both populate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..atlas.atlas_utils import format_region_coverage
from ..stats.graph_metrics import nbs_permutation_test
from .base import BaseAnalysis

logger = logging.getLogger(__name__)

# Connectivity metrics whose matrices are asymmetric/directional. The graph + NBS
# layer is undirected (upper-triangle threshold + symmetrize), so these are
# excluded from it. dPLI is centered at 0.5 (no lead/lag preference), not 0.
_DIRECTED_METRICS = frozenset({"dpli"})


class NetworkAnalysisBase(BaseAnalysis):
    """Common config + NBS execution for graph/NBS/combined network analyses."""

    SELECTABLE = {"metric": "connectivity metric", "band": "frequency band",
                  "hypothesis": "declared hypothesis"}

    # Subclasses override these.
    _default_nbs_threshold: float = 2.5
    _nbs_results_filename: str = "nbs_results.csv"
    # Config block to read first, before this analysis's own block overrides it.
    # The split analyses point this at their combined ``*_network`` block so an
    # existing config drives them with no extra YAML.
    _fallback_config_key: str | None = None

    def _init_network_config(self) -> None:
        """Populate connectivity-metric + NBS settings and the matrix stores.

        Only the *running* analysis's block is lifted to ``config.raw[name]``;
        the combined ``*_network`` block (used as a fall-back so an existing
        config drives the split analyses) is read from the paradigm's
        ``analyses`` map, which is present in ``raw``.
        """
        analyses = self.config.raw.get("analyses", {}) or {}
        cfg: dict = {}
        if self._fallback_config_key:
            cfg.update(analyses.get(self._fallback_config_key, {}) or {})
        cfg.update(analyses.get(self.name, {}) or {})
        cfg.update(self.config.raw.get(self.name, {}) or {})  # lifted own block wins
        self._net_cfg = cfg

        # Network layer runs on every listed connectivity metric; fall back to the
        # legacy single ``metric`` key.
        self._connectivity_metrics = cfg.get("connectivity_metrics") or [
            cfg.get("metric", "imag_coherence")
        ]
        # Restrict to --metric / --select metric=... if requested.
        self._connectivity_metrics = self._select("metric", self._connectivity_metrics)
        # The graph/NBS layer is undirected: it thresholds the upper triangle and
        # symmetrizes (nx.from_numpy_array / np.triu+T). Directed metrics (dPLI,
        # centered at 0.5) would yield meaningless graphs, so drop them with a
        # warning rather than silently produce garbage.
        directed = [m for m in self._connectivity_metrics if m in _DIRECTED_METRICS]
        if directed:
            logger.warning(
                "%s: directed metric(s) %s are not valid for the undirected "
                "graph/NBS layer — skipping. (Directed network analysis is a "
                "separate, future capability.)",
                self.name, directed,
            )
            self._connectivity_metrics = [
                m for m in self._connectivity_metrics if m not in _DIRECTED_METRICS
            ]
        self._nbs_threshold = float(cfg.get("nbs_threshold", self._default_nbs_threshold))
        self._nbs_permutations = int(cfg.get("nbs_permutations", 5000))

        # uid -> band -> conn_metric -> connectivity matrix (filled by subclasses)
        self._conn_matrices: dict[str, dict[str, dict]] = {}
        self._subject_groups: dict[str, str] = {}
        self._nbs_results: dict = {}

    # ----------------------------------------------------------------- NBS --- #
    def _run_nbs(self) -> None:
        """Run NBS for every contrast × band × connectivity metric, then export.

        Keys are ``{contrast}_{band}_{metric}`` so downstream tools (e.g. the
        gallery's NBS renderer) can facet by connectivity metric.
        """
        self._nbs_results = {}
        for contrast in self._pairwise_contrasts():
            group_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            group_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not group_a or not group_b:
                continue
            for band_name in self._selected_bands():
                for metric in self._connectivity_metrics:
                    mats_a = [
                        self._conn_matrices[u][band_name][metric] for u in group_a
                        if metric in self._conn_matrices.get(u, {}).get(band_name, {})
                    ]
                    mats_b = [
                        self._conn_matrices[u][band_name][metric] for u in group_b
                        if metric in self._conn_matrices.get(u, {}).get(band_name, {})
                    ]
                    if mats_a and mats_b:
                        self._nbs_results[f"{contrast.name}_{band_name}_{metric}"] = (
                            nbs_permutation_test(
                                mats_a, mats_b,
                                nbs_threshold=self._nbs_threshold,
                                n_permutations=self._nbs_permutations,
                                seed=42,
                            )
                        )

        # Per-vertex ROI labels, so each subnetwork can report the regions of its
        # participating nodes (vertex modules only; ROI-NBS nodes are ROIs already).
        vertex_rois = self._label_vertex_regions(getattr(self, "_source_coords", None))

        rows = []
        thr = getattr(self, "_nbs_threshold", 0.0)
        for key, nbs in self._nbs_results.items():
            nodes_per_comp = getattr(nbs, "component_nodes", []) or []
            tmat = getattr(nbs, "t_matrix", None)
            for i, (size, pval) in enumerate(
                zip(nbs.component_sizes, nbs.component_pvalues)
            ):
                row = {
                    "key": key, "component": i + 1,
                    "n_edges": size, "p_corrected": pval,
                }
                if i < len(nodes_per_comp):
                    nodes = nodes_per_comp[i]
                    # Edge-direction split (NBS thresholds |t|, so a subnetwork can
                    # be up-, down-, or mixed-regulation; t>0 = group A > group B).
                    if tmat is not None:
                        sub = tmat[np.ix_(nodes, nodes)]
                        ev = sub[np.triu_indices(len(nodes), k=1)]
                        ev = ev[np.abs(ev) > thr]
                        n_pos, n_neg = int((ev > 0).sum()), int((ev < 0).sum())
                        row["n_edges_increase"] = n_pos
                        row["n_edges_decrease"] = n_neg
                        row["direction"] = ("increase" if n_pos and not n_neg else
                                            "decrease" if n_neg and not n_pos else "mixed")
                    if any(r is not None for r in vertex_rois):
                        row["region"] = format_region_coverage(
                            [vertex_rois[v] for v in nodes if v < len(vertex_rois)])
                rows.append(row)
        if rows:
            out = self.tbl_dir / self._nbs_results_filename
            pd.DataFrame(rows).to_csv(out, index=False)
            logger.info("Exported %s (%d rows)", self._nbs_results_filename, len(rows))

    # ------------------------------------------- declarative hypotheses --- #
    def _run_nbs_hypotheses(self) -> None:
        """Run declared design:/hypotheses: over the connectivity matrices.

        Additive: writes ``<name>_hypotheses.csv`` (the edge/NBS subnetwork
        contract) alongside the legacy ``*_nbs_results.csv`` left by
        :meth:`_run_nbs`. A pairwise contrast reproduces that legacy NBS
        bit-exact; omnibus/general-weighted contrasts add the >2-group tests
        the legacy per-pair path cannot express. No-op when nothing is declared.
        """
        from ..hypothesis import write_module_hypotheses_edge

        if not self._conn_matrices or not self._subject_groups:
            return
        matrices_by_cell: dict[tuple[str, str], dict] = {}
        for band_name in self._selected_bands():
            for metric in self._connectivity_metrics:
                cell = {
                    uid: self._conn_matrices[uid][band_name][metric]
                    for uid in self._subject_groups
                    if metric in self._conn_matrices.get(uid, {}).get(band_name, {})
                }
                if cell:
                    matrices_by_cell[(band_name, metric)] = cell
        if not matrices_by_cell:
            return

        wanted = self._selection.get("hypothesis")
        write_module_hypotheses_edge(
            matrices_by_cell, self._subject_groups, self.config, self.tbl_dir,
            prefix=self.name, nbs_threshold=self._nbs_threshold,
            n_perms=self._nbs_permutations,
            hypothesis=",".join(sorted(wanted)) if wanted else None, seed=42,
            coords=getattr(self, "_source_coords", None), atlas_dir=self._atlas_dir,
            # ROI modules: name the matrix indices so the subnetwork-edge sidecar
            # carries roi_i / roi_j (vertex modules have no per-node names).
            node_labels=list(getattr(self, "_roi_labels", None) or []) or None,
        )

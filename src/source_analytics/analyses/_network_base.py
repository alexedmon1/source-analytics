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

import pandas as pd

from ..stats.graph_metrics import nbs_permutation_test
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


class NetworkAnalysisBase(BaseAnalysis):
    """Common config + NBS execution for graph/NBS/combined network analyses."""

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
        for contrast in self.config.contrasts:
            group_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            group_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not group_a or not group_b:
                continue
            for band_name in self.config.bands:
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

        rows = []
        for key, nbs in self._nbs_results.items():
            for i, (size, pval) in enumerate(
                zip(nbs.component_sizes, nbs.component_pvalues)
            ):
                rows.append({
                    "key": key, "component": i + 1,
                    "n_edges": size, "p_corrected": pval,
                })
        if rows:
            out = self.tbl_dir / self._nbs_results_filename
            pd.DataFrame(rows).to_csv(out, index=False)
            logger.info("Exported %s (%d rows)", self._nbs_results_filename, len(rows))

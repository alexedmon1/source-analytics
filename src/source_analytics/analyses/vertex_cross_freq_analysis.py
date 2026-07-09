"""Vertex-level cross-frequency coupling: local PAC maps + AAC/PPC (full resolution).

Vertex mirror of ``roi_cross_freq``. ``--metric`` selects the measure; all run at
**full vertex resolution** (no parcellation — see MS2_CONNECTIVITY_PLAN.md D1):

- **pac** — local within-vertex PAC z-map (one Modulation Index per vertex per
  cross-frequency band pair → whole-brain map). The Tier-A measure for the
  source spatial advantage in cross-frequency coupling.
- **aac** — cross-frequency power–power coupling, vertex×vertex per band pair.
- **ppc** — n:m phase–phase coupling (PLF + surrogate z), vertex×vertex per pair.

For AAC/PPC the per-vertex statistical map is the node coupling strength (mean
off-diagonal); the full matrices are stored for downstream edge-level analysis.
Group differences in the per-vertex maps are tested with cluster-based
permutation testing. References: see CONNECTIVITY_METHODS.md.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.pac import get_valid_pac_pairs, compute_local_pac_vertices
from ..spectral.cross_freq import compute_aac, compute_ppc
from .roi_cross_freq_analysis import _nm_ratio
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_band_comparison
from pathlib import Path
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _node_strength(mat: np.ndarray) -> np.ndarray:
    """Per-node coupling strength = mean off-diagonal value over the row."""
    n = mat.shape[0]
    if n <= 1:
        return np.zeros(n)
    off = mat.copy()
    np.fill_diagonal(off, 0.0)
    return off.sum(axis=1) / (n - 1)


class VertexCrossFreqAnalysis(BaseAnalysis):
    """Vertex-level cross-frequency coupling (local PAC, AAC, n:m PPC)."""

    name = "vertex_cross_freq"
    SELECTABLE = {"metric": "coupling measure", "band": "frequency band",
                  "hypothesis": "declared hypothesis"}

    _CROSS_FREQ_METRICS = ["pac", "aac", "ppc"]

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._map_rows: list[dict] = []
        self._sfreq: float | None = None
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._subject_groups: dict[str, str] = {}
        # uid -> "{metric}|{freq_pair}" -> per-vertex map
        self._subject_maps: dict[str, dict[str, np.ndarray]] = {}
        # uid -> "{metric}|{freq_pair}" -> full matrix (aac/ppc only)
        self._matrices: dict[str, dict[str, np.ndarray]] = {}
        self._cluster_results: dict = {}

        cfg = config.raw.get(self.name, {})
        self._metrics: list[str] = list(self._CROSS_FREQ_METRICS)
        self._ppc_surrogates = int(cfg.get("ppc_surrogates", 100))
        self._pac_surrogates = int(cfg.get("pac_surrogates", 100))
        self._n_permutations = int(cfg.get("n_permutations", 1000))
        wb_cfg = config.vertex
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))

    def setup(self) -> None:
        self._metrics = self._select("metric", self._CROSS_FREQ_METRICS)
        self._map_rows.clear()
        self._subject_groups.clear()
        self._subject_maps.clear()
        self._matrices.clear()
        self._cluster_results.clear()
        self._source_coords = None
        self._vertex_indices = None

    # ------------------------------------------------------------ processing
    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses()
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()
        if self._sfreq is None:
            self._sfreq = sfreq

        if self._vertex_indices is None:
            mask = self.config.get_vertex_mask(coords)
            self._vertex_indices = np.where(mask)[0]
            self._source_coords = coords[mask]
            if self.config.has_vertex_filter:
                logger.info("Vertex filter: %d/%d vertices retained",
                            len(self._vertex_indices), len(coords))
        stc_data = stc_data[self._vertex_indices]

        pairs = get_valid_pac_pairs(self._selected_bands())
        if not pairs:
            logger.warning("No valid cross-frequency band pairs for %s", subject.subject_id)
            return

        self._subject_groups[uid] = subject.group
        self._subject_maps.setdefault(uid, {})
        self._matrices.setdefault(uid, {})
        bands = self.config.bands

        for phase_band, amp_band in pairs:
            band_x, band_y = bands[phase_band], bands[amp_band]
            freq_pair = f"{phase_band}-{amp_band}"

            for metric in self._metrics:
                if metric == "pac":
                    z, _mi = compute_local_pac_vertices(
                        stc_data, sfreq, band_x, band_y,
                        n_surrogates=self._pac_surrogates,
                    )
                    vmap = z
                elif metric == "aac":
                    mat = compute_aac(stc_data, sfreq, band_x, band_y)
                    self._matrices[uid][f"aac|{freq_pair}"] = mat
                    vmap = _node_strength(mat)
                else:  # ppc
                    if self._ppc_surrogates > 0:
                        plf, _z = compute_ppc(
                            stc_data, sfreq, band_x, band_y,
                            n=_nm_ratio(band_x, band_y)[0], m=1,
                            n_surrogates=self._ppc_surrogates, seed=42,
                        )
                    else:
                        plf = compute_ppc(stc_data, sfreq, band_x, band_y,
                                          n=_nm_ratio(band_x, band_y)[0], m=1)
                    self._matrices[uid][f"ppc|{freq_pair}"] = plf
                    vmap = _node_strength(plf)

                self._subject_maps[uid][f"{metric}|{freq_pair}"] = vmap
                for vi in range(len(vmap)):
                    self._map_rows.append({
                        "subject": uid, "group": subject.group,
                        "vertex_idx": int(self._vertex_indices[vi]),
                        "metric": metric, "freq_pair": freq_pair,
                        "value": float(vmap[vi]),
                    })

    # ----------------------------------------------------------- aggregate
    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"
        if not self._map_rows:
            logger.warning("No vertex cross-frequency data collected")
            return
        pd.DataFrame(self._map_rows).to_csv(
            data_dir / "vertex_cross_freq_maps.csv", index=False)
        logger.info("Exported vertex_cross_freq_maps.csv (%d rows)", len(self._map_rows))

        if self._source_coords is not None:
            cdf = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            cdf.index.name = "vertex_idx"
            cdf.to_csv(data_dir / "source_coords.csv")
        if any(self._matrices.values()):
            with open(data_dir / "vertex_cross_freq_matrices.pkl", "wb") as f:
                pickle.dump(self._matrices, f)
            logger.info("Saved AAC/PPC matrices pkl")

    # ----------------------------------------------------------- statistics
    def _reload_maps_from_disk(self) -> bool:
        """Reconstruct per-subject CFC maps + coords + groups from persisted CSVs
        so statistics/figures are regenerable via --steps (no reprocessing)."""
        data_dir = self.output_dir / "data"
        maps_csv = data_dir / "vertex_cross_freq_maps.csv"
        coords_csv = data_dir / "source_coords.csv"
        if not maps_csv.exists():
            logger.warning("No persisted maps at %s; cannot reload", maps_csv)
            return False
        if coords_csv.exists():
            self._source_coords = pd.read_csv(coords_csv)[["x", "y", "z"]].to_numpy(dtype=float)
        df = pd.read_csv(maps_csv)
        self._subject_maps = {}
        self._subject_groups = {}
        for (uid, group), g in df.groupby(["subject", "group"], sort=False):
            self._subject_groups[uid] = group
            m: dict = {}
            for (metric, freq_pair), gg in g.groupby(["metric", "freq_pair"], sort=False):
                gg = gg.sort_values("vertex_idx")
                m[f"{metric}|{freq_pair}"] = gg["value"].to_numpy(dtype=float)
            self._subject_maps[uid] = m
        logger.info("Reloaded %d subjects' CFC maps from %s",
                    len(self._subject_maps), maps_csv)
        return True

    def statistics(self) -> None:
        if not self._subject_maps:
            self._reload_maps_from_disk()
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run statistics")
            return
        coords = self._source_coords
        all_stats = []

        # keys are "{metric}|{freq_pair}" — test each across each contrast
        keys = sorted({k for m in self._subject_maps.values() for k in m})
        for contrast in self._pairwise_contrasts():
            uids_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            uids_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not uids_a or not uids_b:
                continue
            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            for key in keys:
                metric, freq_pair = key.split("|", 1)
                data_a = np.array([self._subject_maps[u][key] for u in uids_a
                                   if key in self._subject_maps.get(u, {})])
                data_b = np.array([self._subject_maps[u][key] for u in uids_b
                                   if key in self._subject_maps.get(u, {})])
                if data_a.size == 0 or data_b.size == 0:
                    continue

                result = cluster_permutation_test(
                    data_a, data_b, coords,
                    n_perms=self._n_permutations,
                    threshold=self._cluster_threshold,
                    distance_mm=self._adjacency_distance, seed=42,
                )
                g_map = hedges_g(data_a, data_b)
                self._cluster_results[f"{contrast.name}_{key}"] = {
                    "result": result, "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0), "group_labels": (label_a, label_b),
                    "metric": metric, "freq_pair": freq_pair,
                }
                for vi in range(len(result.t_map)):
                    all_stats.append({
                        "contrast": contrast.name, "metric": metric,
                        "freq_pair": freq_pair, "vertex_idx": vi,
                        "value_a": float(data_a.mean(axis=0)[vi]),
                        "value_b": float(data_b.mean(axis=0)[vi]),
                        "t": float(result.t_map[vi]), "p": float(result.p_map[vi]),
                        "hedges_g": float(g_map[vi]),
                        "cluster_id": int(result.cluster_labels[vi]),
                    })

        if all_stats:
            pd.DataFrame(all_stats).to_csv(
                self.tbl_dir / "vertex_cross_freq_stats.csv", index=False)
            logger.info("Exported vertex_cross_freq_stats.csv (%d rows)", len(all_stats))

        # --- Declarative hypotheses (hypothesis layer; additive, map+cluster) ---
        from ..hypothesis import write_module_hypotheses_perm

        if self._subject_groups:
            # cells keyed by (freq_pair, metric); the per-vertex coupling map is
            # the unit-of-test, same contract as vertex_connectivity FCD.
            maps_by_cell: dict[tuple[str, str], dict] = {}
            for key in keys:
                metric, freq_pair = key.split("|", 1)
                cell = {uid: m[key] for uid, m in self._subject_maps.items() if key in m}
                if cell:
                    maps_by_cell[(freq_pair, metric)] = cell
            wanted_hyp = self._selection.get("hypothesis")
            write_module_hypotheses_perm(
                maps_by_cell, self._subject_groups, coords, self.config,
                self.tbl_dir, prefix="vertex_cross_freq",
                n_perms=self._n_permutations, threshold=self._cluster_threshold,
                distance_mm=self._adjacency_distance,
                hypothesis=",".join(sorted(wanted_hyp)) if wanted_hyp else None,
            )

        self._save_cluster_state()

    def figures(self) -> None:
        if not self._cluster_results:
            self._load_cluster_state()
        if self._source_coords is None:
            return
        coords = self._source_coords
        for key, info in self._cluster_results.items():
            result = info["result"]
            metric, freq_pair = info["metric"], info["freq_pair"]
            safe = f"{metric}_{freq_pair}".lower().replace(" ", "_")
            plot_band_comparison(
                coords=coords, mean_a=info["mean_a"], mean_b=info["mean_b"],
                t_map=result.t_map, cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"{metric.upper()} — {freq_pair}",
                group_labels=info["group_labels"],
                output_path=self.fig_dir / f"cfc_{safe}.png",
            )

    def summary(self) -> None:
        data_dir = self.output_dir / "data"
        cfg = dict(self.config.raw)
        if self._sfreq is not None:
            cfg["sfreq"] = self._sfreq
        with open(data_dir / "study_config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        lines = [
            "# Vertex Cross-Frequency Coupling Summary", "",
            f"**Study**: {self.config.name}",
            f"**Metrics**: {', '.join(self._metrics)}",
            f"**Resolution**: full vertex (no parcellation)", "",
            "## Methods", "",
            "Cross-frequency coupling at full vertex resolution. PAC = local "
            "within-vertex Modulation Index z-map (Tort 2010); AAC = cross-"
            "frequency power-power coupling (Bruns 2000 / Masimore 2004); PPC = "
            "n:m phase-phase coupling with surrogate z (Palva 2005). Per-vertex "
            "maps tested with cluster-based permutation. See CONNECTIVITY_METHODS.md.",
            "",
        ]
        (self.tbl_dir / "vertex_cross_freq_summary.md").write_text("\n".join(lines))

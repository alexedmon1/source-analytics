"""Vertex-level directed connectivity: DTF outflow / inflow / net flow.

The vertex companion to :class:`ROIDirectedAnalysis`. Fits one ridge-regularized
MVAR per subject over the dorsal source vertices and reads out the directed
transfer function (DTF), then reduces the directed matrix to three per-vertex
summary maps that are tested with the same spatial cluster-permutation machinery
as the other vertex modules:

  - **outflow**  — mean DTF *from* a vertex to the rest (causal-source / driver strength)
  - **inflow**   — mean DTF *into* a vertex from the rest (receiver strength)
  - **netflow**  — outflow − inflow (positive = net driver, negative = net receiver)

Ridge regularization is mandatory here: dorsal source vertices are strongly
collinear (mean inter-vertex |corr| ≈ 0.64 from source leakage), which makes the
plain least-squares MVAR explosive — see :mod:`..spectral.directed`.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.directed import (
    fit_mvar,
    dtf_spectrum,
    mvar_spectral_radius,
    DEFAULT_ORDER,
    DEFAULT_RIDGE,
)
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_band_comparison
from .base import BaseAnalysis

logger = logging.getLogger(__name__)

_MEASURES = ["outflow", "inflow", "netflow"]


class VertexDirectedAnalysis(BaseAnalysis):
    """All-to-all vertex DTF reduced to per-vertex outflow/inflow/netflow maps."""

    name = "vertex_directed"
    SELECTABLE = {"measure": "directed summary", "band": "frequency band"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._sfreq: float | None = None
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        # uid -> band -> measure -> per-vertex array
        self._subject_maps: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._subject_groups: dict[str, str] = {}
        # uid -> band -> directed matrix
        self._dtf_matrices: dict[str, dict[str, np.ndarray]] = {}
        self._rows: list[dict] = []
        self._cluster_results: dict = {}

        cfg = config.raw.get(self.name, {})
        self._mvar_order = int(cfg.get("mvar_order", DEFAULT_ORDER))
        self._mvar_ridge = float(cfg.get("mvar_ridge", DEFAULT_RIDGE))
        self._n_freqs = int(cfg.get("n_freqs", 128))
        self._n_permutations = int(cfg.get("n_permutations", 1000))
        self._measures = list(_MEASURES)

        wb_cfg = config.vertex
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))

    def setup(self) -> None:
        self._measures = self._select("measure", _MEASURES)
        self._subject_maps.clear()
        self._subject_groups.clear()
        self._dtf_matrices.clear()
        self._rows.clear()
        self._cluster_results.clear()
        self._source_coords = None
        self._vertex_indices = None

    @staticmethod
    def _reduce(mat: np.ndarray) -> dict[str, np.ndarray]:
        """Per-vertex directed summaries from a DTF matrix (mat[i,j] = j -> i)."""
        n = mat.shape[0]
        m = mat.copy()
        np.fill_diagonal(m, 0.0)
        inflow = m.sum(axis=1) / (n - 1)    # row i: total inflow to vertex i
        outflow = m.sum(axis=0) / (n - 1)   # col i: total outflow from vertex i
        return {"outflow": outflow, "inflow": inflow, "netflow": outflow - inflow}

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses()  # signed
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()
        if self._sfreq is None:
            self._sfreq = sfreq

        if self._vertex_indices is None:
            mask = self.config.get_vertex_mask(coords)
            self._vertex_indices = np.where(mask)[0]
            self._source_coords = coords[mask]
            if self.config.has_vertex_filter:
                logger.info(
                    "Vertex filter: %d/%d vertices retained",
                    len(self._vertex_indices), len(coords),
                )
        stc_data = stc_data[self._vertex_indices]

        # One ridge-MVAR fit per subject; DTF read out across all bands.
        A, _ = fit_mvar(stc_data, order=self._mvar_order, ridge=self._mvar_ridge)
        radius = mvar_spectral_radius(A)
        if radius >= 1.0:
            logger.warning(
                "%s: MVAR unstable (spectral radius %.2f) — DTF unreliable; "
                "raise vertex_directed.mvar_ridge or lower mvar_order.", uid, radius,
            )
        freqs = np.linspace(0, sfreq / 2, self._n_freqs)
        dtf = dtf_spectrum(A, sfreq, freqs)  # (n, n, n_freqs), [i,j] = j -> i

        self._subject_groups[uid] = subject.group
        subj_maps: dict[str, dict[str, np.ndarray]] = {}
        subj_mats: dict[str, np.ndarray] = {}

        for band_name, (fmin, fmax) in self._selected_bands().items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            if not idx.any():
                continue
            mat = dtf[:, :, idx].mean(axis=2)
            subj_mats[band_name] = mat
            maps = self._reduce(mat)
            subj_maps[band_name] = maps
            for vi in range(len(self._vertex_indices)):
                row = {
                    "subject": uid, "group": subject.group,
                    "vertex_idx": int(self._vertex_indices[vi]), "band": band_name,
                }
                for meas in self._measures:
                    row[meas] = float(maps[meas][vi])
                self._rows.append(row)

        self._subject_maps[uid] = subj_maps
        self._dtf_matrices[uid] = subj_mats

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"
        df = pd.DataFrame(self._rows)
        if df.empty:
            logger.warning("No vertex directed data collected")
            return
        df.to_csv(data_dir / "vertex_directed.csv", index=False)
        logger.info("Exported vertex_directed.csv (%d rows)", len(df))

        if self._source_coords is not None:
            cdf = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            cdf.index.name = "vertex_idx"
            cdf.to_csv(data_dir / "source_coords.csv")

        if self._dtf_matrices:
            with open(data_dir / "vertex_dtf_matrices.pkl", "wb") as f:
                pickle.dump(self._dtf_matrices, f)
            logger.info("Saved vertex_dtf_matrices.pkl")

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run statistics")
            return
        coords = self._source_coords
        all_stats = []

        for contrast in self.config.contrasts:
            uids_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            uids_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not uids_a or not uids_b:
                continue
            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            for band_name in self._selected_bands():
                for meas in self._measures:
                    data_a = np.array([
                        self._subject_maps[u][band_name][meas] for u in uids_a
                        if band_name in self._subject_maps.get(u, {})
                    ])
                    data_b = np.array([
                        self._subject_maps[u][band_name][meas] for u in uids_b
                        if band_name in self._subject_maps.get(u, {})
                    ])
                    if data_a.size == 0 or data_b.size == 0:
                        continue

                    result = cluster_permutation_test(
                        data_a, data_b, coords,
                        n_perms=self._n_permutations,
                        threshold=self._cluster_threshold,
                        distance_mm=self._adjacency_distance,
                        seed=42,
                    )
                    g_map = hedges_g(data_a, data_b)
                    self._cluster_results[f"{contrast.name}_{band_name}_{meas}"] = {
                        "result": result,
                        "mean_a": data_a.mean(axis=0),
                        "mean_b": data_b.mean(axis=0),
                        "group_labels": (label_a, label_b),
                        "band": band_name, "measure": meas,
                    }
                    for vi in range(len(result.t_map)):
                        all_stats.append({
                            "contrast": contrast.name, "band": band_name,
                            "measure": meas, "vertex_idx": vi,
                            "value_a": float(data_a.mean(axis=0)[vi]),
                            "value_b": float(data_b.mean(axis=0)[vi]),
                            "t": float(result.t_map[vi]),
                            "p": float(result.p_map[vi]),
                            "hedges_g": float(g_map[vi]),
                            "cluster_id": int(result.cluster_labels[vi]),
                        })

        if all_stats:
            pd.DataFrame(all_stats).to_csv(
                self.tbl_dir / "vertex_directed_stats.csv", index=False,
            )
            logger.info("Exported vertex_directed_stats.csv (%d rows)", len(all_stats))

    def figures(self) -> None:
        if self._source_coords is None:
            return
        coords = self._source_coords
        for key, info in self._cluster_results.items():
            result = info["result"]
            safe = key.lower().replace(" ", "_")
            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"], mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"DTF {info['measure']} — {info['band']}",
                group_labels=info["group_labels"],
                output_path=self.fig_dir / f"dtf_{safe}.png",
            )

    def summary(self) -> None:
        lines = [
            "# Vertex Directed (DTF) Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Vertex DTF — outflow / inflow / netflow",
            f"**MVAR**: order {self._mvar_order}, ridge {self._mvar_ridge}",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Output Files",
            "",
            "- `data/vertex_directed.csv` — per-subject per-vertex outflow/inflow/netflow",
            "- `data/vertex_dtf_matrices.pkl` — full directed matrices",
            "- `data/source_coords.csv` — vertex coordinates (mm)",
            "- `tables/vertex_directed_stats.csv` — cluster-permutation statistics",
            "- `figures/dtf_*.png` — glass-brain directed maps",
            "",
        ]
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines))
        logger.info("Wrote %s", self.output_dir / "ANALYSIS_SUMMARY.md")

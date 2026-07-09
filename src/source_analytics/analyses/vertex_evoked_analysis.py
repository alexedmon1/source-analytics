"""Vertex Evoked response analysis: per-vertex ITC, ERSP, STP for trial paradigms.

The vertex-level companion to :class:`ROIEvokedAnalysis`. Computes Morlet
time-frequency measures (inter-trial coherence, event-related spectral
perturbation, single-trial power) for every source vertex of a trial-based
(evoked) paradigm, then tests group differences with the same spatial
cluster-permutation machinery used by the other vertex modules.

Requires an ``evoked`` section in the study YAML (epoch_samples, sfreq, baseline,
tf_params, measures) — identical schema to ``roi_evoked`` — and a vertex paradigm
whose source reconstructions hold concatenated trial epochs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.tfr import (
    morlet_tfr_avg_power_itc,
    compute_ersp,
    extract_measure_in_band,
)
from ..stats.cluster_permutation import cluster_permutation_test, hedges_g
from ..viz.glass_brain import plot_band_comparison
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


class VertexEvokedAnalysis(BaseAnalysis):
    """Per-vertex ITC / ERSP / STP for evoked (trial-based) paradigms.

    Python computes the Morlet TFR and extracts a scalar measure per vertex;
    group differences are tested per measure with cluster-based permutation over
    the source grid (no R — the maps are vertex-level, like vertex_connectivity).
    """

    name = "vertex_evoked"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._measure_rows: list[dict] = []
        self._sfreq: float | None = None
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        # uid -> measure_name -> per-vertex value array
        self._subject_measures: dict[str, dict[str, np.ndarray]] = {}
        self._subject_groups: dict[str, str] = {}
        self._cluster_results: dict = {}

        wb_cfg = config.vertex
        self._n_permutations = int(config.raw.get("vertex_evoked", {}).get(
            "n_permutations", 1000))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))

    def _get_evoked_config(self) -> dict:
        """Get and validate the evoked config section (same schema as roi_evoked)."""
        evoked = self.config.evoked
        if not evoked:
            evoked = self.config.raw.get(self.name, {})
        if not evoked:
            evoked = self.config.raw.get("evoked", {})
        if not evoked:
            raise ValueError(
                "No 'evoked' section in study config. Evoked analysis requires "
                "epoch_samples, sfreq, baseline, tf_params, and measures."
            )
        required = ["epoch_samples", "sfreq", "baseline", "tf_params", "measures"]
        missing = [k for k in required if k not in evoked]
        if missing:
            raise ValueError(f"Evoked config missing required keys: {missing}")
        return evoked

    def setup(self) -> None:
        self._measure_rows.clear()
        self._subject_measures.clear()
        self._subject_groups.clear()
        self._cluster_results.clear()
        self._source_coords = None
        self._vertex_indices = None
        self._get_evoked_config()

    def process_subject(self, subject: SubjectInfo) -> None:
        evoked_cfg = self._get_evoked_config()
        epoch_samples = int(evoked_cfg["epoch_samples"])
        sfreq = float(evoked_cfg["sfreq"])
        baseline = tuple(evoked_cfg["baseline"])
        tf_params = evoked_cfg["tf_params"]
        measures = evoked_cfg["measures"]
        self._sfreq = sfreq

        fmin, fmax = tf_params["freq_range"]
        freqs = np.arange(fmin, fmax + 1, 1.0)
        n_cycles_cfg = tf_params.get("n_cycles", 7)
        if n_cycles_cfg == "adaptive":
            n_cycles = np.maximum(freqs / 2.0, 3.0)
        else:
            n_cycles = float(n_cycles_cfg)
        xmin = baseline[0]

        loader = SubjectLoader(subject.data_dir)
        # (n_vertices, n_epochs, epoch_samples) — signed for phase-based ITC
        epochs = loader.load_source_epochs(epoch_samples, magnitude=False)
        coords = loader.load_source_coords()

        # Vertex filter (compute mask once from the first subject)
        if self._vertex_indices is None:
            mask = self.config.get_vertex_mask(coords)
            self._vertex_indices = np.where(mask)[0]
            self._source_coords = coords[mask]
            if self.config.has_vertex_filter:
                logger.info(
                    "Vertex filter: %d/%d vertices retained",
                    len(self._vertex_indices), len(coords),
                )
        epochs = epochs[self._vertex_indices]

        uid = f"{subject.group}_{subject.subject_id}"
        self._subject_groups[uid] = subject.group
        n_vertices = epochs.shape[0]
        n_epochs = epochs.shape[1]

        # Per-vertex value array for each configured measure
        subj = {m["name"]: np.full(n_vertices, np.nan) for m in measures}

        for vi in range(n_vertices):
            avg_power, itc_map = morlet_tfr_avg_power_itc(
                epochs[vi], sfreq, freqs, n_cycles,
            )
            stp_map = avg_power
            ersp_map = compute_ersp(avg_power, sfreq, baseline, xmin=xmin)
            measure_maps = {"itc": itc_map, "ersp": ersp_map, "stp": stp_map}

            for mdef in measures:
                mtype, mname = mdef["type"], mdef["name"]
                band = tuple(mdef["band"])
                time_window = tuple(mdef["time_window"])
                if mtype not in measure_maps:
                    logger.warning("Unknown measure type '%s' — skipping", mtype)
                    continue
                value = extract_measure_in_band(
                    measure_maps[mtype], freqs, sfreq, band, time_window, xmin=xmin,
                )
                subj[mname][vi] = value
                self._measure_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": int(self._vertex_indices[vi]),
                    "measure_name": mname,
                    "measure_type": mtype,
                    "band_lo": band[0], "band_hi": band[1],
                    "time_lo": time_window[0], "time_hi": time_window[1],
                    "value": float(value),
                    "n_epochs": n_epochs,
                })
            del avg_power, itc_map, ersp_map, stp_map

        self._subject_measures[uid] = subj

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"
        df = pd.DataFrame(self._measure_rows)
        if df.empty:
            logger.warning("No vertex evoked measure data collected")
            return
        df.to_csv(data_dir / "vertex_evoked_measures.csv", index=False)
        logger.info("Exported vertex_evoked_measures.csv (%d rows)", len(df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def _reload_maps_from_disk(self) -> bool:
        """Reconstruct per-subject evoked measures + coords + groups from the
        persisted CSVs, so statistics/figures are regenerable via --steps."""
        data_dir = self.output_dir / "data"
        csv = data_dir / "vertex_evoked_measures.csv"
        coords_csv = data_dir / "source_coords.csv"
        if not csv.exists():
            logger.warning("No persisted measures at %s; cannot reload", csv)
            return False
        if coords_csv.exists():
            self._source_coords = pd.read_csv(coords_csv)[["x", "y", "z"]].to_numpy(dtype=float)
        df = pd.read_csv(csv)
        self._measure_rows = df.to_dict("records")
        self._subject_measures = {}
        self._subject_groups = {}
        for (uid, group), g in df.groupby(["subject", "group"], sort=False):
            self._subject_groups[uid] = group
            m: dict = {}
            for mname, gg in g.groupby("measure_name", sort=False):
                m[mname] = gg.sort_values("vertex_idx")["value"].to_numpy(dtype=float)
            self._subject_measures[uid] = m
        logger.info("Reloaded %d subjects' evoked measures from %s",
                    len(self._subject_measures), csv)
        return True

    def statistics(self) -> None:
        if not self._subject_measures:
            self._reload_maps_from_disk()
        if self._source_coords is None:
            logger.error("No source coordinates — cannot run statistics")
            return
        coords = self._source_coords
        measure_names = sorted({m["measure_name"] for m in self._measure_rows})
        all_stats = []

        for contrast in self._pairwise_contrasts():
            uids_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            uids_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not uids_a or not uids_b:
                continue
            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            for mname in measure_names:
                data_a = np.array([
                    self._subject_measures[u][mname] for u in uids_a
                    if mname in self._subject_measures.get(u, {})
                ])
                data_b = np.array([
                    self._subject_measures[u][mname] for u in uids_b
                    if mname in self._subject_measures.get(u, {})
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
                self._cluster_results[f"{contrast.name}_{mname}"] = {
                    "result": result,
                    "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0),
                    "group_labels": (label_a, label_b),
                    "measure": mname,
                }
                for vi in range(len(result.t_map)):
                    all_stats.append({
                        "contrast": contrast.name,
                        "measure": mname,
                        "vertex_idx": vi,
                        "value_a": float(data_a.mean(axis=0)[vi]),
                        "value_b": float(data_b.mean(axis=0)[vi]),
                        "t": float(result.t_map[vi]),
                        "p": float(result.p_map[vi]),
                        "hedges_g": float(g_map[vi]),
                        "cluster_id": int(result.cluster_labels[vi]),
                    })

        if all_stats:
            pd.DataFrame(all_stats).to_csv(
                self.tbl_dir / "vertex_evoked_stats.csv", index=False,
            )
            logger.info("Exported vertex_evoked_stats.csv (%d rows)", len(all_stats))

        self._save_cluster_state()

    def figures(self) -> None:
        if not self._cluster_results:
            self._load_cluster_state()
        if self._source_coords is None:
            return
        coords = self._source_coords
        for key, info in self._cluster_results.items():
            result = info["result"]
            measure = info["measure"]
            safe = key.lower().replace(" ", "_")
            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"{measure.upper()}",
                group_labels=info["group_labels"],
                output_path=self.fig_dir / f"evoked_{safe}.png",
            )

    def summary(self) -> None:
        lines = [
            "# Vertex Evoked Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Per-vertex ITC / ERSP / STP (trial paradigm)",
            f"**Permutations**: {self._n_permutations}",
            "",
            "## Output Files",
            "",
            "- `data/vertex_evoked_measures.csv` — per-subject per-vertex measures",
            "- `data/source_coords.csv` — vertex coordinates (mm)",
            "- `tables/vertex_evoked_stats.csv` — cluster-permutation statistics",
            "- `figures/evoked_*.png` — glass-brain measure maps",
            "",
        ]
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("\n".join(lines))
        logger.info("Wrote %s", self.output_dir / "ANALYSIS_SUMMARY.md")

"""Electrode-level (sensor) connectivity analysis — the source-vs-sensor comparator.

Computes all-to-all functional connectivity between the raw scalp electrodes
(the 30-channel MEA) and derives per-channel Functional Connectivity Density
(FCD), mirroring :class:`VertexConnectivityAnalysis` at the sensor level.

This is the **comparator** for the MS2 connectivity-methods thesis: source-
localized (vertex) connectivity recovers spatial structure that sensor-level
connectivity blurs. To make the head-to-head honest, the FC metrics, the FCD
threshold, and the connectivity kernel are *identical* to the vertex module —
the only thing that changes is the node set (30 electrodes vs ~215 vertices).

Like :class:`ElectrodeAnalysis`, this uses ``subject_roster.csv`` (via
``electrode.subject_roster`` in the study config) to map each discovered
subject to its raw ``.set/.fdt`` file, and reuses the shared epoch sampler.

Statistics are sensor-space mass-univariate: per-channel Welch t (group A vs B)
with Benjamini-Hochberg FDR across the 30 channels per band x metric. (Vertex
uses spatial cluster-permutation over the source grid; the sensor montage is too
coarse and irregular for an equivalent cluster test, so per-channel FDR is the
standard sensor-space control — matching the existing electrode band-power
topomap pipeline.)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_set
from ..spectral.epoch_sampler import sample_epochs
from ..spectral.vertex_connectivity import (
    compute_vertex_connectivity_matrix_multi,
    compute_fcd,
    FCD_CENTER,
)
from ..stats.cluster_permutation import hedges_g
from ..viz.brain_roi import fdr_bh
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


class ElectrodeConnectivityAnalysis(BaseAnalysis):
    """All-to-all electrode (sensor) connectivity + FCD — source-vs-sensor comparator.

    Python computes per-channel FCD maps and the full connectivity matrices.
    Group differences in FCD are tested per channel (Welch t + BH-FDR).

    Requires ``electrode.subject_roster`` in the study config (same roster the
    other electrode modules use).
    """

    name = "electrode_connectivity"
    SELECTABLE = {"metric": "connectivity metric", "band": "frequency band",
                  "hypothesis": "declared hypothesis"}

    # FC-six (matches the vertex module's headline metric set). Computed in one
    # shared STFT pass; --metric only restricts which are emitted.
    _FC_SIX = ["aec", "imag_coherence", "pli", "wpli", "dwpli", "dpli"]

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        ec_cfg = config.raw.get("electrode_connectivity", {})
        metrics_cfg = ec_cfg.get("metrics")
        if metrics_cfg is not None:
            self._metrics = list(metrics_cfg)
        else:
            self._metrics = list(self._FC_SIX)
        self._fcd_threshold = float(ec_cfg.get("fcd_threshold", 0.05))
        # Cluster-permutation knobs for the hypothesis layer (sensor space). The
        # montage is head-normalised (not mm), so adjacency_distance defaults to None
        # → auto-computed from the channel coords (≈1.6× median nearest-neighbour).
        self._n_permutations = int(ec_cfg.get("n_permutations", 1000))
        self._cluster_threshold = float(ec_cfg.get("cluster_threshold", 2.0))
        _adj = ec_cfg.get("adjacency_distance", None)
        self._adjacency_distance = float(_adj) if _adj is not None else None

        self._roster: pd.DataFrame | None = None
        self._sfreq: float | None = None
        self._ch_names: list[str] | None = None
        self._ch_coords: np.ndarray | None = None
        # uid -> band -> metric -> per-channel FCD vector
        self._fcd: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._subject_groups: dict[str, str] = {}
        # uid -> band -> metric -> full (n_ch, n_ch) matrix
        self._conn_matrices: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        self._metrics = self._select("metric", self._metrics)
        self._fcd.clear()
        self._subject_groups.clear()
        self._conn_matrices.clear()
        self._ch_names = None
        self._ch_coords = None

        roster_path = self.config.electrode.get("subject_roster")
        if not roster_path:
            raise ValueError(
                "electrode.subject_roster not set in study config. "
                "Add 'electrode: {subject_roster: /path/to/subject_roster.csv}'."
            )
        roster_path = Path(roster_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Subject roster not found: {roster_path}")
        self._roster = pd.read_csv(roster_path)
        required_cols = {"subject_id", "eeg_filename", "eeg_dir"}
        missing = required_cols - set(self._roster.columns)
        if missing:
            raise ValueError(
                f"Subject roster missing required columns: {missing}. "
                f"Available: {list(self._roster.columns)}"
            )
        logger.info(
            "Loaded subject roster: %d entries from %s",
            len(self._roster), roster_path,
        )

    # ------------------------------------------------------- roster / loading
    def _find_eeg_path(self, subject: SubjectInfo) -> Path | None:
        """Look up the raw EEG file path for a subject from the roster."""
        roster_group = subject.pipeline_dir.parent.name  # e.g., "KO ICV"
        matches = self._roster[
            (self._roster["subject_id"] == subject.subject_id)
            & (self._roster["group"] == roster_group)
        ]
        if matches.empty:
            matches = self._roster[self._roster["subject_id"] == subject.subject_id]
        if matches.empty:
            matches = self._roster[
                self._roster["subject_id"] == subject.pipeline_dir.name
            ]
        if matches.empty:
            logger.warning(
                "Subject %s not found in roster, skipping electrode connectivity",
                subject.subject_id,
            )
            return None
        if len(matches) > 1:
            logger.warning(
                "Multiple roster matches for %s in group %s, using first",
                subject.subject_id, roster_group,
            )
        row = matches.iloc[0]
        eeg_path = Path(row["eeg_dir"]) / row["eeg_filename"]
        if not eeg_path.exists():
            logger.warning("Raw EEG file not found: %s", eeg_path)
            return None
        return eeg_path

    def _get_electrode_draws(
        self, data: np.ndarray, sfreq: float,
    ) -> list[np.ndarray]:
        """Apply epoch sampling to raw electrode data.

        Returns a list of 2-D arrays (n_channels, n_samples), one per
        bootstrap draw. No-op (single original array) when sampling is off.
        """
        if not self._epoch_equalize:
            return [data]
        n_bootstrap = self._epoch_n_bootstrap
        if n_bootstrap <= 0:
            return [data]
        if n_bootstrap == 1:
            epochs = sample_epochs(
                data, sfreq,
                epoch_duration_sec=self._epoch_duration_sec,
                n_epochs=self._epoch_n_epochs,
                seed=self._epoch_seed,
            )
            n_ep, n_ch, ep_len = epochs.shape
            return [epochs.transpose(1, 0, 2).reshape(n_ch, n_ep * ep_len)]

        rng = np.random.default_rng(self._epoch_seed)
        draw_seeds = rng.integers(0, 2**31, size=n_bootstrap)
        draws: list[np.ndarray] = []
        for s in draw_seeds:
            epochs = sample_epochs(
                data, sfreq,
                epoch_duration_sec=self._epoch_duration_sec,
                n_epochs=self._epoch_n_epochs,
                seed=int(s),
            )
            n_ep, n_ch, ep_len = epochs.shape
            draws.append(epochs.transpose(1, 0, 2).reshape(n_ch, n_ep * ep_len))
        logger.info(
            "Electrode bootstrap: %d draws x %d epochs", n_bootstrap, self._epoch_n_epochs,
        )
        return draws

    # ------------------------------------------------------------- per subject
    def _compute_subject(self, subject: SubjectInfo):
        """Pure per-subject sensor connectivity + FCD compute (parallel-safe).

        Channel-layout reference locking / mismatch-skip happens in
        :meth:`_merge_subject` (serial in the parent), preserving the exact
        first-subject-wins reference behaviour.
        """
        eeg_path = self._find_eeg_path(subject)
        if eeg_path is None:
            return None

        target_sfreq = (
            self.config.electrode.get("target_sfreq")
            or self.config.raw.get(self.name, {}).get("target_sfreq")
        )
        data, sfreq, ch_names, ch_coords = load_eeglab_set(
            eeg_path, target_sfreq=target_sfreq,
        )

        uid = f"{subject.group}_{subject.subject_id}"
        draws = self._get_electrode_draws(data, sfreq)

        subject_fcd: dict[str, dict[str, np.ndarray]] = {}
        subject_conn: dict[str, dict[str, np.ndarray]] = {}

        for band_name, (fmin, fmax) in self._selected_bands().items():
            # Average connectivity matrices across bootstrap draws.
            avg: dict[str, np.ndarray] | None = None
            for draw_data in draws:
                conn = compute_vertex_connectivity_matrix_multi(
                    draw_data, sfreq, (fmin, fmax), metrics=self._metrics,
                )
                if avg is None:
                    avg = {m: mat.copy() for m, mat in conn.items()}
                else:
                    for m, mat in conn.items():
                        avg[m] += mat
            if avg is None:
                continue
            if len(draws) > 1:
                for m in avg:
                    avg[m] /= len(draws)

            band_fcd = {}
            band_conn = {}
            for metric, mat in avg.items():
                # Identical FCD definition to the vertex module so the
                # source-vs-sensor comparison is apples-to-apples (incl. the
                # directed-metric center, e.g. dPLI on |dPLI-0.5|).
                band_fcd[metric] = compute_fcd(
                    mat, threshold=self._fcd_threshold,
                    center=FCD_CENTER.get(metric),
                )
                band_conn[metric] = mat
            subject_fcd[band_name] = band_fcd
            subject_conn[band_name] = band_conn

        return {
            "uid": uid, "group": subject.group, "sfreq": float(sfreq),
            "ch_names": list(ch_names), "ch_coords": ch_coords,
            "subject_fcd": subject_fcd, "subject_conn": subject_conn,
        }

    def _merge_subject(self, payload) -> None:
        if self._sfreq is None:
            self._sfreq = payload["sfreq"]
        elif payload["sfreq"] != self._sfreq:
            logger.warning("Subject %s has sfreq=%.0f, expected %.0f",
                           payload["uid"], payload["sfreq"], self._sfreq)
        # Lock channel layout from the first merged subject; skip mismatches so
        # the per-channel FCD vectors stack into a coherent (n_subj, n_ch).
        if self._ch_names is None:
            self._ch_names = list(payload["ch_names"])
            self._ch_coords = payload["ch_coords"]
        elif list(payload["ch_names"]) != self._ch_names:
            logger.warning("Subject %s channel layout differs from reference; skipping",
                           payload["uid"])
            return
        uid = payload["uid"]
        self._subject_groups[uid] = payload["group"]
        self._fcd[uid] = payload["subject_fcd"]
        self._conn_matrices[uid] = payload["subject_conn"]

    # ------------------------------------------------------------- aggregate
    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        rows: list[dict] = []
        for uid, bands in self._fcd.items():
            group = self._subject_groups.get(uid, "")
            for band_name, metrics in bands.items():
                for metric, fcd in metrics.items():
                    for ci, ch in enumerate(self._ch_names or []):
                        rows.append({
                            "subject": uid,
                            "group": group,
                            "channel": ch,
                            "band": band_name,
                            "metric": metric,
                            "fcd": float(fcd[ci]),
                        })
        if not rows:
            logger.warning("No electrode connectivity data collected")
            return

        fcd_df = pd.DataFrame(rows)
        fcd_df.to_csv(data_dir / "electrode_fcd.csv", index=False)
        logger.info("Exported electrode_fcd.csv (%d rows)", len(fcd_df))

        # Channel layout (names + montage coords) for the topomap comparator.
        if self._ch_names is not None:
            ch_rows = []
            for ci, ch in enumerate(self._ch_names):
                row = {"channel": ch}
                if self._ch_coords is not None:
                    row["x"], row["y"], row["z"] = (
                        float(self._ch_coords[ci, 0]),
                        float(self._ch_coords[ci, 1]),
                        float(self._ch_coords[ci, 2]),
                    )
                ch_rows.append(row)
            pd.DataFrame(ch_rows).to_csv(
                data_dir / "electrode_layout.csv", index=False,
            )

        # Full matrices for downstream edge/circos views.
        if self._conn_matrices:
            with open(data_dir / "electrode_connectivity_matrices.pkl", "wb") as f:
                pickle.dump(
                    {"channels": self._ch_names, "matrices": self._conn_matrices}, f,
                )
            logger.info("Saved electrode_connectivity_matrices.pkl")

    # ------------------------------------------------------------- statistics
    def statistics(self) -> None:
        if self._ch_names is None:
            logger.error("No channel layout — cannot run statistics")
            return

        n_ch = len(self._ch_names)
        all_stats: list[dict] = []

        for contrast in self._pairwise_contrasts():
            uids_a = [u for u, g in self._subject_groups.items() if g == contrast.group_a]
            uids_b = [u for u, g in self._subject_groups.items() if g == contrast.group_b]
            if not uids_a or not uids_b:
                continue

            for band_name in self._selected_bands():
                for metric in self._metrics:
                    data_a = np.array([
                        self._fcd[u][band_name][metric]
                        for u in uids_a
                        if metric in self._fcd.get(u, {}).get(band_name, {})
                    ])
                    data_b = np.array([
                        self._fcd[u][band_name][metric]
                        for u in uids_b
                        if metric in self._fcd.get(u, {}).get(band_name, {})
                    ])
                    if data_a.shape[0] < 2 or data_b.shape[0] < 2:
                        continue

                    # Per-channel Welch t (group A vs B), BH-FDR across channels.
                    t_vals, p_vals = sp_stats.ttest_ind(
                        data_a, data_b, axis=0, equal_var=False,
                    )
                    t_vals = np.nan_to_num(np.asarray(t_vals), nan=0.0)
                    p_vals = np.where(np.isnan(p_vals), 1.0, p_vals)
                    q_vals = fdr_bh(p_vals)
                    g_map = hedges_g(data_a, data_b)

                    mean_a = data_a.mean(axis=0)
                    mean_b = data_b.mean(axis=0)
                    for ci in range(n_ch):
                        all_stats.append({
                            "contrast": contrast.name,
                            "band": band_name,
                            "metric": metric,
                            "channel": self._ch_names[ci],
                            "fcd_a": float(mean_a[ci]),
                            "fcd_b": float(mean_b[ci]),
                            "t": float(t_vals[ci]),
                            "p": float(p_vals[ci]),
                            "q_fdr": float(q_vals[ci]),
                            "hedges_g": float(g_map[ci]),
                        })

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(
                self.tbl_dir / "electrode_connectivity_stats.csv", index=False,
            )
            logger.info(
                "Exported electrode_connectivity_stats.csv (%d rows)", len(stats_df),
            )

        # --- Declarative hypotheses (hypothesis layer; additive, sensor map+cluster) ---
        # Source-vs-sensor head-to-head: per-channel FCD tested with the SAME cluster
        # permutation adapter as vertex_connectivity. Montage adjacency auto-scaled
        # from the (head-normalised) channel coords unless configured.
        from ..hypothesis import write_module_hypotheses_perm

        if self._ch_coords is not None and self._subject_groups:
            coords = np.asarray(self._ch_coords, dtype=float)
            adj = self._adjacency_distance
            if adj is None:
                from scipy.spatial.distance import cdist
                dm = cdist(coords, coords)
                np.fill_diagonal(dm, np.inf)
                adj = float(1.6 * np.median(dm.min(axis=1)))
            maps_by_cell = {
                (band_name, metric): {
                    uid: self._fcd[uid][band_name][metric]
                    for uid in self._subject_groups
                }
                for band_name in self._selected_bands()
                for metric in self._metrics
            }
            wanted_hyp = self._selection.get("hypothesis")
            write_module_hypotheses_perm(
                maps_by_cell, self._subject_groups, coords, self.config,
                self.tbl_dir, prefix="electrode_connectivity",
                n_perms=self._n_permutations, threshold=self._cluster_threshold,
                distance_mm=adj,
                hypothesis=",".join(sorted(wanted_hyp)) if wanted_hyp else None,
                node_labels=self._channel_region_labels(),
            )

    def _channel_region_labels(self) -> list[str | None] | None:
        """Per-channel electrode-region label (Left Frontal, Right Frontal, …)
        aligned to ``self._ch_names``, from the study's ``electrode_categories``
        grouping. Used to name which sensor region each cluster covers (the
        montage has no brain atlas). None if no grouping is configured."""
        categories = self.config.raw.get("electrode_categories") or {}
        if not categories or not self._ch_names:
            return None
        ch_to_region = {
            ch: region for region, chans in categories.items() for ch in chans
        }
        return [ch_to_region.get(ch) for ch in self._ch_names]

    def figures(self) -> None:
        """Sensor topomaps are rendered by the source-vs-sensor comparator
        (source-lightbox / build_electrode_topomaps), which consumes
        ``electrode_connectivity_stats.csv`` + ``electrode_layout.csv``."""
        pass

    # ---------------------------------------------------------------- summary
    def summary(self) -> None:
        tbl_dir = self.tbl_dir
        lines = [
            "# Electrode Connectivity Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: All-to-all sensor connectivity + FCD (source-vs-sensor comparator)",
            f"**Metrics**: {', '.join(self._metrics)}",
            f"**FCD threshold**: {self._fcd_threshold}",
            f"**Channels**: {len(self._ch_names) if self._ch_names else 0}",
            "",
            "## Methods",
            "",
            "Connectivity was computed between all pairs of scalp electrodes using "
            f"{', '.join(self._metrics)} (identical kernel and FCD threshold to the "
            "vertex module). Per-channel Functional Connectivity Density (FCD) was "
            f"derived as the fraction of connections exceeding {self._fcd_threshold}. "
            "Group differences in per-channel FCD were tested with Welch t-tests and "
            "Benjamini-Hochberg FDR correction across channels.",
            "",
        ]

        stats_csv = tbl_dir / "electrode_connectivity_stats.csv"
        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            lines += ["## Results", ""]
            for keys, sub in stats_df.groupby(["contrast", "band", "metric"]):
                label = " / ".join(str(k) for k in keys)
                n_sig = int((sub["q_fdr"] < 0.05).sum())
                lines.append(
                    f"- **{label}**: {n_sig}/{len(sub)} channels FDR-significant"
                )
            lines.append("")

        lines += [
            "## Output Files",
            "",
            "- `data/electrode_fcd.csv` — per-subject per-channel FCD values",
            "- `data/electrode_connectivity_matrices.pkl` — full connectivity matrices",
            "- `data/electrode_layout.csv` — channel names + montage coordinates",
            "- `tables/electrode_connectivity_stats.csv` — per-channel FCD statistics",
            "",
        ]

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)

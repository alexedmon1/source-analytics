"""Electrode-level evoked response analysis: ITC, ERSP, STP for trial-based paradigms.

Mirrors ROIEvokedAnalysis but operates on raw scalp EEG channels
instead of source-localized ROI time courses.  Uses ``electrode.subject_roster``
to map each discovered subject to its raw ``.set/.fdt`` file.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_epochs
from ..spectral.tfr import (
    morlet_tfr_avg_power_itc,
    compute_ersp,
    debias_itc,
    extract_measure_in_band,
    extract_measure_in_tiles,
    resolve_n_cycles,
)
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


class ElectrodeEvokedAnalysis(BaseAnalysis):
    """Electrode-level ITC, ERSP, and STP analysis for evoked (trial-based) paradigms.

    Combines the electrode data loading approach of ``ElectrodeAnalysis``
    with the TFR computation pipeline of ``ROIEvokedAnalysis``.

    Requires:
    - ``evoked`` section in the study YAML (epoch_samples, sfreq, baseline, tf_params, measures)
    - ``electrode.subject_roster`` pointing to a CSV with columns:
      subject_id, group, eeg_filename, eeg_dir
    """

    name = "electrode_evoked"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._measure_rows: list[dict] = []
        self._tfr_rows: list[dict] = []
        self._sfreq: float | None = None
        self._roster: pd.DataFrame | None = None

    def _get_evoked_config(self) -> dict:
        """Get and validate the evoked config section.

        Checks (in order): config.evoked, config.raw["roi_evoked"]
        (shared evoked params), config.raw[self.name],
        config.raw["evoked"].
        """
        def _has_evoked_keys(d: dict) -> bool:
            return bool(d) and "epoch_samples" in d

        evoked = self.config.evoked if _has_evoked_keys(self.config.evoked) else {}
        if not evoked:
            evoked = self.config.raw.get(self.name, {})
            if not _has_evoked_keys(evoked):
                evoked = {}
        if not evoked:
            evoked = self.config.raw.get("evoked", {})
        if not evoked:
            # Share evoked params with roi_evoked (may be in raw or analyses dict)
            evoked = self.config.raw.get("roi_evoked", {})
        if not evoked:
            analyses = self.config.raw.get("analyses", {})
            evoked = analyses.get("roi_evoked", {})
        if not evoked:
            raise ValueError(
                "No 'evoked' section in study config. "
                "Electrode evoked analysis requires epoch_samples, sfreq, "
                "baseline, tf_params, and measures."
            )
        required = ["epoch_samples", "sfreq", "baseline", "tf_params", "measures"]
        missing = [k for k in required if k not in evoked]
        if missing:
            raise ValueError(f"Evoked config missing required keys: {missing}")
        return evoked

    def setup(self) -> None:
        self._measure_rows.clear()
        self._tfr_rows.clear()

        # Validate evoked config early
        self._get_evoked_config()

        # Load and validate subject roster
        # Check config.electrode first, then analysis-specific raw config
        roster_path = self.config.electrode.get("subject_roster")
        if not roster_path:
            roster_path = self.config.raw.get(self.name, {}).get("subject_roster")
        if not roster_path:
            raise ValueError(
                "electrode.subject_roster not set in study config. "
                "Add 'electrode: {subject_roster: /path/to/roster.csv}' "
                "to analysis.yaml."
            )
        roster_path = Path(roster_path)
        if not roster_path.exists():
            raise FileNotFoundError(f"Subject roster not found: {roster_path}")

        self._roster = pd.read_csv(roster_path)
        logger.info(
            "Loaded subject roster: %d entries from %s",
            len(self._roster), roster_path,
        )

        required_cols = {"subject_id", "eeg_filename", "eeg_dir"}
        missing = required_cols - set(self._roster.columns)
        if missing:
            raise ValueError(
                f"Subject roster missing required columns: {missing}. "
                f"Available: {list(self._roster.columns)}"
            )

    def _find_eeg_path(self, subject: SubjectInfo) -> Path | None:
        """Look up the raw EEG file path for a subject from the roster."""
        roster_group = subject.pipeline_dir.parent.name
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
                "Subject %s not found in roster, skipping electrode evoked",
                subject.subject_id,
            )
            return None

        row = matches.iloc[0]
        eeg_path = Path(row["eeg_dir"]) / row["eeg_filename"]
        if not eeg_path.exists():
            logger.warning("Raw EEG file not found: %s", eeg_path)
            return None
        return eeg_path

    def process_subject(self, subject: SubjectInfo) -> None:
        eeg_path = self._find_eeg_path(subject)
        if eeg_path is None:
            return

        evoked_cfg = self._get_evoked_config()
        epoch_samples = evoked_cfg["epoch_samples"]
        sfreq = float(evoked_cfg["sfreq"])
        baseline = tuple(evoked_cfg["baseline"])
        tf_params = evoked_cfg["tf_params"]
        measures = evoked_cfg["measures"]

        self._sfreq = sfreq

        # Load epoched electrode data
        target_sfreq = (
            self.config.electrode.get("target_sfreq")
            or self.config.raw.get(self.name, {}).get("target_sfreq")
        )
        epochs_3d, file_sfreq, ch_names, file_epoch_samples = load_eeglab_epochs(
            eeg_path, target_sfreq=target_sfreq,
        )

        # Validate epoch structure
        if file_epoch_samples != epoch_samples:
            logger.warning(
                "Subject %s: epoch_samples=%d in file, expected %d from config. Using file value.",
                subject.subject_id, file_epoch_samples, epoch_samples,
            )
            epoch_samples = file_epoch_samples

        if file_sfreq != sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f from config",
                subject.subject_id, file_sfreq, sfreq,
            )

        n_epochs = epochs_3d.shape[0]
        n_channels = epochs_3d.shape[1]

        # Build frequency vector
        fmin, fmax = tf_params["freq_range"]
        freqs = np.arange(fmin, fmax + 1, 1.0)

        # n_cycles: scalar, "adaptive", or [lo, hi] as a linear ramp (see
        # spectral.tfr.resolve_n_cycles). Shared with roi_evoked so the three
        # modules cannot drift apart on the config contract.
        n_cycles = resolve_n_cycles(freqs, tf_params.get("n_cycles", 7))

        # Compute xmin from baseline
        xmin = baseline[0]

        uid = f"{subject.group}_{subject.subject_id}"

        # Optional: export full TF maps for one channel
        export_ch = evoked_cfg.get("export_tfr_channel", None)

        # Process one channel at a time to limit memory
        for ch_idx, ch_name in enumerate(ch_names):
            ch_epochs = epochs_3d[:, ch_idx, :]  # (n_epochs, epoch_samples)

            # Skip channels with bad data
            if np.all(ch_epochs == 0) or np.any(np.isnan(ch_epochs)):
                logger.warning(
                    "Subject %s channel %s has bad data, skipping",
                    subject.subject_id, ch_name,
                )
                continue

            logger.debug(
                "  %s: %s (%d epochs x %d samples)",
                uid, ch_name, n_epochs, epoch_samples,
            )

            # Compute avg power and ITC
            avg_power, itc_map = morlet_tfr_avg_power_itc(
                ch_epochs, sfreq, freqs, n_cycles,
            )

            stp_map = avg_power
            ersp_map = compute_ersp(avg_power, sfreq, baseline, xmin=xmin)

            measure_maps = {"itc": itc_map, "ersp": ersp_map, "stp": stp_map}

            # ITC is biased upward at low trial counts — pure noise gives about
            # 1/sqrt(n) — so subjects with different trial counts are not on the
            # same scale. Offered alongside raw ITC rather than replacing it.
            if n_epochs >= 2:
                measure_maps["itc_debiased"] = debias_itc(itc_map, n_epochs)

            for mdef in measures:
                mtype = mdef["type"]
                mname = mdef["name"]

                if mtype not in measure_maps:
                    logger.warning("Unknown measure type '%s' — skipping", mtype)
                    continue

                # One rectangle, or a union of tiles for a response that
                # sweeps in frequency over time (the chirp).
                tiles = mdef.get("tiles")
                if tiles:
                    value = extract_measure_in_tiles(
                        measure_maps[mtype], freqs, sfreq, tiles, xmin=xmin
                    )
                    band = (min(t["band"][0] for t in tiles),
                            max(t["band"][1] for t in tiles))
                    time_window = (min(t["time_window"][0] for t in tiles),
                                   max(t["time_window"][1] for t in tiles))
                else:
                    band = tuple(mdef["band"])
                    time_window = tuple(mdef["time_window"])
                    value = extract_measure_in_band(
                        measure_maps[mtype], freqs, sfreq, band, time_window,
                        xmin=xmin,
                    )

                self._measure_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "channel": ch_name,
                    "measure_name": mname,
                    "measure_type": mtype,
                    "band_lo": band[0],
                    "band_hi": band[1],
                    "time_lo": time_window[0],
                    "time_hi": time_window[1],
                    "value": value,
                    "n_epochs": n_epochs,
                })

            # Export full TF maps for selected channel
            if export_ch and ch_name == export_ch:
                times = xmin + np.arange(epoch_samples) / sfreq
                for fi, freq in enumerate(freqs):
                    for ti, t in enumerate(times):
                        if ti % max(1, int(sfreq / 100)) != 0:
                            continue
                        self._tfr_rows.append({
                            "subject": uid,
                            "group": subject.group,
                            "freq": float(freq),
                            "time": float(t),
                            "itc": float(itc_map[fi, ti]),
                            "ersp": float(ersp_map[fi, ti]),
                            "stp": float(stp_map[fi, ti]),
                        })

            del avg_power, itc_map, ersp_map, stp_map

    def aggregate(self) -> None:
        """Export CSVs for R consumption."""
        data_dir = self.output_dir / "data"

        measures_df = pd.DataFrame(self._measure_rows)
        if measures_df.empty:
            logger.warning("No electrode evoked measure data collected")
            return

        measures_df.to_csv(
            data_dir / "electrode_evoked_measures.csv", index=False
        )
        logger.info(
            "Exported electrode_evoked_measures.csv (%d rows)", len(measures_df)
        )

        if self._tfr_rows:
            tfr_df = pd.DataFrame(self._tfr_rows)
            tfr_df.to_csv(data_dir / "electrode_evoked_tfr.csv", index=False)
            logger.info(
                "Exported electrode_evoked_tfr.csv (%d rows)", len(tfr_df)
            )

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Regenerate R figures from existing data/tables."""
        self._call_r_figures_only(
            "electrode_evoked_analysis.R", "electrode_evoked_measures.csv"
        )

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "electrode_evoked_measures.csv").exists():
            logger.error(
                "electrode_evoked_measures.csv not found — skipping R analysis"
            )
            return

        try:
            r_dir = find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "electrode_evoked_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Write config YAML for R
        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
            "--fig-dir", str(self.fig_dir),
            "--tbl-dir", str(self.tbl_dir),
        ]
        cmd.extend(self._r_no_figures_flags())

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    logger.info("[R] %s", line)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.info("[R] %s", line)
            if result.returncode != 0:
                logger.error(
                    "R script failed with exit code %d", result.returncode
                )
        except FileNotFoundError:
            logger.error(
                "Rscript not found. Install R to enable statistics and visualization."
            )
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")

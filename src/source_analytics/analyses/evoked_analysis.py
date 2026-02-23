"""Evoked response analysis: ITC, ERSP, STP for trial-based paradigms."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.tfr import (
    morlet_tfr_avg_power_itc,
    compute_ersp,
    extract_measure_in_band,
)
from .base import BaseAnalysis, find_r_script_dir

logger = logging.getLogger(__name__)


class EvokedAnalysis(BaseAnalysis):
    """ITC, ERSP, and STP analysis for evoked (trial-based) paradigms.

    Requires an ``evoked`` section in the study YAML config specifying
    epoch_samples, sfreq, baseline, tf_params, and measures.

    Python computes TFR and extracts scalar measures per ROI.
    R (lme4, ggplot2) handles statistics and visualization.
    """

    name = "evoked"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._measure_rows: list[dict] = []
        self._tfr_rows: list[dict] = []  # optional full TF maps for viz
        self._sfreq: float | None = None

    def _get_evoked_config(self) -> dict:
        """Get and validate the evoked config section."""
        evoked = self.config.evoked
        if not evoked:
            raise ValueError(
                "No 'evoked' section in study config. "
                "Evoked analysis requires epoch_samples, sfreq, baseline, "
                "tf_params, and measures."
            )
        required = ["epoch_samples", "sfreq", "baseline", "tf_params", "measures"]
        missing = [k for k in required if k not in evoked]
        if missing:
            raise ValueError(f"Evoked config missing required keys: {missing}")
        return evoked

    def setup(self) -> None:
        self._measure_rows.clear()
        self._tfr_rows.clear()
        # Validate config early
        self._get_evoked_config()

    def process_subject(self, subject: SubjectInfo) -> None:
        evoked_cfg = self._get_evoked_config()
        epoch_samples = evoked_cfg["epoch_samples"]
        sfreq = float(evoked_cfg["sfreq"])
        baseline = tuple(evoked_cfg["baseline"])
        tf_params = evoked_cfg["tf_params"]
        measures = evoked_cfg["measures"]

        self._sfreq = sfreq

        # Build frequency vector
        fmin, fmax = tf_params["freq_range"]
        # 1 Hz spacing is sufficient for wavelet analysis
        freqs = np.arange(fmin, fmax + 1, 1.0)

        # Determine n_cycles
        n_cycles_cfg = tf_params.get("n_cycles", 7)
        if n_cycles_cfg == "adaptive":
            n_cycles = freqs / 2.0  # freq/2 gives 2 cycles at 4 Hz, 60 at 120 Hz
            n_cycles = np.maximum(n_cycles, 3.0)  # minimum 3 cycles
        else:
            n_cycles = float(n_cycles_cfg)

        # Compute xmin from baseline (epoch starts at baseline[0])
        xmin = baseline[0]

        # Load ROI epochs
        loader = SubjectLoader(subject.data_dir)
        roi_epochs = loader.load_roi_epochs(epoch_samples, signed=True)

        uid = f"{subject.group}_{subject.subject_id}"
        n_rois = len(roi_epochs)

        # Determine which ROI to export full TF maps for (first auditory ROI found)
        export_roi = evoked_cfg.get("export_tfr_roi", None)

        # Process one ROI at a time to limit memory
        for roi_idx, (roi_name, epochs) in enumerate(roi_epochs.items()):
            n_epochs = epochs.shape[0]
            logger.debug(
                "  %s: %s (%d epochs x %d samples)",
                uid, roi_name, n_epochs, epoch_samples,
            )

            # Compute avg power and ITC in a single memory-efficient pass
            # (MNE never materializes the full n_epochs x n_freqs x n_times array)
            avg_power, itc_map = morlet_tfr_avg_power_itc(
                epochs, sfreq, freqs, n_cycles,
            )

            # STP is identical to avg_power (mean |TFR|^2 across trials)
            stp_map = avg_power

            # ERSP is baseline-corrected avg power in dB
            ersp_map = compute_ersp(avg_power, sfreq, baseline, xmin=xmin)

            # Extract scalar measures
            measure_maps = {"itc": itc_map, "ersp": ersp_map, "stp": stp_map}

            for mdef in measures:
                mtype = mdef["type"]
                mname = mdef["name"]
                band = tuple(mdef["band"])
                time_window = tuple(mdef["time_window"])

                if mtype not in measure_maps:
                    logger.warning("Unknown measure type '%s' — skipping", mtype)
                    continue

                value = extract_measure_in_band(
                    measure_maps[mtype], freqs, sfreq, band, time_window, xmin=xmin
                )

                self._measure_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "roi": roi_name,
                    "measure_name": mname,
                    "measure_type": mtype,
                    "band_lo": band[0],
                    "band_hi": band[1],
                    "time_lo": time_window[0],
                    "time_hi": time_window[1],
                    "value": value,
                    "n_epochs": n_epochs,
                })

            # Export full TF maps for selected ROI (for visualization)
            if export_roi and roi_name == export_roi:
                times = xmin + np.arange(epoch_samples) / sfreq
                for fi, freq in enumerate(freqs):
                    for ti, t in enumerate(times):
                        # Subsample time to keep CSV manageable
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

            # Free memory before next ROI
            del avg_power, itc_map, ersp_map, stp_map

    def aggregate(self) -> None:
        """Export CSVs for R consumption."""
        data_dir = self.output_dir / "data"

        # Scalar measures CSV
        measures_df = pd.DataFrame(self._measure_rows)
        if measures_df.empty:
            logger.warning("No evoked measure data collected")
            return

        measures_df.to_csv(data_dir / "evoked_measures.csv", index=False)
        logger.info("Exported evoked_measures.csv (%d rows)", len(measures_df))

        # Optional full TF maps CSV
        if self._tfr_rows:
            tfr_df = pd.DataFrame(self._tfr_rows)
            tfr_df.to_csv(data_dir / "evoked_tfr.csv", index=False)
            logger.info("Exported evoked_tfr.csv (%d rows)", len(tfr_df))

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Regenerate R figures from existing data/tables."""
        self._call_r_figures_only("evoked_analysis.R", "evoked_measures.csv")

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        if not (data_dir / "evoked_measures.csv").exists():
            logger.error("evoked_measures.csv not found — skipping R analysis")
            return

        try:
            r_dir = find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "evoked_analysis.R"
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
                logger.error("R script failed with exit code %d", result.returncode)
        except FileNotFoundError:
            logger.error("Rscript not found. Install R to enable statistics and visualization.")
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")

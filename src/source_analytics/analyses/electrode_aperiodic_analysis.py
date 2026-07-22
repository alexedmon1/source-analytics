"""Electrode-level aperiodic (1/f) spectral parameter analysis.

Mirrors the ROI aperiodic analysis but operates on raw scalp EEG channels.
Fits specparam to each channel's PSD and exports per-channel aperiodic
parameters (exponent, offset) for LMM analysis in R.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.electrode_loader import load_eeglab_set
from ..spectral.epoch_sampler import sample_epochs
from ..spectral.psd import compute_psd
from ..spectral.aperiodic import fit_aperiodic, resolve_freq_range
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class ElectrodeAperiodicAnalysis(BaseAnalysis):
    """Electrode-level aperiodic spectral parameter analysis.

    Python computes per-channel PSD, fits specparam, exports CSVs.
    R handles LMM statistics (group * channel) and visualization.

    Requires ``electrode.subject_roster`` in the study config pointing
    to a CSV with columns: ``subject_id, group, eeg_filename, eeg_dir``.
    """

    name = "electrode_aperiodic"
    SELECTABLE = {"hypothesis": "declared hypothesis"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._subject_aperiodic: list[dict] = []
        self._subject_groups: dict[str, str] = {}
        self._sfreq: float | None = None
        self._roster: pd.DataFrame | None = None
        self._freq_range = resolve_freq_range(config.raw.get("electrode_aperiodic"))

    def setup(self) -> None:
        self._subject_aperiodic.clear()
        self._subject_groups.clear()

        # Load subject roster
        roster_path = self.config.electrode.get("subject_roster")
        if roster_path is None:
            logger.error("No electrode.subject_roster in config")
            return
        roster_path = Path(roster_path)
        if not roster_path.exists():
            logger.error("Subject roster not found: %s", roster_path)
            return
        self._roster = pd.read_csv(roster_path)
        logger.info("Loaded subject roster: %d rows", len(self._roster))

    def _find_eeg_path(self, subject: SubjectInfo) -> Path | None:
        """Resolve the raw EEG .set file path for a subject."""
        if self._roster is None:
            return None

        # Map analysis group back to roster group
        group_mapping = self.config.groups
        roster_group = group_mapping.get(subject.group, subject.group)

        matches = self._roster[
            (self._roster["group"] == roster_group)
            & (self._roster["subject_id"] == subject.subject_id)
        ]
        if matches.empty:
            matches = self._roster[
                self._roster["subject_id"] == subject.pipeline_dir.name
            ]
        if matches.empty:
            logger.warning(
                "Subject %s not found in roster, skipping",
                subject.subject_id,
            )
            return None

        row = matches.iloc[0]
        eeg_path = Path(row["eeg_dir"]) / row["eeg_filename"]
        if not eeg_path.exists():
            logger.warning("Raw EEG file not found: %s", eeg_path)
            return None
        return eeg_path

    def _get_electrode_draws(
        self, data: np.ndarray, sfreq: float,
    ) -> list[np.ndarray]:
        """Apply epoch sampling to raw electrode data."""
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
            "Electrode aperiodic bootstrap: %d draws × %d epochs",
            n_bootstrap, self._epoch_n_epochs,
        )
        return draws

    def process_subject(self, subject: SubjectInfo) -> None:
        eeg_path = self._find_eeg_path(subject)
        if eeg_path is None:
            return

        target_sfreq = (
            self.config.electrode.get("target_sfreq")
            or self.config.raw.get(self.name, {}).get("target_sfreq")
        )
        data, sfreq, ch_names, _ = load_eeglab_set(
            eeg_path, target_sfreq=target_sfreq,
        )

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"
        self._subject_groups[uid] = subject.group

        draws = self._get_electrode_draws(data, sfreq)

        # Fit aperiodic per channel per draw, then average
        ch_params_accum: dict[str, list[dict]] = {}

        for draw_data in draws:
            for ch_idx, ch_name in enumerate(ch_names):
                ch_data = draw_data[ch_idx, :]

                if np.all(ch_data == 0) or np.any(np.isnan(ch_data)):
                    continue

                freqs, psd = compute_psd(ch_data, sfreq, fmin=1.0, fmax=100.0)
                params = fit_aperiodic(freqs, psd, freq_range=self._freq_range)
                ch_params_accum.setdefault(ch_name, []).append(params)

        # Average across draws
        for ch_name, params_list in ch_params_accum.items():
            n = len(params_list)
            self._subject_aperiodic.append({
                "subject": uid,
                "group": subject.group,
                "channel": ch_name,
                "exponent": sum(p["exponent"] for p in params_list) / n,
                "offset": sum(p["offset"] for p in params_list) / n,
                # See roi_aperiodic: report offset_centered with the exponent.
                "offset_centered": sum(p["offset_centered"] for p in params_list) / n,
                "r_squared": sum(p["r_squared"] for p in params_list) / n,
                "n_peaks": sum(p["n_peaks"] for p in params_list) / n,
                "error": sum(p["error"] for p in params_list) / n,
                "method": params_list[0]["method"],
                "fit_fmin": params_list[0]["fit_fmin"],
                "fit_fmax": params_list[0]["fit_fmax"],
            })

    def aggregate(self) -> None:
        """Export aperiodic_params.csv for R consumption."""
        data_dir = self.output_dir / "data"

        df = pd.DataFrame(self._subject_aperiodic)
        if df.empty:
            logger.warning("No electrode aperiodic data collected")
            return

        df.to_csv(data_dir / "electrode_aperiodic_params.csv", index=False)
        logger.info(
            "Exported electrode_aperiodic_params.csv (%d rows)", len(df),
        )

    def statistics(self) -> None:
        """Delegated to R."""
        pass

    def figures(self) -> None:
        """Delegated to R."""
        pass

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        csv_path = data_dir / "electrode_aperiodic_params.csv"
        if not csv_path.exists():
            logger.error(
                "electrode_aperiodic_params.csv not found — skipping R analysis",
            )
            return

        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "electrode_aperiodic_analysis.R"
        if not r_script.exists():
            logger.warning(
                "R script not found: %s — skipping R statistics", r_script,
            )
            return

        config_path = data_dir / "study_config.yaml"
        import yaml

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

        # Manual hypothesis selection (--hypothesis NAME[,NAME]) passed through to R.
        wanted_hyp = self._selection.get("hypothesis")
        if wanted_hyp:
            cmd.extend(["--hypothesis", ",".join(sorted(wanted_hyp))])

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
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
                    "R script failed with exit code %d", result.returncode,
                )
        except FileNotFoundError:
            logger.error(
                "Rscript not found. Install R to enable statistics.",
            )
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")

"""ROI PSD Analysis module: computes PSD, exports CSVs, calls R for stats/viz."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.psd import compute_psd_multiroi
from ..spectral.band_power import extract_band_power_multiroi, relative_power_kwargs
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    # Walk up from this file to find the R/ directory
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent  # src/../..
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    # Fallback: check common locations
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class ROIPsdAnalysis(BaseAnalysis):
    """ROI-level power spectral density analysis with group comparisons.

    Python computes PSD and band power, exports CSVs.
    R (lme4, ggplot2) handles statistics and visualization.
    """

    name = "roi_psd"
    SELECTABLE = {"band": "frequency band", "hypothesis": "declared hypothesis"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._subject_band_power: list[dict] = []
        self._subject_psd_curves: list[dict] = []
        self._subject_groups: dict[str, str] = {}
        self._sfreq: float | None = None

    def setup(self) -> None:
        self._subject_band_power.clear()
        self._subject_psd_curves.clear()
        self._subject_groups.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)

        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir, rois=self.config.rois,
        )
        sfreq = loader.load_sfreq()
        draws = self._equalize_roi_timeseries(roi_ts, sfreq)

        if self._sfreq is None:
            self._sfreq = sfreq
        elif sfreq != self._sfreq:
            logger.warning(
                "Subject %s has sfreq=%.0f, expected %.0f",
                subject.subject_id, sfreq, self._sfreq,
            )

        uid = f"{subject.group}_{subject.subject_id}"
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        self._subject_groups[uid] = subject.group

        # Compute PSD per bootstrap draw and average
        all_psd_arrays: dict[str, list[np.ndarray]] = {}
        all_band_abs: dict[tuple[str, str], list[float]] = {}
        all_band_rel: dict[tuple[str, str], list[float]] = {}
        freqs_out: np.ndarray | None = None

        for draw_ts in draws:
            roi_psds = compute_psd_multiroi(draw_ts, sfreq, fmax=fmax)

            for roi_name, (freqs, psd) in roi_psds.items():
                freqs_out = freqs
                all_psd_arrays.setdefault(roi_name, []).append(psd)

            band_power = extract_band_power_multiroi(
                roi_psds, self._selected_bands(),
                **relative_power_kwargs(self.config.raw.get("relative_power")))
            for roi_name, bp_dict in band_power.items():
                for band_name, power_vals in bp_dict.items():
                    key = (roi_name, band_name)
                    all_band_abs.setdefault(key, []).append(power_vals["absolute"])
                    all_band_rel.setdefault(key, []).append(power_vals["relative"])

        # Average PSD curves across draws
        for roi_name, psd_list in all_psd_arrays.items():
            avg_psd = np.mean(psd_list, axis=0)
            for i, freq in enumerate(freqs_out):
                self._subject_psd_curves.append({
                    "subject": uid,
                    "group": subject.group,
                    "roi": roi_name,
                    "freq_hz": float(freq),
                    "psd": float(avg_psd[i]),
                })

        # Average band power across draws
        for (roi_name, band_name), abs_list in all_band_abs.items():
            self._subject_band_power.append({
                "subject": uid,
                "group": subject.group,
                "roi": roi_name,
                "band": band_name,
                "absolute": float(np.mean(abs_list)),
                "relative": float(np.mean(all_band_rel[(roi_name, band_name)])),
            })

    def aggregate(self) -> None:
        """Export CSVs for R consumption."""
        data_dir = self.output_dir / "data"

        # Band power CSV
        band_df = pd.DataFrame(self._subject_band_power)
        if band_df.empty:
            logger.warning("No band power data collected")
            return

        band_df.to_csv(data_dir / "band_power.csv", index=False)
        logger.info("Exported band_power.csv (%d rows)", len(band_df))

        # PSD curves CSV
        psd_df = pd.DataFrame(self._subject_psd_curves)
        if not psd_df.empty:
            psd_df.to_csv(data_dir / "psd_curves.csv", index=False)
            logger.info("Exported psd_curves.csv (%d rows)", len(psd_df))

    def statistics(self) -> None:
        """Delegated to R — this is a no-op in Python."""
        pass

    def figures(self) -> None:
        """Regenerate R figures from existing data/tables."""
        self._call_r_figures_only("roi_psd_analysis.R", "band_power.csv")

    def summary(self) -> None:
        """Call Rscript for statistics, figures, and summary."""
        data_dir = self.output_dir / "data"

        # Verify CSVs exist
        if not (data_dir / "band_power.csv").exists():
            logger.error("band_power.csv not found — skipping R analysis")
            return

        # Find R scripts
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return

        r_script = r_dir / "roi_psd_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

        # Find the study config YAML path
        # Copy config to data dir so R can read it
        config_path = data_dir / "study_config.yaml"
        import yaml
        # Always write config so sfreq is up-to-date
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Call Rscript
        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
            "--fig-dir", str(self.fig_dir),
            "--tbl-dir", str(self.tbl_dir),
        ]
        cmd.extend(self._r_no_figures_flags())
        cmd.extend(self._r_roi_categories_flags())

        # Manual hypothesis selection: pass --hypothesis NAME[,NAME] through to R.
        # Read directly from the active selection (not routed through _select,
        # since the hypotheses are resolved R-side, not in Python).
        wanted_hyp = self._selection.get("hypothesis")
        if wanted_hyp:
            cmd.extend(["--hypothesis", ",".join(sorted(wanted_hyp))])

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
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

        # Render brain mosaics from posthoc effect sizes
        if self._generate_figures:
            self._render_brain_mosaics()

    def _render_brain_mosaics(self) -> None:
        """Render brain ROI mosaics from ROI PSD posthoc CSVs."""
        from ..viz.brain_roi import render_posthoc_mosaics

        tbl_dir = self.tbl_dir
        fig_dir = self.fig_dir

        posthoc_csv = tbl_dir / "roi_psd_posthoc_roi.csv"
        if not posthoc_csv.exists():
            logger.info("No ROI PSD posthoc ROI CSV — skipping brain mosaics")
            return

        roi_cats = self.config.roi_categories
        if not roi_cats:
            logger.info("No roi_categories in config — skipping brain mosaics")
            return

        render_posthoc_mosaics(
            posthoc_csv,
            roi_cats,
            fig_dir,
            analysis_name="roi_psd",
            effect_col="hedges_g",
            roi_col="roi",
            facet_cols=["contrast", "band", "power_type"],
            colorbar_label="Hedges' g",
        )

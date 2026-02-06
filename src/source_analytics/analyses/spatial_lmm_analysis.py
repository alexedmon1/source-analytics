"""Spatial Linear Mixed Effects Model analysis.

Fits a single model per band using nlme::gls with exponential spatial
correlation structure, accounting for spatial autocorrelation and avoiding
the multiple comparison problem inherent in vertex-wise testing.

The heavy lifting is done in R (nlme::gls). Python handles data preparation,
figure generation, and orchestration.
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
from ..io.loader import SubjectLoader
from ..spectral.vertex import compute_psd_vertices, extract_band_power_vertices
from ..spectral.epoch_sampler import sample_epochs, get_epoch_config
from ..viz.glass_brain import plot_glass_brain
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Cannot find R/ scripts directory")


class SpatialLMMAnalysis(BaseAnalysis):
    """Spatial mixed effects model analysis (R-primary)."""

    name = "spatial_lmm"

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._power_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._vertex_indices: np.ndarray | None = None
        self._sfreq: float | None = None

        # Config
        slmm_cfg = config.raw.get("spatial_lmm", {})
        self._correlation_structure = slmm_cfg.get("correlation_structure", "exponential")
        self._spatial_range_mm = float(slmm_cfg.get("spatial_range_mm", 3.0))

        wb_cfg = config.wholebrain
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        self._epoch_config = get_epoch_config(wb_cfg)

    def setup(self) -> None:
        self._power_rows.clear()
        self._source_coords = None
        self._vertex_indices = None

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses(magnitude=True)
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        if self._sfreq is None:
            self._sfreq = sfreq

        # Apply vertex filter (compute mask once from first subject)
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
        coords = self._source_coords

        # Compute PSD
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        if self._epoch_config is not None:
            epochs = sample_epochs(
                stc_data, sfreq,
                epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
                n_epochs=self._epoch_config.get("n_epochs", 80),
                seed=self._epoch_config.get("seed", 42),
            )
            all_psd = []
            for ep in epochs:
                f, p = compute_psd_vertices(ep, sfreq, fmax=fmax)
                all_psd.append(p)
            freqs = f
            psd = np.mean(all_psd, axis=0)
        else:
            freqs, psd = compute_psd_vertices(stc_data, sfreq, fmax=fmax)

        band_power = extract_band_power_vertices(
            freqs, psd, self.config.bands, noise_exclude=self._noise_exclude,
        )

        n_vertices = stc_data.shape[0]
        for band_name, bp in band_power.items():
            for vi in range(n_vertices):
                self._power_rows.append({
                    "subject": uid,
                    "group": subject.group,
                    "vertex_idx": int(self._vertex_indices[vi]),
                    "x": float(coords[vi, 0]),
                    "y": float(coords[vi, 1]),
                    "z": float(coords[vi, 2]),
                    "band": band_name,
                    "relative": float(bp["relative"][vi]),
                    "dB": float(bp["dB"][vi]),
                    "absolute": float(bp["absolute"][vi]),
                })

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        power_df = pd.DataFrame(self._power_rows)
        if power_df.empty:
            logger.warning("No spatial LMM data collected")
            return
        power_df.to_csv(data_dir / "spatial_lmm_data.csv", index=False)
        logger.info("Exported spatial_lmm_data.csv (%d rows)", len(power_df))

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def statistics(self) -> None:
        """Delegate all statistical analysis to R."""
        pass

    def figures(self) -> None:
        """Delegate to R (variograms, etc). Python generates residual maps if available."""
        tbl_dir = self.output_dir / "tables"
        fig_dir = self.output_dir / "figures"

        residuals_csv = tbl_dir / "spatial_residuals.csv"
        if residuals_csv.exists() and self._source_coords is not None:
            resid_df = pd.read_csv(residuals_csv)
            for band in resid_df["band"].unique():
                sub = resid_df[resid_df["band"] == band]
                mean_resid = sub.groupby("vertex_idx")["residual"].mean().values
                if len(mean_resid) == len(self._source_coords):
                    safe_name = band.lower().replace(" ", "_")
                    plot_glass_brain(
                        coords=self._source_coords,
                        values=mean_resid,
                        title=f"Spatial Residuals — {band}",
                        output_path=fig_dir / f"spatial_residuals_{safe_name}.png",
                        cmap="RdBu_r",
                    )

    def summary(self) -> None:
        """Run R script for spatial GLS fitting + report generation."""
        data_dir = self.output_dir / "data"

        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.warning(str(e))
            self._write_python_summary()
            return

        r_script = r_dir / "spatial_lmm_analysis.R"
        if not r_script.exists():
            logger.warning("R script not found: %s", r_script)
            self._write_python_summary()
            return

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
        ]

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200,  # 20 min for complex models
            )
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    logger.info("[R] %s", line)
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.info("[R] %s", line)
            if result.returncode != 0:
                logger.error("R script failed (exit %d)", result.returncode)
                self._write_python_summary()
        except FileNotFoundError:
            logger.warning("Rscript not found — writing Python summary")
            self._write_python_summary()
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 1200 seconds")
            self._write_python_summary()

    def _write_python_summary(self) -> None:
        lines = [
            "# Spatial LMM Analysis Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Spatial Linear Mixed Effects Model",
            f"**Correlation structure**: {self._correlation_structure}",
            f"**Spatial range**: {self._spatial_range_mm} mm",
            "",
            "## Methods",
            "",
            "Spatial generalized least squares (nlme::gls) was used to model vertex-level "
            "band power as a function of group, with an exponential spatial correlation "
            "structure (`corExp(form = ~x+y+z | subject)`). This single-model approach "
            "accounts for spatial autocorrelation and avoids the multiple comparison "
            "problem inherent in vertex-wise testing.",
            "",
            "## Status",
            "",
            "R analysis not available. Data exported to `data/spatial_lmm_data.csv` "
            "for manual R analysis.",
            "",
        ]

        if self._epoch_config is not None:
            lines.insert(-2,
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )

        lines.extend([
            "## Output Files",
            "",
            "- `data/spatial_lmm_data.csv` — per-subject per-vertex band power with coordinates",
            "- `data/source_coords.csv` — vertex coordinates (mm)",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)

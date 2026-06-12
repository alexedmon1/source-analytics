"""ROI cross-frequency coupling: PAC, AAC, and n:m PPC.

Home for cross-frequency coupling measures, selectable via ``--metric``:

- **pac**: Tort-2010 phase–amplitude Modulation Index (surrogate z-scored),
  one value per ROI per cross-frequency band pair.
- **aac**: cross-frequency amplitude–amplitude coupling, ROI×ROI per band pair
  (edge-level; envelope of the slow band vs envelope of the fast band).
- **ppc**: n:m phase–phase coupling, ROI×ROI per band pair (edge-level).

All three share the signed (phase-preserving) ROI front-end and the same valid
cross-frequency band pairs (``get_valid_pac_pairs``). PAC keeps its R statistics
+ brain mosaics; AAC/PPC emit edge CSVs (group statistics ride the connectivity
/ gating R path).
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
from ..spectral.pac import compute_pac_multiroi, get_valid_pac_pairs
from ..spectral.cross_freq import compute_aac, compute_ppc
from ..viz.constants import CC_ROIS
from .base import BaseAnalysis

logger = logging.getLogger(__name__)


def _find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package."""
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent  # src/../..
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


def _nm_ratio(band_x: tuple[float, float], band_y: tuple[float, float]) -> tuple[int, int]:
    """Integer n:m harmonic ratio for a (slow, fast) band pair from centers.

    Returns ``(n, m)`` with ``m = 1`` and ``n = round(center_y / center_x)`` so
    that ``n·φ_slow − φ_fast ≈ 0`` under perfect n:1 locking.
    """
    cx = sum(band_x) / 2.0
    cy = sum(band_y) / 2.0
    n = max(1, int(round(cy / cx))) if cx > 0 else 1
    return n, 1


class ROICrossFreqAnalysis(BaseAnalysis):
    """ROI-level cross-frequency coupling (PAC, AAC, n:m PPC).

    ``--metric`` selects which coupling measure(s) to compute; all share the
    signed-ROI Hilbert front-end and the valid cross-frequency band pairs.
    """

    name = "roi_cross_freq"
    SELECTABLE = {"metric": "coupling measure", "band": "frequency band"}

    # Coupling measures this module can produce.
    _CROSS_FREQ_METRICS = ["pac", "aac", "ppc"]

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._pac_rows: list[dict] = []
        self._aac_rows: list[dict] = []
        self._ppc_rows: list[dict] = []
        self._sfreq: float | None = None
        self._metrics: list[str] = list(self._CROSS_FREQ_METRICS)
        cfg = config.raw.get(self.name, {})
        # Surrogate count for PPC significance (Palva 2005). 0 = raw PLF only.
        self._ppc_surrogates = int(cfg.get("ppc_surrogates", 200))

    def setup(self) -> None:
        # Restrict to --metric / --select metric=... (pac/aac/ppc).
        self._metrics = self._select("metric", self._CROSS_FREQ_METRICS)
        self._pac_rows.clear()
        self._aac_rows.clear()
        self._ppc_rows.clear()

    # ------------------------------------------------------------ processing
    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir,
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
        pairs = get_valid_pac_pairs(self._selected_bands())
        if not pairs:
            logger.warning(
                "No valid cross-frequency band pairs for %s with bands: %s",
                subject.subject_id, list(self._selected_bands().keys()),
            )
            return

        if "pac" in self._metrics:
            self._process_pac(uid, subject.group, draws, sfreq, pairs)
        if "aac" in self._metrics:
            self._process_edge_metric("aac", uid, subject.group, draws, sfreq, pairs)
        if "ppc" in self._metrics:
            self._process_edge_metric("ppc", uid, subject.group, draws, sfreq, pairs)

    def _process_pac(self, uid, group, draws, sfreq, pairs) -> None:
        """Tort MI z-scores per ROI per band pair (unchanged from roi_pac)."""
        accum: dict[tuple[str, str], list[dict]] = {}
        for draw_ts in draws:
            results = compute_pac_multiroi(draw_ts, sfreq, pairs, self.config.bands)
            for row in results:
                accum.setdefault((row["roi"], row["freq_pair"]), []).append(row)

        for rows in accum.values():
            n = len(rows)
            self._pac_rows.append({
                "subject": uid,
                "group": group,
                "roi": rows[0]["roi"],
                "phase_band": rows[0]["phase_band"],
                "amp_band": rows[0]["amp_band"],
                "freq_pair": rows[0]["freq_pair"],
                "mi": sum(r["mi"] for r in rows) / n,
                "z_score": sum(r["z_score"] for r in rows) / n,
                "surr_mean": sum(r["surr_mean"] for r in rows) / n,
                "surr_std": sum(r["surr_std"] for r in rows) / n,
            })

    def _process_edge_metric(self, metric, uid, group, draws, sfreq, pairs) -> None:
        """AAC / PPC: ROI×ROI matrix per band pair, averaged over draws → edges.

        Corpus-callosum WM tracts are excluded (these are edge measures, like
        the FC connectivity module). PPC additionally emits a surrogate z-score
        (``ppc_z``) for significance (Palva 2005).
        """
        rows_out = self._aac_rows if metric == "aac" else self._ppc_rows
        bands = self.config.bands

        for phase_band, amp_band in pairs:
            band_x = bands[phase_band]
            band_y = bands[amp_band]
            n, m = _nm_ratio(band_x, band_y) if metric == "ppc" else (1, 1)

            accum: dict[str, np.ndarray] | None = None
            roi_names = None
            for draw_ts in draws:
                names = [r for r in sorted(draw_ts.keys()) if r not in CC_ROIS]
                roi_names = names
                data = np.vstack([draw_ts[r] for r in names])
                if metric == "aac":
                    out = {"aac": compute_aac(data, sfreq, band_x, band_y)}
                elif self._ppc_surrogates > 0:
                    plf, z = compute_ppc(
                        data, sfreq, band_x, band_y, n=n, m=m,
                        n_surrogates=self._ppc_surrogates, seed=42,
                    )
                    out = {"ppc": plf, "ppc_z": z}
                else:
                    out = {"ppc": compute_ppc(data, sfreq, band_x, band_y, n=n, m=m)}
                if accum is None:
                    accum = {k: v.copy() for k, v in out.items()}
                else:
                    for k in accum:
                        accum[k] += out[k]
            if accum is None:
                continue
            for k in accum:
                accum[k] /= len(draws)

            freq_pair = f"{phase_band}-{amp_band}"
            n_rois = len(roi_names)
            for i in range(n_rois):
                for j in range(n_rois):
                    row = {
                        "subject": uid,
                        "group": group,
                        "phase_band": phase_band,
                        "amp_band": amp_band,
                        "freq_pair": freq_pair,
                        "n": n, "m": m,
                        "roi_x": roi_names[i],
                        "roi_y": roi_names[j],
                    }
                    for k in accum:
                        row[k] = float(accum[k][i, j])
                    rows_out.append(row)

    # ----------------------------------------------------------- aggregate
    def aggregate(self) -> None:
        """Export per-metric CSVs for the selected coupling measures."""
        data_dir = self.output_dir / "data"

        if self._pac_rows:
            col_order = [
                "subject", "group", "roi", "phase_band", "amp_band",
                "freq_pair", "mi", "z_score", "surr_mean", "surr_std",
            ]
            pac_df = pd.DataFrame(self._pac_rows)
            pac_df = pac_df[[c for c in col_order if c in pac_df.columns]]
            pac_df.to_csv(data_dir / "pac_values.csv", index=False)
            logger.info("Exported pac_values.csv (%d rows)", len(pac_df))

        for metric, rows in (("aac", self._aac_rows), ("ppc", self._ppc_rows)):
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(data_dir / f"{metric}_edges.csv", index=False)
                logger.info("Exported %s_edges.csv (%d rows)", metric, len(df))

    def statistics(self) -> None:
        """Delegated to R (PAC) / gating engine (AAC, PPC)."""
        pass

    def figures(self) -> None:
        """Regenerate PAC R figures from existing data/tables."""
        if "pac" in self._metrics:
            self._call_r_figures_only("roi_pac_analysis.R", "pac_values.csv")

    # -------------------------------------------------------------- summary
    def summary(self) -> None:
        """PAC statistics + figures via R; AAC/PPC emit matrices only (for now)."""
        if "pac" in self._metrics:
            self._run_pac_r()
        if self._aac_rows or self._ppc_rows:
            logger.info(
                "AAC/PPC edge CSVs written; group statistics run via the "
                "connectivity/gating R path (no dedicated R script yet)."
            )

    def _run_pac_r(self) -> None:
        data_dir = self.output_dir / "data"
        if not (data_dir / "pac_values.csv").exists():
            logger.error("pac_values.csv not found -- skipping PAC R analysis")
            return
        try:
            r_dir = _find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return
        r_script = r_dir / "roi_pac_analysis.R"
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return

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
        cmd.extend(self._r_roi_categories_flags())

        logger.info("Calling R: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            for stream in (result.stdout, result.stderr):
                if stream:
                    for line in stream.strip().split("\n"):
                        if line.strip():
                            logger.info("[R] %s", line)
            if result.returncode != 0:
                logger.error("R script failed with exit code %d", result.returncode)
        except FileNotFoundError:
            logger.error("Rscript not found. Install R to enable PAC statistics.")
        except subprocess.TimeoutExpired:
            logger.error("PAC R script timed out")
            return

        if self._generate_figures:
            self._render_brain_mosaics()

    def _render_brain_mosaics(self) -> None:
        """Render brain ROI mosaics from PAC region-level posthoc CSVs."""
        from ..viz.brain_roi import render_posthoc_mosaics

        posthoc_csv = self.tbl_dir / "roi_pac_posthoc_region.csv"
        if not posthoc_csv.exists():
            logger.info("No PAC posthoc region CSV — skipping brain mosaics")
            return
        roi_cats = self.config.roi_categories
        if not roi_cats:
            logger.info("No roi_categories in config — skipping brain mosaics")
            return
        render_posthoc_mosaics(
            posthoc_csv, roi_cats, self.fig_dir,
            analysis_name="roi_cross_freq",
            effect_col="hedges_g", roi_col="region",
            facet_cols=["contrast", "freq_pair"],
            colorbar_label="Hedges' g",
        )

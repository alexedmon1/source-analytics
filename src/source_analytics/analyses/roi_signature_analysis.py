"""ROI-level neural-signature classification analysis.

The source-side counterpart of ``electrode_signature``: the SAME classifiers,
cross-validation and permutation testing, on per-parcel relative band power from
the source-localized ROI time series instead of per-electrode band power. The
feature estimator matches ``electrode_signature`` exactly (Welch PSD per series,
relative band power, this module's own ``epoch_sampling``), so where both ran on
the same epochs the accuracies are directly comparable.

Unlike ``vertex_signature`` it needs no vertex-level estimate, so it runs on any
ROI output, including Monte Carlo operators, which never build one.

``electrode_signature`` renders the comparison when both ran in the same
paradigm. The sensor signature reads the raw recordings, so one run can serve
every source arm: when it ran in another paradigm, name that paradigm with
``sensor_paradigm:`` and this module renders the comparison itself.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.band_power import extract_band_power, relative_power_kwargs
from ..spectral.psd import compute_psd
from .electrode_signature_analysis import (
    ElectrodeSignatureAnalysis,
    render_signature_comparison,
)

logger = logging.getLogger(__name__)


class ROISignatureAnalysis(ElectrodeSignatureAnalysis):
    """Per-parcel neural-signature (classification) analysis."""

    name = "roi_signature"

    _SUMMARY_TITLE = "ROI Neural Signature Analysis Summary"
    _SUMMARY_ANALYSIS = "Source-level (ROI) neural signature (classification)"
    _SUMMARY_METHODS = (
        "Each classifier, with LOOCV, was trained to distinguish groups from the "
        "spatial pattern of per-ROI relative band power in the source-localized "
        "time series. Significance was assessed by permutation testing. The feature "
        "estimator matches electrode_signature, so the `signature_source_vs_sensor` "
        "comparison is like for like.")

    def setup(self) -> None:
        # No electrode roster: the features come from the ROI time series.
        self._feature_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._subject_order.clear()
        self._ch_names = None
        self._ch_coords = None
        self._signature_results.clear()

    def process_subject(self, subject: SubjectInfo) -> None:
        loader = SubjectLoader(subject.data_dir)
        roi_ts = loader.load_or_extract_roi_timeseries(
            signed=True, atlas_dir=self._atlas_dir, rois=self.config.rois)
        sfreq = loader.load_sfreq()
        if self._sfreq is None:
            self._sfreq = sfreq
        rois = list(roi_ts)
        if self._ch_names is None:
            self._ch_names = rois   # feature order; statistics() matches by name

        data = np.stack([np.asarray(roi_ts[r], dtype=float) for r in rois])
        fmax = max(hi for _, hi in self.config.bands.values()) + 10
        rel: dict[tuple[str, str], list[float]] = {}
        for draw in self._get_draws(data, sfreq):
            for i, roi in enumerate(rois):
                x = draw[i, :]
                if np.all(x == 0) or np.any(np.isnan(x)):
                    continue
                freqs, psd = compute_psd(x, sfreq, fmax=fmax)
                bp = extract_band_power(
                    freqs, psd, self._selected_bands(),
                    **relative_power_kwargs(self.config.raw.get("relative_power")))
                for band, vals in bp.items():
                    rel.setdefault((roi, band), []).append(vals["relative"])

        uid = f"{subject.group}_{subject.subject_id}"
        band_power: dict[str, dict[str, float]] = {}
        for (roi, band), vals in rel.items():
            value = float(np.mean(vals))
            band_power.setdefault(band, {})[roi] = value
            self._feature_rows.append({"subject": uid, "group": subject.group,
                                       "roi": roi, "band": band, "relative": value})
        self._subject_groups[uid] = subject.group
        self._subject_order.append(uid)
        self._subject_data[uid] = {"band_power": band_power}

    def aggregate(self) -> None:
        super().aggregate()
        if self._ch_names:
            # The results pickle stores weights without names; figures-only runs
            # need the order back.
            pd.DataFrame({"roi": self._ch_names}).to_csv(
                self.output_dir / "data" / "roi_order.csv", index=False)

    def _load_state_from_disk(self) -> bool:
        ok = super()._load_state_from_disk()
        order = self.output_dir / "data" / "roi_order.csv"
        if ok and order.exists():
            self._ch_names = pd.read_csv(order)["roi"].tolist()
        return ok

    def _plot_importance_topomap(self, values: np.ndarray, title: str, out_path: Path) -> None:
        """|weight| per parcel, ranked (parcels have no montage positions)."""
        if not self._ch_names:
            return
        w = np.abs(np.asarray(values, dtype=float))
        ok = ~np.isnan(w)
        if not ok.any():
            return
        names, vals = np.asarray(self._ch_names)[ok], w[ok]
        order = np.argsort(vals)
        fig, ax = plt.subplots(figsize=(6, max(3.0, 0.28 * len(vals))))
        ax.barh(names[order], vals[order], color="#C0392B")
        ax.set_xlabel("|weight|")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _render_source_vs_sensor(self, fig_dir: Path) -> None:
        """Compare against a sensor signature from ANOTHER paradigm, if named.

        In the same paradigm ``electrode_signature`` renders the comparison;
        doing it here too would duplicate it.
        """
        para = self.config.raw.get(self.name, {}).get("sensor_paradigm")
        if not para or para == self.config.paradigm_name:
            return
        sensor_csv = (self.tbl_dir.parent.parent / para / "electrode_signature"
                      / "electrode_signature_results.csv")
        source_csv = self.tbl_dir / f"{self.name}_results.csv"
        if not (sensor_csv.exists() and source_csv.exists()):
            logger.info("Source-vs-sensor comparison skipped: %s not found", sensor_csv)
            return
        render_signature_comparison(source_csv, sensor_csv, self.tbl_dir, fig_dir,
                                    source_module=self.name)

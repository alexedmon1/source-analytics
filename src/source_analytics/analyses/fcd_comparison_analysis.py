"""Source-vs-sensor functional-connectivity-density (FCD) comparison.

Compares vertex-level (source) FCD against electrode-level (sensor) FCD — the
head-to-head behind the MS2 thesis that source connectivity recovers spatial
structure that sensor connectivity blurs. Reads the per-element FCD tables that
``vertex_connectivity`` and ``electrode_connectivity`` already persist, and for
each subject × band × metric derives two per-subject summaries:

* ``fcd_mean`` — mean FCD over the map's units (global connectivity density).
  FCD is degree/(n-1)-normalized, so ``fcd_mean`` is directly comparable across
  the two resolutions (30 channels vs. hundreds of vertices).
  [Tomasi & Volkow 2010, PNAS 107(21):9885-9890 — FCD mapping.]
* ``fcd_cv`` — coefficient of variation (SD/mean) of FCD across units: the
  *spatial heterogeneity* of connectivity density. Higher CV = more structure
  (hubs vs. a flat field); the "source resolves, sensor blurs" quantity.
  [Standard CV; see e.g. Everitt & Skrondal, Cambridge Dict. of Statistics.]

For each (band, metric) it reports the cross-subject concordance (Pearson r,
source vs. sensor) of each summary, and the per-contrast group effect (Hedges g
+ 95% CI) at BOTH levels with a sign-concordance flag. NB the absolute CV is not
comparable across resolutions (more units sample the FCD field more finely), but
each group contrast is WITHIN a level, so the source-vs-sensor comparison of the
group EFFECT is resolution-fair — the same logic as ``electrode_comparison`` for
spectral power.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..viz.constants import metric_display, order_bands
from .base import BaseAnalysis
from .electrode_comparison_analysis import _hedges_g_ci

logger = logging.getLogger(__name__)


def _fcd_summaries(fcd_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-subject FCD summaries from a long per-element FCD table.

    ``fcd_df`` has columns subject, group, band, metric, fcd (one row per spatial
    unit — channel or vertex). Returns one row per (subject, group, band, metric)
    with ``<prefix>_mean`` (mean FCD) and ``<prefix>_cv`` (SD/mean over units).
    """
    def _agg(g: pd.DataFrame) -> pd.Series:
        vals = g["fcd"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        mean = float(np.mean(vals)) if vals.size else np.nan
        if vals.size < 2 or not np.isfinite(mean) or mean == 0:
            cv = np.nan
        else:
            cv = float(np.std(vals, ddof=1) / mean)
        return pd.Series({f"{prefix}_mean": mean, f"{prefix}_cv": cv})

    return (
        fcd_df.groupby(["subject", "group", "band", "metric"], sort=False)
        .apply(_agg)
        .reset_index()
    )


class FCDComparisonAnalysis(BaseAnalysis):
    """Compare source (vertex) vs sensor (electrode) FCD, per band × metric."""

    name = "fcd_comparison"
    SELECTABLE = {"metric": "connectivity metric", "band": "frequency band"}

    # (prefix, native summary column) — the two per-subject FCD summaries compared
    _SUMMARIES = ("mean", "cv")

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._sensor_df: pd.DataFrame | None = None
        self._source_df: pd.DataFrame | None = None
        self._comparison_df: pd.DataFrame | None = None
        cfg = config.raw.get(self.name, {})
        self._metric_filter = cfg.get("metrics")

    def setup(self) -> None:
        base = self.config.output_dir
        sensor_csv = base / "electrode_connectivity" / "data" / "electrode_fcd.csv"
        source_csv = base / "vertex_connectivity" / "data" / "vertex_fcd.csv"
        if not sensor_csv.exists():
            raise FileNotFoundError(
                f"{sensor_csv} not found — run 'electrode_connectivity' first."
            )
        if not source_csv.exists():
            raise FileNotFoundError(
                f"{source_csv} not found — run 'vertex_connectivity' first."
            )
        self._sensor_df = pd.read_csv(sensor_csv)
        self._source_df = pd.read_csv(source_csv)
        logger.info(
            "Loaded FCD: sensor=%d rows, source=%d rows",
            len(self._sensor_df), len(self._source_df),
        )

    def process_subject(self, subject: SubjectInfo) -> None:  # noqa: D401
        """No-op — reads persisted FCD tables."""

    def aggregate(self) -> None:
        if self._sensor_df is None or self._source_df is None:
            return
        sensor = _fcd_summaries(self._sensor_df, "sensor")
        source = _fcd_summaries(self._source_df, "source")
        comp = source.merge(sensor, on=["subject", "group", "band", "metric"], how="inner")

        # metric intersection, honoring --metric / config metrics filter
        metrics = sorted(comp["metric"].unique())
        if self._metric_filter:
            metrics = [m for m in metrics if m in set(self._metric_filter)]
        metrics = self._select("metric", metrics)
        comp = comp[comp["metric"].isin(metrics)]

        bands = self._select("band", order_bands(comp["band"].unique(), self.config))
        comp = comp[comp["band"].isin(bands)]

        self._comparison_df = comp.reset_index(drop=True)
        out = self.output_dir / "data" / "fcd_subject_summary.csv"
        self._comparison_df.to_csv(out, index=False)
        logger.info("Exported fcd_subject_summary.csv (%d rows)", len(self._comparison_df))

    def statistics(self) -> None:
        from scipy import stats as sp_stats

        comp = self._comparison_df
        if comp is None or comp.empty:
            logger.warning("No FCD comparison data — skipping statistics")
            return

        rows = []
        for (band, metric), bdata in comp.groupby(["band", "metric"], sort=False):
            row: dict = {"band": band, "metric": metric, "n_subjects": len(bdata)}
            # cross-subject concordance (source vs sensor) of each summary
            for s in self._SUMMARIES:
                valid = bdata[[f"sensor_{s}", f"source_{s}"]].dropna()
                if len(valid) > 2:
                    r, p = sp_stats.pearsonr(valid[f"sensor_{s}"], valid[f"source_{s}"])
                else:
                    r, p = np.nan, np.nan
                row[f"corr_{s}_r"] = r
                row[f"corr_{s}_p"] = p

            base_row = dict(row)
            for contrast in self._pairwise_contrasts():
                ga = bdata[bdata["group"] == contrast.group_a]
                gb = bdata[bdata["group"] == contrast.group_b]
                r = dict(base_row)
                r["contrast"] = contrast.name
                for s in self._SUMMARIES:
                    sen_g, sen_lo, sen_hi = _hedges_g_ci(
                        ga[f"sensor_{s}"].values, gb[f"sensor_{s}"].values)
                    src_g, src_lo, src_hi = _hedges_g_ci(
                        ga[f"source_{s}"].values, gb[f"source_{s}"].values)
                    r[f"sensor_{s}_g"] = sen_g
                    r[f"sensor_{s}_ci_lo"] = sen_lo
                    r[f"sensor_{s}_ci_hi"] = sen_hi
                    r[f"source_{s}_g"] = src_g
                    r[f"source_{s}_ci_lo"] = src_lo
                    r[f"source_{s}_ci_hi"] = src_hi
                    r[f"{s}_concordant"] = bool(
                        np.isfinite(sen_g) and np.isfinite(src_g)
                        and np.sign(sen_g) == np.sign(src_g)
                    )
                rows.append(r)

        stats_df = pd.DataFrame(rows)
        if not stats_df.empty:
            path = self.tbl_dir / "fcd_comparison_stats.csv"
            stats_df.to_csv(path, index=False)
            logger.info("Exported fcd_comparison_stats.csv (%d rows)", len(stats_df))
        self._stats_df = stats_df

    def figures(self) -> None:
        # Regenerable from persisted data: reload the per-subject summary if the
        # in-memory frame is absent (e.g. `--steps figures` standalone).
        comp = self._comparison_df
        if comp is None or comp.empty:
            summ = self.output_dir / "data" / "fcd_subject_summary.csv"
            if summ.exists():
                comp = pd.read_csv(summ)
        if comp is None or comp.empty:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for metric, mdata in comp.groupby("metric", sort=False):
            # (1) concordance scatter: source vs sensor mean FCD, colored by band
            fig, ax = plt.subplots(figsize=(5, 5))
            bands = order_bands(mdata["band"].unique(), self.config)
            cmap = plt.get_cmap("viridis", max(len(bands), 1))
            for i, band in enumerate(bands):
                d = mdata[mdata["band"] == band][["sensor_mean", "source_mean"]].dropna()
                if not d.empty:
                    ax.scatter(d["sensor_mean"], d["source_mean"], s=18,
                               color=cmap(i), alpha=0.7, label=band)
            lims = [
                float(np.nanmin([mdata["sensor_mean"].min(), mdata["source_mean"].min()])),
                float(np.nanmax([mdata["sensor_mean"].max(), mdata["source_mean"].max()])),
            ]
            if np.all(np.isfinite(lims)):
                ax.plot(lims, lims, ls="--", c="grey", lw=0.8)
            ax.set_xlabel("Sensor mean FCD")
            ax.set_ylabel("Source mean FCD")
            ax.set_title(f"Source vs sensor mean FCD — {metric_display(metric)}")
            ax.legend(fontsize=7, title="Band")
            fig.tight_layout()
            fig.savefig(self.fig_dir / f"fcd_concordance_mean_{metric}.png", dpi=200)
            plt.close(fig)

            # (2) spatial heterogeneity: mean CV by group, source vs sensor, per band
            fig, ax = plt.subplots(figsize=(6, 4))
            grp = (mdata.groupby("band")[["sensor_cv", "source_cv"]]
                   .mean().reindex(bands))
            x = np.arange(len(bands))
            ax.bar(x - 0.2, grp["sensor_cv"], width=0.4, label="Sensor", color="#888")
            ax.bar(x + 0.2, grp["source_cv"], width=0.4, label="Source", color="#B2182B")
            ax.set_xticks(x)
            ax.set_xticklabels(bands, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("FCD spatial CV (SD/mean)")
            ax.set_title(f"FCD spatial heterogeneity — {metric_display(metric)}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(self.fig_dir / f"fcd_heterogeneity_{metric}.png", dpi=200)
            plt.close(fig)

    def summary(self) -> None:
        stats_df = getattr(self, "_stats_df", None)
        lines = ["# FCD Source-vs-Sensor Comparison\n"]
        if stats_df is None or stats_df.empty:
            lines.append("*No comparison statistics computed.*\n")
        else:
            n_mean_conc = int(stats_df.get("mean_concordant", pd.Series(dtype=bool)).sum())
            n_cv_conc = int(stats_df.get("cv_concordant", pd.Series(dtype=bool)).sum())
            lines.append(
                f"{len(stats_df)} band × metric × contrast cells. "
                f"Group-effect sign-concordance source-vs-sensor: "
                f"mean FCD {n_mean_conc}/{len(stats_df)}, "
                f"spatial CV {n_cv_conc}/{len(stats_df)}.\n"
            )
            lines.append(
                "Tables: `fcd_comparison_stats.csv` (concordance r + per-contrast "
                "Hedges g at both levels), `fcd_subject_summary.csv` (per-subject "
                "mean + CV).\n"
            )
        (self.output_dir / "ANALYSIS_SUMMARY.md").write_text("".join(lines), encoding="utf-8")

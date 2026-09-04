"""Vertex-level spectral parameterization analysis.

Fits aperiodic (1/f) models at each vertex to decompose the power spectrum
into aperiodic (1/f) and oscillatory components.  Detects peaks in every
configured frequency band and tests group differences in aperiodic
exponent, offset, and per-band peak presence using cluster permutation
and chi-squared tests.
"""

from __future__ import annotations

import logging
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sp_stats

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from ..io.loader import SubjectLoader
from ..spectral.aperiodic import band_peak_reachability, resolve_freq_range
from ..spectral.vertex import compute_psd_vertices
from ..spectral.vertex_aperiodic import fit_aperiodic_vertices
from ..spectral.epoch_sampler import sample_epochs
from ..stats.cluster_permutation import (
    cluster_permutation_test,
    has_significant_cluster as _has_significant_cluster,
    hedges_g,
)
from ..viz.glass_brain import plot_glass_brain, plot_band_comparison
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


class VertexSpecparamAnalysis(BaseAnalysis):
    """Vertex-level spectral parameterization analysis."""

    name = "vertex_specparam"
    SELECTABLE = {"hypothesis": "declared hypothesis"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._param_rows: list[dict] = []
        self._peak_rows: list[dict] = []
        self._source_coords: np.ndarray | None = None
        self._sfreq: float | None = None
        self._subject_data: dict[str, dict] = {}
        self._subject_groups: dict[str, str] = {}

        # Config
        sp_cfg = config.raw.get("vertex_specparam", {})
        # Shared package default (2-50 Hz). The previous default here was
        # 1-100 Hz, which spans the 57-63 Hz notch and the >80 Hz roll-off and
        # collapsed the vertex fits to r^2~0.16 / exponent~0.04 (flat).
        self._freq_range = resolve_freq_range(sp_cfg)
        self._peak_width_limits = tuple(sp_cfg.get("peak_width_limits", [1.0, 12.0]))
        self._max_n_peaks = int(sp_cfg.get("max_n_peaks", 6))
        # Separate, wider window for PEAK detection. The narrow aperiodic window
        # makes every band outside it structurally undetectable, which would turn
        # the fit-window choice into an unfalsifiable assertion: the borders are
        # justified by where the peaks are, so the peaks must be measured over a
        # window that does not presuppose the answer. Unset => single fit.
        self._peak_freq_range = (
            resolve_freq_range(sp_cfg, key="peak_freq_range")
            if sp_cfg.get("peak_freq_range") else self._freq_range
        )

        # Frequency bands for peak detection
        self._bands = dict(config.bands)
        # Only bands the PEAK window can reach get peak columns — an unreachable
        # band's absence is structural, and emitting False for it fabricates a
        # measured null (see spectral.aperiodic.band_peak_reachability).
        self._band_reach = band_peak_reachability(self._bands, self._peak_freq_range)
        self._peak_bands = {
            n: b for n, b in self._bands.items() if self._band_reach[n]["reachable"]
        }
        self._band_keys = {
            name: name.lower().replace(" ", "_") for name in self._peak_bands
        }

        wb_cfg = config.vertex
        self._n_permutations = int(wb_cfg.get("n_permutations", 1000))
        self._adjacency_distance = float(wb_cfg.get("adjacency_distance_mm", 5.0))
        self._cluster_threshold = float(wb_cfg.get("cluster_threshold", 2.0))
        self._noise_exclude = wb_cfg.get("noise_exclude_hz")
        if self._noise_exclude is not None:
            self._noise_exclude = tuple(self._noise_exclude)

        # Global epoch_sampling → vertex: block → per-analysis block (see base).
        self._epoch_config = self._vertex_epoch_config()
        self._cluster_results: dict = {}

    def setup(self) -> None:
        self._param_rows.clear()
        self._peak_rows.clear()
        self._subject_data.clear()
        self._subject_groups.clear()
        self._source_coords = None
        self._cluster_results.clear()

    def _compute_subject(self, subject: SubjectInfo):
        """Pure per-subject specparam/FOOOF compute (parallel-safe)."""
        loader = SubjectLoader(subject.data_dir)
        uid = f"{subject.group}_{subject.subject_id}"

        stc_data = loader.load_source_timecourses()
        sfreq = loader.load_sfreq()
        coords = loader.load_source_coords()

        # Compute PSD — must span the WIDER of the two windows
        fmax = max(self._freq_range[1], self._peak_freq_range[1]) + 10
        if self._epoch_config is not None:
            epochs = sample_epochs(
                stc_data, sfreq,
                epoch_duration_sec=self._epoch_config.get("epoch_duration_sec", 2.0),
                n_epochs=self._epoch_config.get("n_epochs", 80),
                seed=self._epoch_config.get("seed", 42),
                n_bootstrap=self._epoch_config.get("n_bootstrap", 1),
            )
            all_psd = []
            for ep in epochs:
                f, p = compute_psd_vertices(ep, sfreq, fmax=fmax)
                all_psd.append(p)
            freqs = f
            psd = np.mean(all_psd, axis=0)
        else:
            freqs, psd = compute_psd_vertices(stc_data, sfreq, fmax=fmax)

        # Fit specparam at each vertex
        params = fit_aperiodic_vertices(
            freqs, psd,
            freq_range=self._freq_range,
            max_n_peaks=self._max_n_peaks,
            peak_width_limits=self._peak_width_limits,
            bands=self._bands,
            peak_freq_range=self._peak_freq_range,
        )

        param_rows: list[dict] = []
        n_vertices = psd.shape[0]
        for vi in range(n_vertices):
            row = {
                "subject": uid,
                "group": subject.group,
                "vertex_idx": vi,
                "exponent": float(params["exponent"][vi]),
                "offset": float(params["offset"][vi]),
                # Offset at the fit-window centre — the one to report alongside
                # exponent (the 1 Hz-referenced offset is mechanically coupled to
                # the slope). Matches roi_aperiodic/electrode_aperiodic.
                "offset_centered": float(params["offset_centered"][vi]),
                "r_squared": float(params["r_squared"][vi]),
                "n_peaks": int(params["n_peaks"][vi]),
                "n_peaks_wide": int(params["n_peaks_wide"][vi]),
                "method": params["method"][vi],
                "fit_fmin": float(self._freq_range[0]),
                "fit_fmax": float(self._freq_range[1]),
                "peak_fmin": float(self._peak_freq_range[0]),
                "peak_fmax": float(self._peak_freq_range[1]),
            }
            for key in self._band_keys.values():
                row[f"has_{key}_peak"] = bool(params[f"has_{key}_peak"][vi])
                row[f"{key}_peak_freq"] = float(params[f"{key}_peak_freq"][vi])
                row[f"{key}_peak_power"] = float(params[f"{key}_peak_power"][vi])
            param_rows.append(row)

        # Long-format inventory of every peak found by the peak-window fit.
        # This is the raw material for the fit-window diagnostic: where the
        # oscillations actually are, independent of any band definition.
        peak_rows = [
            {"subject": uid, "group": subject.group, **pk}
            for vertex_peaks in params["peaks_all"] for pk in vertex_peaks
        ]

        return {
            "uid": uid, "group": subject.group, "sfreq": float(sfreq),
            "source_coords": coords, "params": params, "param_rows": param_rows,
            "peak_rows": peak_rows,
        }

    def _merge_subject(self, payload) -> None:
        uid = payload["uid"]
        if self._sfreq is None:
            self._sfreq = payload["sfreq"]
        if self._source_coords is None:
            self._source_coords = payload["source_coords"]
        self._subject_groups[uid] = payload["group"]
        self._subject_data[uid] = payload["params"]
        self._param_rows.extend(payload["param_rows"])
        self._peak_rows.extend(payload.get("peak_rows", []))

    def aggregate(self) -> None:
        data_dir = self.output_dir / "data"

        param_df = pd.DataFrame(self._param_rows)
        if param_df.empty:
            logger.warning("No specparam data collected")
            return
        param_df.to_csv(data_dir / "vertex_specparam.csv", index=False)
        logger.info("Exported vertex_specparam.csv (%d rows)", len(param_df))

        if self._peak_rows:
            peak_df = pd.DataFrame(self._peak_rows)
            peak_df.to_csv(data_dir / "peak_inventory.csv", index=False)
            logger.info("Exported peak_inventory.csv (%d peaks)", len(peak_df))
            self._write_fit_window_diagnostic(peak_df)

        if self._source_coords is not None:
            coords_df = pd.DataFrame(self._source_coords, columns=["x", "y", "z"])
            coords_df.index.name = "vertex_idx"
            coords_df.to_csv(data_dir / "source_coords.csv")

    def _write_fit_window_diagnostic(self, peak_df: pd.DataFrame) -> None:
        """Check the aperiodic fit window against where the oscillations are.

        Gerster et al. 2022: "Oscillations crossing the fitting range borders
        must be avoided for all investigated power spectra" — a peak sitting on
        a border produces large exponent error. That rule is testable, and this
        table tests it on the study's own spectra instead of asserting it.

        A peak is treated as crossing a border when its support (centre
        frequency +/- half the specparam bandwidth, i.e. +/-1 SD of the fitted
        Gaussian) straddles that border. This catches both a peak centred inside
        the window whose tail leaks out, and one centred outside whose tail
        leaks in.
        """
        fmin, fmax = self._freq_range
        n_fits = len(self._param_rows) or 1

        cf = peak_df["center_frequency"].to_numpy(dtype=float)
        bw = peak_df["bandwidth"].to_numpy(dtype=float)
        half = np.where(np.isfinite(bw), bw / 2.0, 0.0)
        lo_edge, hi_edge = cf - half, cf + half
        cross_lo = (lo_edge < fmin) & (hi_edge > fmin)
        cross_hi = (lo_edge < fmax) & (hi_edge > fmax)

        def _row(name, blo, bhi, mask, reach):
            sub_cf = cf[mask]
            n = int(mask.sum())
            return {
                "band": name,
                "band_lo": blo,
                "band_hi": bhi,
                "aperiodic_fmin": float(fmin),
                "aperiodic_fmax": float(fmax),
                "peak_fmin": float(self._peak_freq_range[0]),
                "peak_fmax": float(self._peak_freq_range[1]),
                "reachable": reach.get("reachable", True),
                "censored": reach.get("censored", False),
                "frac_visible": reach.get("frac_visible", 1.0),
                "n_peaks": n,
                "peaks_per_fit": float(n / n_fits),
                "cf_median": float(np.median(sub_cf)) if n else float("nan"),
                "cf_p5": float(np.percentile(sub_cf, 5)) if n else float("nan"),
                "cf_p95": float(np.percentile(sub_cf, 95)) if n else float("nan"),
                "n_cross_fmin": int((cross_lo & mask).sum()),
                "n_cross_fmax": int((cross_hi & mask).sum()),
                "frac_crossing": float(
                    ((cross_lo | cross_hi) & mask).sum() / n) if n else 0.0,
            }

        rows = [_row(
            "ALL", float(self._peak_freq_range[0]), float(self._peak_freq_range[1]),
            np.ones(len(cf), dtype=bool), {},
        )]
        for name, (blo, bhi) in self._bands.items():
            reach = self._band_reach[name]
            in_band = (cf >= blo) & (cf <= bhi)
            rows.append(_row(name, float(blo), float(bhi), in_band, reach))

        # Peaks the aperiodic window deliberately excludes — context for the
        # crossing counts: a window that excludes a lot of oscillatory activity
        # is only defensible if that activity is clear of the borders.
        outside = (cf < fmin) | (cf > fmax)
        diag = pd.DataFrame(rows)
        diag.to_csv(self.tbl_dir / "fit_window_diagnostic.csv", index=False)

        n_total = len(cf)
        n_cross = int((cross_lo | cross_hi).sum())
        logger.info(
            "Fit-window diagnostic: %d peaks over %.4g-%.4g Hz; %d (%.1f%%) "
            "cross the aperiodic borders %.4g/%.4g Hz; %d (%.1f%%) lie outside "
            "the aperiodic window entirely.",
            n_total, self._peak_freq_range[0], self._peak_freq_range[1],
            n_cross, 100.0 * n_cross / max(n_total, 1), fmin, fmax,
            int(outside.sum()), 100.0 * outside.sum() / max(n_total, 1),
        )

    def statistics(self) -> None:
        if self._source_coords is None:
            logger.error("No source coordinates")
            return

        coords = self._source_coords
        tbl_dir = self.tbl_dir
        all_stats = []

        for contrast in self._pairwise_contrasts():
            group_a_uids = [
                uid for uid, g in self._subject_groups.items() if g == contrast.group_a
            ]
            group_b_uids = [
                uid for uid, g in self._subject_groups.items() if g == contrast.group_b
            ]

            if not group_a_uids or not group_b_uids:
                continue

            label_a = self.config.get_group_label(contrast.group_a)
            label_b = self.config.get_group_label(contrast.group_b)

            # Cluster permutation on exponent and offset maps
            for param_name in ["exponent", "offset"]:
                data_a = np.array([
                    self._subject_data[uid][param_name] for uid in group_a_uids
                ])
                data_b = np.array([
                    self._subject_data[uid][param_name] for uid in group_b_uids
                ])

                result = cluster_permutation_test(
                    data_a, data_b, coords,
                    n_perms=self._n_permutations,
                    threshold=self._cluster_threshold,
                    distance_mm=self._adjacency_distance,
                    seed=42,
                )

                g_map = hedges_g(data_a, data_b)

                self._cluster_results[f"{contrast.name}_{param_name}"] = {
                    "result": result,
                    "mean_a": data_a.mean(axis=0),
                    "mean_b": data_b.mean(axis=0),
                    "group_labels": (label_a, label_b),
                    "param": param_name,
                }

                for vi in range(len(result.t_map)):
                    cid = int(result.cluster_labels[vi])
                    cp = float(result.cluster_pvalues[cid - 1]) if cid > 0 else float("nan")
                    all_stats.append({
                        "contrast": contrast.name,
                        "parameter": param_name,
                        "vertex_idx": vi,
                        "t": float(result.t_map[vi]),
                        "p": float(result.p_map[vi]),
                        "hedges_g": float(g_map[vi]),
                        "cluster_id": cid,
                        # Corrected per-cluster p and significance — cluster_id alone
                        # is NOT significance (clusters are candidates pre-permutation).
                        "cluster_p": cp,
                        "significant": bool(cid > 0 and cp < 0.05),
                    })

            # Per-band chi-squared tests on peak presence + optional
            # cluster permutation on peak power
            n_a, n_b = len(group_a_uids), len(group_b_uids)
            all_chi2_stats: list[dict] = []

            # Detectable bands only. A band the peak window cannot reach has no
            # column to test, and testing it would emit p=1.0 at every vertex —
            # a null the data never had the power to reject.
            for band_name in self._peak_bands:
                key = self._band_keys[band_name]
                col = f"has_{key}_peak"

                peak_a = np.array([
                    self._subject_data[uid][col] for uid in group_a_uids
                ])  # (n_a, n_vertices) bool
                peak_b = np.array([
                    self._subject_data[uid][col] for uid in group_b_uids
                ])

                rate_a = peak_a.mean(axis=0)
                rate_b = peak_b.mean(axis=0)

                for vi in range(len(rate_a)):
                    a_yes = int(peak_a[:, vi].sum())
                    a_no = n_a - a_yes
                    b_yes = int(peak_b[:, vi].sum())
                    b_no = n_b - b_yes

                    table = np.array([[a_yes, a_no], [b_yes, b_no]])
                    if table.sum() > 0 and np.all(table.sum(axis=0) > 0):
                        chi2, p_val, _, _ = sp_stats.chi2_contingency(
                            table, correction=True,
                        )
                    else:
                        chi2, p_val = 0.0, 1.0

                    all_chi2_stats.append({
                        "contrast": contrast.name,
                        "band": band_name,
                        "band_key": key,
                        "vertex_idx": vi,
                        "rate_a": float(rate_a[vi]),
                        "rate_b": float(rate_b[vi]),
                        "chi2": float(chi2),
                        "p": float(p_val),
                    })

                # Cluster permutation on peak power for bands with
                # enough detected peaks (>=10% of vertices overall)
                overall_rate = np.concatenate([peak_a, peak_b]).mean(axis=0)
                if overall_rate.mean() >= 0.10:
                    power_a = np.array([
                        self._subject_data[uid][f"{key}_peak_power"]
                        for uid in group_a_uids
                    ])
                    power_b = np.array([
                        self._subject_data[uid][f"{key}_peak_power"]
                        for uid in group_b_uids
                    ])
                    power_a = np.nan_to_num(power_a, nan=0.0)
                    power_b = np.nan_to_num(power_b, nan=0.0)

                    result = cluster_permutation_test(
                        power_a, power_b, coords,
                        n_perms=self._n_permutations,
                        threshold=self._cluster_threshold,
                        distance_mm=self._adjacency_distance,
                        seed=42,
                    )
                    g_map = hedges_g(power_a, power_b)

                    self._cluster_results[
                        f"{contrast.name}_{key}_peak_power"
                    ] = {
                        "result": result,
                        "mean_a": power_a.mean(axis=0),
                        "mean_b": power_b.mean(axis=0),
                        "group_labels": (label_a, label_b),
                        "param": f"{key}_peak_power",
                    }

                    for vi in range(len(result.t_map)):
                        cid = int(result.cluster_labels[vi])
                        cp = float(result.cluster_pvalues[cid - 1]) if cid > 0 else float("nan")
                        all_stats.append({
                            "contrast": contrast.name,
                            "parameter": f"{key}_peak_power",
                            "vertex_idx": vi,
                            "t": float(result.t_map[vi]),
                            "p": float(result.p_map[vi]),
                            "hedges_g": float(g_map[vi]),
                            "cluster_id": cid,
                            "cluster_p": cp,
                            "significant": bool(cid > 0 and cp < 0.05),
                        })

            if all_chi2_stats:
                chi2_df = pd.DataFrame(all_chi2_stats)
                chi2_df.to_csv(tbl_dir / "band_peak_chi2.csv", index=False)
                # Backward compat: gamma-only subset
                gamma_keys = [
                    k for k in self._band_keys.values() if "gamma" in k
                ]
                gamma_sub = chi2_df[chi2_df["band_key"].isin(gamma_keys)]
                if not gamma_sub.empty:
                    gamma_sub.to_csv(
                        tbl_dir / "gamma_peak_chi2.csv", index=False,
                    )

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_df.to_csv(tbl_dir / "vertex_specparam_stats.csv", index=False)
            logger.info("Exported vertex_specparam_stats.csv (%d rows)", len(stats_df))

        # Save cluster results for --steps figures support
        if self._cluster_results:
            data_dir = self.output_dir / "data"
            pkl_data = {}
            for key, info in self._cluster_results.items():
                result = info["result"]
                pkl_data[key] = {
                    "t_map": result.t_map,
                    "p_map": result.p_map,
                    "cluster_labels": result.cluster_labels,
                    "cluster_pvalues": result.cluster_pvalues,
                    "cluster_stats": result.cluster_stats,
                    "n_clusters": result.n_clusters,
                    "n_permutations": result.n_permutations,
                    "mean_a": info["mean_a"],
                    "mean_b": info["mean_b"],
                    "group_labels": info["group_labels"],
                    "param": info["param"],
                }
            with open(data_dir / "specparam_cluster_results.pkl", "wb") as f:
                pickle.dump(pkl_data, f)
            logger.info("Saved specparam_cluster_results.pkl")

        # --- Declarative hypotheses (hypothesis layer; additive, map+cluster) ---
        # exponent/offset are broadband per-vertex maps (no band dimension).
        from ..hypothesis import write_module_hypotheses_perm

        if self._source_coords is not None and self._subject_groups:
            maps_by_cell = {
                ("broadband", param): {
                    uid: self._subject_data[uid][param]
                    for uid in self._subject_groups
                }
                for param in ["exponent", "offset"]
            }
            wanted_hyp = self._selection.get("hypothesis")
            write_module_hypotheses_perm(
                maps_by_cell, self._subject_groups, self._source_coords, self.config,
                self.tbl_dir, prefix="vertex_specparam",
                n_perms=self._n_permutations, threshold=self._cluster_threshold,
                distance_mm=self._adjacency_distance,
                hypothesis=",".join(sorted(wanted_hyp)) if wanted_hyp else None,
                atlas_dir=self._atlas_dir,
            )

    def _load_state_from_disk(self) -> bool:
        """Load saved state from pickle for --steps figures support."""
        from ..stats.cluster_permutation import ClusterResult

        data_dir = self.output_dir / "data"
        pkl_path = data_dir / "specparam_cluster_results.pkl"
        if not pkl_path.exists():
            logger.warning("No saved state at %s; skipping figures", pkl_path)
            return False

        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        for key, d in saved.items():
            self._cluster_results[key] = {
                "result": ClusterResult(
                    t_map=d["t_map"],
                    p_map=d["p_map"],
                    cluster_labels=d["cluster_labels"],
                    cluster_pvalues=d["cluster_pvalues"],
                    cluster_stats=d["cluster_stats"],
                    n_clusters=d.get("n_clusters", 0),
                    n_permutations=d.get("n_permutations", 0),
                ),
                "mean_a": d["mean_a"],
                "mean_b": d["mean_b"],
                "group_labels": d["group_labels"],
                "param": d["param"],
            }

        # Load source coords
        coords_csv = data_dir / "source_coords.csv"
        if coords_csv.exists():
            coords_df = pd.read_csv(coords_csv)
            self._source_coords = coords_df[["x", "y", "z"]].values

        logger.info("Loaded vertex_specparam state from %s", pkl_path)
        return True

    def figures(self) -> None:
        # Load from disk if in-memory state is missing (--steps support)
        if not self._cluster_results or self._source_coords is None:
            if not self._load_state_from_disk():
                return

        if self._source_coords is None:
            return

        coords = self._source_coords
        fig_dir = self.fig_dir

        for key, info in self._cluster_results.items():
            result = info["result"]
            if not _has_significant_cluster(result):
                continue
            param = info["param"]
            contrast = info.get("contrast", key)
            group_labels = info["group_labels"]

            safe = f"{contrast}_{param}".lower().replace(" ", "_")
            plot_band_comparison(
                coords=coords,
                mean_a=info["mean_a"],
                mean_b=info["mean_b"],
                t_map=result.t_map,
                cluster_labels=result.cluster_labels,
                cluster_pvalues=result.cluster_pvalues,
                band_name=f"{param} — {contrast}",
                group_labels=group_labels,
                output_path=fig_dir / f"specparam_{safe}.png",
            )

        self._plot_fit_window_diagnostic()

        # Per-band peak presence maps
        band_keys = self._band_keys if self._band_keys else {}
        # Detect band keys from CSV if in-memory state is empty
        if not band_keys:
            data_dir = self.output_dir / "data"
            csv_path = data_dir / "vertex_specparam.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                import re
                for col in df.columns:
                    m = re.match(r"has_(.+)_peak$", col)
                    if m:
                        band_keys[m.group(1)] = m.group(1)  # key == key

        for band_name, key in (
            self._band_keys.items() if self._band_keys else
            {k: k for k in band_keys}.items()
        ):
            col = f"has_{key}_peak"
            if self._subject_data:
                try:
                    all_peaks = np.array([
                        d[col].astype(float) for d in self._subject_data.values()
                    ])
                    mean_rate = all_peaks.mean(axis=0)
                except KeyError:
                    continue
            else:
                data_dir = self.output_dir / "data"
                csv_path = data_dir / "vertex_specparam.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if col not in df.columns:
                    continue
                mean_rate = df.groupby("vertex_idx")[col].mean().values

            plot_glass_brain(
                coords=coords,
                values=mean_rate,
                title=f"{band_name} Peak Presence Rate",
                output_path=fig_dir / f"{key}_peak_presence.png",
                cmap="YlOrRd",
                vlim=(0, 1),
            )

    def _plot_fit_window_diagnostic(self) -> None:
        """The fit-window justification figure: peaks vs the aperiodic borders.

        Regenerated from ``data/peak_inventory.csv`` so ``--steps figures``
        alone reproduces it (no in-memory state).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        csv_path = self.output_dir / "data" / "peak_inventory.csv"
        if not csv_path.exists():
            return
        peaks = pd.read_csv(csv_path)
        if peaks.empty:
            return

        fmin, fmax = self._freq_range
        pmin, pmax = self._peak_freq_range
        cf = peaks["center_frequency"].to_numpy(dtype=float)
        bw = peaks["bandwidth"].to_numpy(dtype=float)
        half = np.where(np.isfinite(bw), bw / 2.0, 0.0)
        crossing = ((cf - half < fmin) & (cf + half > fmin)) | \
                   ((cf - half < fmax) & (cf + half > fmax))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bins = np.linspace(pmin, pmax, 80)
        ax.hist(cf[~crossing], bins=bins, color="#4C78A8",
                label=f"clear of borders (n={int((~crossing).sum())})")
        ax.hist(cf[crossing], bins=bins, color="#E45756",
                label=f"support crosses a border (n={int(crossing.sum())})")

        ax.axvspan(pmin, fmin, color="0.85", zorder=0)
        ax.axvspan(fmax, pmax, color="0.85", zorder=0)
        for border in (fmin, fmax):
            ax.axvline(border, color="k", ls="--", lw=1.5)
        ax.set_xlim(pmin, pmax)
        ax.set_xlabel("Peak centre frequency (Hz)")
        ax.set_ylabel("Peaks detected")
        ax.set_title(
            f"Fit-window check — peaks detected over {pmin:g}-{pmax:g} Hz "
            f"vs aperiodic window {fmin:g}-{fmax:g} Hz (dashed)"
        )
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "fit_window_diagnostic.png", dpi=150)
        plt.close(fig)

    def summary(self) -> None:
        data_dir = self.output_dir / "data"

        config_path = data_dir / "study_config.yaml"
        config_data = dict(self.config.raw)
        if self._sfreq is not None:
            config_data["sfreq"] = self._sfreq
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        try:
            r_dir = _find_r_script_dir()
            r_script = r_dir / "vertex_specparam_analysis.R"
            if r_script.exists():
                cmd = [
                    "Rscript", str(r_script),
                    "--data-dir", str(data_dir),
                    "--config", str(config_path),
                    "--output-dir", str(self.output_dir),
                    "--fig-dir", str(self.fig_dir),
                    "--tbl-dir", str(self.tbl_dir),
                ]
                cmd.extend(self._r_no_figures_flags())
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        self._write_python_summary()

    def _fit_window_summary_lines(self) -> list[str]:
        """Report the border check before the results it underwrites."""
        diag_csv = self.tbl_dir / "fit_window_diagnostic.csv"
        if not diag_csv.exists():
            return []
        diag = pd.read_csv(diag_csv)
        all_rows = diag[diag["band"] == "ALL"]
        if all_rows.empty:
            return []
        a = all_rows.iloc[0]

        frac = float(a["frac_crossing"])
        if frac < 0.05:
            verdict = "SATISFIED — the borders sit in spectral gaps on this data."
        elif frac < 0.15:
            verdict = ("MARGINAL — a minority of peaks touch a border, so the "
                       "exponents carry some added error.")
        else:
            verdict = ("VIOLATED — peaks sit on the borders; the window needs "
                       "revisiting for this dataset.")

        lines = [
            "## Fit-Window Diagnostic",
            "",
            "Gerster et al. (2022): oscillations crossing the fit borders must be "
            "avoided, since a peak on a border inflates exponent error. A peak "
            "counts as crossing when its support (centre frequency ± half the "
            "specparam bandwidth) straddles a border.",
            "",
            f"**{int(a['n_peaks'])} peaks** detected over "
            f"{a['peak_fmin']:g}–{a['peak_fmax']:g} Hz. "
            f"**{int(a['n_cross_fmin']) + int(a['n_cross_fmax'])} ({100 * frac:.1f}%)** "
            f"cross an aperiodic border ({a['aperiodic_fmin']:g} / "
            f"{a['aperiodic_fmax']:g} Hz).",
            "",
            f"**Verdict: {verdict}**",
            "",
            "| Band | Range (Hz) | Reachable | Censored | Peaks | Median CF | Crossing |",
            "|------|-----------|-----------|----------|-------|-----------|----------|",
        ]
        for _, r in diag[diag["band"] != "ALL"].iterrows():
            cf_med = "—" if pd.isna(r["cf_median"]) else f"{r['cf_median']:.1f}"
            lines.append(
                f"| {r['band']} | {r['band_lo']:g}–{r['band_hi']:g} | "
                f"{'yes' if r['reachable'] else '**no**'} | "
                f"{'**yes**' if r['censored'] else 'no'} | "
                f"{int(r['n_peaks'])} | {cf_med} | {100 * r['frac_crossing']:.1f}% |"
            )
        lines.extend([
            "",
            "Unreachable bands emit no peak columns at all — their absence is a "
            "property of the window, not a measurement. Censored bands extend "
            "past the peak window, so their detection rates are a lower bound.",
            "",
        ])
        return lines

    def _write_python_summary(self) -> None:
        tbl_dir = self.tbl_dir

        lines = [
            "# Spectral Parameterization (Vertex-Level) Summary",
            "",
            f"**Study**: {self.config.name}",
            "**Analysis**: Vertex-level spectral parameterization (aperiodic + peaks)",
            f"**Aperiodic fit range**: {self._freq_range[0]:g}-{self._freq_range[1]:g} Hz",
            f"**Peak detection range**: {self._peak_freq_range[0]:g}-"
            f"{self._peak_freq_range[1]:g} Hz"
            + (" (separate wider fit)"
               if self._peak_freq_range != self._freq_range else " (same fit)"),
            f"**Max peaks**: {self._max_n_peaks}",
            f"**Peak width limits**: {self._peak_width_limits[0]}-{self._peak_width_limits[1]} Hz",
            "",
            "## Methods",
            "",
            "Spectral parameterization (specparam/FOOOF) was applied to the PSD at "
            "each source vertex to decompose the spectrum into aperiodic (1/f) and "
            "oscillatory components. Peaks were detected in all configured frequency "
            "bands to determine whether power elevations reflect true oscillatory "
            "peaks or broadband spectral shifts. Group differences in aperiodic "
            "exponent and offset were tested using cluster-based permutation testing. "
            "Per-band peak presence rates were compared using per-vertex chi-squared "
            "tests. For bands with sufficient peak prevalence (>=10% of vertices), "
            "cluster permutation was also applied to peak power maps.",
            "",
        ]

        if self._epoch_config is not None:
            lines.append(
                f"**Epoch sampling**: {self._epoch_config.get('n_epochs', 80)} epochs "
                f"of {self._epoch_config.get('epoch_duration_sec', 2.0)}s"
            )
            lines.append("")

        lines.extend(self._fit_window_summary_lines())

        # Specparam stats
        stats_csv = tbl_dir / "vertex_specparam_stats.csv"
        if stats_csv.exists():
            stats_df = pd.read_csv(stats_csv)
            lines.append("## Aperiodic Parameter Results")
            lines.append("")
            for param in stats_df["parameter"].unique():
                sub = stats_df[stats_df["parameter"] == param]
                n_clust = len(set(sub["cluster_id"]) - {0})
                lines.append(f"- **{param}**: {n_clust} clusters found")
            lines.append("")

        # Per-band chi-squared results
        chi2_csv = tbl_dir / "band_peak_chi2.csv"
        if not chi2_csv.exists():
            chi2_csv = tbl_dir / "gamma_peak_chi2.csv"  # backward compat
        if chi2_csv.exists():
            chi2_df = pd.read_csv(chi2_csv)
            lines.append("## Peak Presence by Band")
            lines.append("")
            if "band" in chi2_df.columns:
                for band_name in chi2_df["band"].unique():
                    sub = chi2_df[chi2_df["band"] == band_name]
                    n_sig = len(sub[sub["p"] < 0.05])
                    lines.append(
                        f"- **{band_name}**: {n_sig}/{len(sub)} vertices with "
                        "significant group differences (uncorrected p<0.05)"
                    )
            else:
                n_sig = len(chi2_df[chi2_df["p"] < 0.05])
                lines.append(
                    f"- {n_sig}/{len(chi2_df)} vertices with significant "
                    "group differences (uncorrected p<0.05)"
                )
            lines.append("")

        lines.extend([
            "## Output Files",
            "",
            "- `data/vertex_specparam.csv` — per-subject per-vertex specparam parameters",
            "- `tables/vertex_specparam_stats.csv` — cluster permutation results",
            "- `tables/band_peak_chi2.csv` — per-band peak presence chi-squared tests",
            "- `figures/specparam_*.png` — aperiodic parameter glass brains",
            "- `figures/{band}_peak_presence.png` — per-band peak prevalence maps",
            "",
        ])

        summary_path = self.output_dir / "ANALYSIS_SUMMARY.md"
        summary_path.write_text("\n".join(lines))
        logger.info("Wrote %s", summary_path)

"""End-to-end smoke tests for the R entry points against synthetic CSVs.

These pin the audit fixes on the R side:

- the evoked scripts derive their contrast list from ``design:``/``hypotheses:``
  (they used to iterate ``config$contrasts``, which is NULL for modern configs);
- ``roi_connectivity_analysis.R`` runs to completion on a single-metric CSV
  (it used to require ``coherence`` + ``imag_coherence`` columns);
- ``roi_cross_freq_edges_analysis.R`` gives AAC/PPC a hypothesis-layer path;
- ``roi_transfer_entropy_analysis.R`` writes ``roi_directed_*`` tables and
  tests a DTF-only edge CSV.

Skipped when ``Rscript`` is not on PATH.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

R_DIR = Path(__file__).resolve().parent.parent / "R"

pytestmark = pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")

GROUPS = {"WT_VEH": 6, "KO_VEH": 6}
ROIS = ["Motor_L", "Motor_R", "Hipp_L", "Hipp_R", "Thal_L", "Thal_R"]
BANDS = {"Theta": [4, 10], "Alpha": [10, 13], "Low Gamma": [30, 55]}


def _subjects():
    return [(f"{g}_s{i:02d}", g) for g, n in GROUPS.items() for i in range(n)]


def _run(script: str, data_dir: Path, config: Path, out: Path, *extra: str) -> str:
    cmd = [
        "Rscript", str(R_DIR / script),
        "--data-dir", str(data_dir), "--config", str(config),
        "--output-dir", str(out), "--fig-dir", str(out / "figures"),
        "--tbl-dir", str(out / "tables"), "--no-figures", *extra,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, f"{script} failed:\n{res.stdout}\n{res.stderr}"
    return res.stdout + res.stderr


@pytest.fixture
def design_config(tmp_path) -> Path:
    """A modern config: design:/hypotheses: only, no legacy contrasts: block."""
    cfg = {
        "name": "Smoke",
        "groups": {"WT_VEH": "WT", "KO_VEH": "KO"},
        "group_order": ["WT_VEH", "KO_VEH"],
        "group_colors": {"WT_VEH": "#3498DB", "KO_VEH": "#E74C3C"},
        "bands": BANDS,
        "roi_categories": {"Motor": ROIS[:2], "Hipp": ROIS[2:4], "Thal": ROIS[4:]},
        "sfreq": 500,
        "design": {"factor": "group", "reference": "WT_VEH", "levels": ["WT_VEH", "KO_VEH"]},
        "hypotheses": [
            {"name": "group_omnibus", "kind": "omnibus"},
            {"name": "disease_effect", "kind": "contrast",
             "weights": {"KO_VEH": 1, "WT_VEH": -1}, "label": "KO vs WT"},
        ],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_evoked_scripts_derive_contrasts_from_design_spec(tmp_path, design_config):
    rng = np.random.default_rng(0)
    for level, col, units in (("roi", "roi", ROIS), ("electrode", "channel", ["Fz", "Cz", "Pz"])):
        rows = []
        for uid, g in _subjects():
            for m, mt in (("itc_theta", "itc"), ("ersp_gamma", "ersp")):
                for u in units:
                    rows.append({
                        "subject": uid, "group": g, col: u, "measure_name": m,
                        "measure_type": mt, "band_lo": 4, "band_hi": 10,
                        "time_lo": 0.0, "time_hi": 0.3,
                        "value": rng.normal(0.5 + (0.2 if g == "KO_VEH" else 0), 0.1),
                        "n_epochs": 50,
                    })
        out = tmp_path / level
        data = out / "data"
        data.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(data / f"{level}_evoked_measures.csv", index=False)

        log = _run(f"{level}_evoked_analysis.R", data, design_config, out)
        assert "Contrasts: 1" in log
        omnibus = pd.read_csv(out / "tables" / f"{level}_evoked_omnibus.csv")
        assert len(omnibus) > 0, "descriptive LMM table is empty — contrast loop did not run"
        assert set(omnibus["contrast"]) == {"disease_effect"}


def test_connectivity_script_runs_on_single_metric_csv(tmp_path, design_config):
    rng = np.random.default_rng(1)
    rows = []
    for uid, g in _subjects():
        for b in BANDS:
            for r1, r2 in itertools.combinations(ROIS, 2):
                rows.append({"subject": uid, "group": g, "band": b, "roi1": r1, "roi2": r2,
                             "aec": rng.normal(0.2, 0.03)})
    out = tmp_path / "conn"
    data = out / "data"
    data.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data / "roi_connectivity_edges.csv", index=False)

    log = _run("roi_connectivity_analysis.R", data, design_config, out)
    assert "Metrics present: aec" in log
    summary = (out / "ANALYSIS_SUMMARY.md").read_text()
    assert "AEC" in summary
    assert "Imaginary Coherence" not in summary


def test_cross_freq_edges_script_writes_hypothesis_tables(tmp_path, design_config):
    rng = np.random.default_rng(2)
    pairs = [("Theta", "Low Gamma"), ("Alpha", "Low Gamma")]
    data = tmp_path / "xf" / "data"
    data.mkdir(parents=True)
    for metric in ("aac", "ppc"):
        rows = []
        for uid, g in _subjects():
            for pb, ab in pairs:
                for rx in ROIS:
                    for ry in ROIS:
                        row = {"subject": uid, "group": g, "phase_band": pb, "amp_band": ab,
                               "freq_pair": f"{pb}-{ab}", "n": 4, "m": 1,
                               "roi_x": rx, "roi_y": ry,
                               metric: rng.normal(0.3 + (0.15 if g == "KO_VEH" else 0), 0.05)}
                        if metric == "ppc":
                            row["ppc_z"] = rng.normal(1, 1)
                        rows.append(row)
        pd.DataFrame(rows).to_csv(data / f"{metric}_edges.csv", index=False)

    out = tmp_path / "xf"
    _run("roi_cross_freq_edges_analysis.R", data, design_config, out, "--metric", "aac,ppc")
    tbl = out / "tables"
    for m in ("aac", "ppc"):
        for tier in ("global", "directed_edges", "region"):
            path = tbl / f"roi_cross_freq_{m}_{tier}_hypotheses.csv"
            assert path.exists(), path.name
            df = pd.read_csv(path)
            assert {"hypothesis", "kind", "band", "q_value", "significant"} <= set(df.columns)
            assert set(df["band"]) <= {f"{pb}-{ab}" for pb, ab in pairs}
    # PPC carries the surrogate z as a second DV
    ppc_edges = pd.read_csv(tbl / "roi_cross_freq_ppc_directed_edges_hypotheses.csv")
    assert set(ppc_edges["dv"]) == {"ppc", "ppc_z"}
    assert "Amplitude-Amplitude Coupling" in (out / "ANALYSIS_SUMMARY.md").read_text()


@pytest.mark.parametrize("cols", [["dtf"], ["te", "net_te", "dtf"]])
def test_directed_script_uses_canonical_prefix_and_tests_every_dv(tmp_path, design_config, cols):
    rng = np.random.default_rng(3)
    rows = []
    for uid, g in _subjects():
        for b in BANDS:
            for r1 in ROIS:
                for r2 in ROIS:
                    if r1 == r2:
                        continue
                    row = {"subject": uid, "group": g, "band": b,
                           "source_roi": r1, "target_roi": r2}
                    for c in cols:
                        row[c] = rng.normal(0.1 + (0.03 if g == "KO_VEH" else 0), 0.02)
                    rows.append(row)
    out = tmp_path / "dir"
    data = out / "data"
    data.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data / "roi_transfer_entropy_edges.csv", index=False)

    _run("roi_transfer_entropy_analysis.R", data, design_config, out)
    tbl = out / "tables"
    assert not list(tbl.glob("roi_transfer_entropy_*"))
    for name in ("roi_directed_global_hypotheses.csv",
                 "roi_directed_directed_edges_hypotheses.csv",
                 "roi_directed_region_hypotheses.csv",
                 "roi_directed_omnibus_lmm.csv"):
        assert (tbl / name).exists(), name
    edges = pd.read_csv(tbl / "roi_directed_directed_edges_hypotheses.csv")
    assert set(edges["dv"]) == set(cols)


# ---------------------------------------------------------------------------
# ROI categories: the study config wins over the --roi-categories file.
#
# The file is looked up in the atlas *directory*, which several atlases share
# (allen32/allen26/allen64 all live in allen/), so for an allen26 study it was
# allen32's partition. It used to REPLACE the study's categories: two allen26
# categories matched no parcel and vanished, and Deep Subcortical was built
# from 4 of its 8 parcels.
# ---------------------------------------------------------------------------

def _psd_band_power_csv(data_dir: Path) -> None:
    rng = np.random.default_rng(7)
    rows = []
    for subj, grp in _subjects():
        for roi in ROIS:
            for band in BANDS:
                shift = 0.5 if (grp == "KO_VEH" and roi.startswith("Motor")) else 0.0
                rows.append({"subject": subj, "group": grp, "roi": roi, "band": band,
                             "absolute": float(rng.normal(-200.0 + shift, 1.0)),
                             "relative": float(rng.uniform(0.1, 0.3))})
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "band_power.csv", index=False)


def _psd_region_labels(out: Path) -> set[str]:
    labels: set[str] = set()
    for f in (out / "tables").glob("roi_psd*region*.csv"):
        d = pd.read_csv(f)
        for col in ("spatial", "region"):
            if col in d.columns:
                labels |= set(d[col].dropna().astype(str))
    return labels


def test_psd_region_tier_uses_study_categories_over_the_atlas_file(tmp_path, design_config):
    other = tmp_path / "another_atlas_categories.yaml"
    other.write_text(yaml.safe_dump({"Bogus": ["Motor_L", "Hipp_L"], "Absent": ["Not_A_Parcel"]}))
    data, out = tmp_path / "data", tmp_path / "out"
    _psd_band_power_csv(data)
    log = _run("roi_psd_analysis.R", data, design_config, out, "--roi-categories", str(other))
    assert "Using roi_categories from the study config" in log
    labels = _psd_region_labels(out)
    assert labels, "no region-level table was written"
    assert labels <= {"Motor", "Hipp", "Thal"}, labels


def test_psd_region_tier_falls_back_to_the_file_when_config_has_none(tmp_path, design_config):
    cfg = yaml.safe_load(design_config.read_text())
    cfg.pop("roi_categories")
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump(cfg))
    cats = tmp_path / "cats.yaml"
    cats.write_text(yaml.safe_dump({"Front": ROIS[:2], "Back": ROIS[2:]}))
    data, out = tmp_path / "data", tmp_path / "out"
    _psd_band_power_csv(data)
    log = _run("roi_psd_analysis.R", data, bare, out, "--roi-categories", str(cats))
    assert "Loaded roi_categories from:" in log
    labels = _psd_region_labels(out)
    assert labels and labels <= {"Front", "Back"}, labels

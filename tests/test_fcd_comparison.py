"""Tests for the source-vs-sensor FCD comparison module."""

import numpy as np
import pandas as pd

from source_analytics.config import StudyConfig
from source_analytics.core import StudyAnalyzer
from source_analytics.analyses.fcd_comparison_analysis import _fcd_summaries


def test_fcd_summaries_mean_and_cv():
    df = pd.DataFrame({
        "subject": ["A"] * 3 + ["B"] * 3,
        "group": ["KO_VEH"] * 3 + ["WT_VEH"] * 3,
        "band": ["Alpha"] * 6, "metric": ["aec"] * 6,
        "fcd": [0.2, 0.4, 0.6, 0.5, 0.5, 0.5],
    })
    s = _fcd_summaries(df, "sensor")
    a = s[s.subject == "A"].iloc[0]
    b = s[s.subject == "B"].iloc[0]
    assert abs(a.sensor_mean - 0.4) < 1e-9
    assert abs(a.sensor_cv - 0.5) < 1e-9          # std=0.2, mean=0.4
    assert abs(b.sensor_cv) < 1e-12                 # flat map -> zero heterogeneity


def test_fcd_summaries_edge_cases():
    # single unit -> CV NaN; all-zero -> mean 0, CV NaN
    df = pd.DataFrame({
        "subject": ["A", "B", "B"], "group": ["KO_VEH", "WT_VEH", "WT_VEH"],
        "band": ["Alpha"] * 3, "metric": ["aec"] * 3, "fcd": [0.3, 0.0, 0.0],
    })
    s = _fcd_summaries(df, "source")
    assert np.isnan(s[s.subject == "A"].iloc[0].source_cv)   # <2 units
    assert np.isnan(s[s.subject == "B"].iloc[0].source_cv)   # mean 0


def _write_fcd(path, subjects, n_units, band, metric, base, rng):
    rows = []
    for subj, group, mean in subjects:
        vals = np.clip(rng.normal(mean, 0.05, n_units), 0, 1)
        for u in range(n_units):
            rows.append({"subject": subj, "group": group, "band": band,
                         "metric": metric, "fcd": float(vals[u])})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fcd_comparison_end_to_end(sample_config_yaml):
    config = StudyConfig.from_yaml(sample_config_yaml)
    base = config.output_dir
    rng = np.random.default_rng(0)

    # KO higher FCD than WT at BOTH levels (concordant group effect)
    subs = [(f"KO_VEH_{i}", "KO_VEH", 0.6) for i in range(5)] + \
           [(f"WT_VEH_{i}", "WT_VEH", 0.4) for i in range(5)]
    _write_fcd(base / "electrode_connectivity" / "data" / "electrode_fcd.csv",
               subs, 30, "Alpha", "aec", 0.5, rng)
    _write_fcd(base / "vertex_connectivity" / "data" / "vertex_fcd.csv",
               subs, 200, "Alpha", "aec", 0.5, rng)

    StudyAnalyzer(config).run_analysis("fcd_comparison")

    summ = pd.read_csv(base / "fcd_comparison" / "data" / "fcd_subject_summary.csv")
    assert set(["sensor_mean", "sensor_cv", "source_mean", "source_cv"]).issubset(summ.columns)
    assert len(summ) == 10  # one row per subject (Alpha x aec)

    tbl = config.results_dir / "tables" / (config.paradigm_name or "") / \
        "fcd_comparison" / "fcd_comparison_stats.csv"
    s = pd.read_csv(tbl)
    for col in ("band", "metric", "contrast", "corr_mean_r",
                "sensor_mean_g", "source_mean_g", "sensor_cv_g", "source_cv_g",
                "mean_concordant", "cv_concordant"):
        assert col in s.columns, f"missing {col}"
    row = s[s.contrast == "disease_effect"].iloc[0]
    # KO>WT at both levels -> mean effect same sign -> concordant
    assert row["mean_concordant"]
    assert np.sign(row["sensor_mean_g"]) == np.sign(row["source_mean_g"])

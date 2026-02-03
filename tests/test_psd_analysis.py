"""Integration test for the PSD analysis pipeline (Python CSV export + R stats)."""

from pathlib import Path

import pandas as pd

from source_analytics.config import StudyConfig
from source_analytics.core import StudyAnalyzer


def test_psd_csv_export(sample_config_yaml):
    """Test that Python correctly exports CSVs for R."""
    config = StudyConfig.from_yaml(sample_config_yaml)
    analyzer = StudyAnalyzer(config)

    issues = analyzer.validate()
    assert len(issues) == 0, f"Validation issues: {issues}"

    # Run PSD analysis (R will be called but may fail in CI — that's ok)
    analyzer.run_analysis("psd")

    psd_dir = config.output_dir / "psd"
    assert psd_dir.exists()

    # Check Python CSV exports exist
    assert (psd_dir / "data" / "band_power.csv").exists()
    assert (psd_dir / "data" / "psd_curves.csv").exists()
    assert (psd_dir / "data" / "study_config.yaml").exists()

    # Check band_power.csv structure
    bp = pd.read_csv(psd_dir / "data" / "band_power.csv")
    assert len(bp) > 0
    assert set(bp.columns) == {"subject", "group", "roi", "band", "absolute", "relative", "dB"}
    assert set(bp["group"].unique()) == {"KO_VEH", "WT_VEH"}

    # Check psd_curves.csv structure
    psd = pd.read_csv(psd_dir / "data" / "psd_curves.csv")
    assert len(psd) > 0
    assert set(psd.columns) == {"subject", "group", "roi", "freq_hz", "psd"}

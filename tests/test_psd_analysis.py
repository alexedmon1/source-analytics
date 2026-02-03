"""Integration test for the PSD analysis pipeline."""

from pathlib import Path

from source_analytics.config import StudyConfig
from source_analytics.core import StudyAnalyzer


def test_psd_analysis_integration(sample_config_yaml):
    """Full end-to-end test: config -> discover -> PSD -> stats -> figures -> summary."""
    config = StudyConfig.from_yaml(sample_config_yaml)
    analyzer = StudyAnalyzer(config)

    # Validate
    issues = analyzer.validate()
    assert len(issues) == 0, f"Validation issues: {issues}"

    # Run PSD analysis
    analyzer.run_analysis("psd")

    # Check outputs exist
    psd_dir = config.output_dir / "psd"
    assert psd_dir.exists()
    assert (psd_dir / "data" / "band_power_all.csv").exists()
    assert (psd_dir / "tables" / "psd_statistics.csv").exists()
    assert (psd_dir / "ANALYSIS_SUMMARY.md").exists()

    # Check figures were generated
    fig_dir = psd_dir / "figures"
    assert fig_dir.exists()
    png_files = list(fig_dir.glob("*.png"))
    assert len(png_files) > 0

    # Check statistics CSV has content
    import pandas as pd
    stats_df = pd.read_csv(psd_dir / "tables" / "psd_statistics.csv")
    assert len(stats_df) > 0
    assert "q_value" in stats_df.columns
    assert "hedges_g" in stats_df.columns
    assert "lmm_z" in stats_df.columns

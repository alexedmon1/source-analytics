"""R is handed the EFFECTIVE ROI categories, not whatever ``raw`` happens to hold.

A profile narrows ``config.roi_categories`` but not ``config.raw``, and the atlas
default never reaches ``raw`` at all. R used to fill that gap from the category
file in the atlas *directory*, which for allen26 was allen32's.
"""

from __future__ import annotations

from dataclasses import replace

from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
from source_analytics.config import StudyConfig


def test_r_config_carries_the_narrowed_categories(sample_config_yaml, tmp_path):
    cfg = StudyConfig.from_yaml(sample_config_yaml)
    motor = {"Motor": list(cfg.roi_categories["Motor"])}
    analysis = ROIPsdAnalysis(replace(cfg, roi_categories=motor), tmp_path / "out")
    assert analysis._r_config_data()["roi_categories"] == motor
    # raw alone still holds the full map -- which is exactly what R used to get.
    assert "Subcortical" in cfg.raw["roi_categories"]


def test_r_config_leaves_raw_untouched(sample_config_yaml, tmp_path):
    cfg = StudyConfig.from_yaml(sample_config_yaml)
    before = dict(cfg.raw)
    ROIPsdAnalysis(cfg, tmp_path / "out")._r_config_data()["roi_categories"] = {}
    assert cfg.raw == before

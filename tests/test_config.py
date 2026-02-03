"""Tests for config loading and validation."""

from source_analytics.config import StudyConfig


def test_load_config(sample_config_yaml):
    config = StudyConfig.from_yaml(sample_config_yaml)
    assert config.name == "Test Study"
    assert len(config.groups) == 2
    assert len(config.contrasts) == 1
    assert config.contrasts[0].name == "disease_effect"
    assert len(config.bands) == 4
    assert config.bands["Theta"] == (4, 8)


def test_config_validation(sample_config_yaml):
    config = StudyConfig.from_yaml(sample_config_yaml)
    warnings = config.validate()
    # Should have no warnings with valid config
    assert len(warnings) == 0


def test_group_labels(sample_config_yaml):
    config = StudyConfig.from_yaml(sample_config_yaml)
    assert config.get_group_label("WT_VEH") == "WT Vehicle"
    assert config.get_group_label("UNKNOWN") == "UNKNOWN"

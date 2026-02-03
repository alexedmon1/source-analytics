"""Tests for I/O modules."""

from source_analytics.io.loader import SubjectLoader
from source_analytics.io.discovery import discover_subjects


def test_subject_loader(synthetic_subject_dir):
    loader = SubjectLoader(synthetic_subject_dir)
    roi_ts = loader.load_roi_timeseries()
    assert len(roi_ts) == 8
    assert "Primary_Motor_Cortex_L" in roi_ts
    assert roi_ts["Primary_Motor_Cortex_L"].ndim == 1


def test_load_sfreq(synthetic_subject_dir):
    loader = SubjectLoader(synthetic_subject_dir)
    sfreq = loader.load_sfreq()
    assert sfreq == 500.0


def test_discover_subjects(synthetic_study_dir):
    subjects = discover_subjects(
        synthetic_study_dir,
        group_mapping={"KO ICV": "KO_VEH", "WT ICV": "WT_VEH"},
    )
    assert len(subjects) == 10
    groups = {s.group for s in subjects}
    assert groups == {"KO_VEH", "WT_VEH"}


def test_available_files(synthetic_subject_dir):
    loader = SubjectLoader(synthetic_subject_dir)
    files = loader.available_files
    assert "step6_roi_timeseries_magnitude.pkl" in files
    assert "step1_info.pkl" in files

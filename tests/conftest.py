"""Synthetic data fixtures for testing."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


ROI_NAMES = [
    "Primary_Motor_Cortex_L",
    "Primary_Motor_Cortex_R",
    "Hippocampus_L",
    "Hippocampus_R",
    "Thalamus_L",
    "Thalamus_R",
    "Cortex_Auditory_L",
    "Cortex_Auditory_R",
]

SFREQ = 500.0
DURATION = 10.0  # seconds


def _make_roi_timeseries(sfreq: float, duration: float, n_rois: int = 8) -> dict[str, np.ndarray]:
    """Generate synthetic ROI timeseries with realistic spectral content."""
    rng = np.random.default_rng(42)
    n_times = int(sfreq * duration)
    t = np.arange(n_times) / sfreq

    roi_ts = {}
    for i, name in enumerate(ROI_NAMES[:n_rois]):
        # 1/f noise + some oscillatory components
        signal = rng.standard_normal(n_times) * 0.1
        signal += 0.5 * np.sin(2 * np.pi * 10 * t)  # alpha
        signal += 0.3 * np.sin(2 * np.pi * 40 * t)  # gamma
        signal += 0.2 * np.sin(2 * np.pi * 6 * t)   # theta
        roi_ts[name] = np.abs(signal).astype(np.float32)  # magnitude

    return roi_ts


def _make_info(sfreq: float) -> MagicMock:
    """Create a mock MNE Info object."""
    info = {"sfreq": sfreq, "ch_names": [f"elec_{i:03d}" for i in range(32)]}
    return info


@pytest.fixture
def synthetic_subject_dir(tmp_path):
    """Create a temporary subject data directory with synthetic data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    roi_ts = _make_roi_timeseries(SFREQ, DURATION)
    with open(data_dir / "step6_roi_timeseries_magnitude.pkl", "wb") as f:
        pickle.dump(roi_ts, f)

    info = _make_info(SFREQ)
    with open(data_dir / "step1_info.pkl", "wb") as f:
        pickle.dump(info, f)

    return data_dir


@pytest.fixture
def synthetic_study_dir(tmp_path):
    """Create a multi-subject study directory for discovery testing."""
    root = tmp_path / "study"
    groups = {"WT ICV": 5, "KO ICV": 5}

    for group_name, n in groups.items():
        group_dir = root / group_name
        for i in range(n):
            subj_dir = group_dir / f"subj_{i:03d}"
            data_dir = subj_dir / "data"
            data_dir.mkdir(parents=True)

            # Add slight group difference in gamma for KO
            roi_ts = _make_roi_timeseries(SFREQ, DURATION)
            if "KO" in group_name:
                for name in roi_ts:
                    t = np.arange(len(roi_ts[name])) / SFREQ
                    roi_ts[name] = roi_ts[name] + 0.2 * np.abs(np.sin(2 * np.pi * 45 * t)).astype(np.float32)

            with open(data_dir / "step6_roi_timeseries_magnitude.pkl", "wb") as f:
                pickle.dump(roi_ts, f)

            info = _make_info(SFREQ)
            with open(data_dir / "step1_info.pkl", "wb") as f:
                pickle.dump(info, f)

    return root


@pytest.fixture
def sample_config_yaml(tmp_path, synthetic_study_dir):
    """Create a sample study YAML config pointing to synthetic data."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    config_text = f"""
name: "Test Study"
output_dir: "{output_dir}"

groups:
  WT_VEH: "WT Vehicle"
  KO_VEH: "KO Vehicle"

group_order: [WT_VEH, KO_VEH]

group_colors:
  WT_VEH: "#3498DB"
  KO_VEH: "#E74C3C"

contrasts:
  - name: disease_effect
    group_a: KO_VEH
    group_b: WT_VEH

bands:
  Theta: [4, 8]
  Alpha: [8, 13]
  Beta: [13, 30]
  Gamma: [30, 55]

roi_categories:
  Motor:
    - Primary_Motor_Cortex_L
    - Primary_Motor_Cortex_R
  Subcortical:
    - Hippocampus_L
    - Hippocampus_R
    - Thalamus_L
    - Thalamus_R

discovery:
  root_dir: "{synthetic_study_dir}"
  group_mapping:
    "KO ICV": KO_VEH
    "WT ICV": WT_VEH
"""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_text)
    return config_path

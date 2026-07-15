"""Tests for StudyConfig.for_profile() — the profile narrowing layer.

A profile only ever *narrows*. The load-bearing case is ROIs: the ROI set is the
FDR correction family, so a profile that drops ROIs changes every q-value. See
source-analytics/PROFILE_PROVENANCE_PLAN.md and FORGE/treatment/REPORT_PLAN.md §10b.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.config import StudyConfig
from source_analytics.io.loader import SubjectLoader

CONFIG_TEXT = """
name: "Profile Test Study"

groups:
  WT_VEH: "WT Vehicle"
  KO_VEH: "KO Vehicle"
group_order: [WT_VEH, KO_VEH]
group_colors:
  WT_VEH: "#000000"
  KO_VEH: "#E0007A"

bands:
  Delta: [1, 4]
  Alpha: [10, 13]
  Low Gamma: [30, 55]
  High Gamma: [65, 80]

roi_categories:
  Motor: [Motor_L, Motor_R]
  Visual: [Visual_L, Visual_R]
  Deep: [Thalamus_L, Thalamus_R]

design:
  factor: group
  reference: WT_VEH
  levels: [WT_VEH, KO_VEH]
  fdr: {scope: band, method: BH}

hypotheses:
  - name: disease_effect
    kind: contrast
    weights: {KO_VEH: 1, WT_VEH: -1}
  - name: dose_effect
    kind: contrast
    weights: {KO_VEH: 1, WT_VEH: -1}

curated:
  bands:
    Delta: [1, 4]
    Alpha: [10, 13]
    Low Gamma: [30, 45]
  roi_categories:
    Motor: [Motor_L, Motor_R]
    Visual: [Visual_L, Visual_R]
  include_hypotheses: [disease_effect]
  include_analyses: [roi_psd]

paradigms:
  resting:
    data_dir: ./derivatives
    analyses:
      roi_psd: {}
      roi_aperiodic: {}
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "study.yaml"
    p.write_text(CONFIG_TEXT)
    return StudyConfig.from_yaml(p)


# ---- the default profile must be untouched ------------------------------- #


def test_default_profile_is_unnarrowed(cfg):
    assert cfg.profile_name is None
    assert cfg.rois == []
    assert cfg.include_analyses is None
    assert len(cfg.bands) == 4
    assert len(cfg.design_spec.hypotheses) == 2
    assert cfg.results_dir.name == "results"


def test_default_profile_paths_have_no_profile_segment(cfg):
    a = cfg.for_paradigm_analysis("resting", "roi_psd")
    assert a.results_dir.name == "results"
    assert a.output_dir.name == "resting"
    assert a.rois == []


# ---- narrowing ----------------------------------------------------------- #


def test_for_profile_narrows_bands_and_may_redefine_edges(cfg):
    p = cfg.for_profile("curated")
    assert list(p.bands) == ["Delta", "Alpha", "Low Gamma"]
    # Same name, different edges — a replacement, not a subset.
    assert p.bands["Low Gamma"] == (30, 45)
    assert cfg.bands["Low Gamma"] == (30, 55), "parent config must not be mutated"


def test_for_profile_derives_rois_from_roi_categories(cfg):
    p = cfg.for_profile("curated")
    assert p.rois == ["Motor_L", "Motor_R", "Visual_L", "Visual_R"]
    assert "Thalamus_L" not in p.rois
    assert set(p.roi_categories) == {"Motor", "Visual"}


def test_for_profile_filters_hypotheses(cfg):
    p = cfg.for_profile("curated")
    assert [h.name for h in p.design_spec.hypotheses] == ["disease_effect"]
    # The parent spec is untouched (replace(), not mutate).
    assert len(cfg.design_spec.hypotheses) == 2


def test_for_profile_sets_profile_segment_on_both_trees(cfg):
    p = cfg.for_profile("curated")
    assert p.profile_name == "curated"
    assert p.results_dir.name == "curated"
    assert p.results_dir.parent.name == "results"
    assert p.output_dir.name == "curated"


def test_narrowing_survives_paradigm_scoping(cfg):
    """for_profile() is applied to the root config; for_paradigm*() rebuilds
    StudyConfig field-by-field, so it must carry the narrowing through."""
    a = cfg.for_profile("curated").for_paradigm_analysis("resting", "roi_psd")
    assert a.profile_name == "curated"
    assert len(a.bands) == 3
    assert len(a.rois) == 4
    assert len(a.design_spec.hypotheses) == 1
    assert a.results_dir.name == "curated"
    # Profile segment sits ABOVE the paradigm: analytics/<profile>/<paradigm>
    assert a.output_dir.name == "resting"
    assert a.output_dir.parent.name == "curated"


def test_for_paradigm_also_carries_profile(cfg):
    a = cfg.for_profile("curated").for_paradigm("resting")
    assert a.profile_name == "curated"
    assert len(a.rois) == 4
    assert a.include_analyses == ["roi_psd"]


def test_include_analyses_gates_get_paradigm_analyses(cfg):
    assert cfg.get_paradigm_analyses("resting") == ["roi_psd", "roi_aperiodic"]
    p = cfg.for_profile("curated")
    assert p.get_paradigm_analyses("resting") == ["roi_psd"]
    assert p.get_paradigm_analyses("unknown_paradigm") is None


# ---- validation: fail loudly, never silently narrow ---------------------- #


def test_unknown_profile_raises(cfg):
    with pytest.raises(ValueError, match="No profile block 'nope:'"):
        cfg.for_profile("nope")


def test_unknown_roi_raises(cfg, tmp_path):
    text = CONFIG_TEXT.replace("    Visual: [Visual_L, Visual_R]\n  include_hypotheses",
                               "    Visual: [Not_A_Roi]\n  include_hypotheses")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="unknown ROI"):
        StudyConfig.from_yaml(p).for_profile("curated")


def test_unknown_hypothesis_raises(cfg, tmp_path):
    text = CONFIG_TEXT.replace("include_hypotheses: [disease_effect]",
                               "include_hypotheses: [disease_effect, bogus]")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="unknown hypothesis"):
        StudyConfig.from_yaml(p).for_profile("curated")


def test_duplicate_roi_across_categories_raises(cfg, tmp_path):
    text = CONFIG_TEXT.replace("    Visual: [Visual_L, Visual_R]\n  include_hypotheses",
                               "    Visual: [Motor_L]\n  include_hypotheses")
    p = tmp_path / "bad.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="more than one category"):
        StudyConfig.from_yaml(p).for_profile("curated")


# ---- the loader filter: what actually makes the FDR family smaller ------- #


def test_restrict_rois_noop_when_unset():
    ts = {"a": np.zeros(3), "b": np.zeros(3)}
    assert SubjectLoader._restrict_rois(ts, None) is ts
    assert SubjectLoader._restrict_rois(ts, []) is ts


def test_restrict_rois_subsets_and_orders():
    ts = {"a": np.zeros(3), "b": np.ones(3), "c": np.zeros(3)}
    out = SubjectLoader._restrict_rois(ts, ["c", "a"])
    assert list(out) == ["c", "a"]
    assert "b" not in out


def test_restrict_rois_raises_on_missing():
    """Must raise, not silently return fewer ROIs — a dropped ROI would shrink
    the FDR family with nothing in the output to show for it."""
    ts = {"a": np.zeros(3)}
    with pytest.raises(KeyError, match="not present in the data"):
        SubjectLoader._restrict_rois(ts, ["a", "gone"])

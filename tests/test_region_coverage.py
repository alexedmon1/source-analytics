"""Tests for cluster anatomical-coverage formatting (atlas_utils.format_region_coverage)."""

from source_analytics.atlas.atlas_utils import format_region_coverage


def test_lists_rois_above_5pct_and_counts_the_rest():
    # 20 vertices: Motor_R x6 (30%), Somatosensory_L x3 (15%), then 11 ROIs x1 (5% each).
    labels = (["Motor_R"] * 6 + ["Somatosensory_L"] * 3
              + [f"ROI_{i}" for i in range(11)])
    assert len(labels) == 20
    out = format_region_coverage(labels)
    # >5% listed with share of the TOTAL cluster vertices, ordered high->low.
    assert out.startswith("Motor_R 30%, Somatosensory_L 15%")
    # the eleven 5% ROIs are not listed (5% is not > 5%) but counted.
    assert "(+11 ROIs ≤5%)" in out
    assert "ROI_0" not in out


def test_fully_diffuse_when_nothing_exceeds_threshold():
    labels = [f"ROI_{i}" for i in range(10)]  # each 10% ... wait, 10 distinct = 10% each
    # Make each ≤5%: 20 distinct ROIs, one vertex each.
    labels = [f"ROI_{i}" for i in range(20)]
    out = format_region_coverage(labels)
    assert out == "20 ROIs ≤5%"


def test_empty_cluster_is_blank():
    assert format_region_coverage([]) == ""


def test_percentages_are_of_total_including_unlabeled():
    # 10 vertices, 4 unlabeled (None): Motor_R is 4/10 = 40%, not 4/6.
    labels = ["Motor_R"] * 4 + [None] * 4 + ["Visual_Parietal_L"] * 2
    out = format_region_coverage(labels)
    assert "Motor_R 40%" in out and "Visual_Parietal_L 20%" in out

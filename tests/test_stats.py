"""Tests for statistical modules."""

import numpy as np
import pandas as pd

from source_analytics.stats.parametric import ttest_between_groups, lmm_group_effect
from source_analytics.stats.effect_size import cohens_d, hedges_g
from source_analytics.stats.correction import fdr_correction


def test_ttest_significant():
    rng = np.random.default_rng(42)
    a = rng.normal(10, 1, 30)
    b = rng.normal(8, 1, 30)
    result = ttest_between_groups(a, b)
    assert result.p_value < 0.05
    assert result.statistic > 0
    assert result.n_a == 30
    assert result.n_b == 30


def test_ttest_not_significant():
    rng = np.random.default_rng(42)
    a = rng.normal(10, 1, 10)
    b = rng.normal(10, 1, 10)
    result = ttest_between_groups(a, b)
    assert result.p_value > 0.05


def test_cohens_d():
    a = np.array([10, 11, 12, 13, 14])
    b = np.array([8, 9, 10, 11, 12])
    d = cohens_d(a, b)
    assert d > 0  # a > b
    assert 0.5 < abs(d) < 2.0  # reasonable range


def test_hedges_g():
    a = np.array([10, 11, 12, 13, 14])
    b = np.array([8, 9, 10, 11, 12])
    g = hedges_g(a, b)
    d = cohens_d(a, b)
    # Hedges' g should be slightly smaller than Cohen's d (bias correction)
    assert abs(g) < abs(d)


def test_fdr_correction():
    pvals = np.array([0.001, 0.01, 0.04, 0.5, 0.9])
    rejected, qvals = fdr_correction(pvals)
    # First p-value should still be significant
    assert rejected[0]
    # Last should not
    assert not rejected[-1]
    # q-values should be monotonically related to p-values
    assert all(qvals[i] <= qvals[i + 1] for i in range(len(qvals) - 1) if not np.isnan(qvals[i]))


def test_fdr_with_nans():
    pvals = np.array([0.01, np.nan, 0.5])
    rejected, qvals = fdr_correction(pvals)
    assert not np.isnan(qvals[0])
    assert np.isnan(qvals[1])


def test_lmm_group_effect():
    rng = np.random.default_rng(42)
    n_per_group = 15
    data = []
    for i in range(n_per_group):
        # Add subject-level random intercept to make random effects estimable
        subj_offset_wt = rng.normal(0, 1.5)
        subj_offset_ko = rng.normal(0, 1.5)
        for roi in ["ROI_A", "ROI_B", "ROI_C", "ROI_D", "ROI_E"]:
            data.append({"subject": f"wt_{i}", "group": "WT", "roi": roi,
                         "value": 5 + subj_offset_wt + rng.normal(0, 0.5)})
            data.append({"subject": f"ko_{i}", "group": "KO", "roi": roi,
                         "value": 8 + subj_offset_ko + rng.normal(0, 0.5)})

    df = pd.DataFrame(data)
    result = lmm_group_effect(df)
    assert result.test == "lmm"
    # With realistic random effects, the model should converge and detect the group effect
    assert not np.isnan(result.p_value)
    assert result.p_value < 0.05  # clear group difference

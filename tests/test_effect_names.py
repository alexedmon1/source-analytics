"""Cohen's d versus Hedges' g in the legacy post-hoc tables (KNOWN_ISSUES 1, 2026-11-20).

The uncorrected standardized difference is written as ``cohens_d``; tables written before carry the same value
as ``hedges_g``, and the figure readers take either. roi_network's global table holds a corrected g and keeps it.
"""
from __future__ import annotations

import pandas as pd

from source_analytics.viz.figure_registry import TABLE_SCHEMAS, effect_symbol, load_posthoc


def _table(tmp_path, analysis, effect_col):
    schema = TABLE_SCHEMAS[analysis]
    pd.DataFrame({"contrast": ["d"], "dv": ["absolute"], "band": ["Alpha"], "estimate": [0.5],
                  "p_value": [0.01], "q_value": [0.02], "significant": ["TRUE"],
                  effect_col: [0.8]}).to_csv(tmp_path / schema.posthoc_file, index=False)


def test_a_table_written_before_the_rename_is_read_as_cohens_d(tmp_path):
    _table(tmp_path, "roi_psd", "hedges_g")
    df = load_posthoc(tmp_path, "roi_psd")
    assert TABLE_SCHEMAS["roi_psd"].effect_col == "cohens_d"
    assert df["cohens_d"].tolist() == [0.8] and "hedges_g" not in df.columns


def test_a_current_table_is_read_unchanged(tmp_path):
    _table(tmp_path, "roi_psd", "cohens_d")
    assert load_posthoc(tmp_path, "roi_psd")["cohens_d"].tolist() == [0.8]


def test_the_corrected_hedges_g_keeps_its_name_and_letter():
    assert TABLE_SCHEMAS["roi_network"].effect_col == "hedges_g"
    assert effect_symbol("hedges_g") == "g" and effect_symbol("cohens_d") == "d"

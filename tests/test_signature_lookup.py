"""The source-vs-sensor decoding comparison pairs tables from ONE paradigm.

Locks out: electrode_signature globbed the whole results tree for
``vertex_signature_results.csv`` and took the first hit. A stale vertex table
from an earlier study version sat in that tree and would have been merged
against the new run's sensor results, labelled as if it were its source side.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from source_analytics.analyses.electrode_signature_analysis import (
    ElectrodeSignatureAnalysis,
    render_signature_comparison,
)


class _Sensor(ElectrodeSignatureAnalysis):
    """Just enough of the module to exercise the lookup (no config, no IO)."""

    def __init__(self, tbl_dir: Path):
        self._tbl = tbl_dir

    @property
    def tbl_dir(self) -> Path:
        return self._tbl


def _table(path: Path, accuracy: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "contrast": ["disease_effect"] * 2, "band": ["Theta", "Beta"],
        "classifier": ["svm_linear"] * 2, "accuracy": [accuracy] * 2,
        "balanced_accuracy": [accuracy] * 2, "p_value": [0.01, 0.5], "auc": [accuracy] * 2,
    }).to_csv(path, index=False)
    return path


def test_other_paradigms_are_never_searched(tmp_path):
    tables = tmp_path / "results" / "tables"
    _table(tables / "vertex" / "vertex_signature" / "vertex_signature_results.csv", 0.9)
    assert _Sensor(tables / "cartesian_mc" / "electrode_signature")._find_source_results() is None


def test_roi_signature_in_the_same_paradigm_is_preferred(tmp_path):
    paradigm = tmp_path / "results" / "tables" / "cartesian_mc"
    roi = _table(paradigm / "roi_signature" / "roi_signature_results.csv", 0.8)
    _table(paradigm / "vertex_signature" / "vertex_signature_results.csv", 0.7)
    found = _Sensor(paradigm / "electrode_signature")._find_source_results()
    assert found == ("roi_signature", roi)


def test_the_comparison_records_which_source_it_used(tmp_path):
    source = _table(tmp_path / "source.csv", 0.8)
    sensor = _table(tmp_path / "sensor.csv", 0.6)
    out = tmp_path / "out"
    out.mkdir()
    render_signature_comparison(source, sensor, out, out, source_module="roi_signature")
    merged = pd.read_csv(out / "signature_source_vs_sensor.csv")
    assert set(merged["source_module"]) == {"roi_signature"}
    assert merged["accuracy_gain"].round(6).tolist() == [0.2, 0.2]

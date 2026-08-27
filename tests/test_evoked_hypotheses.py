"""Verification of the declared-hypothesis wiring on the evoked modules.

``roi_evoked`` / ``electrode_evoked`` were the two modules
``docs/methods/HYPOTHESIS.md`` §9 listed as deferred, for "long-format DV": they
export one ``value`` column faceted by ``measure_name`` instead of one column
per DV. The wiring resolves that by making the measure a FACET, so each measure
gets its own FDR family across the spatial grid.

Checks (synthetic measure table, 3 groups x 2 measures x 3 ROIs, no band axis):
  1. the additive ``<name>_hypotheses.csv`` is written, with a row per
     (hypothesis x measure x spatial) cell;
  2. spatial AND measure specificity — the planted cell is significant, and
     neither a null ROI nor the null measure is;
  3. the band coordinate is null, because evoked measures carry no band axis;
  4. each measure is its own FDR family (a null measure cannot dilute a
     signal measure, nor vice versa);
  5. the CSV fallback drives ``--steps statistics`` with no rows in memory;
  6. a study that renames the design factor still resolves;
  7. nothing is written when no hypotheses are declared;
  8. ``electrode_evoked`` behaves identically on ``channel``.

Run: uv run --extra dev pytest tests/test_evoked_hypotheses.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from source_analytics.analyses._evoked_hypotheses import write_evoked_hypotheses
from source_analytics.config import DesignSpec, Hypothesis

GROUPS = ["Vehicle", "AUT00206", "AUT00201"]
ROIS = ["Auditory_L", "Auditory_R", "Visual_L"]
MEASURES = ["itc_onset", "erp_n1_amp"]

# The planted effect: AUT00206 > Vehicle, on itc_onset, in Auditory_L only.
SIGNAL = ("itc_onset", "Auditory_L", "AUT00206", 1.4)


def _measure_rows(spatial_col: str = "roi", seed: int = 7) -> list[dict]:
    """Long evoked rows; one measure x one spatial unit x one group carries signal."""
    rng = np.random.default_rng(seed)
    m_sig, sp_sig, g_sig, amp = SIGNAL
    rows = []
    sid = 0
    for g in GROUPS:
        for _ in range(12):
            sid += 1
            uid = f"m{sid:03d}"
            for sp in ROIS:
                for m in MEASURES:
                    val = 1.0 + rng.normal(0, 0.5)
                    if m == m_sig and sp == sp_sig and g == g_sig:
                        val += amp
                    rows.append({
                        "subject": uid, "group": g, spatial_col: sp,
                        "measure_name": m, "measure_type": "itc",
                        "band_lo": np.nan, "band_hi": np.nan,
                        "time_lo": 0.0, "time_hi": 0.1,
                        "value": val, "n_epochs": 40,
                    })
    return rows


def _hyp() -> Hypothesis:
    return Hypothesis(name="drug_effect", kind="contrast",
                      weights={"AUT00206": 1.0, "Vehicle": -1.0})


def _analysis(tmp_path, *, rows, spatial_col="roi", name="roi_evoked",
              hyps=None, factor="group", selection=None):
    """Minimal stand-in for the BaseAnalysis surface the wiring touches."""
    spec = DesignSpec(factor=factor, reference="Vehicle", levels=GROUPS,
                      hypotheses=_hyp_list(hyps))
    tbl = tmp_path / "tables"
    tbl.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        name=name,
        config=SimpleNamespace(design_spec=spec, bands={}),
        tbl_dir=tbl,
        output_dir=tmp_path,
        _selection=selection or {},
        _measure_rows=rows,
    )


def _hyp_list(hyps):
    if hyps is None:
        return [_hyp()]
    return hyps


def _run(tmp_path, **kw):
    spatial_col = kw.pop("spatial_col", "roi")
    name = kw.get("name", "roi_evoked")
    an = _analysis(tmp_path, spatial_col=spatial_col, **kw)
    write_evoked_hypotheses(
        an, spatial_col=spatial_col, measures_csv=f"{name}_measures.csv"
    )
    return an.tbl_dir / f"{name}_hypotheses.csv"


# --------------------------------------------------------------------------- #

def test_writes_table_with_cell_per_measure_and_spatial(tmp_path):
    out = _run(tmp_path, rows=_measure_rows())
    assert out.exists(), "additive hypotheses CSV was not written"
    df = pd.read_csv(out)
    # one hypothesis x 2 measures x 3 ROIs
    assert len(df) == len(MEASURES) * len(ROIS)
    assert set(df["measure_name"]) == set(MEASURES)
    assert set(df["spatial"]) == set(ROIS)
    assert set(df["hypothesis"]) == {"drug_effect"}


def test_measure_and_spatial_specificity(tmp_path):
    df = pd.read_csv(_run(tmp_path, rows=_measure_rows()))
    m_sig, sp_sig, _, _ = SIGNAL

    def cell(m, sp):
        r = df[(df.measure_name == m) & (df.spatial == sp)]
        assert len(r) == 1
        return r.iloc[0]

    # planted cell recovered, and in the planted direction
    hit = cell(m_sig, sp_sig)
    assert bool(hit["significant"]), "planted cell not recovered"
    assert hit["estimate"] > 0

    # same measure, other ROIs: null
    for sp in ROIS:
        if sp != sp_sig:
            assert not bool(cell(m_sig, sp)["significant"])
    # the other measure: null everywhere, including the signal ROI
    for sp in ROIS:
        assert not bool(cell("erp_n1_amp", sp)["significant"])


def test_band_coordinate_is_null(tmp_path):
    """Evoked measures fix their own band/time window, so there is no band axis."""
    df = pd.read_csv(_run(tmp_path, rows=_measure_rows()))
    assert df["band"].isna().all()


def test_each_measure_is_its_own_fdr_family(tmp_path):
    """A null measure must not dilute a signal measure's correction, or vice versa."""
    df = pd.read_csv(_run(tmp_path, rows=_measure_rows()))
    fam_sizes = df.groupby("measure_name")["fdr_family"].nunique()
    assert (fam_sizes == 1).all(), "a measure's cells split across FDR families"

    # The family is the spatial grid within one measure: q == BH over 3 ROIs.
    for m in MEASURES:
        sub = df[df.measure_name == m].sort_values("spatial")
        p = sub["p_value"].to_numpy()
        n = len(p)
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        expect = np.minimum.accumulate(ranked[::-1])[::-1]
        got = sub["q_value"].to_numpy()[order]
        assert np.allclose(got, expect, atol=1e-12), f"{m}: q is not BH over its ROIs"


def test_csv_fallback_when_no_rows_in_memory(tmp_path):
    """--steps statistics on its own reads the exported measures CSV."""
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(_measure_rows()).to_csv(data / "roi_evoked_measures.csv", index=False)

    out = _run(tmp_path, rows=[])          # nothing in memory
    assert out.exists()
    df = pd.read_csv(out)
    m_sig, sp_sig, _, _ = SIGNAL
    hit = df[(df.measure_name == m_sig) & (df.spatial == sp_sig)].iloc[0]
    assert bool(hit["significant"])


def test_missing_rows_and_missing_csv_is_a_noop(tmp_path):
    out = _run(tmp_path, rows=[])
    assert not out.exists()


def test_renamed_design_factor_resolves(tmp_path):
    """Rows always carry 'group'; a study naming its factor otherwise still works."""
    out = _run(tmp_path, rows=_measure_rows(), factor="treatment")
    df = pd.read_csv(out)
    m_sig, sp_sig, _, _ = SIGNAL
    assert bool(df[(df.measure_name == m_sig) & (df.spatial == sp_sig)].iloc[0]["significant"])


def test_no_declared_hypotheses_writes_nothing(tmp_path):
    out = _run(tmp_path, rows=_measure_rows(), hyps=[])
    assert not out.exists()


def test_hypothesis_selection_filters(tmp_path):
    other = Hypothesis(name="other_effect", kind="contrast",
                       weights={"AUT00201": 1.0, "Vehicle": -1.0})
    out = _run(tmp_path, rows=_measure_rows(), hyps=[_hyp(), other],
               selection={"hypothesis": frozenset({"drug_effect"})})
    df = pd.read_csv(out)
    assert set(df["hypothesis"]) == {"drug_effect"}


def test_electrode_arm_behaves_identically_on_channel(tmp_path):
    """The electrode comparator is the same wiring over 'channel'."""
    out = _run(tmp_path, rows=_measure_rows(spatial_col="channel"),
               spatial_col="channel", name="electrode_evoked")
    df = pd.read_csv(out)
    m_sig, sp_sig, _, _ = SIGNAL
    assert len(df) == len(MEASURES) * len(ROIS)
    assert bool(df[(df.measure_name == m_sig) & (df.spatial == sp_sig)].iloc[0]["significant"])


def test_missing_spatial_column_is_a_noop(tmp_path):
    rows = _measure_rows()
    for r in rows:
        r.pop("roi")
    out = _run(tmp_path, rows=rows)
    assert not out.exists()


def test_fdr_family_label_separates_the_measures(tmp_path):
    """Two measures are two FDR families, and must not share one label.

    Regression for the facet-identity defect: ``fdr_family`` encodes family
    IDENTITY so q-values from different families are provably non-comparable
    (REPORT_PLAN §10b). The member hash cannot do it here — both families span
    the same ROI cells — so the facet has to appear in the label. Before the fix
    both measures emitted ``key=all|drug_effect|NA members=cell[3] hash=...``,
    byte-identical across genuinely different families.
    """
    df = pd.read_csv(_run(tmp_path, rows=_measure_rows()))
    per_measure = df.groupby("measure_name")["fdr_family"].agg(set)
    assert all(len(s) == 1 for s in per_measure), "a measure split across labels"
    labels = {m: next(iter(s)) for m, s in per_measure.items()}
    assert len(set(labels.values())) == len(MEASURES), (
        f"distinct FDR families share one label: {labels}"
    )
    for m, lab in labels.items():
        assert f"measure_name={m}" in lab, f"facet identity missing from {lab!r}"

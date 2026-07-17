"""Verification of the hypothesis-layer tabular adapter (Python, graph modules).

Checks (synthetic long nodal table, 4 groups × 3 ROIs × 1 band × 1 facet):
  1. contrast estimate/SE/t on the signal ROI match an independent pooled-SD
     one-way computation (the emmeans/`.adapt_cell` formula);
  2. spatial specificity — the planted-signal ROI is significant, a null ROI is not;
  3. omnibus yields a finite F with partial ω² in [0, 1);
  4. equivalence emits a TOST verdict + margin;
  5. declarative FDR scope: per-band recovers a signal the hypothesis-wide family dilutes;
  6. the standard schema (+ legacy aliases) is written.

Run: uv run --no-sync pytest tests/test_tabular.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from source_analytics.config import DesignSpec, Hypothesis
from source_analytics.hypothesis.tabular import (
    _adapt_cell,
    _apply_fdr,
    _fdr_family_label,
    write_module_hypotheses_tabular,
)

GROUPS = ["WT_VEH", "KO_VEH", "KO_HD_ICV", "KO_HD_IV"]


def _long_df(seed: int = 11) -> pd.DataFrame:
    """4 groups × 12 subjects × ROIs {A,B,C}; ROI A carries a KO_VEH>WT_VEH signal."""
    rng = np.random.default_rng(seed)
    eff = {"A": {"KO_VEH": 1.2}, "B": {}, "C": {}}  # signal only on A, KO_VEH
    rows = []
    sid = 0
    for g in GROUPS:
        for _ in range(12):
            sid += 1
            uid = f"s{sid:03d}"
            for roi in ("A", "B", "C"):
                val = 2.0 + eff[roi].get(g, 0.0) + rng.normal(0, 0.5)
                rows.append({
                    "subject": uid, "group": g, "band": "B1",
                    "conn_metric": "coh", "graph_metric": "degree",
                    "roi": roi, "value": val,
                })
    return pd.DataFrame(rows)


def _cfg(hyps, fdr=None) -> SimpleNamespace:
    spec = DesignSpec(factor="group", reference="WT_VEH", levels=GROUPS,
                      fdr=fdr or {}, hypotheses=hyps)
    return SimpleNamespace(design_spec=spec)


def test_contrast_matches_pooled_one_way():
    df = _long_df()
    a = df[(df.roi == "A")]
    spec = DesignSpec(factor="group", reference="WT_VEH", levels=GROUPS)
    hyp = Hypothesis(name="disease_effect", kind="contrast",
                     weights={"KO_VEH": 1.0, "WT_VEH": -1.0})
    by_group = {g: a.loc[a.group == g, "value"].to_numpy() for g in GROUPS}
    res = _adapt_cell(by_group, hyp, GROUPS)

    # Independent pooled-SD one-way contrast (value ~ group, all 4 groups).
    arrays = [by_group[g] for g in GROUPS]
    ns = np.array([x.size for x in arrays])
    N, k = int(ns.sum()), len(arrays)
    sse = sum(((x - x.mean()) ** 2).sum() for x in arrays)
    s = np.sqrt(sse / (N - k))
    mk, mw = by_group["KO_VEH"].mean(), by_group["WT_VEH"].mean()
    est = mk - mw
    se = s * np.sqrt(1 / by_group["KO_VEH"].size + 1 / by_group["WT_VEH"].size)
    assert abs(res["estimate"] - est) < 1e-9
    assert abs(res["SE"] - se) < 1e-9
    assert abs(res["stat"] - est / se) < 1e-9
    assert res["df"] == N - k
    assert res["effect_size_type"] == "hedges_g"
    assert res["group_a"] == "KO_VEH" and res["group_b"] == "WT_VEH"


def test_spatial_specificity_and_schema(tmp_path):
    df = _long_df()
    hyps = [
        Hypothesis(name="group_omnibus", kind="omnibus", role="phenotype"),
        Hypothesis(name="disease_effect", kind="contrast", role="phenotype",
                   label="Disease effect", weights={"KO_VEH": 1.0, "WT_VEH": -1.0}),
    ]
    out = write_module_hypotheses_tabular(
        df, _cfg(hyps), tmp_path, prefix="roi_graph",
        value_col="value", spatial_col="roi",
        facet_cols=("conn_metric", "graph_metric"), band_col="band",
    )
    assert out is not None
    # native hypothesis schema
    for col in ("hypothesis", "kind", "band", "spatial", "estimate", "p_value",
                "q_value", "significant", "effect_size", "stat",
                "conn_metric", "graph_metric"):
        assert col in out.columns, f"missing column {col}"
    # legacy aliases dropped (native schema is the sole contract)
    for col in ("contrast", "roi", "t_ratio", "hedges_g", "p_fdr", "power_type"):
        assert col not in out.columns, f"legacy alias {col} should be dropped"
    # signal ROI A is significant for disease_effect; null ROI C is not
    de = out[out.hypothesis == "disease_effect"]
    a = de[de.spatial == "A"].iloc[0]
    c = de[de.spatial == "C"].iloc[0]
    assert a["significant"] and not c["significant"]
    assert a["estimate"] > 0.5  # KO_VEH > WT_VEH
    # omnibus rows finite F, partial omega^2 in [0,1)
    om = out[out.hypothesis == "group_omnibus"]
    assert len(om) == 3 and np.isfinite(om["stat"]).all()
    assert ((om["effect_size"] >= 0) & (om["effect_size"] < 1)).all()


def test_scalar_no_spatial(tmp_path):
    """vertex_graph shape: global scalar per facet, spatial_col=None → one cell/facet/band."""
    rng = np.random.default_rng(3)
    rows = []
    sid = 0
    for g in GROUPS:
        for _ in range(12):
            sid += 1
            for cm in ("coh", "wpli"):
                for gmet in ("global_efficiency", "modularity"):
                    sig = 0.8 if (cm == "coh" and gmet == "global_efficiency" and g == "KO_VEH") else 0.0
                    rows.append({
                        "subject": f"s{sid:03d}", "group": g, "band": "B1",
                        "conn_metric": cm, "graph_metric": gmet,
                        "value": 1.0 + sig + rng.normal(0, 0.4),
                    })
    df = pd.DataFrame(rows)
    hyps = [Hypothesis(name="disease_effect", kind="contrast", role="phenotype",
                       weights={"KO_VEH": 1.0, "WT_VEH": -1.0})]
    out = write_module_hypotheses_tabular(
        df, _cfg(hyps), tmp_path, prefix="vertex_graph",
        value_col="value", spatial_col=None,
        facet_cols=("conn_metric", "graph_metric"), band_col="band",
    )
    assert out is not None
    assert out["spatial"].isna().all()              # no spatial dimension
    assert len(out) == 4                            # 2 conn × 2 graph facets, 1 band
    sig = out[(out.conn_metric == "coh") & (out.graph_metric == "global_efficiency")].iloc[0]
    null = out[(out.conn_metric == "wpli") & (out.graph_metric == "modularity")].iloc[0]
    assert sig["estimate"] > 0.4 and sig["significant"]
    assert not null["significant"]


def test_equivalence_verdict(tmp_path):
    df = _long_df()
    hyps = [Hypothesis(name="norm", kind="equivalence", role="normalization",
                       weights={"KO_HD_ICV": 1.0, "WT_VEH": -1.0},
                       margin={"mode": "sd", "value": 2.0})]
    out = write_module_hypotheses_tabular(
        df, _cfg(hyps), tmp_path, prefix="roi_graph",
        value_col="value", spatial_col="roi",
        facet_cols=("conn_metric", "graph_metric"), band_col="band",
    )
    assert out is not None
    assert "equivalent" in out.columns
    assert out["stat_type"].eq("tost").all()
    assert (out["margin_used"] > 0).all()


def test_fdr_scope_family_size():
    # One signal cell among weak cells in band b1; b2 all weak. Per-band recovers it;
    # the hypothesis-wide family (n=10) dilutes it. (Same logic as the R §9 test.)
    rows = [
        {"band": "b1", "spatial": s, "p_value": p}
        for s, p in zip("rstuv", [0.008, 0.04, 0.2, 0.4, 0.6])
    ] + [
        {"band": "b2", "spatial": s, "p_value": p}
        for s, p in zip("rstuv", [0.50, 0.60, 0.7, 0.8, 0.9])
    ]
    band_rows = [dict(r) for r in rows]
    hyp_rows = [dict(r) for r in rows]
    _apply_fdr(band_rows, "BH", "band")
    _apply_fdr(hyp_rows, "BH", "hypothesis")
    sig_band = band_rows[0]  # b1/r, p=0.008
    sig_hyp = hyp_rows[0]
    assert sig_band["significant"] and not sig_hyp["significant"]
    assert sum(r["significant"] for r in band_rows) > sum(r["significant"] for r in hyp_rows)
    none_rows = [dict(r) for r in rows]
    _apply_fdr(none_rows, "BH", "none")
    assert all(abs(r["q_value"] - r["p_value"]) < 1e-12 for r in none_rows)


def test_fdr_family_label_qualified_and_member_hash():
    """W1: fdr_family is fully qualified (scope/method/key/members/hash) and the
    hash encodes member-set IDENTITY, so a 20-ROI family is never confused with a
    32-ROI one even when scope/method/band match (REPORT_PLAN §10b).

    These exact strings are also asserted, byte-for-byte, by the R side in
    tests/test_hypothesis.R — the label builders must stay in lockstep.
    """
    lab = _fdr_family_label(
        "BH", "band", "Alpha", "disease_effect", "relative",
        ["Motor_L", "Auditory_L", "Auditory_R"],
        ["Motor_L", "Auditory_L", "Auditory_R"], "roi",
    )
    assert lab == (
        "scope=band method=BH key=Alpha|disease_effect|relative "
        "members=roi[3] hash=7463bb9d"
    )

    # band-less (aperiodic) hypothesis-scope family: band coerces to the NA token
    lab2 = _fdr_family_label(
        "BH", "hypothesis", None, "hd_icv_rescue", "offset",
        [None, None], ["Auditory_L", "Auditory_R"], "roi",
    )
    assert lab2 == (
        "scope=hypothesis method=BH key=all|hd_icv_rescue|offset "
        "members=cell[2] hash=11f8cfba"
    )

    # §10b property: same band/hyp/dv/scope, different ROI membership → different hash
    rois20 = [f"R{i:02d}" for i in range(20)]
    rois32 = [f"R{i:02d}" for i in range(32)]
    h20 = _fdr_family_label("BH", "band", "Alpha", "h", "rel", rois20, rois20, "roi")
    h32 = _fdr_family_label("BH", "band", "Alpha", "h", "rel", rois32, rois32, "roi")
    assert "members=roi[20]" in h20 and "members=roi[32]" in h32
    assert h20.rsplit("hash=", 1)[1] != h32.rsplit("hash=", 1)[1]

    # membership identity is order-invariant (sorted before hashing)
    shuffled = list(reversed(rois20))
    assert _fdr_family_label("BH", "band", "Alpha", "h", "rel", shuffled, shuffled, "roi") == h20

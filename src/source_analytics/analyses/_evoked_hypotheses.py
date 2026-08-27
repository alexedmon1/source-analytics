"""Declared-hypothesis wiring shared by the evoked modules.

``roi_evoked`` and ``electrode_evoked`` are near-twins: the same measure
extraction over a different spatial unit (ROI vs channel), writing the same
row schema. They therefore need the same hypothesis wiring, and it lives here
so the two cannot drift — the same reason ``_network_base`` exists for the
graph/NBS pair.

**Why these two were deferred.** ``docs/methods/HYPOTHESIS.md`` §9 lists
``roi_evoked`` / ``electrode_evoked`` as deferred with the reason "long-format
DV". Every other emmeans-tabular module exports one column per dependent
variable (``roi_psd`` has a column per band-power), so the module hands
``write_module_hypotheses`` a ``dv_cols`` vector. The evoked modules instead
export a single ``value`` column with ``measure_name`` telling you what it
holds. There is exactly one DV and 20-odd measures inside it.

**The resolution is that the measure is a FACET, not a DV.** The tabular adapter
already carries ``facet_cols``, which runs an independent FDR family per facet
combination across the band x spatial grid. Pointing that at ``measure_name``
gives one family per measure across the ROI/channel grid, which is the only
defensible family here: the measures are on incomparable scales (ITC is 0-1,
ERSP is dB, ERP amplitude is signal units, ERP latency is seconds), so a family
spanning them would pool quantities with no common null. The declared ``fdr:``
scope still modulates *within* a measure.

**There is no band axis.** Each measure definition already fixes its own band
and time window — ``band_lo``/``band_hi`` are properties of the measure, not a
factor to test across — so no ``band`` column is exported and the band
coordinate stays null. That is a genuine absence, not a dropped axis.

This is **additive**. It does not replace the descriptive ``group * roi`` LMM
the R scripts run: that is a whole-brain omnibus over every spatial unit, while
this is the a-priori path, one declared contrast/omnibus/equivalence per
``hypotheses:`` entry. Both tables are written.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Every evoked measure row carries the group under this key, whatever the study
# calls its design factor.
_ROW_GROUP_KEY = "group"


def evoked_hypothesis_frame(analysis, measures_csv: str) -> pd.DataFrame | None:
    """The measure table to test, from memory or from the exported CSV.

    Prefers the in-memory rows, so a full run needs no round-trip. Falls back to
    the module's exported measures CSV so ``--steps statistics`` works on its own
    against an earlier run's export.

    Returns ``None`` (having logged why) when there is nothing testable.
    """
    rows = getattr(analysis, "_measure_rows", None)
    if rows:
        df = pd.DataFrame(rows)
    else:
        csv = Path(analysis.output_dir) / "data" / measures_csv
        if not csv.exists():
            logger.warning(
                "No measure rows in memory and no %s — skipping %s hypotheses.",
                csv.name, analysis.name,
            )
            return None
        df = pd.read_csv(csv)

    if df.empty:
        return None

    # The design factor is whatever the spec names it; rows always carry the
    # group under "group", so alias when a study renames the factor.
    factor = analysis.config.design_spec.factor
    if factor not in df.columns:
        if _ROW_GROUP_KEY not in df.columns:
            logger.warning(
                "%s measure table has neither '%s' nor '%s' — skipping hypotheses.",
                analysis.name, factor, _ROW_GROUP_KEY,
            )
            return None
        df[factor] = df[_ROW_GROUP_KEY]
    return df


def write_evoked_hypotheses(analysis, *, spatial_col: str, measures_csv: str):
    """Run every declared hypothesis over an evoked measure table.

    Writes the additive ``<name>_hypotheses.csv``. No-op when nothing is
    declared. See the module docstring for why the measure is the facet.

    Parameters
    ----------
    analysis
        The calling :class:`BaseAnalysis`; supplies config, ``tbl_dir``, ``name``
        and the ``--hypothesis`` selection.
    spatial_col
        The spatial unit column — ``"roi"`` or ``"channel"``.
    measures_csv
        Basename of the module's exported measures CSV, used as the fallback
        source when no rows are in memory.
    """
    spec = analysis.config.design_spec
    if spec is None or not spec.hypotheses:
        return None

    df = evoked_hypothesis_frame(analysis, measures_csv)
    if df is None:
        return None
    if spatial_col not in df.columns:
        logger.warning(
            "%s measure table has no '%s' column — skipping hypotheses.",
            analysis.name, spatial_col,
        )
        return None

    from ..hypothesis import write_module_hypotheses_tabular

    wanted = analysis._selection.get("hypothesis")
    return write_module_hypotheses_tabular(
        df, analysis.config, analysis.tbl_dir, prefix=analysis.name,
        value_col="value", spatial_col=spatial_col,
        facet_cols=("measure_name",),
        # No band axis: absent from the frame, so the adapter runs a single
        # null-band pass rather than iterating bands.
        band_col="band",
        hypothesis=",".join(sorted(wanted)) if wanted else None,
    )

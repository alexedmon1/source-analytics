"""Minimum detectable effect (MDE) for the ``hypothesis`` layer's tabular rows.

Every tabular inferential row says how large a true difference the design would
have detected, so a null reads as "nothing as large as X" rather than as absence.
DUET (OPEN.md #151) reads each finding's size against this number, and only the
tool that fitted the model can supply it: the SE and df the design produced are
here, not in the record.

The contract (identical names in source-analytics, neurofaune and neurovrai):

- ``SE`` -- this tool's existing standard-error column *is* the contract's ``se``;
  it is not renamed and not duplicated.
- ``mde`` = ``(t.ppf(1 - alpha/2, df) + t.ppf(power, df)) * SE``, a positive number
  in the estimate's units; normal quantiles when ``df`` is missing or infinite.
- ``mde_percent`` = ``100 * mde / |reference mean|``; NaN for slopes, omnibus tests,
  or a reference mean within one residual SD of zero (``|reference mean| <
  residual SD`` of the fitted model; the pooled SD for a two-group test). For
  change scores, centred measures, dB or log ratios a percent of a near-zero mean
  is a huge meaningless number. The rule is shared with neurofaune and neurovrai.
- ``mde_power`` = 0.80 and ``mde_alpha`` = 0.05, constant, so the CSV describes itself.
- ``mde_note`` -- empty when ``mde`` is computed, otherwise ``"not computed: <reason>"``.
  ``mde`` is never NaN without a note.

Power 0.80 at two-sided alpha 0.05, **nominal**: no multiple-comparison correction,
by the user's decision. ``q_value`` is a separate column and the MDE does not try to
anticipate it. Cluster, vertex, permutation and NBS results carry the note instead
of a number: they have no single estimate with a standard error.

The R side (``R/hypothesis.R``: ``mde_from_se`` / ``.add_mde``) computes the same
thing; ``tests/test_power.py`` and ``tests/test_hypothesis.R`` pin the same
reference values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as st

MDE_POWER = 0.80
MDE_ALPHA = 0.05


def mde_from_se(se: float, df: float | None = None, *,
                alpha: float = MDE_ALPHA, power: float = MDE_POWER) -> float:
    """MDE for an estimate with standard error ``se`` on ``df`` degrees of freedom.

    Normal quantiles when ``df`` is None/NaN/inf. NaN when ``se`` is not a positive
    finite number or ``df`` is not positive.
    """
    try:
        se = float(se)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(se) or se <= 0:
        return float("nan")
    d = float("nan") if df is None else float(df)
    if math.isnan(d) or math.isinf(d):
        q = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    elif d <= 0:
        return float("nan")
    else:
        q = st.t.ppf(1 - alpha / 2, d) + st.t.ppf(power, d)
    return float(q * se)


def reference_weights(weights: dict[str, float], groups: list[str],
                      reference: str | None = None) -> np.ndarray | None:
    """Weights over ``groups`` whose combination of group means is the reference mean.

    The design's reference level when the contrast weights it; otherwise the
    negative-weight side (|w|-weighted), the baseline the estimate is taken from.
    None when the contrast has no negative side. Mirrors R ``.mde_reference_weights``.
    """
    wv = np.array([float(weights.get(g, 0.0)) for g in groups])
    rw = np.zeros(len(groups))
    if reference is not None and reference in groups and wv[groups.index(reference)] != 0:
        rw[groups.index(reference)] = 1.0
    elif np.any(wv < 0):
        neg = wv < 0
        rw[neg] = np.abs(wv[neg]) / np.abs(wv[neg]).sum()
    else:
        return None
    return rw


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def mde_columns(se: float, df: float | None, ref_mean: float | None = None,
                resid_sd: float | None = None) -> dict[str, Any]:
    """The five MDE columns for one row with a standard error.

    ``mde_percent`` needs both ``ref_mean`` and ``resid_sd`` (the fitted model's
    residual SD) and is left NaN when ``|ref_mean| < resid_sd``.
    """
    m = mde_from_se(se, df)
    pct = float("nan")
    if (math.isfinite(m) and _finite(ref_mean) and _finite(resid_sd)
            and abs(float(ref_mean)) >= float(resid_sd)):
        pct = 100.0 * m / abs(float(ref_mean))
    return {
        "mde": m,
        "mde_percent": pct,
        "mde_power": MDE_POWER,
        "mde_alpha": MDE_ALPHA,
        "mde_note": "" if math.isfinite(m) else "not computed: no finite standard error",
    }


def mde_not_computed(reason: str) -> dict[str, Any]:
    """The five MDE columns for a row that has no single estimate with an SE."""
    return {
        "mde": float("nan"),
        "mde_percent": float("nan"),
        "mde_power": MDE_POWER,
        "mde_alpha": MDE_ALPHA,
        "mde_note": f"not computed: {reason}",
    }

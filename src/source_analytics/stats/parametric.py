"""Parametric statistical tests: t-tests and linear mixed models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Container for a statistical test result."""

    test: str
    statistic: float
    p_value: float
    df: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    n_a: int | None = None
    n_b: int | None = None
    group_a_mean: float | None = None
    group_b_mean: float | None = None
    group_a_std: float | None = None
    group_b_std: float | None = None
    converged: bool = True


def ttest_between_groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
    equal_var: bool = False,
) -> TestResult:
    """Independent samples t-test (Welch's by default).

    Parameters
    ----------
    values_a, values_b : ndarray
        Values for each group (one per subject).
    equal_var : bool
        If False (default), use Welch's t-test.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)

    # Remove NaNs
    values_a = values_a[~np.isnan(values_a)]
    values_b = values_b[~np.isnan(values_b)]

    if len(values_a) < 2 or len(values_b) < 2:
        return TestResult(
            test="welch_ttest" if not equal_var else "student_ttest",
            statistic=np.nan,
            p_value=np.nan,
            n_a=len(values_a),
            n_b=len(values_b),
        )

    t_stat, p_val = stats.ttest_ind(values_a, values_b, equal_var=equal_var)

    # Welch-Satterthwaite degrees of freedom
    if not equal_var:
        n1, n2 = len(values_a), len(values_b)
        v1, v2 = np.var(values_a, ddof=1), np.var(values_b, ddof=1)
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2
    else:
        df = len(values_a) + len(values_b) - 2

    return TestResult(
        test="welch_ttest" if not equal_var else "student_ttest",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=float(df),
        n_a=len(values_a),
        n_b=len(values_b),
        group_a_mean=float(np.mean(values_a)),
        group_b_mean=float(np.mean(values_b)),
        group_a_std=float(np.std(values_a, ddof=1)),
        group_b_std=float(np.std(values_b, ddof=1)),
    )


def lmm_group_effect(
    df: pd.DataFrame,
    value_col: str = "value",
    group_col: str = "group",
    subject_col: str = "subject",
    roi_col: str | None = None,
) -> TestResult:
    """Fit a linear mixed model: value ~ group + (1|subject).

    Optionally nests ROI within subject: value ~ group + (1|subject/roi).

    Parameters
    ----------
    df : DataFrame
        Long-format data with columns for value, group, subject, and optionally roi.
    value_col, group_col, subject_col, roi_col : str
        Column names.

    Returns
    -------
    TestResult
        With LMM z-statistic and p-value for the group effect.
    """
    import statsmodels.formula.api as smf

    work = df[[value_col, group_col, subject_col]].dropna().copy()
    if roi_col and roi_col in df.columns:
        work[roi_col] = df[roi_col]

    if len(work) < 4:
        return TestResult(test="lmm", statistic=np.nan, p_value=np.nan, converged=False)

    # Ensure group is categorical
    work[group_col] = work[group_col].astype("category")

    formula = f"{value_col} ~ {group_col}"

    if roi_col and roi_col in work.columns:
        # Nested random effect: (1|subject) + (1|subject:roi)
        re_formula = f"1"
        groups = work[subject_col]
    else:
        re_formula = "1"
        groups = work[subject_col]

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.mixedlm(formula, work, groups=groups, re_formula=re_formula)
            # Try LBFGS first, fall back to Powell if singular
            try:
                result = model.fit(reml=True, method="lbfgs")
            except Exception:
                result = model.fit(reml=True, method="powell")

        # The group effect is the second parameter (first is intercept)
        group_params = [p for p in result.params.index if p != "Intercept" and "Group Var" not in p]
        if not group_params:
            group_params = [p for p in result.params.index if p != "Intercept"]

        if group_params:
            param_name = group_params[0]
            z_stat = float(result.tvalues[param_name])
            p_val = float(result.pvalues[param_name])
        else:
            z_stat = np.nan
            p_val = np.nan

        return TestResult(
            test="lmm",
            statistic=z_stat,
            p_value=p_val,
            converged=result.converged,
        )
    except Exception as e:
        logger.warning("LMM failed: %s", e)
        return TestResult(test="lmm", statistic=np.nan, p_value=np.nan, converged=False)

# Known issues

Recorded 2026-10-20, while adding the minimum detectable effect (MDE) to the
hypothesis tables (`feat/power-mde`). None is fixed yet; each is to be fixed
later.

## Defects

1. **`hedges_g` is not Hedges' g.** The hypothesis layer labels
   `effect_size_type = "hedges_g"` on a value computed as the estimate divided by
   the residual SD, with no small-sample correction (`R/hypothesis.R`, lines 412
   and 659; `src/source_analytics/hypothesis/tabular.py`, line 256). That is an
   uncorrected standardized difference (Cohen's d against the model's residual
   SD). Either apply the correction or rename the type.

2. **`uv.lock` is out of step with `pyproject.toml`.** The lock records
   source-analytics 0.8.0 and `pyproject.toml` says 0.8.2, so any `uv run` or
   `uv sync` rewrites `uv.lock`, leaving every checkout dirty. Re-lock and commit.

## Follow-ups to the MDE work

3. **`mde_percent` of a dB or log-scale measure means nothing.** Many measures
   here are in dB (spectral power) or on a log scale. A percent of a dB mean is
   not a percent of anything, and the near-zero rule (blank when the reference
   mean is within one residual SD of zero) does not catch it: a mean of -20 dB
   with an SD of 3 passes. `mde_percent` should be blank for measures declared
   as log or dB scale, which needs a scale per measure in the study config.

4. **Two rules choose the reference group.** The hypothesis layer divides by the
   `design.reference` group's mean when the contrast weights it; the older
   posthoc tables (`R/stats_utils.R`) divide by `emmean_b`. For a contrast like
   `{WT: +1, KO: -1}` with reference WT, the two give percents of different
   groups' means. One rule should hold everywhere.

5. **Not yet covered:** the posthoc functions defined inside individual module
   scripts (roi_aperiodic, electrode_aperiodic region_nested, roi/electrode_evoked
   channel and region, roi_pac region, roi_transfer_entropy, roi_connectivity)
   write tables without the MDE columns. The three `run_global_ttests` functions
   are retired and not called.

6. **Multiple-comparison correction.** The MDE uses a nominal alpha of 0.05 by
   decision; with BH correction across a band x spatial family the real bar is
   stricter, so the MDE is optimistic. To be added later.

7. **TOST rows** carry the MDE of the two-sided difference test, not of the
   equivalence test itself.

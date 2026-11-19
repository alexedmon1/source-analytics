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

2. **Fixed 2026-11-03** (re-locked with the `results` extra). **`uv.lock` is out of step with `pyproject.toml`.** The lock records
   source-analytics 0.8.0 and `pyproject.toml` says 0.8.2, so any `uv run` or
   `uv sync` rewrites `uv.lock`, leaving every checkout dirty. Re-lock and commit.

## Follow-ups to the MDE work

3. **`mde_percent` of a dB or log-scale measure means nothing.** Many measures
   here are in dB (spectral power) or on a log scale. A percent of a dB mean is
   not a percent of anything, and the near-zero rule (blank when the reference
   mean is within one residual SD of zero) does not catch it: a mean of -20 dB
   with an SD of 3 passes. `mde_percent` should be blank for measures declared
   as log or dB scale, which needs a scale per measure in the study config.

4. **Fixed 2026-11-19** (decided: the design reference everywhere). **Two rules
   chose the reference group.** The hypothesis layer divided by the
   `design.reference` group's mean when the contrast weights it; the older
   posthoc tables (`R/stats_utils.R`) divided by `emmean_b`. Every post-hoc table
   now uses the hypothesis layer's rule through `.add_posthoc_mde` in
   `R/hypothesis.R`: `emmean_a` when `group_a` is the reference, else `emmean_b`.

5. **Fixed 2026-11-19.** The post-hoc tables the module scripts still write
   (roi/electrode_evoked ROI or channel and region, electrode_aperiodic
   region_nested) carry the MDE columns. The rest of the list was already
   covered or is dead code: roi_aperiodic's and roi_pac's post-hoc tables are
   hypothesis-layer rows, and the post-hoc functions left in roi_aperiodic,
   roi_pac, roi_transfer_entropy and roi_connectivity are not called. The three
   `run_global_ttests` functions are retired and not called.

6. **Multiple-comparison correction.** The MDE uses a nominal alpha of 0.05 by
   decision; with BH correction across a band x spatial family the real bar is
   stricter, so the MDE is optimistic. To be added later.

7. **TOST rows** carry the MDE of the two-sided difference test, not of the
   equivalence test itself.

# Pending: settle the vertex split, then rename the ROI analyses to "source"

**Status:** pending, not started. Written 2026-09-11.

v0.7.1 works as it is, and nothing on this page changes it. When this work is
done or dropped, move this page to the [archive](../archive/README.md).

## Current state

- **v0.7.1 (tagged):** the vertex analyses are in core, and the ROI analyses are
  named `roi_*`.
- **`main`:** v0.7.1, plus nibabel as a core dependency.
- **PR #5** (`feat/vertex-plugin`,
  [alexedmon1/source-analytics#5](https://github.com/alexedmon1/source-analytics/pull/5))
  makes three changes:
  - It moves the vertex analyses into the private `source-analytics-vertex`
    plugin: the 12 vertex analyses, including `fcd_comparison`, plus
    `spectral.vertex`, `spectral.vertex_aperiodic`, the five `R/vertex_*.R`
    scripts and the glass-brain figures.
  - It adds the `source_analytics.plugins` entry-point hook.
  - It bumps the version to 0.8.0.

  The PR is open and mergeable. **It is on hold until Decision 1.**

## Decision 1: where the vertex analyses live

Vertex maps depend on one fixed placement of sources. They are being replaced by
ROI analyses on Monte Carlo source operators, which average over many source
draws and better match the electrode array's spatial resolution. The open
question is whether a comparison of vertex against Monte Carlo results has to be
published.

| Option | What it means | Catch |
|---|---|---|
| **A.** Merge PR #5 | Vertex runs from the plugin | Published results need public code. v0.7.1 is public; the plugin repo is private and would have to be made public |
| **B.** Close PR #5, keep vertex in core | The hook could still be merged on its own | The vertex code and its tests stay in the maintained package |
| **C.** Retire vertex | Merge PR #5 and leave the plugin unmaintained | None |

PR #5 works whether a comparison is published (A) or not (C); only B requires
closing it. Either way, v0.7.1 stays available to reproduce earlier vertex
results.

## Decision 2: rename `roi_*` to `source_*`

With vertex out of core, every source-level analysis runs on parcel time series,
however the sources were placed: predefined per ROI, Cartesian Monte Carlo, or
shell Monte Carlo. "ROI" as the name of the analysis family then no longer tells
anything apart. It also collides with source-localization's `roi_based`
placement method.

After the rename, source-analytics has three groups: **Source** analyses,
**Electrode** analyses, and the comparison modules.

### Design

- **Analysis names:** `roi_psd` → `source_psd`, and the same for `aperiodic`,
  `connectivity`, `cross_freq`, `directed`, `graph`, `nbs`, `network`,
  `signature` and `evoked`.
- **Metadata:** `level: "roi"` becomes `"source"`, and `source-analytics list`
  shows "Source Level".
- **Keep "ROI" and "region" where they mean an atlas parcel:** the `roi`
  columns, the region tier, the `roi_categories` config, and the per-ROI rows.
  That is the spatial unit, and the word is still correct there.
- **Everything else follows the new names:** R scripts, table-file prefixes
  (`roi_psd_region_hypotheses.csv` → `source_psd_region_hypotheses.csv`), class
  names and tests.
- **Old `roi_*` names keep working as deprecated aliases,** through
  `_DEPRECATED_NAMES`. An alias writes into the canonical `source_*` folder, so
  rerunning an old config changes its output paths.
- **source-localization keeps `roi_based`** as the name of its predefined
  placement method.
- **Release:** fold the rename into v0.8.0 if PR #5 hasn't merged yet, so there
  is one breaking release. Otherwise release it as v0.9.0.

### Size, as of 2026-09-11

- **source-analytics:** 15 files named `roi_*` and about 540 references.
- **source-lightbox:** about 24 hard-coded references in 7 files, mostly
  `render.py` and `summarize.py`.

### Steps

1. On a branch, rename the modules, classes, R scripts, registry and metadata
   keys, and table prefixes. Add the aliases and update the tests.
2. Check that no numbers change. Run one study arm on v0.7.1 and on the branch.
   After mapping the names, compare the tables cell for cell.
3. Make source-lightbox accept both `roi_*` and `source_*` names, since frozen
   studies keep `roi_*`.
4. Add a CHANGELOG entry under "Behaviour changes" and tag the release.
5. Studies switch to the new names at their next re-freeze. Frozen outputs are
   not touched.

# PROFILE_PROVENANCE_PLAN.md — `--profile` runs + output provenance

Scoping doc. Nothing here is implemented. Driver: FORGE MS2 needs a curated
**Report** build alongside the full **Exploratory** one
(`FORGE/treatment/REPORT_PLAN.md`, §10b). This doc scopes the source-analytics
side only.

## 1. Why this exists (the finding that forces it)

Verified empirically against `roi_psd_hypotheses.csv` (BH reproduced by hand):
**the FDR correction unit is `band × hypothesis × dv`, over the ROI set** —
91/91 families match; pooling hypotheses matches 0/13; pooling bands 0/7.

⇒ **The ROI set IS the FDR family.** The report narrows 32 → 20 ROIs, so every
report q-value differs from its exploratory counterpart — including for the four
bands whose edges are unchanged. The report therefore **cannot** be a row-filter
of the exploratory tables; it must be its own run. Narrowing hypotheses, bands,
or DVs *are* clean filters (family is per-hypothesis, per-band, per-dv), but the
ROI axis is not.

**Consequence:** two runs of one pipeline, two result trees, numbers that are
legitimately different and must never be compared cell-to-cell. Nothing today
lets an output say which run produced it or what family it was corrected in —
that gap is the whole point of this work. (The correction unit above had to be
reverse-engineered by re-running BH three ways; the table itself does not say.)

**Non-goal: caching/reuse.** Since the report recomputes everything anyway, there
is no inheritance layer to build. This is about *identification*, not reuse.

## 2. What the code actually looks like (verified 2026-07-15)

- **`fdr_family` has exactly two builders**, not one per module:
  `R/hypothesis.R:146 .apply_fdr` and `hypothesis/tabular.py:96 _apply_fdr`.
  Modules never build the string themselves.
- **But neither builder knows the hypothesis or dv.** The real family boundary is
  implicit in *caller loop nesting*: R `write_module_hypotheses` (R/hypothesis.R:428)
  loops `for hyp × for dv` and calls `run_hypothesis` → `.apply_fdr` once per
  (hyp, dv); Python `write_module_hypotheses_tabular:297` calls `_apply_fdr` once
  per (hyp, facet). The builders only ever see pre-filtered rows and label them
  with the one key they can see (`band`). **The string is under-descriptive, not
  wrong** — so fixing it is labelling-only and changes no q-value.
- **Supported fdr scopes:** `hypothesis | band | spatial | none` (R/hypothesis.R:116,
  tabular.py:42). None of them names the member set.
- **Every module reads `config.bands`, `config.roi_categories`, and
  `config.design_spec.hypotheses` exclusively** — no module re-parses raw YAML or
  hardcodes them. This is the key enabler: narrowing them on a derived
  `StudyConfig` propagates everywhere for free.
- **`for_paradigm()` / `for_paradigm_analysis()`** (config.py:531, 587) are the
  established "copy the dataclass, override some fields" pattern, and they pass
  `bands` / `roi_categories` / `design_spec` through **unchanged** — the natural
  seam for a `.for_profile()` sibling.
- **`get_paradigm_analyses()`** (config.py:521) is the single choke point for
  "which analyses run", used by both `cmd_run` loop sites.
- **`fig_dir` / `tbl_dir`** are computed properties on `BaseAnalysis` (base.py:90-100)
  — one choke point for output paths.
- **Writing is NOT centralized**: ~35 `to_csv` call sites across 20 modules, no
  shared writer. This is the single most expensive fact in this doc.
- **`exclude_analyses` does not exist in source-analytics** — it is a *lightbox*
  key. There is no allowlist/denylist for analyses here today.
- **No provenance machinery of any kind exists**: no run-id, config hash, git sha,
  or version stamping in any output. Greenfield.

### ⚠ Two traps found

1. **`__version__` is stale and would make any compute_key lie.**
   `src/source_analytics/__init__.py:3` says `0.3.0`; `pyproject.toml:7` says
   `0.4.0`; FORGE pins `v0.6.0`; git HEAD is `v0.6.0-22-geefb983`. A key built
   from `__version__` would silently claim 0.3.0 for all of it. **Fix version
   stamping (single-source it from package metadata + git describe) before it
   feeds a hash.**
2. **`viz/summary_figures.py::_find_coords` walks up exactly 4 levels** looking
   for a dir named `results` (line ~473), and infers `paradigm = tbl_dir.parent.name`
   (lines 202, 298, 481). Inserting a profile level *inside* results survives with
   **zero slack** (4 hops exactly reaches `results/`) and the paradigm inference
   still works — but any further nesting silently breaks coordinate lookup, and
   "silently" is the operative word: `_find_coords` returns `None` and glass
   brains just don't render.

## 3. Work items

### W1 — Fully-qualify `fdr_family` · **S** · non-breaking
Thread `hypothesis` + `dv` (+ facet keys) into the two builders, or prefix the
string at the call site where they're already in scope. Include the **member set
identity**, not just its size — e.g.
`scope=band method=BH key=Alpha|disease_effect|relative members=roi[20] hash=…`.
- Touch: `R/hypothesis.R:146` + call sites (`run_hypothesis:380`,
  `write_module_hypotheses:428-434`); `hypothesis/tabular.py:96` + call site (`:297`).
- **q-values do not change** — the family boundary already is (hyp, dv, band); only
  the label improves. Safe to ship independently, no re-run needed.
- Out of scope (separate FDR paths, no `fdr_family` column — leave alone):
  `roi_network_analysis.py:181,235` (scipy `false_discovery_control`, legacy
  `*_stats.csv`) and `R/stats_utils.R` (Holm, legacy pre-hypothesis tables).
- Risk: the two implementations must stay in lockstep; they already duplicate BH
  (`_p_adjust` is a hand-rolled `p.adjust`). Add a parity test.

### W2 — Profile-separated outputs · **S if root-override, L if per-row column**
Two separable halves — do not conflate:
- **W2a — separate trees · S. [RESOLVED 2026-07-15 — see §5.1]** The profile overrides
  `results_dir`/`output_dir` on the derived config (`.for_profile()`) to a **nested,
  asymmetric** tree: the default profile keeps `results/` unchanged; a named profile
  gets `results/<profile>/`. So `base.py` `fig_dir`/`tbl_dir`, `cli.py:427-430` (a
  second hardcoded copy of that path construction), `summary_figures.py`'s depth walk,
  and **source-lightbox's `scanner.py`** (which hardcodes `figures/<paradigm>/<analysis>`
  by depth) all keep working untouched — the curated build just points its lightbox
  `results:` entry at `results/<profile>/`.
  - **Hard constraint: exactly ONE level.** `_find_coords`'s 4-hop walk reaches
    `results/` from `results/<profile>/tables/<par>/<mod>` with **zero slack**.
    `results/profiles/<name>/…` breaks it silently. Do not nest deeper without
    fixing `_find_coords` first.
  - FORGE's first profile id is **`external`** (not `report` — `analyses/report/` is
    the IRL REPORTING contract; see §5.2).
- **W2b — `profile` column on every table · L.** Needs all ~35 `to_csv` sites, since
  there is no shared writer. **Prerequisite:** introduce `BaseAnalysis._write_table()`
  and migrate modules onto it; then the column is one edit. Defer — W2a + W1's
  member-set hash already make a row self-identifying in practice.

### W3 — `compute_key` provenance · **M** · greenfield
Hash of the inputs that actually determine a number: dv definition, band edges,
member/ROI set, hypothesis, model spec, subject set, estimator params, tool
version. Same key ⇒ same number; different key ⇒ never compare.
- **Blocked on the `__version__` trap above.**
- Cheaper first cut: a **run-level `run_manifest.json`** per profile tree (profile,
  resolved config hash, git describe, subject set, band/ROI/hypothesis sets,
  timestamp) instead of a per-row column. Gets 80% of the identification value at
  ~10% of the cost, and composes with the IRL `MANIFEST.sha256` freeze.

### W4 — `--profile report` run mode · **S/M** · the gate for the report
`StudyConfig.for_profile(name)`, sibling to `for_paradigm`:
- filter `bands` (dict subset by key), `roi_categories` (dict subset),
  `design_spec.hypotheses` (filter by name → new `DesignSpec`). **Propagates to all
  modules with zero per-module changes** (§2).
- `include_analyses` allowlist → intersect in `get_paradigm_analyses()` (one choke point).
- `connectivity_metrics` → **compile down to the existing `--select`/`--metric`
  machinery** (`cli.py:53-110 _parse_selection` → `BaseAnalysis._select`) rather than
  new plumbing; per-analysis config lives in `raw[analysis]`, not as dataclass fields.
- Blast radius: **`config.py` + `cli.py` only.** No changes to `analyses/*`,
  `stats/*`, `hypothesis/*`.
- Compose as `config.for_paradigm_analysis(p, a).for_profile(name)`.
- Tests: no test covers `for_paradigm*` narrowing today. `tests/test_select.py` is the
  precedent to mirror; add `test_config_profile.py`.

### ⚠ Gaps in FORGE's `report:` block that SA cannot honour yet
- **`dvs: [absolute, relative, delta_ref]` has no SA counterpart.** There is no
  first-class DV concept in config — `dv` exists only as an output *column*; the DV
  set is decided inside each module. And `delta_ref` **does not exist at all** (it's
  REPORT_PLAN R2 work). So `dvs:` is declarative intent today, not an enforceable
  narrowing. Either add a real DV concept or accept it as documentation.
- **`emphasis: effect_size` and `circos: false` are lightbox-side**, not SA.

## 4. Recommended sequence

1. **W1** — cheap, non-breaking, immediately stops the "what family is this?"
   archaeology. Ship independently of everything else.
2. **W4** — the actual gate on the report existing at all. Cheap because of §2.
3. **W2a** — profile trees via results-root override (pairs naturally with W4).
4. **W3-lite** — `run_manifest.json` per tree; fix `__version__` first.
5. **W2b / W3-full** — per-row `profile` + `compute_key`, only after
   `_write_table()` centralization makes it a one-line change.

**Report-critical path = W4 + W2a.** W1 and W3 are provenance hygiene that pay off
across both builds and are not strictly required to produce the report.

## 5. Decisions

### 5.1 Freeze boundary — **[RESOLVED 2026-07-15]** nested, asymmetric, one tag
Settled against the FORGE IRL spine (`treatment/spine-conventions.md`):
```
analyses/results/                  ← default profile (exploratory), paths UNCHANGED
analyses/results/<profile>/        ← named profile (e.g. external/)
analyses/report/                   ← the spine's REPORTING contract (unrelated)
```
- The freeze `find` already runs over `analyses/` (manifest paths are relative to
  `analyses/` and cover both `results/` and `report/` today), so a nested profile is
  **auto-covered by the existing `MANIFEST.sha256` under a single `analyses-vN`** — no
  spine amendment, no second tag. Atomic freeze is correct: both profiles are one
  snapshot off one `preprocessing-vN`.
- **Asymmetric on purpose**: the default profile is the analysis of record (feeds the
  manuscript); named profiles are derived deliverables. Zero path churn.
- ⇒ **`for_profile()` overrides `results_dir` → `results_dir / profile`** (and the
  analytics `output_dir` likewise) **only when a profile is named**; the default path
  must stay byte-identical to today.

### 5.2 Profile naming — **[RESOLVED 2026-07-15]** id ≠ UI label
FORGE's curated profile id is **`external`**, not `report`: `analyses/report/` is
already the IRL REPORTING contract (`results.json` → manuscript). The gallery still
labels the view **"Report"**. Generalization for SA: **profile ids are arbitrary
strings; never assume `report`**, and keep the id out of user-facing labels.

### 5.3 Still open
1. **Does `dvs:` become a real config concept**, or stay documentation until R2
   builds `delta_ref`? (SA has no DV concept today — see §3 gaps.)
2. **W1 string format** — human-readable (`members=roi[20] hash=ab12cd`) vs. a
   machine-parseable JSON blob in the column. Readability vs. tooling.
3. **Split `MANIFEST.sha256`?** (FORGE-side finding, affects any IRL study using SA.)
   The manifest mixes 53 tracked files with 582 gitignored figures, so the spine's
   "must pass" boundary check can never pass on a fresh clone, and passes locally only
   until the next `--steps figures` rewrites `fig_dir`. Proposal: a tracked manifest
   (portable, must pass) + a figures manifest (informational). Profiles double the
   figure count, so this gets worse before it gets better.

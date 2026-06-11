# Hypothesis-Testing Contrasts — Design Plan

**Status:** design, pre-implementation (2026-06-09). For review before any code.
**Repos:** source-analytics (schema + stats engine), source-lightbox (presentation).
**Branches:** source-analytics on `analysis-grouping` (as-is); source-lightbox on `main`.

---

## 1. Problem

Today every contrast is an independent `group_a` vs `group_b` difference test. In
`config.py` the `Contrast` dataclass keeps only `name, group_a, group_b` — all other
fields are dropped. The 11 FORGE contrasts run the same omnibus→post-hoc pipeline
symmetrically; the `group:` field only buckets them for display.

In a **treatment-rescue design** that makes most output noise. "Treatment vs WT vehicle"
(the Normalization tier) is run as a standalone difference test *everywhere*, but that
comparison is only meaningful **conditionally** — at the (analysis, band, region) cells
where (a) there is a real disease effect and (b) treatment moved KO toward WT. Nobody
cares whether an arbitrary treatment differs from WT at a cell that was never abnormal.

What carries a hypothesis:

| Tier | Hypothesis | Contrast | Test |
|------|-----------|----------|------|
| **H1 Phenotype** | KO veh ≠ WT veh — where is there a deficit? | KO_VEH vs WT_VEH | difference |
| **H2 Rescue** | *given H1*, treatment ≠ KO veh | treatment vs KO_VEH | difference (2-sided, sign reported) |
| **H3 Normalization** | *given H1+H2*, treatment ≈ WT veh ("restored") | treatment vs WT_VEH | **equivalence (TOST)** |

## 2. Design principles

1. **The hypothesis structure lives in the study YAML, never in code.** The engine is
   generic: it reads gating dependencies and test types and executes them. A different
   study with a different design just writes different contrasts.
2. **Gating shrinks the testing family.** A gated contrast is only tested at cells that
   survived its parent(s). This is both the multiplicity story (fewer tests, honest FDR)
   and the noise story (off-footprint comparisons never happen).
3. **Backward compatible.** A contrast with no `role`/`gate_on`/`test` behaves exactly as
   today (independent two-sided difference test). Existing studies are unaffected.

## 3. YAML schema

New optional per-contrast fields (added to the `Contrast` dataclass + serialized to
`study_config.yaml` so R sees them):

| Field | Values | Default | Meaning |
|-------|--------|---------|---------|
| `role` | `phenotype` \| `rescue` \| `normalization` \| `exploratory` | `exploratory` | Semantic tag; drives source-lightbox grouping. Engine treats as opaque. |
| `test` | `difference` \| `equivalence` | `difference` | Difference = LMM/emmeans contrast (current). Equivalence = TOST. |
| `gate_on` | contrast name or list of names | none | Restrict testing/reporting to the intersection of the named contrasts' FDR-significant cell masks. |
| `equivalence_margin` | `{mode, value}` | — | Required when `test: equivalence`. See §4.3. |

Study-level optional default block (so margins aren't repeated per contrast):

```yaml
hypothesis_testing:
  default_equivalence_margin: { mode: gap_fraction, value: 0.25 }
  gate_alpha: 0.05          # significance threshold defining a parent's mask
```

### 3.1 FORGE contrasts rewritten (illustrative)

```yaml
contrasts:
  # ── H1: phenotype (defines the disease footprint) ───────────────
  - name: disease_effect
    label: "Disease effect (KO vs WT)"
    group: "Disease effect"
    group_a: KO_VEH
    group_b: WT_VEH
    role: phenotype
    test: difference

  # ── H2: rescue (gated on phenotype) ─────────────────────────────
  - name: hd_icv_rescue
    label: "HD-ICV rescue (vs KO veh)"
    group: "Treatment rescue (vs KO veh)"
    group_a: KO_HD_ICV
    group_b: KO_VEH
    role: rescue
    gate_on: disease_effect
  - name: hd_iv_rescue   {…same shape, KO_HD_IV…}
  - name: ld_rescue      {…same shape, KO_LD_IV_ICV…}

  # ── H3: normalization (gated on phenotype + matching rescue, TOST) ──
  - name: hd_icv_normalization
    label: "HD-ICV normalization to WT"
    group: "Normalization to WT"
    group_a: KO_HD_ICV
    group_b: WT_VEH
    role: normalization
    test: equivalence
    gate_on: [disease_effect, hd_icv_rescue]
    # equivalence_margin omitted → inherits hypothesis_testing.default_equivalence_margin
  - name: hd_iv_normalization  {…gate_on: [disease_effect, hd_iv_rescue]…}
  - name: ld_normalization     {…gate_on: [disease_effect, ld_rescue]…}

  # ── Route & dose: treatment-vs-treatment, exploratory (ungated) ──
  - name: route_comparison  { group_a: KO_HD_ICV, group_b: KO_HD_IV, role: exploratory }
  - name: dose_icv          { … role: exploratory }
  - name: dose_iv           { … role: exploratory }
```

This replaces the old `hd_icv_vs_wt / hd_iv_vs_wt / ld_vs_wt` difference contrasts (the
noise) with gated equivalence `*_normalization` contrasts.

## 4. Engine semantics (source-analytics, mostly `R/stats_utils.R`)

### 4.1 Dependency ordering & masks
- Build a DAG from `gate_on`; topologically sort contrasts so parents run first. Error on
  cycles or unknown names (`config.py` validation).
- A contrast's **mask** = the set of cells (granularity per module, §5) where it is
  significant at `gate_alpha` after its own FDR. Persist masks per (analysis, power_type,
  contrast) in memory across the contrast loop.
- A gated contrast's testable cells = ∩ of its parents' masks. Cells outside the mask are
  not tested and not FDR-counted (smaller family → more power on the cells that matter).
- Emit the mask membership in the output tables (`gated_in: TRUE/FALSE`, `gate_parents:`)
  so the provenance is auditable and source-lightbox can render it.

### 4.2 Difference test (unchanged)
Existing omnibus LMM + emmeans pairwise, but the post-hoc cell loop is intersected with
the gate mask. `role: rescue` is two-sided; the signed estimate + CI are reported (no
direction requirement, per decision).

### 4.3 Equivalence test (TOST) — supports both margin modes
Conclude "equivalent to WT / normalized" when the (1−2·gate_alpha) CI of
mean(group_a) − mean(group_b) lies entirely within ±margin. Margin per cell:

- **`mode: gap_fraction`** — `margin = value × |Δ_disease(cell)|`, where `Δ_disease` is the
  KO_VEH−WT_VEH estimate at that cell (available because `disease_effect` is a gate parent
  and already estimated). Interpretation: "treatment closed ≥(1−value) of the deficit."
  Per-cell, scales with each region/band's deficit.
- **`mode: sd`** — `margin = value × pooled_SD(cell)` (Cohen's-d bounds, e.g. ±0.5).
  Standard, scale-free, doesn't adapt to deficit size.

Both produce the same TOST verdict columns (`tost_lower_p`, `tost_upper_p`, `equivalent`,
`margin_used`). A study may set a default and override per contrast.

### 4.4 Outputs
Per-cell stat tables gain: `role`, `test`, `gated_in`, `gate_parents`, and for equivalence
`equivalent` + `margin_used`. The rescue verdict per cell becomes a small enum the gallery
can map: `not_in_phenotype` / `rescued_normalized` / `rescued_not_normalized` /
`not_rescued`.

## 5. Per-module applicability ("the types of analyses")

The contrast machinery is consumed by many modules; the **cell mask granularity** differs:

| Module(s) | Mask granularity | Fit |
|-----------|------------------|-----|
| roi_psd, roi_aperiodic, electrode_psd, electrode_aperiodic | band × ROI/channel | clean |
| roi_connectivity, roi_transfer_entropy | band × edge | clean |
| roi_pac | band-pair × ROI | clean |
| vertex_spatial | band × vertex | clean |
| vertex_cluster | significant **clusters** | gate = restrict to disease clusters (cluster-level mask) |
| roi_nbs / vertex_nbs | significant **sub-network components** | gate = test rescue within disease components |
| roi_graph / vertex_graph | band × node metric | clean (node = cell) |
| vertex_mvpa | classification accuracy (not a per-cell contrast) | **out of framework** — decodability is a different hypothesis; leave as-is |

**Phase 1 targets** the "clean" LMM/cell modules. Cluster/NBS gating is Phase 2 (needs the
parent's cluster/component geometry, not just a cell list). MVPA is explicitly out.

## 6. source-lightbox tie-in

- `manifest.py`: carry `role` + `gate_on` + the verdict columns through per analysis/contrast.
- Presentation: replace flat per-contrast pages with a **rescue map** per (analysis, band):
  the disease footprint (H1) as the canvas; each treatment overlaid as
  rescued-normalized / rescued-not-normalized / not-rescued. Exploratory (route/dose) stay
  available but clearly secondary. Ungated treatment-vs-WT difference noise is gone.
- Keep it generic: the gallery keys on `role`/verdict columns, not FORGE group names.

## 7. Files & phases

**Phase 0 — schema/plumbing (lock the YAML contract):** `config.py` `Contrast` (+ validation,
DAG check), `study_config.yaml` serialization, manifest pass-through. Fields parsed, not yet
acted on. *Verifiable: study_config.yaml round-trips the new fields into R.*

**Phase 1 — engine on clean modules:** `R/stats_utils.R` (ordering, masks, gate intersection,
TOST w/ both margin modes), update the LMM analysis callers. *Verifiable: FORGE rerun shows
gated tables + equivalence verdicts.*

**Phase 2 — source-lightbox rescue-map presentation.**

**Phase 3 — cluster/NBS gating** (vertex_cluster, roi_nbs/vertex_nbs).

## 8. Decisions locked / open

Locked (2026-06-09): hierarchy is YAML-declarative; normalization = gated TOST; rescue =
two-sided (any difference, sign reported); margin supports **both** `gap_fraction` and `sd`.

Locked (2026-06-11): default margin = **`gap_fraction 0.25`** ("closed ≥75% of the deficit");
route/dose contrasts **kept as `exploratory`** (ungated difference, FDR-reported).

Open: exact rescue-map visual (Phase 2).

## 9. Progress

**Phase 0 — DONE (2026-06-11).** Schema + plumbing landed, fields parsed/validated but not
yet acted on by the engine.
- `config.py`: `Contrast` gained `label/group/role/test/gate_on/equivalence_margin` + a
  `Contrast.from_dict()` parser; `_validate_margin()` and `_validate_contrast_graph()`
  enforce enum/margin shape, unknown-`gate_on`, equivalence-margin-resolvability, and
  gate cycle detection. `StudyConfig` gained a `hypothesis_testing` field, parsed in both
  unified/legacy loaders and propagated through `for_paradigm`/`for_paradigm_analysis`.
- R-facing serialization needed **no change**: analysis writers dump `dict(config.raw)`, so
  the new YAML keys reach R untouched (verified by round-trip test).
- FORGE `study_treatment.yaml` rewritten: `disease_effect` (phenotype) → `*_rescue`
  (rescue, gate_on disease_effect) → `*_normalization` (equivalence, gate_on [disease,
  matching rescue]) → route/dose (exploratory). Added `hypothesis_testing` default block.
  **Note:** old `hd_icv_vs_wt`/`hd_iv_vs_wt`/`ld_vs_wt` contrasts renamed to
  `*_normalization` — output table contrast-name keys change; MS2 citations of these need
  updating when the pipeline is re-run.
- source-lightbox: `contrast_meta` ({role,test,gate_on}) threaded cli → `BuildConfig` →
  `build_manifest` → `manifest["contrast_meta"]`; consumed by Phase 2 presentation.
- Tests: `tests/test_hypothesis_contrasts.py` (12 cases) green; source-lightbox suite green.

**Phase 1 (next)** — engine in `R/stats_utils.R`: topo order, per-cell masks, gate ∩, TOST
(both margin modes), wire into the LMM analysis callers.

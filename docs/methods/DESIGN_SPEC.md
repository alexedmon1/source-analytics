# Design & Hypothesis Spec — declarative hypotheses, manual testing, pluggable engines

**Status:** design proposal (supersedes the gating half of `HYPOTHESIS_CONTRASTS_PLAN.md`).
**Repos:** source-analytics (spec schema + loaders + per-family adapters + module wiring).
**Author/date:** drafted 2026-06-22.
**Usage / reference:** see `HYPOTHESIS.md` (this doc is the *why*; that one is the *how*).

---

## 1. Motivation

The current contrast system is *already* declarative — `study_treatment.yaml`'s `contrasts:`
block names each comparison and the R callers compile it to `emmeans`. The problem is the
layer bolted on top: `gate_on`, `role`-driven chaining, and `test: equivalence` auto-fire into
**verdict columns** (`gated_in`, `equivalent`, rescue verdicts). That moves the scientific
judgment into code — a TOST margin or a gate threshold becomes a silent yes/no, and the real
interpretation (is this band meaningful? is the region plausible? is the sign sensible?) never
gets made by a human.

This spec keeps the declarative *statement* of hypotheses and throws away the automated
*adjudication*. You declare what each hypothesis is; you run them **one at a time, by name**;
you read the result and interpret. Equivalence (TOST) is a kind **you invoke deliberately**,
never a gate.

Inspired by `neuroaider`'s declarative ergonomics, but emitting **analysis-native specs**, not
FSL vest files — and not tied to any one stats engine (§4).

## 2. Core architecture: kind × adapter × payload

Three orthogonal axes:

1. **Kind** — what *kind* of question (omnibus / contrast / regression / equivalence). §5.
2. **Adapter** — what *engine* runs it, selected by the module you run in (emmeans for R LMM
   modules; permutation for network/vertex modules). §6.
3. **Payload** — the portable description a kind carries (a `weights` map, a `groups` set, or a
   `predictor` name). Backend-agnostic.

The same hypothesis declaration runs under any adapter. `disease_effect = {KO_VEH:1, WT_VEH:-1}`
is tested by emmeans in `roi_psd` and by permutation in `vertex_network` — same declaration,
native engine each time. **No single stats vocabulary is privileged; the weights-dict /
group-set / predictor is the shared language, not `emmeans`.**

### 2.1 Module identity & integration — `hypothesis`

The module is named **`hypothesis`** (matching the `--hypothesis` flag, the `run_hypothesis()`
runner, and the `hypotheses:` config block). It is a **shared inference layer**, the third peer
to the two that already exist — `R/stats_utils.R` (R) and `src/source_analytics/stats/` (Python).

**It is NOT a registry analysis module.** It never appears in `ANALYSIS_METADATA` (core.py),
processes no subjects, and produces no figures. Analysis modules *call into* it from their
`statistics()` lifecycle step — exactly as they already source `stats_utils.R` or import
`stats/`.

```
study.yaml: design: + hypotheses:
   │  parsed by config.py → StudyConfig (new DesignSpec accessor; the existing
   │  Contrast dataclass is generalized to carry kind + weights/groups/predictor)
   ▼
hypothesis layer ─────────────────────────────────────────────────────────────
   R/hypothesis.R               load_design_spec() + run_hypothesis() + EMMEANS adapter
                                (sourced by roi_psd_analysis.R, electrode_analysis.R, …)
   src/source_analytics/        load_design_spec() + run_hypothesis() + PERMUTATION adapter
     hypothesis/                (built on stats/cluster_permutation.py, stats/graph_metrics.py;
       __init__.py              imported by vertex_network_analysis.py, …)
       spec.py    ← loader/dataclasses
       emmeans.py ← thin shim invoking R when a Py module wants the LMM adapter (rare)
       permutation.py ← perm adapter (omnibus-F, weighted contrast, Freedman–Lane)
─────────────────────────────────────────────────────────────────────────────
   ▲                                   ▲
   roi_psd_analysis.R                  vertex_network_analysis.py
   (statistics step → emmeans adapter) (statistics step → permutation adapter)
```

**Invocation path.** `--hypothesis <name>` rides the existing selection plumbing
(`BaseAnalysis.run(select=…)`, `_select`) the same way `--metric`/`--band` do, scoping a run to
one named hypothesis. Inside `statistics()`, the module calls `run_hypothesis(hyp, ctx)`; the
runner picks the adapter from the module's level/engine. This **replaces** the inline
`emmeans(fit, pairwise ~ group)` in `stats_utils.R` (R modules) and **gives the Python perm
modules a matching entry point they lack today**.

**Ownership split.** The *spec* (loader, dataclasses, kind/payload validation) is shared and
language-mirrored (`spec.py` ⇄ the parsing half of `hypothesis.R`). The *adapters* are
engine-specific: the emmeans adapter is R-only; the permutation adapter is Python-only. A
module uses whichever adapter matches where its statistic is computed — it does not need both.

## 3. Design principles

1. **One language-neutral spec, pluggable adapters.** A single YAML block is the source of
   truth; each analysis family supplies an adapter that knows how to run each kind.
2. **A contrast is a named map `group level → weight`.** Pairwise (`{KO_VEH:1, WT_VEH:-1}`) is
   the common case; the same form expresses averages and any linear combination — for free.
3. **Nothing auto-fires.** No `gate_on`, no verdict columns, no contrast DAG. You choose which
   hypothesis to run.
4. **`role` survives only as a label.** It groups output for display; the engine treats it as
   opaque, never control flow.
5. **Per-family escape hatches.** A kind's payload covers the common case; each adapter exposes
   the full power of its engine for the rest (raw `emmeans:` string for R; perm knobs for
   Python). The spec never *limits* the test you can run.

## 4. The design block

```yaml
design:
  factor: group                  # the categorical factor most kinds are taken over
  reference: WT_VEH              # reference level (orientation; relevel)
  levels: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # optional explicit order; else sorted
  covariates:                    # NUISANCE adjustment — adjusted for, not tested (§7)
    - n_epochs                   #   continuous → mean-centered by the loader
    - sex                        #   categorical → entered as a factor
```

**Model fit is shared.** The model is fit once over all `design` groups (e.g.
`dv ~ group * roi + n_epochs + sex + (1|subject)` for an R module); omnibus, contrast,
regression, and equivalence kinds all read off that single fit. This is required because a
≥3-group omnibus must see every group in one model, and post-hoc contrasts should share its
error term (standard ANOVA→emmeans workflow). The model *formula* stays module-owned; the spec
supplies only the factor, the covariates, and (for regression) the predictor.

> **Legacy note.** The retired system fit a *separate 2-group model per pairwise contrast*
> (`run_posthoc_global`, `stats_utils.R:582`), so its SEs were pooled over only 2 groups.
> Shared-fit SEs differ slightly. Already-published FORGE manuscript numbers used the per-pair
> fit; a `fit_scope: per_contrast` opt-in preserves exact reproduction where needed.

## 5. Hypothesis kinds

```yaml
hypotheses:
  # ── OMNIBUS — "do these groups differ at all?" (ANOVA / permutation-F) ──
  - name: group_omnibus
    kind: omnibus
    groups: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # default: all design levels
    role: phenotype

  # ── CONTRAST — a specific linear comparison (post-hoc); weights need not be pairwise ──
  - name: ko_vs_wt
    kind: contrast
    weights: { KO_VEH: 1, WT_VEH: -1 }
    role: phenotype
  - name: treated_vs_ko_veh
    kind: contrast
    weights: { KO_HD_ICV: 0.5, KO_HD_IV: 0.5, KO_VEH: -1 }   # avg of treated vs KO veh
    role: rescue

  # ── REGRESSION — effect (slope) of a continuous predictor ──
  - name: dose_response
    kind: regression
    predictor: dose            # continuous column; slope tested ≠ 0
    by: group                  # optional: per-group slopes (emtrends ~ group) + their contrast
    role: exploratory

  # ── EQUIVALENCE — TOST that a contrast lies within a margin ──
  - name: hd_icv_normalization
    kind: equivalence
    weights: { KO_HD_ICV: 1, WT_VEH: -1 }
    margin: { mode: sd, value: 0.25 }              # §8
    role: normalization
```

### 5.1 Payload by kind

| kind | required payload | optional | tests |
|---|---|---|---|
| `omnibus` | — (defaults to all `design.levels`) | `groups: [...]` subset | any difference among the groups |
| `contrast` | `weights: {level: w}` | — | the linear combination = 0 |
| `regression` | `predictor: <col>` | `by: <factor>`, `groups: [...]` | slope = 0 (or slopes equal, if `by`) |
| `equivalence` | `weights: {level: w}` + `margin` | — | \|contrast\| < margin (TOST) |

Common to all: `name` (unique, the manual handle), `label`, `role` (display tag only).
Legacy `group_a`/`group_b` is accepted as sugar for `kind: contrast, weights:{group_a:1,
group_b:-1}`.

## 6. Adapters

An adapter implements every kind in its paradigm. Selected by the module the hypothesis runs in.

**The divide is the inference machinery, not the spatial level.** The two adapters split modules
by *how* they test, which does NOT coincide with roi/vertex/electrode: `electrode_connectivity`
runs on the **permutation** adapter (cluster maps), alongside `vertex_connectivity`, while
`electrode_psd` runs on the **emmeans** adapter, alongside `roi_psd`. A hypothesis declaration is
the *same* across all of them (that uniformity is what makes the MS2 source-vs-sensor head-to-head
apples-to-apples); only the result contract and method knobs differ, and they follow the adapter.

### 6.1 emmeans adapter (R LMM modules: roi_psd, roi_aperiodic, electrode_psd, …)

| kind | implementation |
|---|---|
| omnibus | `emmeans::joint_tests(fit)` on the `group` term (or `anova` on the reduced model) → F, df1, df2, p |
| contrast | `emmeans(fit, ~ group)` → `contrast(emm, method = list(name = weight_vector))` → estimate, SE, df, CI, t, p, Hedges g |
| regression | `emmeans::emtrends(fit, ~1, var = predictor)` (slope) or `emtrends(fit, ~ group, var = predictor)` + `contrast` when `by:` set |
| equivalence | `tost_equivalent(estimate, SE, df, margin, alpha)` on the contrast (`stats_utils.R:742`, kept) |

Escape hatch: `emmeans: "<raw emmeans/contrast expression>"` on a hypothesis runs verbatim —
full access to interaction contrasts, `poly` trends, custom `emtrends`, etc. R-only, power user.

### 6.2 permutation adapter (Python modules: vertex/electrode connectivity maps, network graph/NBS)

Group labels are permuted; the module's per-unit statistic (per-vertex value, graph metric,
edge weight) is recomputed each permutation to build the null.

| kind | implementation |
|---|---|
| omnibus | permutation **F** across the listed groups: observed between-group F of the statistic vs the null from relabeling all involved groups |
| contrast | permuted **weighted group-contrast** of the statistic: `Σ wᵢ · stat(groupᵢ)`, null from relabeling |
| regression | permutation test of the **slope** of the statistic on `predictor` (permute the predictor / residuals) |
| equivalence | TOST from the permutation CI of the contrast vs the margin |

Escape hatch: the module's existing perm knobs (`n_permutations`, cluster-forming threshold,
which metric) stay in the module config; the hypothesis only supplies the contrast/groups.

### 6.3 Result contracts — tabular vs map (adapter-keyed)

The runner returns an **adapter-appropriate** result; the hypothesis is the same noun, the result
shape is the adapter's. Forcing a permutation map into the per-cell table shape is the mismatch we
avoid.

- **emmeans adapter → tabular.** One row per band × spatial cell: `estimate, SE, df, CI, stat,
  p_value, q_value` (within-run BH-FDR), `effect_size` (+type), `significant`. (Implemented — the
  `roi_psd_hypotheses.csv` schema.)
- **permutation adapter → map + clusters.** A per-vertex statistic map plus a surviving-cluster
  table: `cluster_id, extent (n vertices), mass, peak_stat, cluster_p` (max-statistic / TFCE
  corrected — NOT per-cell FDR, which is the wrong family at vertex density), and the thresholded
  map itself. Equivalence/regression maps follow the same map+cluster shape.

Both carry the shared hypothesis metadata (`hypothesis, kind, role, label, test, band`) so results
align across adapters for the head-to-head, even though one is a table and the other a map.

Method knobs (cluster-forming threshold, TFCE, spatial adjacency, `n_permutations`) are **module/
adapter config**, never hypothesis fields — they describe the test, not the question.

### 6.4 Covariates under permutation — Freedman–Lane

Adjusting for a `design.covariate` in a permutation test is **not** "add a column." The adapter
uses the **Freedman–Lane** procedure (Freedman & Lane 1983; Winkler et al. 2014, *NeuroImage*
92:381–397 — the scheme FSL `randomise` uses): regress the nuisance covariate(s) out of the
data, permute the *residuals*, then re-add the nuisance fit before recomputing the statistic.
The emmeans adapter handles covariates the ordinary way (extra terms in the LMM formula,
marginalized/held-at-mean by emmeans).

## 7. The runner & manual control

```
run_hypothesis(hyp, module_ctx) ->
    adapter  = module_ctx.adapter            # emmeans | permutation, from the module
    model    = adapter.fit(data, design)     # shared fit over all design groups
    dispatch on hyp.kind -> standard result row
```

- **CLI:** `--hypothesis <name>` runs exactly one, by name; composes with the existing
  `--metric` / `--band` / `--select DIM=val` axes.
- **No `--hypothesis`:** the module *lists* available hypotheses and runs none. There is no
  "run all + adjudicate" path — deleted on purpose.
- **Analysis loop (external):** a research loop drives `run_hypothesis` across modules — run the
  omnibus, read it, decide whether the post-hoc contrasts follow, run the ones you choose, log the
  reasoning. The runner is the action layer; the loop is the process; the registry is the record (a
  mini pre-registration). The judgment stays with the human/loop, not in a gate. This loop tooling
  lives in a separate, local workflow and is intentionally not part of this repo.

## 8. Equivalence margins (manual primitive)

`tost_equivalent()` / `.equivalence_margin()` (`stats_utils.R:742`, `:754`) are kept. Modes:

- `sd`: `value × residual SD at cell`. Self-contained; recommended default.
- `gap_fraction`: `value × |reference estimate|` — needs a reference effect. Without auto-gating
  the reference is **declared explicitly**: `margin: {mode: gap_fraction, value: 0.25,
  ref: ko_vs_wt}` — one declared cross-hypothesis read, not a hidden gate. The runner fetches
  the `ref` hypothesis's per-cell estimate.

TOST verdict only describes the chosen contrast; it adjudicates nothing else.

## 9. Migration & what is retired

**Migrated:** `study_treatment.yaml` `contrasts:` + `hypothesis_testing:` → `design:` +
`hypotheses:` (each pairwise `group_a`/`group_b` → a `kind: contrast` weights-dict; `gate_on`
dropped; `role`/`label` kept; `equivalence_margin` → `margin`). A new `group_omnibus` is added.

**Retired (deleted):**
- `apply_hypothesis_gating()` (`stats_utils.R:779`) — gating orchestration + verdict columns.
- `build_rescue_verdicts()` (`stats_utils.R:838`) — rescue/normalization adjudication.
- `_validate_contrast_graph()` gating-DAG/cycle logic (`config.py:119`); `gate_on`,
  `gate_alpha`, `default_equivalence_margin`.

**Kept as primitives:** `tost_equivalent()`, `.equivalence_margin()`,
`run_omnibus_lmm()`/`run_posthoc_*()` (the LMM/emmeans machinery the emmeans adapter builds on).

**Tests:** `tests/test_gating.R` → `tests/test_hypothesis.R` (weight-vector alignment;
shared-fit contrast estimate vs a fixture; omnibus F vs `anova`; TOST verdict; Freedman–Lane
exchangeability on a synthetic covariate).

## 10. Resolved decisions (2026-06-22)

1. **§10.1 Effect sizes per kind.** Hedges g for contrasts (existing). **Omnibus → partial ω²**;
   **regression → standardized β**. Each adapter computes its kind's effect size in-paradigm
   (emmeans: ω² from the F/df; permutation: ω² from the null F-distribution).
2. **§10.2 Multiple-comparison scope.** **FDR within a single `run_hypothesis` call's cells**
   (e.g. bands × ROIs), with the family reported in the output. Cross-hypothesis correction is
   the analyst's call, logged by the external analysis loop — never silent, never automatic.
   **Family scope is declarative** (the emmeans adapter): a `fdr:` block — study-level under
   `design:` and/or per-hypothesis (the per-hypothesis one overrides field-by-field) — sets
   `scope` (`hypothesis` = the whole band×spatial grid, the default and most conservative;
   `band` = a family per band/freq_pair; `spatial`; `none` = no correction) and `method`
   (`BH` default | `BY` | `holm` | `bonferroni` | `none`). Aggressiveness is driven by family
   SIZE, not just method — `scope: band` is the principled lever when bands/freq-pairs are
   pre-specified independent hypotheses (e.g. PAC). Declaring it in the spec keeps the family
   definition pre-registered. The permutation/map adapter uses cluster-extent correction, so
   `fdr:` is a no-op there. Example:
   ```yaml
   design:
     fdr: { scope: hypothesis, method: BH }   # study default (= pre-toggle behaviour)
   hypotheses:
     - name: disease_effect
       fdr: { scope: band }                   # per-freq_pair override
   ```
3. **§10.3 Adapter/module order.** emmeans adapter + `roi_psd` first; **`vertex_network` (graph
   metrics) second** for the permutation adapter.

## 11. Build order (after approval)

1. Spec schema + loader (`R/hypothesis.R` + `hypothesis/spec.py`) + the **emmeans adapter** +
   runner; wire into `roi_psd`; verify a FORGE rerun reproduces the legacy `ko_vs_wt` estimate
   (per-pair mode) and that `group_omnibus` matches `anova`/`joint_tests`.
2. **Permutation adapter** (omnibus-F + weighted contrast + Freedman–Lane covariates); wire into
   **`vertex_network`** (graph metrics) first.
3. Migrate `study_treatment.yaml`; delete the retired gating code + `test_gating.R`.
4. Roll `--hypothesis` across the remaining ROI/electrode/vertex callers. (The analysis loop that
   drives it is an external, local workflow — out of scope for this repo.)

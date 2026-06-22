# The `hypothesis` layer — usage & reference

Declarative hypotheses, tested one at a time, by hand. This is the practical guide;
for the *why* and the design decisions see **`DESIGN_SPEC.md`**.

---

## 1. What it is

`hypothesis` is a **shared inference layer** — a third peer to `R/stats_utils.R` and
`src/source_analytics/stats/`. It is **not** a registry analysis module: it processes no
subjects, makes no figures, and never appears in `core.py`'s `ANALYSIS_METADATA`. Analysis
modules *call into* it from their `statistics()` step.

You **declare** your hypotheses once (in the study YAML), then **run them one at a time, by
name**, and read the result. Nothing auto-fires — there is no gating, no chained verdicts, no
"run everything and adjudicate." The scientific judgment stays with you.

| piece | where | status |
|---|---|---|
| spec loader + emmeans adapter + runner + module helper | `R/hypothesis.R` | ✅ built |
| `Hypothesis` / `DesignSpec` dataclasses, `StudyConfig.design_spec` | `src/source_analytics/config.py` | ✅ built |
| permutation adapter (vertex / connectivity maps) | `src/source_analytics/hypothesis/` | ⏳ planned |
| kept primitives `tost_equivalent()` / `.equivalence_margin()` | `R/stats_utils.R` | ✅ reused |

## 2. Declaring hypotheses (the spec)

Two YAML blocks in the study config:

```yaml
design:
  factor: group              # the categorical factor hypotheses are taken over
  reference: WT_VEH          # reference level (effect orientation)
  levels: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # optional explicit order
  covariates: [n_epochs]     # optional nuisance covariates (mean-centered)

hypotheses:
  - name: group_omnibus
    kind: omnibus
    groups: [WT_VEH, KO_VEH, KO_HD_ICV, KO_HD_IV]   # default: all design levels
    role: phenotype

  - name: disease_effect
    kind: contrast
    weights: { KO_VEH: 1, WT_VEH: -1 }              # the portable contrast payload
    role: phenotype

  - name: dose_response
    kind: regression
    predictor: dose          # continuous column; slope tested
    role: exploratory

  - name: hd_icv_normalization
    kind: equivalence
    weights: { KO_HD_ICV: 1, WT_VEH: -1 }
    margin: { mode: sd, value: 0.25 }
    role: normalization
```

**Backward compatible.** A study still using the legacy `contrasts:` block (with
`group_a`/`group_b`) works unchanged — each contrast is lifted to a `kind: contrast` whose
weights are `{group_a: +1, group_b: -1}`. You do not have to migrate to start using the layer.

### Field reference

| field | applies to | meaning |
|---|---|---|
| `name` | all | unique id; the handle you pass to `--hypothesis` |
| `kind` | all | `omnibus` \| `contrast` \| `regression` \| `equivalence` (default `contrast`) |
| `weights` | contrast, equivalence | `{level: weight}`; a linear combination of group means |
| `groups` | omnibus | the levels the F-test spans (default: all `design.levels`) |
| `predictor` | regression | continuous column whose slope is tested |
| `by` | regression | optional factor for per-group slopes |
| `margin` | equivalence | `{mode: sd|gap_fraction, value: ...}` (see §6) |
| `label`, `role` | all | display only; `role` groups output, never controls flow |

## 3. The four kinds

| kind | question | payload | effect size |
|---|---|---|---|
| **omnibus** | "do these groups differ at all?" | `groups` | partial ω² |
| **contrast** | a specific linear comparison (post-hoc) | `weights` | Hedges g |
| **regression** | slope of a continuous predictor | `predictor` (+`by`) | standardized β |
| **equivalence** | is a contrast within a margin? (TOST) | `weights` + `margin` | — |

The omnibus → contrast flow is the standard ANOVA → post-hoc workflow, **driven by you**: run
the omnibus, read it, then run whichever post-hoc contrasts *you* judge to follow. No gate
decides that for you.

## 4. Running them

`--hypothesis NAME[,NAME]` runs one (or a few) by name; with no flag, the module runs **all**
declared hypotheses. It composes with the existing `--metric` / `--band` / `--select` axes.

```bash
# one hypothesis
uv run --no-sync source-analytics run --study study_treatment.yaml \
    --paradigm resting --analysis roi_psd --hypothesis disease_effect

# a band-scoped single cell
... --analysis roi_psd --hypothesis disease_effect --band "Low Gamma"
```

Output is **additive**: a new `<module>_hypotheses.csv` under the module's tables dir, written
*alongside* the legacy omnibus/posthoc tables (which are untouched). Schema:

```
hypothesis, kind, role, band, spatial, estimate, SE, df, df_num, stat, stat_type,
p_value, estimate_lcl, estimate_ucl, effect_size, effect_size_type, label, test,
q_value, significant, fdr_family, dv
```

One row per **band × spatial cell × DV**. `q_value`/`significant` are **within-run** BH-FDR,
corrected across that hypothesis's band × spatial family (named in `fdr_family`). Cross-
hypothesis correction is your call, made explicitly — never silent.

## 5. Adapters & result contracts

The same hypothesis declaration runs under whichever **adapter** matches where a module's
statistic is computed — the divide is the inference machinery, **not** the spatial level.

| adapter | modules | result contract |
|---|---|---|
| **emmeans** (R LMM) ✅ | roi_psd, roi_aperiodic, electrode_psd, electrode_aperiodic, … | **tabular** — per-cell estimate/CI/p/FDR/effect (the schema above) |
| **permutation** (Python) ⏳ | all vertex_*, electrode_connectivity, roi_connectivity/directed/cross_freq/graph | **map + clusters** — per-unit statistic map + cluster extent/mass/cluster-p (max-stat/TFCE corrected, *not* per-cell FDR) |

`electrode_connectivity` runs on the **permutation** adapter (it is a map), alongside
`vertex_connectivity` — not with `electrode_psd`. Method knobs (cluster threshold, TFCE,
adjacency, `n_permutations`) live in **module config**, never in the hypothesis.

## 6. Equivalence (TOST)

`kind: equivalence` runs a TOST verdict for the chosen contrast — it adjudicates nothing else.
Margin modes:

- `sd`: `value × residual SD` — self-contained; recommended.
- `gap_fraction`: `value × |reference estimate|` — needs a reference effect; declare it
  explicitly (`margin: { mode: gap_fraction, value: 0.25, ref: disease_effect }`), never a
  hidden gate. (The reference wiring is added when a study uses it.)

## 7. API reference

**R (`R/hypothesis.R`)**

```r
spec <- parse_design_spec(config)        # parse design:/hypotheses: (config = read_yaml'd)

run_hypothesis(data, hyp, spec,          # run ONE hypothesis -> tidy data.frame
               dv_col = "dv", spatial_col = "roi",
               band_col = "band", bands = NULL,
               fit_scope = "shared",     # or "per_contrast" (reproduces legacy per-pair SEs)
               fdr_method = "BH")        # hyp = a parsed def or a name in spec$hypotheses

write_module_hypotheses(df, config, tbl_dir, prefix, dv_cols,   # the module one-liner
                        spatial_col = "roi", band_col = "band",
                        hypothesis = NULL, fit_scope = "shared")
```

`fit_scope = "shared"` (default) fits all design groups in one model — required for a ≥3-group
omnibus, and gives post-hoc contrasts a common error term. `"per_contrast"` subsets to the
contrast's groups and reproduces the legacy per-pair estimates bit-exact.

**Python (`config.py`)**

```python
cfg = StudyConfig.from_yaml("study.yaml")
cfg.design_spec                 # DesignSpec | None  (parsed design:/hypotheses:, or lifted contrasts:)
cfg.design_spec.hypotheses      # list[Hypothesis]   (.name/.kind/.weights/.groups/.predictor/.margin)
cfg.referenced_groups()         # set[str] — every group named by a contrast/hypothesis (subject discovery)
```

## 8. Wiring a new (emmeans-tabular) module

Four steps — see `roi_psd_analysis.R` / `electrode_aperiodic_analysis.R` for live examples:

1. **R:** `source(file.path(script_dir, "hypothesis.R"))` near the other sources.
2. **R:** add `parser$add_argument("--hypothesis", default = NULL)`.
3. **R:** after the module's stat-table exports, call `write_module_hypotheses(df, config,
   tbl_dir, prefix = "<module>", dv_cols = c(...), spatial_col = "...", band_col = "band"|NULL,
   hypothesis = args$hypothesis)`.
4. **Python:** add `"hypothesis": "declared hypothesis"` to the module's `SELECTABLE`, and pass
   `--hypothesis` through to the Rscript command from `self._selection.get("hypothesis")`.

This fits modules with a modest spatial cardinality (≤ ~32 ROIs / ~30 channels). Edge/node
**maps** (connectivity, directed, graph) do **not** use this path — they belong to the
permutation adapter (a `group × edge` LMM is infeasible and wrong for them).

## 9. Status

- ✅ **emmeans adapter** built + verified (bit-exact vs legacy on real data); wired into
  `roi_psd`, `roi_aperiodic`, `electrode_psd`, `electrode_aperiodic` — every applicable
  emmeans-tabular module.
- ⏳ **permutation adapter** (`hypothesis/` Python): omnibus-F + weighted contrast + Freedman–
  Lane covariates + map/cluster contract; wire `vertex_network` first, then the rest of the
  map family (vertex_*, connectivity, directed, graph).
- ⏳ **deferred:** `roi_evoked` / `electrode_evoked` (long-format DV; no data in the resting
  study). **specials:** `vertex_mvpa` (decoding), `vertex_spatial` (GLS), `electrode_comparison`
  (agreement — may not take hypotheses).
- ⏳ **migration / retirement:** move `study_treatment.yaml` to `design:`/`hypotheses:` and
  delete the retired gating code (`apply_hypothesis_gating`, `build_rescue_verdicts`,
  `gate_on`).

See `DESIGN_SPEC.md` for the full architecture and the retired auto-gating rationale.

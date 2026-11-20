# Coverage table: every outcome, frequentist and Bayesian (19 Nov 2026)

What source-analytics can test today, outcome by outcome, with its frequentist path (linear mixed models, permutation
tests and the hypothesis layer) and with the Bayesian engine. It is the first step of bayes-sl's close-out plan
section 9 (decision D10: the Bayesian analyses must be available for every outcome source-analytics handles). It
describes `main` at 81339ff; nothing was run to make it, so every mark rests on the code and the tests named.
Checked against the code on 20 Nov: `electrode_signature`'s statistics are tested (corrected below), and the
Bayesian evidence for `roi_psd` and `roi_aperiodic` is the E0 replay, not a recorded G1 result.

**Marks.**
- **works**: an automated test runs this path, or it has been checked against a reference on real data (named).
- **runs, untested**: the code path exists and has been used, but no test runs it and no reference check is recorded.
- **possible, untested**: nothing stops it (for the engine: a declared `csv:` table can carry it), but nobody has
  run it.
- **missing**: no path.
- **n/a**: no inferential test by design.

**How the engine reaches an outcome.** Two ways. (1) Built in (`bayes/tables.py`): `roi_psd` (absolute, relative,
delta_ref) and `roi_aperiodic` (exponent, offset). (2) Declared tables (`bayes.tables`, `bayes/declared.py`): a
`source:` table reads a module's `*measures*.csv` (today only `roi_evoked` and `electrode_evoked` write one) or a
built-in table; a `csv:` table reads any per-subject long file. The engine's table has one row per subject x outcome
x (optional) one spatial column, so per-edge outcomes (two spatial columns, `roi1` and `roi2`) do not fit: an edge can
only enter as its own outcome label, or edges averaged together (`combine: mean`).

**Engine scope, all outcomes.** Contrasts, equivalence (with a margin) and regression on covariates; no omnibus.
Gaussian outcome per band; one between-subject factor; no sessions or within-subject factors (G2).

## 1. Modules

There are 15 built-in analyses (16 registry names; `roi_network` runs `roi_graph` + `roi_nbs`). The close-out plan
said 18; the difference is not resolved here (the retired vertex analyses, now in the unmaintained plugin, may have
been counted).

| Module | Per-subject table: outcome columns | Frequentist path | Frequentist | Bayesian | Evidence / note |
|---|---|---|---|---|---|
| `roi_psd` | `band_power.csv`: absolute (dB/Hz), relative (fraction), delta_ref; per ROI x band | LMM + emmeans; hypothesis layer (R) | **works** | absolute: **works**; relative, delta_ref: **possible, untested** | freq: `test_r_scripts_smoke` (3 tests), `test_psd_analysis`, lifecycle run. Bayes: E0 replay of the frozen FORGE G1 fits and gate verdicts (`tests/bayes/test_e0_regression.py`; opt-in, needs `SA_BAYES_E0`; passed on engine 2026.11.09.1, 9-10 Nov), `tests/bayes/test_e5_slow.py` tiny fits (slow). G1's comparison against the emmeans tables (`BAYES_ENGINE.md` section 7) has no recorded result. Relative power is a fraction in 0-1: the Gaussian model is an approximation there |
| `roi_aperiodic` | `aperiodic_params.csv`: exponent, offset; per ROI | LMM; hypothesis layer (R) | **works** | exponent: **works**; offset: **possible, untested** | freq: smoke test (model-warning test). Bayes: E0 replay of the FORGE G1 fits (exponent), as `roi_psd` |
| `electrode_psd` | `electrode_band_power.csv`: absolute, relative, delta_ref; per channel x band | LMM per channel and region model; hypothesis layer (R) | **runs, untested** | **possible, untested** (`csv:` table, `spatial: channel`) | no test runs `electrode_analysis.R`; `test_results_spec` checks only its table specification; used on FORGE |
| `electrode_aperiodic` | `electrode_aperiodic_params.csv`: exponent, offset, offset_centered; per channel | LMM per channel + region model (R) | **works** | **possible, untested** (`csv:`) | freq: `test_electrode_aperiodic_region_posthoc_carries_mde_of_the_reference` |
| `electrode_comparison` | `comparison_data.csv` (electrode vs source power per band), `aperiodic_comparison_data.csv`, `regional_source_power.csv` | Pearson r per band; Hedges g with 95% CI per level; no correction | **runs, untested** | **missing** (a concordance and a two-level comparison, not a group test; would need its own model) | only metadata tested (`test_audit_fixes`) |
| `roi_connectivity` | `roi_connectivity_edges.csv`: coherence, imag_coherence, pli, wpli, dwpli, dpli, aec, partial_corr; per ROI pair x band | descriptive; inference is in `roi_graph` / `roi_nbs` | **n/a** (script smoke-tested) | per edge: **missing**; mean over edges or a chosen edge as an outcome: **possible, untested** | PLI-family values are bounded 0-1 (Gaussian approximation) |
| `roi_graph` | `roi_graph_nodal_metrics.csv`: degree, clustering, betweenness per ROI x band x connectivity metric; `roi_graph_global_metrics.csv`: global_efficiency, modularity, small_worldness | one-way model per ROI; t / F / two one-sided tests; BH (Python `hypothesis/tabular.py`); no regression | **works** (global metrics drawn, not tested: **n/a**) | nodal and global: **possible, untested** (`csv:`; outcome = band x metric) | freq: `test_tabular`, `test_network_split`. Degree and betweenness are counts and skewed |
| `roi_nbs` | edge statistics -> connected components (`roi_nbs_results.csv`) | Network-Based Statistic, permutation FWE; equivalence per edge (TOST) | **works** | **missing** (a component-extent test; a Bayesian counterpart would need a new model of the edge set) | `test_edge`, `test_network_split` |
| `roi_cross_freq` | `pac_values.csv`: mi, z_score per ROI x phase band x amplitude band; `aac_edges.csv`: aac; `ppc_edges.csv`: ppc, ppc_z per ROI pair | PAC: LMM + hypothesis layer; AAC/PPC: hypothesis layer on edges, global and region (R) | **works** | PAC: **possible, untested** (`csv:`, outcome = band pair); AAC/PPC per edge: **missing** | freq: `test_cross_freq_edges_script_writes_hypothesis_tables`, `test_cross_freq` |
| `roi_directed` | `roi_transfer_entropy_edges.csv`: te, net_te, dtf per directed ROI pair x band | TE: Welch per pair, one-sample net TE, region-pair LMM, hypothesis layer (R); DTF: none | TE: **works**; DTF: **missing** (documented) | per pair: **missing**; mean over pairs: **possible, untested** | freq: `test_directed_script_uses_canonical_prefix_and_tests_every_dv` |
| `electrode_connectivity` | `electrode_fcd.csv`: FCD per channel x band; channel-pair matrices | Welch per channel + BH; cluster permutation over the montage (Python `hypothesis/permutation.py`) | **runs, untested** (the permutation engine itself is tested) | FCD per channel: **possible, untested** (`csv:`); pairs: **missing** | `test_permutation` covers the engine; `test_electrode_connectivity_conforms_from_the_cluster_layer` runs the module's permutation hypothesis layer on synthetic FCD values, not the module's own steps |
| `roi_signature` | classification accuracy, feature importance (`roi_signature_results.csv`) | permutation test of the classifier | **works** | **missing** (no group-level outcome per subject to model) | `test_roi_signature_full_lifecycle` |
| `electrode_signature` | same, on electrode band power | permutation test of the classifier | **works** | **missing** (as above) | `test_signature_modules_conform[electrode_signature]` (`test_results_spec.py`) runs its `statistics()` on synthetic band power; `test_signature_lookup` |
| `roi_evoked` | `roi_evoked_measures.csv`: value per ROI x measure (ITC, ERSP, STP windows) | one-way model per ROI x measure; t / F / TOST; BH (Python, tabular); no regression | **works** | **works** (declared `source:` table) | freq: `test_evoked_hypotheses`, smoke test of the R script. Bayes: a97d field job reproduced exactly; deep-midline parcel tables within 0.02 SD (`BAYES_ENGINE.md` 2.1 checks) |
| `electrode_evoked` | `electrode_evoked_measures.csv`: as above per channel | as `roi_evoked` | **works** | **possible, untested** (declared `source:` table, `spatial: channel`) | freq: `test_electrode_arm_behaves_identically_on_channel` |

## 2. Inputs that are not a module

| Input | Frequentist | Bayesian | Evidence / note |
|---|---|---|---|
| Field runs (source-localization field mode, `field_run.json`): any module can read their region time series | **runs, untested** as an analysis; the loader recognises field runs and refuses to pool them with source runs (**works**, `test_run_manifest` field tests) | same as the module run on them (above) | surface-evoked ran `roi_evoked` on field runs; no test runs a module on a field run. Plan section 9 said "a tested frequentist path through `roi_evoked` only"; I found no such test |
| Located-field region values (source-localization `field region-values`: region, p_source, subject, y_db) | **missing** (a97d's frequentist numbers came from a study script) | **works** (`csv:` table, one outcome per region, no spatial) | FORGE's L3 region values: 20 of 20 decisions the same as bayes-sl, estimates within 0.022 SD (`BAYES_ENGINE.md` 2.1); those fits predate engine 2026.11.09.1's no-ROI residual-SD prior |

## 3. Across all outcomes

| Feature | Frequentist | Bayesian |
|---|---|---|
| Contrast | works (every inferential module) | works (engine) |
| Equivalence | works (TOST; margin x residual SD) | works (HDI vs margin x residual SD); the margin's unit is open (D10) |
| Regression on a covariate | works in the R hypothesis layer and permutation engines; **missing** in the tabular path (`roi_graph`, evoked) | works |
| Omnibus | works (F; NBS F) | **missing** (deferred since G1) |
| Sessions / within-subject factors | no explicit model (sessions are coded as groups, e.g. FORGE `G2_D1pre`; a shared subject id would get one random intercept in the LMMs, not checked) | **missing** (G2) |
| Per-ROI single-group claims (a group's level vs 0) | via the hypothesis layer's weights | works with `center: false` (engine 2026.11.10.1); gate rule for these claims open (D10) |
| Convergence rule | lmer warnings kept in `fit_warning` (1f0d) | R-hat < 1.05 in the gate; bulk-ESS rule and model rewrite proposed (bayes-sl convergence report section 6) |

## 4. What this says for the D10 plan

Grouped by what a Bayesian path needs, cheapest first:

1. **Already reachable by a `csv:` table, needs only a test and a reference check** (no engine change): `roi_psd`
   relative and delta_ref, `roi_aperiodic` offset, `electrode_psd`, `electrode_aperiodic`, `electrode_evoked`,
   PAC, `roi_graph` nodal and global metrics, `electrode_connectivity` FCD. Making them built-in tables
   (`BAYES_TABLES`) would let them run without a declaration.
2. **Needs an engine feature:** omnibus; sessions and within-subject factors (G2; the model rewrite's "own mice"
   mean has to be defined per cell first); non-Gaussian outcomes (fractions, bounded coupling values, counts) if
   the Gaussian approximation fails a check.
3. **Needs a new model:** per-edge outcomes (connectivity, AAC/PPC, TE, electrode pairs: two spatial columns, and
   what pools across edges); the NBS counterpart (a sub-network claim).
4. **Probably out of scope for a group model:** the signature modules (classification accuracy) and
   `electrode_comparison` (source-sensor concordance). Proposed: record them as frequentist-only, with the reason,
   unless the user wants otherwise.

Frequentist gaps found on the way (D12: main stays working): no test runs `electrode_psd`'s R script,
`electrode_comparison`'s statistics or `electrode_connectivity` as a module;
no frequentist path for located-field region values; no regression in the tabular path.

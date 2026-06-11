# MS2 Connectivity Build-Out — Implementation Plan

**Status:** planning, pre-implementation (2026-06-11). For execution next session.
**Repos:** source-analytics (kernels + modules), source-lightbox (presentation),
FORGE (`study_treatment.yaml` wiring).
**Rides on:** the hypothesis-testing contrast engine — see `HYPOTHESIS_CONTRASTS_PLAN.md`
(Phase 0 done; Phase 1 engine pending). Connectivity cells = band × edge.

---

## 1. Purpose & thesis

Manuscript 2 is reframed (2026-06-11) as a **connectivity methods paper**. Thesis:

> **Source-localized (vertex) connectivity recovers spatial structure that sensor-level
> connectivity blurs.**

Consequences:
- **ROI level is OUT** of MS2 (MS1 + a separate treatment manuscript already cover ROI
  spectral). MS2 does **not** re-run the ROI paradigm.
- **Vertex / whole-brain is the anchor** for all headline analyses.
- A **new sensor (electrode) level connectivity analysis** is added as the comparator that
  source-localization is shown to improve on. None exists today (electrode modules are
  psd/aperiodic/comparison only).

---

## 2. Scope decisions — recommended defaults

Three decisions were open at the end of 2026-06-11. Recommended defaults below; revisit at
the top of next session and confirm or redline before coding.

### D1 — Node-reduction for the expensive measures

The plain phase/amplitude FC metrics are all-pairs-feasible at vertex scale. The
**directed + cross-frequency** family is not (DTF needs a joint MVAR over all nodes;
pairwise PAC/PPC/comodulograms are N² × frequency-grid). Splitting by feasibility:

**▶ DEFAULT (recommended): two resolution tiers, node-reduction only where forced.**

- **Tier A — whole-brain vertex, all-pairs (carries the spatial thesis):**
  AEC, imaginary coherence, PLI, wPLI, dwPLI, **dPLI**, and **local within-vertex PAC**
  (phase & amplitude from the same vertex → one MI per vertex per band-pair → whole-brain
  PAC map). These are where "vertex beats sensor" is demonstrated.
- **Tier B — reduced-node (parcellated), framed as model-based confirmation:**
  cross-region PAC, comodulograms, phase–phase (n:m) coupling, and DTF. Reduce the ~215
  vertices to the existing **Allen-32 parcellation of the vertex set** (reuse the ROI
  assignment machinery; no new atlas). DTF over ~32 nodes is numerically stable; a
  freq×freq comodulogram per node is tractable.
- Make the reduced-node count / parcellation a **config parameter** (`reduced_nodes:`),
  defaulting to Allen-32, so it's tunable without code changes.

Rationale: dPLI is the cheap directed measure that scales to vertex all-pairs, so the
*directed* whole-brain story is carried by dPLI; DTF is the coarser spectral-Granger
cross-check at parcel resolution. We never parcellate the measures that are supposed to
demonstrate the vertex advantage.

### D2 — Headline of the preliminary assertion

**▶ DEFAULT (recommended): the methods claim is the headline.**

"Vertex (source) connectivity localizes the FXS connectivity phenotype with spatial
specificity that sensor connectivity cannot resolve." The disease contrast
(KO_VEH vs WT_VEH) is the *vehicle/evidence*, run **head-to-head source-vs-sensor**, and the
sensor comparator is therefore the **central** result, not a supporting panel.

Treatment-rescue: retained as **secondary** evidence that the source measures are sensitive
enough to track modulation — NOT the headline. (Open sub-question flagged in §9: confirm
whether treatment rescue stays in MS2 at all, given the rescue/ROI-spectral story is going
to the other manuscript. The gating engine applies to the disease footprint regardless.)

### D3 — First-cut priority

**▶ DEFAULT (recommended): stage it — FC-six first, CFC/DTF second.**

- **Cut 1 (unblocks the preliminary assertion):** the six FC metrics
  (AEC · imcoh · PLI · wPLI · dwPLI · dPLI) at **vertex + sensor**. Kernels mostly exist —
  only wPLI + dPLI to add. All-pairs vertex-feasible. This alone supports the methods claim
  and the writeup.
- **Cut 2 (rounds out the "full connectivity portrait"):** local PAC maps, then the
  Tier-B node-reduced family (cross-region PAC, comodulograms, PPC, DTF).

De-risks the manuscript: the preliminary assertion is defensible from Cut 1 alone; the hard
kernels never block the draft.

---

## 3. Metric inventory (have vs need)

Existing kernels in `src/source_analytics/spectral/`:

| Kernel | file | ROI | Vertex | Sensor |
|---|---|:---:|:---:|:---:|
| coherence, imag_coherence, pli, dwpli, aec, partial_corr | `connectivity.py`, `vertex_connectivity.py` | ✅ | ✅ | ❌ |
| PAC (Tort-2010 MI, surrogate z, comodulogram) | `pac.py` | ✅ | ❌ | ❌ |
| transfer entropy (directed, info-theoretic) | `transfer_entropy.py` | ✅ | ❌ | ❌ |

To build:

| Need | Kernel status | Notes |
|---|---|---|
| **wPLI** | NEW (small) | Vinck 2011 non-debiased; sits beside `_compute_dwpli`. |
| **dPLI** | NEW (small) | Directed PLI, Stam & van Straaten 2012. `dPLI = ⟨H(Im(Sxy))⟩` ∈ [0,1]; **asymmetric**. Shares cross-spectrum with dwPLI. NOT magnitude-weighted (confirmed: plain dPLI, not a hybrid). |
| **cross-frequency AAC** | NEW | Envelope(band X) vs envelope(band Y) correlation; orthogonalized (reuse AEC machinery, different bands). Within-band AAC ≈ AEC. |
| **phase–phase coupling (n:m)** | NEW | `PPC = |⟨exp(i(n·φ1 − m·φ2))⟩|`; needs surrogate significance. |
| **DTF (MVAR)** | NEW (heavy) | Spectral directed influence from a joint MVAR fit. Reduced-node only (D1). Dependency decision — see §7. |
| vertex PAC module | NEW | Wraps `pac.py` at vertex (local maps Tier A; comodulograms Tier B). |
| sensor connectivity module(s) | NEW (mostly wiring) | Reuse ROI ts-based kernels on electrode time series. |

---

## 4. Architecture

### 4.1 Kernels (`spectral/`)
- Extend `connectivity.py` + `vertex_connectivity.py`: add **wPLI** and **dPLI** to the
  metric dispatch (`_SPECTRAL_METRICS` / `_ALL_METRICS` in `vertex_connectivity.py:26-27`;
  the ROI `compute_connectivity_matrix` map). dPLI returns an **asymmetric** matrix — audit
  every downstream consumer that assumes symmetry (network/NBS symmetrization, plotting).
- New `spectral/cfc.py` (or extend `pac.py`): cross-frequency AAC + n:m phase–phase, sharing
  the band-filter/Hilbert front-end with `pac.py`.
- New `spectral/directed.py`: DTF via MVAR (+ optionally a vertex-capable TE path).

### 4.2 Analysis modules (`analyses/`)
- `vertex_connectivity` already multi-metric — adding wPLI/dPLI to the kernel auto-flows
  through its `connectivity_metrics` list (verify the asymmetric dPLI matrix serializes and
  plots correctly).
- **NEW `vertex_pac`** — local within-vertex PAC maps (Tier A) + parcel comodulograms (Tier B).
- **NEW `vertex_cfc`** — cross-frequency AAC + phase–phase coupling (Tier B parcel-level;
  AAC can also run Tier A whole-brain like AEC).
- **NEW `vertex_directed`** — DTF (Tier B) ± vertex TE.
- **NEW `electrode_connectivity`** — the comparator. FC-six on electrode time series
  (reuse `connectivity.py`). Mirror PAC/directed at sensor in Cut 2.

### 4.3 Sensor comparator
Sensor FC is mostly plumbing: load electrode epochs (the electrode module already reads raw
`.set` via the roster — see the electrode roster path note in
`[[project_r460_abi_break_2026-06-08]]`), run the ROI ts-based kernels on the 30-channel
montage. The head-to-head figure (source vs sensor for the same contrast/band/metric) is the
manuscript's central panel — design it in source-lightbox as a paired view.

### 4.4 Config (`study_treatment.yaml`)
- The `vertex` paradigm's `vertex_connectivity` block gains wPLI/dPLI in
  `connectivity_metrics`.
- New analysis blocks: `vertex_pac`, `vertex_cfc`, `vertex_directed`, and a new
  electrode/sensor paradigm entry `electrode_connectivity`.
- New keys: `reduced_nodes:` (default Allen-32) for Tier-B measures; CFC band-pair lists.
- Remove/disable the ROI `resting` paradigm analyses from the MS2 run set (ROI is out) —
  but keep the YAML blocks (other manuscript / reproducibility); just don't run them for MS2.

---

## 5. Integration with the gating engine

Connectivity cell granularity = **band × edge** (per metric). Edge counts are large
(vertex all-pairs ⇒ ~23k edges; parcel ⇒ ~500), so honest multiplicity control matters:
- Phenotype (KO_VEH vs WT_VEH) defines the **disease edge mask** per band × metric.
- Rescue/normalization (if retained, D2) test only within that mask — the gating is what
  makes a vertex-edge rescue analysis reportable.
- dPLI/DTF asymmetry: the "cell" is the **ordered** pair (i→j); phenotype/rescue gating runs
  on directed edges. Confirm the engine's mask keys include direction.
- Edge-level FDR + (for the directed/CFC families) cluster/NBS gating is the **Phase 3**
  bucket of the hypothesis-contrasts plan.

---

## 6. Implementation phases

- **P1 — FC kernels:** add wPLI + dPLI to `connectivity.py` + `vertex_connectivity.py`;
  fix asymmetry-assuming consumers; unit tests (known-signal sanity: zero-lag → PLI≈0,
  dPLI≈0.5; constant lead → dPLI→1). *Verifiable: vertex_connectivity emits 6 metrics.*
- **P2 — sensor comparator:** `electrode_connectivity` module (FC-six on electrode ts);
  wire sensor paradigm; source-vs-sensor paired figure in source-lightbox.
  *Verifiable: head-to-head disease-contrast panel renders.*
- **P3 — local PAC maps:** `vertex_pac` Tier-A whole-brain MI maps. *(Cut 2 begins.)*
- **P4 — Tier-B node-reduced family:** parcellation helper (`reduced_nodes`), then
  cross-region PAC + comodulograms (`vertex_pac`), AAC + PPC (`vertex_cfc`).
- **P5 — DTF:** MVAR fit + DTF in `spectral/directed.py`; `vertex_directed` module at parcel
  resolution. Dependency decision first (§7).
- **P6 — gating integration:** once hypothesis-contrasts Phase 1 lands, route connectivity
  modules through the per-edge gating + TOST; directed-edge mask keys.

P1+P2 = Cut 1 (unblocks preliminary assertion). P3–P5 = Cut 2. P6 depends on the other plan.

---

## 7. Computational & dependency notes

- **DTF dependency:** options — (a) custom Yule-Walker / Nuttall-Strand MVAR + DTF (no dep,
  full control, more code), (b) a library (`scot`, `connectivipy`, or Eden-Kramer
  `spectral_connectivity`). **▶ DEFAULT: evaluate `scot` first; fall back to custom** if the
  ABI/maintenance risk is high (cf. the R-stack `otel` breakage). Decide at P5 start.
- **Vertex all-pairs cost:** FC-six at 215 vertices ≈ 23k edges/band — the eps-clamp/STFT
  path already does dwPLI at this scale (overnight pipeline precedent,
  `[[project_network_split_2026-06-04]]`). dPLI/wPLI are ~free additions (same cross-spectrum).
- **PAC/PPC surrogates** are the expensive multiplier — keep surrogate counts configurable;
  Tier-B parcel resolution keeps it overnight-feasible.
- Reuse the `epoch_sampler` + STFT front-end already shared across the spectral metrics.

## 8. Validation

- Known-signal unit tests per new kernel (phase-lead → dPLI→1; coupled tones → PAC MI > surr;
  n:m locked pair → PPC≈1). Add to `tests/`.
- Cross-check wPLI vs dwPLI and dPLI vs PLI on the same data (monotone relationships).
- Source-vs-sensor sanity: a known volume-conduction-prone metric (coherence) should differ
  more between levels than a VC-robust one (imcoh/PLI) — itself a manuscript talking point.

## 9. Open questions / risks (resolve next session)

1. **Confirm D1/D2/D3 defaults** above.
2. **Does treatment-rescue stay in MS2** or fully move to the other manuscript? (Affects
   whether P6 gating/TOST is in MS2 scope or just the disease phenotype.)
3. **Parcellation choice** for Tier B — Allen-32 (default) vs a finer/data-driven set.
4. **DTF library vs custom** (P5).
5. **Weighted-directional hybrid** — user confirmed plain dPLI; revisit only if reviewers ask.
6. Reconcile this connectivity reframe with the existing MS2 prose
   (`[[project_ms2_resume_2026-05-21]]`, broadband/Phase-2-failure narrative) — the drafted
   Abstract/Results assume the ROI-spectral story.

## 10. Next deliverable

Draft the MS2 **preliminary assertion** once §2 is confirmed. Skeleton:
> In the Fmr1-KO model, [functional connectivity metric(s)] reveal a [hypo/hyper]-connectivity
> phenotype that is **spatially focal at the source (vertex) level** but **smeared/attenuated
> at the sensor level**, demonstrating that source localization provides connectivity
> information unavailable to electrode-space analysis. [Directed measures (dPLI/DTF) further
> localize the directionality of the deficit.]

Fill brackets from Cut-1 results (FC-six, disease contrast, source vs sensor).

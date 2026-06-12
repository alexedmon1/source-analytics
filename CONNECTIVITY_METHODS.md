# Connectivity Methods — Provenance & References

Primary-literature grounding for every connectivity / coupling metric in
`source-analytics`. One entry per metric: **canonical reference**, **defining
equation** (as stated in the source paper), **our implementation**
(`file:function`), and any **deviation** from the canonical method.

This document is the source of truth for method citations in the connectivity
methods manuscript (MS2). Equations were verified against fetched primary
sources (2026-06-12); confidence + the source actually read are noted per metric.
Where our code deviates from the cited paper, it is flagged **⚠ DEVIATION** and
listed in [§ Deviations requiring a decision](#deviations-requiring-a-decision).

Conventions: `S_xy(f)` = cross-spectral density, `S_xx` = auto-spectrum,
`ℑ` = imaginary part, `⟨·⟩` = average over segments/time, `H(·)` = Heaviside step.

---

## Same-frequency functional connectivity (`spectral/connectivity.py`, `vertex_connectivity.py`)

### Magnitude-squared coherence — `coherence`
- **Reference:** classical spectral analysis (Bendat & Piersol, *Random Data*, Wiley; Carter 1987, *Proc. IEEE* 75:236–255). No single EEG originator.
- **Equation:** `COH_xy(f) = |S_xy(f)|² / (S_xx(f)·S_yy(f))`, range 0–1, symmetric.
- **Our code:** `coh_freq = |csd|² / (pxx_i·pxx_j)`, averaged over band. ✅ **Matches.**
- **Note:** maximally sensitive to volume conduction (zero-lag) — the motivation for the imaginary/phase-lag family below. Confidence: high.

### Imaginary coherence — `imag_coherence`
- **Reference:** **Nolte G, Bai O, Wheaton L, Mari Z, Vorbach S, Hallett M (2004).** "Identifying true brain interaction from EEG data using the imaginary part of coherency." *Clin Neurophysiol* 115(10):2292–2307.
- **Equation:** `IC_ij(f) = ℑ(S_ij) / √(S_ii·S_jj)`. Coherency of non-interacting (volume-conduction-only) sources is purely real, so a non-zero imaginary part = genuine lagged interaction. Range −1…1 (we report |IC|).
- **Our code:** `icoh = |ℑ(csd / √(pxx_i·pxx_j))|`. ✅ **Matches** (we take the magnitude).
- **Note:** amplitude-normalized → uncorrelated noise shrinks IC (Vinck 2011 critique). Confidence: high (equation verified secondhand vs ≥2 reproducing sources; Nolte PDF paywalled).

### Phase Lag Index — `pli`
- **Reference:** **Stam CJ, Nolte G, Daffertshofer A (2007).** "Phase lag index…" *Hum Brain Mapp* 28(11):1178–1193.
- **Equation:** `PLI = |⟨sign(ℑ{X})⟩|` (≡ `|⟨sign(Δφ)⟩|`), range 0–1, symmetric. Zero-lag coupling → 0 by construction.
- **Our code:** `|mean(sign(ℑ(csd)))|` per freq, averaged over band. ✅ **Matches.**
- **Note:** sign step-discontinuity makes PLI noise-sensitive near zero lag (→ wPLI). Confidence: high (3 independent confirmations).

### Weighted PLI — `wpli`
- **Reference:** **Vinck M, Oostenveld R, van Wingerden M, Battaglia F, Pennartz CMA (2011).** "An improved index of phase-synchronization…" *NeuroImage* 55(4):1548–1565, **Eq. 8**.
- **Equation:** `wPLI = |E{ℑ{X}}| / E{|ℑ{X}|}`, range 0–1, symmetric.
- **Our code:** `|Σ ℑ| / Σ|ℑ|` per freq, averaged over band (`_compute_pli_family`, `_compute_wpli`). ✅ **Matches Eq. 8.**
- **Note:** mixes phase + amplitude via the |ℑ| weighting. Confidence: high (primary PDF read directly).

### Debiased weighted PLI² — `dwpli`
- **Reference:** **Vinck et al. 2011**, *NeuroImage* 55(4), **Eqs. 31–32** (estimates wPLI², not wPLI).
- **Equation:** `Ω̂_w = [Σ_{j≠k} ℑX_j·ℑX_k] / [Σ_{j≠k}|ℑX_j·ℑX_k|]`. Excludes diagonal j=k self-terms (the source of small-N positive bias). Can return small **negative** values for true ~0 connectivity — expected, not an error.
- **Our code:** `numer = (Σℑ)² − Σℑ²`, `denom = (Σ|ℑ|)² − Σℑ²`. Since `Σ_{j≠k}ℑX_jℑX_k = (Σℑ)² − Σℑ²` and `Σ_{j≠k}|ℑX_jℑX_k| = (Σ|ℑ|)² − Σℑ²`, this **equals Eq. 31**. ✅ **Matches.**
- **⚠ Minor:** we `clip(0,1)` the per-freq value, suppressing the (legitimate) small negatives. Acceptable for reporting; note it estimates the **square**. Confidence: high (primary PDF read directly).

### Directed PLI — `dpli`
- **Reference:** **Stam CJ, van Straaten ECW (2012).** "Go with the flow: …directed phase lag index…" *NeuroImage* 62(3):1415–1428.
- **Equation:** `dPLI_ij = (1/N) Σ_t H(Δφ_ij(t))`, H Heaviside with `H(0)=0.5`. Range 0–1; **asymmetric**. `dPLI_ij > 0.5` ⇒ i phase-leads j; `dPLI_ij + dPLI_ji = 1`. Relation: `PLI = 2·|dPLI − 0.5|`.
- **Our code:** `0.5·(sign(ℑ(S_ij)) + 1)` averaged → maps ℑ>0/=0/<0 to 1/0.5/0 = `H(ℑ(S_ij))`. `S_ij = Z_i·conj(Z_j)` so `phase(S_ij)=φ_i−φ_j`; ℑ>0 ⇒ i leads j ⇒ dPLI>0.5. ✅ **Matches**, convention = "row leads column" (same as dyconnmap). `dpli[j,i]=1−dpli[i,j]`.
- **Note:** directed; **excluded from the undirected graph/NBS layer** (`_DIRECTED_METRICS` guard). Confidence: high for `(1/N)ΣH(Δφ)`; the `H(0)=0.5` micro-detail is standard Heaviside (medium — not re-printed in open sources).

### Orthogonalized amplitude envelope correlation — `aec`
- **Reference:** **Hipp JF, Hawellek DJ, Corbetta M, Siegel M, Engel AK (2012).** "Large-scale cortical correlation structure of spontaneous oscillatory activity." *Nat Neurosci* 15(6):884–890.
- **Equation:** `Y_⊥X(t,f) = imag(Y(t,f)·X(t,f)*/|X(t,f)|)`; then **square → log-transform → Pearson** of log-power envelopes; orthogonalization is directional so **both directions averaged**.
- **Our code (`_orthogonalize_log_power`/`_band_orthogonalized_aec`; `_orth_log_power`/`_compute_aec`):** `Y_⊥X = imag(z_other·conj(z_ref)/|z_ref|)`; `r = Pearson(log|z_ref|², log(Y_⊥X)²)`; average both directions. ✅ **Matches Hipp 2012** (aligned 2026-06-12, deviation A resolved).
- **Test:** `tests/test_aec.py` — genuine lagged amplitude coupling detected; zero-lag (volume-conduction) mixture suppressed. Confidence: high (Hipp PMC3861400 read directly).

### Partial correlation — `partial_corr`
- **Reference:** **Marrelec G, Krainik A, Duffau H, et al. (2006).** "Partial correlation for functional brain interactivity investigation in functional MRI." *NeuroImage* 32(1):228–237. (Statistical identity classical.)
- **Equation:** with precision matrix `Ω = Σ⁻¹ = (p_ij)`: `ρ_ij|rest = −p_ij / √(p_ii·p_jj)`. Distinguishes direct from indirect edges.
- **Our code:** Ledoit-Wolf-style shrinkage of covariance (α=0.01 toward diagonal) → invert → `−P_ij/√(P_ii P_jj)`. ✅ **Matches**, with **shrinkage regularization** (a stabilization choice for #timepoints ≲ #nodes; standard, document it). Confidence: high formula / medium in-paper notation.

---

## Cross-frequency coupling (`spectral/pac.py`, `spectral/cross_freq.py`)

### Phase–amplitude coupling, Modulation Index — `pac` (incl. vertex local-PAC)
- **Reference:** **Tort ABL, Komorowski R, Eichenbaum H, Kopell N (2010).** "Measuring phase-amplitude coupling between neuronal oscillations of different frequencies." *J Neurophysiol* 104(2):1195–1210.
- **Equation:** amplitude-by-phase distribution `P(j)` over N phase bins; `D_KL(P,U)=log(N)−H(P)`; `MI = D_KL(P,U)/log(N)`. Tort used **N=18 bins**; significance via trial-shuffle surrogates. Range 0–1; directional (slow phase modulates fast amplitude).
- **Our code (`_compute_mi_from_phase_amp`, `compute_pac_zscore`, `compute_local_pac_vertices`):** identical KL/log(N) MI, 18-bin default. ✅ **Matches.** Vertex local-PAC = same MI with phase & amplitude from the **same vertex** → one MI per vertex (whole-brain map).
- **⚠ Minor:** surrogates via **circular time-shift** of the amplitude envelope (standard single-trial alternative) rather than Tort's trial-shuffle. Equivalent intent; document. Confidence: high (primary PDF read directly).

### Cross-frequency amplitude–amplitude (power–power) coupling — `aac`
- **Reference:** no single canonical paper. Conceptual primary: **Bruns A, Eckhorn R, Jokeit H, Ebner A (2000).** "Amplitude envelope correlation detects coupling among incoherent brain signals." *NeuroReport* 11(7):1509–1514. Cross-frequency power-comodulogram computation: **Masimore B, Kakalios J, Redish AD (2004).** *J Neurosci Methods* 138(1):97–105.
- **Equation:** correlation between the band-X power envelope and band-Y power envelope across time.
- **Our code (`compute_aac`):** `M[i,j] = Pearson(P_X(i), P_Y(j))`, **power** (squared amplitude) envelopes, **raw** (not orthogonalized), **Pearson**. Symmetric for equal bands. ✅ **Choice fixed 2026-06-12** (deviation B resolved) — power form per the Masimore comodulogram lineage.
- **Note:** raw (un-orthogonalized) envelopes — cross-frequency is partly self-protected from zero-lag leakage but not immune; state explicitly in the manuscript. Confidence: medium (no canonical source; Masimore attribution secondhand).

### n:m phase–phase coupling — `ppc`  ⚠ INCOMPLETE
- **Reference:** concept origin **Tass P, Rosenblum MG, Weule J, et al. (1998).** "Detection of n:m phase locking from noisy data: application to MEG." *Phys Rev Lett* 81(15):3291–3294. Cross-frequency PLV estimator in human EEG/MEG: **Palva JM, Palva S, Kaila K (2005).** "Phase synchrony among neuronal oscillations in the human cortex." *J Neurosci* 25(15):3962–3972.
- **Equation (Palva):** `PLF = |(1/N) Σ_t exp(i·φ_{n,m}(t))|`, `φ_{n,m} = n·φ_x − m·φ_y`. Range 0–1, symmetric. Palva tested **n=1, m∈{1…6}**.
- **Our code (`compute_ppc`):** `|mean exp(i(n·φ_x − m·φ_y))|`. ✅ **Matches the PLF form.** n:m via `_nm_ratio` = `n=round(f_y/f_x), m=1` (harmonic ratio from band centers; band_x=slow).
- **Surrogate significance ✅ ADDED 2026-06-12 (deviation C resolved):** with
  `n_surrogates>0`, `compute_ppc` also returns a **surrogate z-score** —
  band-Y phase is circularly time-shifted by random amounts (preserves each
  signal's phase stats, destroys the cross-frequency relationship), `z = (PLF −
  mean_surr)/std_surr`. The `roi_cross_freq` module emits both `ppc` (PLF) and
  `ppc_z`; surrogate count is `ppc_surrogates` (config, default 200).
- **Note:** n:m convention (`n` on slow band, m=1) differs in labeling from
  Palva's `n=1, m∈{1..6}` but is mathematically equivalent up to band ordering.
  Confidence: high for PLF formula; n:m selection has no field standard.

---

## Directed connectivity (`spectral/transfer_entropy.py`)

### Transfer entropy — `te` / `net_te`
- **Reference:** **Schreiber T (2000).** "Measuring information transfer." *Phys Rev Lett* 85(2):461–464. (arXiv:nlin/0001042.)
- **Equation (Eq. 4):** `T_{J→I} = Σ p(i_{n+1}, i_n^{(k)}, j_n^{(l)}) · log[ p(i_{n+1}|i_n^{(k)}, j_n^{(l)}) / p(i_{n+1}|i_n^{(k)}) ]` — conditional mutual information `I(I_{n+1}; J_n^{(l)} | I_n^{(k)})`. Natural choices `l=k` or `l=1`; worked examples use `k=l=1`. Asymmetric.
- **Our code (`compute_transfer_entropy`, `_te_from_discretized`):** binned (equal-probability, `n_bins=5`), `lag=1` (k=l=1), `TE = H(Y_f,Y_p)+H(Y_p,X_p)−H(Y_p)−H(Y_f,Y_p,X_p)` — the entropy decomposition of Eq. 4. `net_te = te − te.T`. ✅ **Matches** (binned k=l=1 estimator; net-TE is the standard downstream directionality summary).
- **Note:** equal-probability binning + lag=1 are estimator choices; TE has positive finite-sample bias (significance normally vs surrogates — not yet wired). Confidence: high (Schreiber primary PDF read directly).

---

## Deviations requiring a decision

These are where our implementation does not exactly follow the cited paper. Each
needs a call: align the code to the canonical method, or keep the current choice
and cite/justify it.

- **A — AEC vs Hipp 2012. ✅ RESOLVED 2026-06-12 — aligned to Hipp 2012 exactly**
  (`imag(Y·X*/|X|)` orthogonalization + square→log-power→Pearson, both directions
  averaged). Changes previously-computed `aec` values — re-run required.
- **B — AAC design fork. ✅ RESOLVED 2026-06-12 — power / raw / Pearson**
  (Masimore-2004 comodulogram lineage; cite Bruns 2000 + Masimore 2004).
- **C — PPC surrogate significance. ✅ RESOLVED 2026-06-12** — added circular
  time-shift surrogate z-score (`compute_ppc(n_surrogates>0)` → `(plf, z)`;
  module emits `ppc` + `ppc_z`).

**All three deviations (A, B, C) resolved 2026-06-12.**

---

## Sources fetched (2026-06-12)

Primary PDFs read directly: **Vinck 2011** (wPLI/dwPLI Eqs. 8, 31–33);
**Tort 2010** (PMC2941206, MI eqs); **Hipp 2012** (PMC3861400, orthogonalization);
**Schreiber 2000** (arXiv:nlin/0001042, Eq. 4); **Palva 2005** (PMC6724920, PLF).
Verified secondhand vs ≥2 reproducing sources (primary paywalled): **Nolte 2004**,
**Stam 2007**, **Stam & van Straaten 2012** (dPLI; via PMC9584648), **Marrelec 2006**,
**Bruns 2000** (abstract), **Masimore 2004** (attribution via the source-space AAC
literature), **Tass 1998** (via review PMC3346979). Confidence flags per metric above.

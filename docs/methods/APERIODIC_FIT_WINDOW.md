# Choosing the aperiodic fit window

**Default in this package: `12–45 Hz`** (`spectral.aperiodic.DEFAULT_FREQ_RANGE`).
Configurable per analysis — see [Overriding](#overriding) below.

This document explains why, because the choice is consequential and there is no
universal standard to defer to.

---

## There is no standard range, and that is the literature's own position

Gerster et al. (2022) review the fitting ranges used across the field and find
everything from **0.01 Hz to 100 Hz**. They deliberately decline to recommend a
number, advising instead:

> "examining the PSDs of interest carefully and choosing the fitting range that
> best avoids the challenges."

So the defensible standard is a **procedure**, not a constant. Their rules:

1. **Borders must not cross oscillatory peaks.** "Oscillations crossing the
   fitting range borders must be avoided for all investigated power spectra" —
   a peak sitting on a border produces large exponent error.
2. **The upper border must sit below the spectral plateau**, and "as low as
   possible to increase SNR". They define plateau onset operationally as "the
   lowest frequency of a 50 Hz frequency interval with a vanishing exponent."
3. Where an oscillation masks the plateau onset, "the upper fitting range border
   must be lower than the onset of the masking oscillation."

The specparam documentation adds that `[3, 35] Hz` is "a good starting point"
for M/EEG work on low-frequency bands, and warns that ranges wider than ~40 Hz
usually contain a bend that a single exponent cannot describe (requiring knee
mode).

## Why the default is 12–45 Hz

The default targets the package's primary use case: **rodent EEG that has been
high-pass filtered and notch filtered at mains frequency.** Applying the three
rules to such spectra:

| border | value | reason |
|---|---|---|
| lower | **12 Hz** | above the theta/alpha peak (rule 1) and clear of the high-pass roll-off |
| upper | **45 Hz** | below a 57–63 Hz notch and below the high-frequency flattening (rules 2–3) |

**The low border is the one that matters most.** Below the high-pass corner,
power *rises* with frequency — the log-log slope goes **negative**. That segment
cannot be aperiodic neural activity by definition (a 1/f component is
monotonically decreasing), but a fit spanning it will happily average it in and
report a near-zero exponent. This is the single largest source of spuriously
flat exponents.

Rodent practice supports starting well above delta: Bhatt et al. (2026) begin
both aperiodic and periodic assessment at 4 Hz explicitly "to avoid delta
rhythm."

### Worked example (FORGE mouse EEG, 30-channel, 57–63 Hz notch)

Local log-log exponent of the grand-average spectrum, notch masked:

| window | exponent | regime |
|---|---|---|
| 2–4 Hz | **−0.76** | high-pass roll-off — power RISES, not aperiodic |
| 3–6 Hz | 1.00 | shoulder |
| 5–10 Hz | 0.72 | theta/alpha peak region |
| 8–16 Hz | 1.73 | (inflated by the alpha peak rolling off) |
| **12–24 Hz** | **1.09** | clean 1/f |
| **16–32 Hz** | **1.06** | clean 1/f |
| **20–40 Hz** | **0.98** | clean 1/f |
| 30–60 Hz | 0.62 | flattening |
| 50–100 Hz | 0.52 | flattening |

Peak structure (residual against the broadband trend) puts the theta/alpha peak
at **5–11 Hz**, with 11–45 Hz peak-free. Hence 12 Hz as the lower border.

Fitting **2–50 Hz** on these data averages the negative-exponent roll-off, the
genuine 1/f, and the flattening tail, and returns **~0.30** — not because the
brain is unusual but because the window spans three regimes. Progressively
cleaning it: `2–50 → 0.302`, `3–45 → 0.489`, `8–40 → 0.608`.

### Sanity check against published values

Reported mouse EEG aperiodic exponents are **0.737–1.25** across vigilance
states (Kozhemiako et al. 2024). The frequently quoted "1–3" range is a **human**
benchmark and should not be used to judge rodent data. A clean-window FORGE
exponent near 1.0 is squarely normal for mouse.

## The trade-off, stated plainly

12–45 Hz is about **1.9 octaves**. The specparam docs warn that narrow ranges
make the aperiodic component harder to estimate, and that is a real cost: this
default buys an *unbiased* slope at the price of *precision*.

Spectra that are not hemmed in by a high-pass corner, a strong low-frequency
oscillation, a notch, and an early plateau can and should use a wider window.
That is exactly why this is configurable rather than hard-coded.

## A narrow window cannot check its own borders — use two fits

The whole justification above rests on one empirical claim: **no oscillation
crosses 12 Hz or 45 Hz on this data.** That claim is testable, and it must be
tested, because it is dataset-specific — a different strain, age, anaesthetic or
montage moves the peaks, and a window inherited from someone else's study is
then simply wrong.

The problem is that specparam only returns peaks whose centre frequency lies
**inside the fitted window**. Fit at 12–45 Hz and the theta peak you are claiming
to have cleared is invisible *by construction*. The window is being used to
verify itself, and it always passes. Worse, the per-band peak columns for every
band outside the window come back as a uniform `False`, which downstream looks
exactly like a measured null: on FORGE this produced 860 rows of
`chi2 = 0, p = 1.0` for Delta/Theta/High Gamma/Epsilon — a fabricated result for
a comparison the data never had the power to make.

So peak detection gets its **own, wider window**:

```yaml
vertex_specparam:
  freq_range:      [12, 45]   # aperiodic: exponent, offset, r^2
  peak_freq_range: [2, 50]    # peaks only: wide enough to see the borders
```

Both fits run per vertex; the aperiodic estimates still come from the narrow
window and are bit-identical to a single-fit run. Two rules follow:

1. **Unreachable bands emit no columns at all.** A band the peak window cannot
   see is *absent* from the output, never `False`. Absence of a column is
   honest; a `False` is a claim. `band_peak_reachability()` decides this, and
   partially-covered bands are flagged `censored` — their detection rates are a
   lower bound and their peak frequencies are truncated at the window edge.
2. **Peaks from the wide fit are a diagnostic, not a precision measurement.** A
   wide window is a worse 1/f model, and specparam subtracts that model before
   finding peaks. Use these to justify the aperiodic window and to guard band
   power interpretation — not as headline oscillation estimates.

### What the diagnostic answers

`tables/fit_window_diagnostic.csv` and `figures/fit_window_diagnostic.png` test
Gerster's border rule directly. A peak counts as **crossing** when its support
(centre frequency ± half the specparam bandwidth, i.e. ±1 SD of the fitted
Gaussian) straddles a border — which catches both a peak centred inside whose
tail leaks out, and one centred outside whose tail leaks in.

| crossing fraction | reading |
|---|---|
| < 5 % | window **satisfied** — borders sit in spectral gaps |
| 5–15 % | **marginal** — exponents carry some added error |
| > 15 % | **violated** — revisit the window for this dataset |

`data/peak_inventory.csv` holds every detected peak in long format, so the check
can be repeated against any candidate window without re-fitting.

### The second reason peaks are worth keeping

Band power cannot distinguish a narrowband oscillation from a change in the 1/f
background: if the exponent flattens, power rises across every high band with no
oscillation involved. Since aperiodic effects are often the *largest* effects in
a dataset, every band-power result is in principle ambiguous between the two
readings, and peak detection is the only instrument that separates them.

## Report `offset_centered`, not `offset`, next to the exponent

specparam's `offset` is the intercept **extrapolated to 1 Hz** — typically far
below the fit window. A steeper slope therefore *mechanically* forces a higher
intercept, and offset/exponent correlate even when nothing physiological links
them. On FORGE this produced **r(offset, exponent) = 0.96**.

Every fit therefore also returns **`offset_centered`**, the offset re-referenced
to the geometric centre of the fit window:

```
offset_centered = offset − exponent · log10(sqrt(fmin · fmax))
```

On the same data this drops the coupling to **r ≈ 0.60**, which is the genuine
covariance. From a 12–45 Hz window the 1 Hz extrapolation spans ~1.37 decades,
so the higher and narrower the window, the more this matters.

> **Never report offset and exponent effects as two independent findings.**
> Use `offset_centered` when they are discussed together.

## Provenance

Every fit records `fit_fmin` / `fit_fmax` alongside the estimates, so any results
table states the window it came from without reference to the config that
produced it. Two-fit runs also record `peak_fmin` / `peak_fmax`, so a table
carries both windows.

## Overriding

Set `freq_range` in the analysis's config block:

```yaml
roi_aperiodic:
  freq_range: [12, 45]
electrode_aperiodic:
  freq_range: [12, 45]
vertex_specparam:
  freq_range: [12, 45]
  peak_freq_range: [2, 50]   # optional; unset = single fit, peaks from freq_range
```

`resolve_freq_range()` validates the result. It **raises** on a malformed range
(`fmin >= fmax`, non-positive, non-finite) and **warns** — without failing — for
the two silent-corruption cases: an upper border past 50 Hz (mains/notch and
plateau) and a lower border below 4 Hz (high-pass roll-off, delta/theta crossing
the border). Warnings, not errors, because a differently filtered study may
legitimately want a wider window.

## References

- Donoghue T, Haller M, Peterson EJ, Varma P, Sebastian P, Gao R, Noto T,
  Lara AH, Wallis JD, Knight RT, Shestyuk A, Voytek B (2020). *Parameterizing
  neural power spectra into periodic and aperiodic components.* Nature
  Neuroscience 23, 1655–1665. [doi:10.1038/s41593-020-00744-x](https://doi.org/10.1038/s41593-020-00744-x)
  — the specparam/FOOOF algorithm.
- Gerster M, Waterstraat G, Litvak V, Lehnertz K, Schnitzler A, Florin E,
  Curio G, Nikulin V (2022). *Separating Neural Oscillations from Aperiodic 1/f
  Activity: Challenges and Recommendations.* Neuroinformatics 20, 991–1012.
  [doi:10.1007/s12021-022-09581-8](https://doi.org/10.1007/s12021-022-09581-8)
  — border rules, plateau-onset definition, range survey.
- Bhatt N, et al. (2026). *Aperiodicity in Mouse CA1 and DG Power Spectra.*
  eNeuro 13(3). [doi:10.1523/ENEURO.0136-25.2026](https://doi.org/10.1523/ENEURO.0136-25.2026)
  — rodent practice: start above delta; multi-exponent/knee structure.
- Kozhemiako N, et al. (2024). *The aperiodic exponent of neural activity varies
  with vigilance state in mice and men.* PLOS ONE 19(4): e0301406.
  [doi:10.1371/journal.pone.0301406](https://doi.org/10.1371/journal.pone.0301406)
  — mouse EEG exponent reference values.
- specparam documentation, [FAQ](https://fooof-tools.github.io/fooof/faq.html)
  and [Aperiodic Component Fitting tutorial](https://fooof-tools.github.io/fooof/auto_tutorials/plot_05-AperiodicFitting.html).

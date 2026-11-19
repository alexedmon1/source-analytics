# source-analytics

Analysis modules for EEG source-localized ROI and electrode data, driven by a
single study YAML.

The full usage guide — installation, the CLI, the module list, output layout —
lives in the [README](https://github.com/alexedmon1/source-analytics#readme).
This site holds the **methods documentation**: the decisions behind the numbers,
with primary-literature citations.

## Methods

Each page states a canonical reference, what the method does, how it is
implemented here, and any deviation from the canonical form.

| Page | Covers |
|---|---|
| [Aperiodic fit window](methods/APERIODIC_FIT_WINDOW.md) | Why the 1/f fit range defaults to 12–45 Hz; how to override; why to report `offset_centered` |
| [Connectivity metrics](methods/CONNECTIVITY_METHODS.md) | Per-metric provenance: reference, defining equation, implementation, deviations |
| [Hypothesis layer — usage](methods/HYPOTHESIS.md) | Declarative hypotheses: how to declare and run them |
| [Hypothesis layer — design](methods/DESIGN_SPEC.md) | Why the design/hypothesis spec is shaped the way it is |
| [Bayesian engine](methods/BAYES_ENGINE.md) | The Bayesian hypothesis adapter: model, hypothesis kinds, the two prior sets, the simulation gate, declared outcome tables |
| [Bayesian engine catalogue](methods/BAYES_CATALOGUE.md) | Every model, prior and decision rule with its id, version and values (generated from the code) |
| [Benchmark scoring](methods/BENCHMARK.md) | One scoring contract for every method, against simulated cohorts with known truth |

## Guides

| Page | Covers |
|---|---|
| [Bayesian engine user guide](guides/BAYES_USER_GUIDE.md) | From installation to reading a result: declare, gate, fit, read the report, long runs, worked examples |

[Known issues](KNOWN_ISSUES.md) lists the defects and open follow-ups in the
effect sizes and the minimum detectable effect columns.

## Two rules that keep this site useful

1. **The `nav:` in `mkdocs.yml` is the curation.** A page not listed there is not
   part of the site. Adding a document means deciding where it belongs — which is
   what stops reference material from being buried under working notes.
2. **Methods pages are durable; plans are not.** A document that describes *what
   we intend to build* belongs in [the archive](archive/README.md) once built.
   A document that explains *why the code does what it does* belongs in Methods
   and is maintained.

## Publishing

This site is **not published yet** — deployment is gated off while the package is
under active development. See [Publishing this site](PUBLISHING.md) for the
two-step enable and for why versioned docs (`mike`) should be decided at the same
time.

## Local preview

```bash
uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs serve   # http://127.0.0.1:8000
uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs build   # render to site/
```

CI builds this site `--strict` on every push and PR (a broken link fails the
build). Publishing to GitHub Pages is gated off — see
[Publishing this site](PUBLISHING.md).

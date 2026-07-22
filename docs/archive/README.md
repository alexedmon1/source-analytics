# Archive — historical planning documents

These are **completed or superseded** working documents, kept for provenance:
they record why things were built the way they were, and the state of the world
at the time. They are **not maintained** and may describe code that has since
changed.

**Do not treat anything here as current.** For how the code behaves today, see
the [Methods](../index.md#methods) pages and the README.

| Document | Written | Status |
|---|---|---|
| [Profile provenance](PROFILE_PROVENANCE_PLAN.md) | 2026-07-15 | **Implemented** — `--profile` + `for_profile()` shipped |
| [Native schema migration](NATIVE_SCHEMA_MIGRATION.md) | 2026-06 | **Complete** (2026-07-06) — legacy aliases dropped |
| [Hypothesis contrasts](HYPOTHESIS_CONTRASTS_PLAN.md) | 2026-06-09 | **Superseded** by the design spec — its gating half was replaced by declarative `design:`/`hypotheses:` |
| [MS2 connectivity build-out](MS2_CONNECTIVITY_PLAN.md) | 2026-06-11 | **Implemented** — kernels + modules built |
| [Analysis grouping](ANALYSIS_GROUPING_PLAN.md) | 2026-06-03 | **Merged** — the `analysis-grouping` branch landed on `main` |
| [Project status](PROJECT_STATUS.md) | 2026-02-03 | **Stale** — describes v0.2.0; kept only as a snapshot |

## When to move something here

A planning document belongs in the archive as soon as the thing it plans is
built, abandoned, or superseded. If part of it turned out to be durable
explanation rather than a plan, extract that part into a Methods page first —
then archive the rest.

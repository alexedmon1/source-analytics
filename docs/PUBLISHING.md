# Publishing the docs site

**Status: built but not published — deliberately.** The site builds in CI on every
push and PR; deployment is gated off while the package is under active
development. Enabling it is a two-minute job whenever we're ready.

## Why it's deferred

The docs describe behaviour that is still moving. Publishing now would put a
public URL on decisions we're still making, and a stale public methods page is
worse than no public page — it can be cited.

**Trigger to publish:** when `source-analytics` reaches a stable release and the
FORGE analyses stop churning. (`source-localization` has its own trigger: the
validation work in progress there.) There is no rush — the site is already
useful locally.

## How to turn it on

Both steps are required; the deploy job stays skipped until the variable is set.

1. **Settings → Pages → Source = "GitHub Actions"**
2. **Settings → Secrets and variables → Actions → Variables → New variable:**
   `PUBLISH_DOCS` = `true`

The next push to `main` publishes to `https://alexedmon1.github.io/source-analytics/`.
To pause again, set `PUBLISH_DOCS` to anything other than `true` — no need to
touch the workflow.

## Do this at the same time: versioned docs

Consider adding [`mike`](https://github.com/jimporter/mike) before or with the
first publish. It keeps one published copy per release:

```
alexedmon1.github.io/source-analytics/v0.6.0/
alexedmon1.github.io/source-analytics/latest/
```

**This matters here specifically.** Manuscripts pin exact versions — MS1 pins
`source-analytics v0.4.0` and `source-localization v0.2.1` — so a site that only
ever shows `main` will drift away from what those pinned versions actually did.
A reader checking the methods behind a published figure needs the docs *as of
that tag*. With `mike`, a Methods section can cite a permanent versioned URL.

Retro-fitting versioning after publishing means either rewriting URLs people may
already have, or leaving an unversioned root that silently means "latest" — so
it is cheaper to decide this at the same time as step 1 above.

## Local preview (works now, no setup)

```bash
uv run --no-project --with "mkdocs-material>=9.5,<10" mkdocs serve
```

> Note: `uv run --group docs …` currently fails on an unrelated pre-existing
> resolution error (`specparam>=2.0` is unsatisfiable for py3.14/win32 because
> `requires-python` is broader than the dependency supports). The
> `--no-project` invocation above sidesteps it — docs don't import the package.

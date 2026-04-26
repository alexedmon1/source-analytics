# Changelog

## v0.4.0 — 2026-04-25

### Compatibility

- Bump for `source-localization` v0.2.0 Allen32 ROI rename. Four composite ROI labels were renamed upstream for nomenclature accuracy:

| Identifier (v0.2.0) | Legacy alias | Reason |
|---|---|---|
| `Frontal_Anterior_{L,R}` | `Prefrontal_mPFC_{L,R}` | Composite includes ORB and FRP areas, which are lateral / ventral, not medial — the "mPFC" qualifier was misleading. |
| `Basal_Ganglia_{L,R}` | `Striatum_{L,R}` | Composite includes pallidum (globus pallidus), which is anatomically distinct from striatum. |
| `Amygdalar_Complex_{L,R}` | `Amygdala_{L,R}` | Composite includes claustrum and endopiriform nucleus, which are adjacent to but distinct from amygdala. |
| `Brainstem_Tectum_{L,R}` | `Brainstem_{L,R}` | Composite includes superior and inferior colliculi (dorsal-midbrain tectum), not just brainstem proper. |

10-region category `Prefrontal` → `Frontal-Anterior`. Region membership and label IDs (1–32) are unchanged — these are nomenclature clarifications only and have no effect on numerical results.

### No code changes

`source-analytics` consumes ROI labels dynamically from upstream YAML/JSON; no string-literal references to the renamed labels exist in this codebase. This bump documents the compatibility cut so downstream pinning has a clear boundary. Pre-v0.2.0 derivatives (e.g., the FORGE `ms1-v6-frozen` Zenodo deposit) can be read transparently via the `deprecated_aliases` block added in `source-localization` v0.2.0 / `roi_categories.yaml`.

## v0.3.0 — earlier

(no changelog entries before v0.4.0)

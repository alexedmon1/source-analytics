"""What built a localization run: ``data/config_resolved.yaml``.

source-localization 0.4.2+ writes the fully resolved config next to the outputs
it produced — preset, atlas, source-space method, inverse method, orientation,
and (0.5.0+) the source sampling mode. Before that the outputs could not say
what made them, so a manifest is optional and its absence is not an error.

The mode that matters here is **Monte Carlo sampling**
(``source_space.source_sampling: monte_carlo``). It averages the ROI operator
over many sparse source draws instead of solving one arbitrary grid, and is
ROI-only by construction: no single grid is solved, so the run carries no
``step5_stc_*.pkl`` and no vertex set. Its parcel time series are drop-in
compatible with every analysis in this package — they are written to the same
``step6_roi_timeseries_signed.pkl`` in the same epoch-major layout — so nothing
here needs to special-case them to *run*. What it needs is to tell them apart:

- a cohort that mixes fixed-grid and Monte Carlo subjects averages two
  different spatial operators into one group statistic, and
- Monte Carlo flags parcels whose signal the montage cannot separate from a
  neighbour's, and those parcels must not be read as individual results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILE = "config_resolved.yaml"
FIELD_RUN_FILE = "field_run.json"            # source-localization's field mode (project step), 2026-11
MC_REPORT_FILE = "monte_carlo_report.json"

#: ``|cos|`` at or above which source-localization calls two parcels
#: near-collinear. Mirrors ``source_space/realizations.py``.
COLLINEARITY_THRESHOLD = 0.99


@dataclass(frozen=True)
class RunManifest:
    """The resolved config of one source-localization run."""

    path: Path | None = None
    version: str | None = None
    preset: str | None = None
    config_source: str | None = None
    atlas: str | None = None
    bem_type: str | None = None
    source_type: str | None = None
    surface_method: str | None = None
    spacing_mm: float | None = None
    source_sampling: str = "fixed"
    inverse_method: str | None = None
    orientation: str | None = None
    output_variants: tuple[str, ...] = ()
    monte_carlo: dict[str, Any] = field(default_factory=dict)
    #: "source" (localized parcel series) or "field" (projection onto a frozen field template: one virtual channel
    #: per field, no head model in the measurement). A field run is a different readout of the same recording and is
    #: never pooled with a source run, nor with a run on another template or projection.
    readout_mode: str = "source"
    template: str | None = None              # field runs: "<name> v<version> <sha256>"
    projection: str | None = None            # field runs: "<catalogue id> v<version>"

    @property
    def is_monte_carlo(self) -> bool:
        return self.source_sampling == "monte_carlo"

    @property
    def has_vertex_output(self) -> bool:
        """False for Monte Carlo runs, which never solve a single grid."""
        return not self.is_monte_carlo

    def signature(self) -> tuple:
        """The settings two subjects must share to be pooled in one statistic.

        Deliberately excludes the preset name and the package version: a preset
        is a bundle of these values, and two presets that resolve to the same
        geometry are interchangeable. What is compared is what the numbers
        depend on.
        """
        mc = self.monte_carlo
        if self.readout_mode == "field":
            return ("field", self.template, self.projection)
        return (
            self.atlas, self.bem_type, self.source_type, self.surface_method,
            self.spacing_mm, self.source_sampling, self.inverse_method,
            self.orientation,
            (mc.get("n_sources"), mc.get("n_draws")) if self.is_monte_carlo else None,
        )

    def describe(self) -> str:
        """One line, for logs and reports."""
        if self.readout_mode == "field":
            return f"field projection onto {self.template} ({self.projection})"
        bits = [b for b in (self.preset, self.atlas) if b]
        geom = "/".join(b for b in (self.bem_type, self.source_type,
                                    self.surface_method) if b)
        if geom:
            bits.append(geom)
        if self.inverse_method:
            bits.append(self.inverse_method
                        + (f" ({self.orientation})" if self.orientation else ""))
        if self.is_monte_carlo:
            mc = self.monte_carlo
            bits.append(f"Monte Carlo (K={mc.get('n_draws', '?')}, "
                        f"{mc.get('n_sources', '?')} sources/draw)")
        else:
            bits.append("fixed grid")
        return ", ".join(bits) or "unknown run"


def read_field_run(data_dir: str | Path) -> RunManifest | None:
    """Read ``<data_dir>/field_run.json`` (source-localization's field mode), or None if absent or unreadable."""
    path = Path(data_dir) / FIELD_RUN_FILE
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    t, pr = rec.get("template") or {}, rec.get("projection") or {}
    pkg = rec.get("package") or {}
    return RunManifest(path=path, version=pkg.get("version"), readout_mode="field",
                       template=f"{t.get('name')} v{t.get('version')} {t.get('sha256')}",
                       projection=f"{pr.get('id')} v{pr.get('version')}")


def read_run_manifest(data_dir: str | Path) -> RunManifest | None:
    """Read what built a run: ``<data_dir>/field_run.json`` for a field run (it takes precedence), else
    ``<data_dir>/config_resolved.yaml``.

    Returns None when neither is present (a run from before 0.4.2) or cannot be
    parsed. Callers treat None as "unknown", never as "fixed".
    """
    fr = read_field_run(data_dir)
    if fr is not None:
        return fr
    path = Path(data_dir) / MANIFEST_FILE
    if not path.exists():
        return None

    import yaml

    try:
        with open(path) as f:
            snapshot = yaml.safe_load(f) or {}
    except Exception as exc:                      # malformed YAML, unreadable file
        logger.warning("Could not read %s: %s", path, exc)
        return None

    cfg = snapshot.get("config") or {}
    pipeline = cfg.get("pipeline") or {}
    src = cfg.get("source_space") or {}
    surface = src.get("surface") or {}
    inverse = cfg.get("inverse") or {}
    outputs = cfg.get("outputs") or {}
    prov = cfg.get("provenance") or {}

    variants = outputs.get("output_variants", "both")
    if isinstance(variants, str):
        variants = ["signed", "magnitude"] if variants == "both" else [variants]

    return RunManifest(
        path=path,
        version=snapshot.get("source_localization_version"),
        preset=prov.get("preset"),
        config_source=prov.get("config_source"),
        atlas=prov.get("atlas"),
        bem_type=pipeline.get("bem_type"),
        source_type=pipeline.get("source_type"),
        surface_method=surface.get("method"),
        spacing_mm=surface.get("spacing_mm"),
        source_sampling=src.get("source_sampling") or "fixed",
        inverse_method=inverse.get("method"),
        orientation=inverse.get("orientation"),
        output_variants=tuple(variants),
        monte_carlo=dict(src.get("monte_carlo") or {}),
    )


def read_monte_carlo_report(data_dir: str | Path) -> dict | None:
    """Read ``<data_dir>/monte_carlo_report.json``, or None if absent.

    Written by source-localization's ``monte_carlo_roi`` step. Carries the draw
    parameters and, per parcel, the SNR ``gain`` over a single draw plus the
    parcels it is near-collinear with.
    """
    path = Path(data_dir) / MC_REPORT_FILE
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


def unreliable_parcels(report: dict | None) -> dict[str, list[str]]:
    """Parcels the montage cannot separate -> what each is collinear with.

    A parcel listed here shares its dominant sensor topography with another, so
    the inverse splits their common signal arbitrarily and the split moves with
    the source draw. Its *individual* value is not interpretable; the pair taken
    together is. Empty dict when the run is not Monte Carlo.
    """
    if not report:
        return {}
    out: dict[str, list[str]] = {}
    for parcel, entry in (report.get("per_parcel") or {}).items():
        partners = entry.get("collinear_with") or []
        if isinstance(partners, str):
            partners = [partners]
        if partners:
            out[parcel] = list(partners)
    return out


#: Coverage below which a parcel's row is an average over too few draws to be
#: compared as an amplitude. Mirrors the warning in ``realizations.py``.
COVERAGE_THRESHOLD = 0.5


def undersampled_parcels(report: dict | None,
                         threshold: float = COVERAGE_THRESHOLD) -> dict[str, float]:
    """Parcels sampled in fewer than *threshold* of the draws -> their coverage.

    Such a parcel's operator row averages fewer placements than its neighbours'
    and its amplitude is scaled down to match, so a low value there means "rarely
    sampled", not "quiet source". Empty dict when the run is not Monte Carlo.
    """
    if not report:
        return {}
    return {
        parcel: float(entry["coverage"])
        for parcel, entry in (report.get("per_parcel") or {}).items()
        if entry.get("coverage") is not None and float(entry["coverage"]) < threshold
    }


def parcel_caveats(report: dict | None) -> dict[str, str]:
    """Parcel -> why its individual value should not be read at face value.

    Combines the two flags source-localization raises when it builds the Monte
    Carlo operator. Returns an empty dict for a fixed-grid run, so callers can
    apply it unconditionally.
    """
    caveats: dict[str, str] = {}
    for parcel, coverage in undersampled_parcels(report).items():
        caveats[parcel] = (f"sampled in {coverage:.0%} of draws; amplitude is "
                           f"scaled down accordingly")
    for parcel, partners in unreliable_parcels(report).items():
        note = (f"near-collinear with {', '.join(partners)}; the inverse splits "
                f"their shared signal arbitrarily")
        caveats[parcel] = f"{caveats[parcel]}; {note}" if parcel in caveats else note
    return caveats

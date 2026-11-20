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

# Run records source-localization writes beside the outputs (0.9.1 onwards). Names, tags and the rule for a
# run that predates each record mirror source-localization's ``study/run_status.py``; read, not imported, so
# this package does not depend on source-localization.
STANDARDISATION_FILE = "step5_inverse_standardisation.json"   # 0.9.1
DIRECTIONS_FILE = "step6_mc_directions.json"                  # 0.9.2; magnitude_readout from 0.9.13
ROI_COVERAGE_FILE = "step6_roi_coverage.json"                 # 0.9.11
STANDARDISATION_TAG, LEGACY_STANDARDISATION_TAG = "per-source-v1", "per-component-legacy"
DIRECTIONS_TAG, LEGACY_DIRECTIONS_TAG = "three-direction-block-v1", "dominant-direction-legacy"

#: The caveat every region-level result carries (D10 step 1b). Source: source-localization
#: reports/readout_groupdiff_2026-11-20/README.md, finding 5; the 42% is the signed series, which roi_psd reads.
REGION_LEAKAGE_CAVEAT = (
    "Spatial leakage: a difference in one region also appears in most other regions. In source-localization's "
    "simulated two-group studies (20 Nov 2026: FORGE recordings as the noise, allen32, Monte Carlo sLORETA, a "
    "6 dB difference placed in one parcel at a time), 42-100% of the other parcels were also significant, and "
    "the parcel holding the difference had the largest effect in only 1-29% of experiments. A significant "
    "region shows that the groups differ in the source estimate, not that the difference is located in that "
    "region; compare regions with each other only with this in mind."
)

#: inverse.snr when a record leaves it out: 3 until 0.9.12, 1 from 0.9.12 (source-localization decision D4).
#: Records carry every step default from 0.9.11, so the fallback matters for runs made before that.
SNR_DEFAULT_BEFORE_0912, SNR_DEFAULT_FROM_0912 = 3.0, 1.0
#: BEM vertices per surface for a numerical head model whose record predates the setting (0.9.12).
BEM_MESH_BEFORE_0912 = 642

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
    # What the numbers depend on beyond the geometry and the method (source-localization 0.9.1-0.9.13). A
    # record absent from an older run is read as the only value that release had; None where the setting does
    # not apply (a field run, the analytic sphere's mesh) or cannot be told (no version to date the run by).
    regularisation: str | None = None        # "snr 1 (lambda2 1)": lambda2 or 1/snr**2
    bem_mesh_vertices: int | None = None     # numerical head models (ellipsoid, numerical sphere)
    analytic_electrodes: str | None = None   # analytic sphere: projected (0.9.12) or raw
    electrode_centered: bool | None = None   # sphere head models: centred under the electrodes
    inverse_standardisation: str | None = None  # correct / faulty / not affected / unknown
    mc_directions: str | None = None         # Monte Carlo: three / dominant / none / unknown
    mc_readout: str | None = None            # Monte Carlo: mean_of_norms / vector_mean / none / unknown

    @property
    def mc_draws(self) -> tuple | None:
        """(sources per draw, draws) for a Monte Carlo run, else None."""
        if not self.is_monte_carlo:
            return None
        return (self.monte_carlo.get("n_sources"), self.monte_carlo.get("n_draws"))

    @property
    def is_monte_carlo(self) -> bool:
        return self.source_sampling == "monte_carlo"

    @property
    def has_vertex_output(self) -> bool:
        """False for Monte Carlo runs, which never solve a single grid."""
        return not self.is_monte_carlo

    def signature(self) -> tuple:
        """The settings two subjects must share to be pooled in one statistic (:data:`COHORT_FIELDS`).

        Deliberately excludes the preset name and the package version: a preset
        is a bundle of these values, and two presets that resolve to the same
        geometry are interchangeable. What is compared is what the numbers
        depend on.
        """
        return tuple(getattr(self, attr) for attr, _label in COHORT_FIELDS)

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


#: What two subjects must share to be pooled, with the label a refusal names it by. The one list behind both
#: ``RunManifest.signature`` and ``BaseAnalysis``'s cohort check.
COHORT_FIELDS = (
    ("readout_mode", "readout (source or field)"),
    ("template", "field template"),
    ("projection", "field projection"),
    ("atlas", "atlas"),
    ("bem_type", "BEM"),
    ("source_type", "source space"),
    ("surface_method", "surface method"),
    ("spacing_mm", "spacing_mm"),
    ("source_sampling", "source sampling"),
    ("inverse_method", "inverse method"),
    ("orientation", "orientation"),
    ("regularisation", "inverse regularisation (inverse.snr / lambda2)"),
    ("bem_mesh_vertices", "BEM mesh vertices per surface (bem.<type>.n_vertices)"),
    ("analytic_electrodes", "analytic sphere electrodes (bem.sphere.analytic_electrodes)"),
    ("electrode_centered", "sphere centred under the electrodes (bem.sphere.electrode_centered)"),
    ("inverse_standardisation", "inverse standardisation (step5_inverse_standardisation.json)"),
    ("mc_draws", "Monte Carlo (sources per draw, draws)"),
    ("mc_directions", "Monte Carlo directions (step6_mc_directions.json)"),
    ("mc_readout", "Monte Carlo magnitude readout (source_space.monte_carlo.readout)"),
)


def _read_json(path: Path) -> dict | None:
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _version_tuple(version) -> tuple | None:
    import re
    m = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", str(version or ""))
    return tuple(int(x or 0) for x in m.groups()) if m else None


def _number(value) -> str:
    return f"{float(value):g}"


def _regularisation(inverse: dict, version) -> str | None:
    """``lambda2`` as the inverse used it, with the snr it came from: an explicit lambda2 wins, else
    1 / snr**2, the snr defaulting by release (3 before 0.9.12, 1 from it). None when the record has neither
    and no version dates it."""
    lam = inverse.get("lambda2")
    if lam is not None:
        return f"lambda2 {_number(lam)}"
    snr = inverse.get("snr")
    if snr is None:
        v = _version_tuple(version)
        if v is None:
            return None
        snr = SNR_DEFAULT_FROM_0912 if v >= (0, 9, 12) else SNR_DEFAULT_BEFORE_0912
    return f"snr {_number(snr)} (lambda2 {_number(1.0 / float(snr) ** 2)})"


def _head_model_form(cfg: dict) -> tuple[int | None, str | None, bool | None]:
    """(mesh vertices, analytic electrodes, electrode centred) from the resolved config. A record that
    predates a setting (0.9.12) had the only value earlier releases had: 642 vertices, raw electrodes, a sphere
    centred on the brain."""
    bem_type = (cfg.get("pipeline") or {}).get("bem_type")
    if not bem_type:
        return None, None, None
    section = (cfg.get("bem") or {}).get(bem_type) or {}
    numerical = bem_type != "sphere" or section.get("numerical", True) is not False
    mesh = int(section.get("n_vertices") or BEM_MESH_BEFORE_0912) if numerical else None
    analytic = None if numerical else str(section.get("analytic_electrodes") or "raw")
    centred = bool(section.get("electrode_centered", False)) if bem_type == "sphere" else None
    return mesh, analytic, centred


def _orientation(cfg: dict) -> str:
    """The orientation source-localization's inverse step resolves for this config."""
    inv = cfg.get("inverse") or {}
    if inv.get("orientation"):
        return str(inv["orientation"]).lower()
    surf = (cfg.get("source_space") or {}).get("surface") or {}
    anatomical = ((cfg.get("pipeline") or {}).get("source_type") == "surface"
                  and surf.get("method", "anatomical") == "anatomical")
    return "fixed" if anatomical else "free"


def _inverse_standardisation(data_dir: Path, cfg: dict) -> str:
    """correct / faulty / not affected / unknown, as source-localization's ``inverse_status`` decides it: the
    standardisation record (0.9.1), else the configuration (Monte Carlo sampling, fixed or loose orientation
    and beamformers were never affected; free-orientation sLORETA, dSPM, MNE and eLORETA were)."""
    rec = _read_json(data_dir / STANDARDISATION_FILE)
    if rec is not None:
        if rec.get("orientation") in ("fixed", "loose"):
            return "not affected"
        if str(rec.get("method", "")).upper() == "ELORETA":
            # Records before 0.9.2 carry no eloreta_weights: those runs used the scalar weight.
            return {"block": "correct", "legacy_scalar": "faulty"}.get(
                rec.get("eloreta_weights", "legacy_scalar"), "unknown")
        return {STANDARDISATION_TAG: "correct", LEGACY_STANDARDISATION_TAG: "faulty"}.get(
            rec.get("inverse_standardisation"), "unknown")
    sampling = (cfg.get("source_space") or {}).get("source_sampling") or "fixed"
    method = str((cfg.get("inverse") or {}).get("method", "sLORETA")).upper()
    if sampling == "monte_carlo" or method in ("LCMV", "DICS") or _orientation(cfg) in ("fixed", "loose"):
        return "not affected"
    if method in ("SLORETA", "DSPM", "MNE", "ELORETA"):
        return "faulty"
    return "unknown"


def _mc_directions_and_readout(data_dir: Path, cfg: dict) -> tuple[str | None, str | None]:
    """How a Monte Carlo pool's free sources entered the operator, and how its magnitude series was read out;
    (None, None) for fixed sampling. As source-localization's ``mc_directions_status`` / ``mc_readout_status``:
    a run without the directions record predates 0.9.2 and collapsed its free sources (``dominant``; ``none``
    for the fixed-orientation surface pool, which has nothing to collapse), so it has no separate readout; a
    three-direction run whose record has no ``magnitude_readout`` predates 0.9.13 and read the norm of the mean
    vector (``vector_mean``, told apart from an explicit one only in source-localization's QC label)."""
    if ((cfg.get("source_space") or {}).get("source_sampling") or "fixed") != "monte_carlo":
        return None, None
    rec = _read_json(data_dir / DIRECTIONS_FILE)
    if rec is not None:
        directions = {DIRECTIONS_TAG: "three", LEGACY_DIRECTIONS_TAG: "dominant",
                      "none": "none"}.get(rec.get("mc_directions"), "unknown")
        if "magnitude_readout" in rec:
            return directions, str(rec["magnitude_readout"])
        return directions, {"three": "vector_mean"}.get(directions, "none" if directions != "unknown"
                                                        else "unknown")
    source_type = (cfg.get("pipeline") or {}).get("source_type")
    if source_type == "surface" and _orientation(cfg) == "fixed":
        return "none", "none"
    return "dominant", "none"


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

    version = snapshot.get("source_localization_version")
    mesh, analytic, centred = _head_model_form(cfg)
    directions, readout = _mc_directions_and_readout(path.parent, cfg)
    return RunManifest(
        path=path,
        version=version,
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
        regularisation=_regularisation(inverse, version),
        bem_mesh_vertices=mesh,
        analytic_electrodes=analytic,
        electrode_centered=centred,
        inverse_standardisation=_inverse_standardisation(path.parent, cfg),
        mc_directions=directions,
        mc_readout=readout,
    )


def read_empty_parcels(data_dir: str | Path) -> list[str] | None:
    """Parcels a run has no sources in (``step6_roi_coverage.json``, ``parcels_without_sources``), or None when
    the run has no such record (before source-localization 0.9.11, which zero-filled them instead of leaving them
    out of the outputs; or a field run)."""
    rec = _read_json(Path(data_dir) / ROI_COVERAGE_FILE)
    if rec is None:
        return None
    return [str(p) for p in rec.get("parcels_without_sources") or []]


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

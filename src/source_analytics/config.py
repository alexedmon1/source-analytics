"""YAML-driven study configuration loader."""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_atlas_roi_categories(atlas_name: str | None) -> dict[str, list[str]]:
    """Load canonical roi_categories from atlas package data, if available."""
    if not atlas_name:
        return {}
    try:
        from source_analytics.atlas import find_atlas_dir, load_roi_categories

        atlas_dir = find_atlas_dir(atlas_name=atlas_name)
        return load_roi_categories(atlas_dir)
    except Exception:
        return {}


# Hypothesis-testing contrast vocabularies (see HYPOTHESIS_CONTRASTS_PLAN.md).
VALID_CONTRAST_ROLES = {"phenotype", "rescue", "normalization", "exploratory"}
VALID_CONTRAST_TESTS = {"difference", "equivalence"}
VALID_MARGIN_MODES = {"gap_fraction", "sd"}


@dataclass
class Contrast:
    """A between-group pairwise contrast (legacy per-contrast analysis paths).

    The canonical declarative form is now :class:`Hypothesis` (``design:`` /
    ``hypotheses:``). ``Contrast`` remains the simple group_a-vs-group_b record the
    legacy module statistics loops consume; when a study declares ``hypotheses:``
    instead of ``contrasts:``, ``StudyConfig`` *derives* these from the pairwise
    contrast/equivalence hypotheses (see :func:`_contrasts_from_design_spec`).
    ``role``/``test`` are display/semantic tags only — there is no gating.
    """

    name: str
    group_a: str
    group_b: str
    label: str | None = None
    group: str | None = None
    role: str = "exploratory"
    test: str = "difference"

    @classmethod
    def from_dict(cls, c: dict) -> Contrast:
        """Build a Contrast from a raw YAML contrast mapping, with validation."""
        name = c["name"]

        role = c.get("role", "exploratory")
        if role not in VALID_CONTRAST_ROLES:
            raise ValueError(
                f"Contrast '{name}': invalid role '{role}'. "
                f"Expected one of {sorted(VALID_CONTRAST_ROLES)}."
            )

        test = c.get("test", "difference")
        if test not in VALID_CONTRAST_TESTS:
            raise ValueError(
                f"Contrast '{name}': invalid test '{test}'. "
                f"Expected one of {sorted(VALID_CONTRAST_TESTS)}."
            )

        return cls(
            name=name,
            group_a=c["group_a"],
            group_b=c["group_b"],
            label=c.get("label"),
            group=c.get("group"),
            role=role,
            test=test,
        )


def _validate_margin(contrast_name: str, margin: Any) -> None:
    """Validate an equivalence_margin mapping ({mode, value})."""
    if not isinstance(margin, dict):
        raise ValueError(
            f"Contrast '{contrast_name}': equivalence_margin must be a mapping "
            f"with 'mode' and 'value', got {type(margin).__name__}."
        )
    mode = margin.get("mode")
    if mode not in VALID_MARGIN_MODES:
        raise ValueError(
            f"Contrast '{contrast_name}': equivalence_margin.mode '{mode}' "
            f"invalid. Expected one of {sorted(VALID_MARGIN_MODES)}."
        )
    value = margin.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Contrast '{contrast_name}': equivalence_margin.value must be a "
            f"positive number, got {value!r}."
        )


def _contrasts_from_design_spec(spec: "DesignSpec") -> list[Contrast]:
    """Derive legacy pairwise :class:`Contrast` records from a design spec.

    Lets a study that declares only ``design:``/``hypotheses:`` still feed the
    legacy per-contrast analysis loops (which iterate ``config.contrasts``). Only
    *pairwise* contrast/equivalence hypotheses (exactly two opposite-sign weights)
    map to a two-group Contrast; omnibus/regression/multi-group hypotheses have no
    Contrast analogue and are skipped (the new hypothesis layer handles them).
    """
    out: list[Contrast] = []
    for h in spec.hypotheses:
        if h.kind not in ("contrast", "equivalence") or not h.weights:
            continue
        pos = [g for g, w in h.weights.items() if w > 0]
        neg = [g for g, w in h.weights.items() if w < 0]
        if len(h.weights) == 2 and len(pos) == 1 and len(neg) == 1:
            out.append(Contrast(
                name=h.name, group_a=pos[0], group_b=neg[0],
                label=h.label, group=h.role, role=h.role,
                test=h.test or "difference",
            ))
    return out


VALID_KINDS = {"omnibus", "contrast", "regression", "equivalence"}


@dataclass
class Hypothesis:
    """A declarative hypothesis for the ``hypothesis`` inference layer.

    Mirrors the R-side parser in ``R/hypothesis.R``. A hypothesis carries a
    ``kind`` and the portable payload that kind needs (``weights`` for contrast/
    equivalence, ``groups`` for omnibus, ``predictor`` for regression). The legacy
    ``group_a``/``group_b`` contrast form is accepted as pairwise-weights sugar.
    Unlike the retired gating system, nothing here auto-fires — these are run one
    at a time, by name. See DESIGN_SPEC.md.
    """

    name: str
    kind: str = "contrast"
    label: str | None = None
    role: str = "exploratory"
    weights: dict[str, float] | None = None
    groups: list[str] | None = None
    predictor: str | None = None
    by: str | None = None
    test: str | None = None
    margin: dict[str, Any] | None = None
    # Per-hypothesis multiple-comparison override: {scope, method}. None inherits
    # the design-level default ({scope=hypothesis, method=BH}). Applied by the
    # R emmeans adapter (R/hypothesis.R); the permutation/map adapter uses
    # cluster-extent correction, so this field is a no-op there.
    fdr: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, h: dict) -> Hypothesis:
        name = h["name"]
        # Legacy sugar: group_a/group_b in place of weights.
        if "kind" not in h and "weights" not in h and "group_a" in h and "group_b" in h:
            kind = "equivalence" if h.get("test") == "equivalence" else "contrast"
            return cls(
                name=name, kind=kind, label=h.get("label"),
                role=h.get("role", "exploratory"),
                weights={h["group_a"]: 1.0, h["group_b"]: -1.0},
                test=h.get("test"), margin=h.get("equivalence_margin"),
                fdr=h.get("fdr"),
            )
        kind = h.get("kind", "contrast")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"Hypothesis '{name}': invalid kind '{kind}'. "
                f"Expected one of {sorted(VALID_KINDS)}."
            )
        weights = None
        if h.get("weights") is not None:
            weights = {str(k): float(v) for k, v in h["weights"].items()}
        groups = [str(x) for x in h["groups"]] if h.get("groups") else None
        margin = h.get("margin") or h.get("equivalence_margin")
        if kind in ("contrast", "equivalence") and not weights:
            raise ValueError(f"Hypothesis '{name}': kind={kind} requires 'weights'.")
        if kind == "regression" and not h.get("predictor"):
            raise ValueError(f"Hypothesis '{name}': kind=regression requires 'predictor'.")
        if kind == "equivalence" and margin is None:
            raise ValueError(f"Hypothesis '{name}': kind=equivalence requires 'margin'.")
        if margin is not None:
            _validate_margin(name, margin)
        return cls(
            name=name, kind=kind, label=h.get("label"),
            role=h.get("role", "exploratory"), weights=weights, groups=groups,
            predictor=h.get("predictor"), by=h.get("by"), test=h.get("test"),
            margin=margin, fdr=h.get("fdr"),
        )

    def referenced_groups(self) -> set[str]:
        """All group levels this hypothesis names (for subject discovery)."""
        g: set[str] = set()
        if self.weights:
            g |= set(self.weights)
        if self.groups:
            g |= set(self.groups)
        return g

    def pairwise_endpoints(self) -> tuple[str, str] | None:
        """``(group_a, group_b)`` for a simple pairwise contrast, else ``None``.

        ``group_a`` is the positive-weight level, ``group_b`` the negative-weight
        one — the two-sample form the per-vertex/edge map loops use to drive
        ``cluster_permutation_test`` / NBS directly from the declared hypotheses,
        replacing the legacy ``config.contrasts`` bridge. Returns ``None`` for
        omnibus/regression and for non-pairwise (>2-level) weighted contrasts.
        """
        if self.kind not in ("contrast", "equivalence") or not self.weights:
            return None
        pos = [g for g, w in self.weights.items() if w > 0]
        neg = [g for g, w in self.weights.items() if w < 0]
        if len(pos) == 1 and len(neg) == 1:
            return pos[0], neg[0]
        return None


@dataclass
class DesignSpec:
    """Parsed ``design:`` + ``hypotheses:`` blocks (the hypothesis registry).

    Falls back to lifting a legacy ``contrasts:`` block into hypotheses when no
    ``hypotheses:`` block is present, so unmigrated studies still expose a spec.
    Returns ``None`` from :meth:`from_dict` only when neither block exists.
    """

    factor: str = "group"
    reference: str | None = None
    levels: list[str] | None = None
    covariates: list[str] = field(default_factory=list)
    # Study-level multiple-comparison default ({scope, method}); per-hypothesis
    # fdr: overrides it field-by-field. Empty -> {scope=hypothesis, method=BH}.
    fdr: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[Hypothesis] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> DesignSpec | None:
        design = data.get("design") or {}
        raw_h = data.get("hypotheses")
        if raw_h is None:
            legacy = data.get("contrasts")
            if not legacy and not design:
                return None
            raw_h = legacy or []
        hyps = [Hypothesis.from_dict(h) for h in raw_h]
        levels = [str(x) for x in design["levels"]] if design.get("levels") else None
        return cls(
            factor=design.get("factor", "group"),
            reference=design.get("reference"),
            levels=levels,
            covariates=[str(x) for x in (design.get("covariates") or [])],
            fdr=design.get("fdr") or {},
            hypotheses=hyps,
        )


@dataclass
class StudyConfig:
    """Complete study configuration loaded from YAML.

    Attributes
    ----------
    name : str
        Human-readable study name.
    output_dir : Path
        Root directory for analysis outputs (intermediate data).
    results_dir : Path
        Root directory for results (figures, tables).
    groups : dict[str, str]
        Mapping of group_id -> display label.
    group_order : list[str]
        Preferred ordering for plots.
    group_colors : dict[str, str]
        Hex colors per group.
    contrasts : list[Contrast]
        Between-group contrasts.
    bands : dict[str, tuple[float, float]]
        Frequency band definitions.
    roi_categories : dict[str, list[str]]
        Named ROI groupings for regional analysis.
    discovery : dict[str, Any]
        Subject discovery configuration (root_dir, group_mapping, etc.).
    raw : dict
        The raw parsed YAML for extension.
    paradigms : dict[str, dict]
        Per-paradigm configuration sections (multi-paradigm mode).
    paradigm_name : str or None
        Name of the active paradigm (set by ``for_paradigm()``).
    """

    name: str
    output_dir: Path
    results_dir: Path
    groups: dict[str, str]
    group_order: list[str]
    group_colors: dict[str, str]
    bands: dict[str, tuple[float, float]]
    roi_categories: dict[str, list[str]]
    discovery: dict[str, Any]
    group_linetypes: dict[str, str] = field(default_factory=dict)
    vertex_filter: dict[str, Any] = field(default_factory=dict)
    vertex: dict[str, Any] = field(default_factory=dict)
    electrode: dict[str, Any] = field(default_factory=dict)
    evoked: dict[str, Any] = field(default_factory=dict)
    paradigms: dict[str, dict] = field(default_factory=dict, repr=False)
    paradigm_name: str | None = None
    design_spec: DesignSpec | None = None
    # ---- Profile narrowing (set by for_profile(); see PROFILE_PROVENANCE_PLAN.md) ----
    # rois: allowlist restricting which ROIs enter the analysis at all. Empty = all
    # ROIs present in the data (today's behaviour). NOT merely cosmetic: the ROI set
    # is the FDR family, so narrowing it changes every q-value.
    rois: list[str] = field(default_factory=list)
    # Analyses this profile is allowed to run (allowlist). None = all.
    include_analyses: list[str] | None = None
    # Profile id ("external", ...). None = the default/exploratory profile, whose
    # output paths must stay byte-identical to a pre-profile run.
    profile_name: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        """Load a study config from a YAML file.

        Supports two formats:

        **Legacy format** (separate analytics config):
            Has ``output_dir`` and ``discovery`` keys.

        **Unified format** (shared by source-localization, source-analytics,
        source-lightbox):
            Has ``subjects`` as a list of dicts with ``eeg_file`` keys and a
            ``paths`` section.  Detected automatically.

        Path defaults (when keys are absent from the YAML):
        - ``output_dir`` → directory containing the YAML file
        - ``discovery.root_dir`` → ``../derivatives`` relative to the YAML file
        """
        path = Path(path).resolve()
        config_dir = path.parent

        with open(path) as f:
            data = yaml.safe_load(f)

        # Detect unified format: subjects is a list of dicts with eeg_file
        subjects_raw = data.get("subjects", [])
        is_unified = (
            isinstance(subjects_raw, list)
            and len(subjects_raw) > 0
            and isinstance(subjects_raw[0], dict)
            and "eeg_file" in subjects_raw[0]
        )

        # Pass YAML filename stem so output_dir can be derived from it
        data["_config_stem"] = path.stem

        if is_unified:
            return cls._from_unified_yaml(data, config_dir)
        return cls._from_legacy_yaml(data, config_dir)

    @classmethod
    def _from_unified_yaml(cls, data: dict, config_dir: Path) -> StudyConfig:
        """Parse unified study.yaml format."""
        paths = data.get("paths", {})

        # Resolve paths relative to config directory
        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (config_dir / pp).resolve()

        loc_dir = _resolve(paths.get("localization", "./localization"))
        analytics_dir = _resolve(paths.get("analytics", "./analytics"))
        results_dir = _resolve(paths.get("results", "./results"))

        # Build subject_groups dict for discovery: {dir_name: group}
        # Discovery root = localization/derivatives
        # Subjects with exclude: true are omitted from analyses
        discovery_root = loc_dir / "derivatives"
        subject_groups: dict[str, str] = {}
        for s in data.get("subjects", []):
            if s.get("exclude", False):
                continue
            sid = str(s["id"])
            group = s.get("group", "unknown")
            subject_groups[f"sub-{sid}"] = group

        discovery: dict = {
            "root_dir": str(discovery_root),
            "subject_groups": subject_groups,
            "data_subdir": "pipeline/data",
        }

        design_spec = DesignSpec.from_dict(data)

        bands = {
            name: tuple(limits) for name, limits in data.get("bands", {}).items()
        }

        # Map analyses to paradigm structure if no paradigms key
        paradigms: dict[str, dict] = {}
        analyses_raw = data.get("analyses", {})
        if analyses_raw and "paradigms" not in data:
            # Wrap in a single "resting" paradigm
            paradigms["resting"] = {
                "analyses": analyses_raw,
                "data_dir": str(discovery_root),
                "data_subdir": "pipeline/data",
                "subjects": subject_groups,
            }
        for pname, pdata in data.get("paradigms", {}).items():
            pcopy = dict(pdata)
            if "data_dir" in pcopy:
                pcopy["data_dir"] = str((config_dir / pcopy["data_dir"]).resolve())
            paradigms[pname] = pcopy

        roi_categories = data.get("roi_categories") or _load_atlas_roi_categories(
            data.get("pipeline", {}).get("atlas")
        )

        return cls(
            name=data.get("name", "Unnamed Study"),
            output_dir=analytics_dir,
            results_dir=results_dir,
            groups=data.get("groups", {}),
            group_order=data.get("group_order", list(data.get("groups", {}).keys())),
            group_colors=data.get("group_colors", {}),
            group_linetypes=data.get("group_linetypes", {}),
            bands=bands,
            roi_categories=roi_categories,
            discovery=discovery,
            vertex_filter=data.get("vertex_filter", {}),
            vertex=_resolve_vertex_config(data),
            electrode=data.get("electrode", {}),
            evoked=data.get("evoked", {}),
            paradigms=paradigms,
            design_spec=design_spec,
            raw=data,
        )

    @classmethod
    def _from_legacy_yaml(cls, data: dict, config_dir: Path) -> StudyConfig:
        """Parse legacy analytics YAML format."""
        # Resolve output_dir: explicit, or derive from YAML filename so that
        # multiple configs in the same directory never collide.
        # e.g. analytics/study_allen32_roi.yaml → analytics/allen32_roi/
        if "output_dir" in data:
            output_dir = Path(data["output_dir"])
        else:
            stem = config_dir.stem if hasattr(config_dir, "stem") else ""
            # Use the YAML filename (passed via _config_stem) if available
            yaml_stem = data.get("_config_stem")
            if yaml_stem:
                # Strip common prefixes like "study_"
                dirname = yaml_stem
                for prefix in ("study_", "config_"):
                    if dirname.startswith(prefix):
                        dirname = dirname[len(prefix):]
                        break
                output_dir = config_dir / dirname
            else:
                output_dir = config_dir

        # Resolve results_dir: explicit or default to sibling results/
        if "results_dir" in data:
            results_dir = Path(data["results_dir"])
        else:
            results_dir = output_dir.parent / "results"

        # Resolve discovery.root_dir: explicit or default to sibling derivatives/
        discovery = data.get("discovery", {})
        if "root_dir" not in discovery:
            discovery["root_dir"] = str(config_dir.parent / "derivatives")

        design_spec = DesignSpec.from_dict(data)

        bands = {
            name: tuple(limits) for name, limits in data.get("bands", {}).items()
        }

        # Resolve paradigm data_dir paths relative to config file
        paradigms = {}
        for pname, pdata in data.get("paradigms", {}).items():
            pcopy = dict(pdata)
            if "data_dir" in pcopy:
                pcopy["data_dir"] = str((config_dir / pcopy["data_dir"]).resolve())
            # Also resolve per-analysis data_dir when analyses is a dict
            analyses_raw = pcopy.get("analyses")
            if isinstance(analyses_raw, dict):
                resolved = {}
                for aname, acfg in analyses_raw.items():
                    acopy = dict(acfg) if isinstance(acfg, dict) else {}
                    if "data_dir" in acopy:
                        acopy["data_dir"] = str(
                            (config_dir / acopy["data_dir"]).resolve()
                        )
                    resolved[aname] = acopy
                pcopy["analyses"] = resolved
            paradigms[pname] = pcopy

        return cls(
            name=data["name"],
            output_dir=output_dir,
            results_dir=results_dir,
            groups=data.get("groups", {}),
            group_order=data.get("group_order", list(data.get("groups", {}).keys())),
            group_colors=data.get("group_colors", {}),
            group_linetypes=data.get("group_linetypes", {}),
            bands=bands,
            roi_categories=data.get("roi_categories", {}),
            discovery=discovery,
            vertex_filter=data.get("vertex_filter", {}),
            vertex=_resolve_vertex_config(data),
            electrode=data.get("electrode", {}),
            evoked=data.get("evoked", {}),
            paradigms=paradigms,
            design_spec=design_spec,
            raw=data,
        )

    @property
    def has_paradigms(self) -> bool:
        """True if this config defines multiple paradigms."""
        return bool(self.paradigms)

    def get_paradigm_analyses(self, name: str) -> list[str] | None:
        """Return the analyses list for a paradigm, or None.

        When a profile declares ``include_analyses`` (an allowlist), the result is
        intersected with it — order follows the paradigm, and analyses the profile
        doesn't name are simply absent.
        """
        pdata = self.paradigms.get(name)
        if pdata is None:
            return None
        analyses = pdata.get("analyses")
        if isinstance(analyses, dict):
            analyses = list(analyses.keys())
        if analyses is None or self.include_analyses is None:
            return analyses
        allowed = set(self.include_analyses)
        return [a for a in analyses if a in allowed]

    def for_profile(self, name: str) -> StudyConfig:
        """Return a config narrowed by the top-level ``<name>:`` profile block.

        A profile only ever *narrows* — it never redefines shared study facts
        (groups, colors, subjects, design). Apply it to the **root** config, before
        paradigm scoping::

            config.for_profile("external").for_paradigm_analysis(paradigm, analysis)

        so that the profile segment lands above the paradigm in both output trees
        (``results/<profile>/tables/<paradigm>/<module>``). ``for_paradigm*`` carry
        the narrowed fields through.

        Recognised keys (all optional):

        ``bands``
            Replaces the band set. Bands may redefine edges under the same name
            (e.g. a report's ``Low Gamma: [30, 45]`` vs the study's ``[30, 55]``),
            so this is a replacement, not a subset.
        ``roi_categories``
            Replaces the category→ROI map AND derives :attr:`rois` from it (the
            flattened values), which is what actually restricts the analysis. ROIs
            are the FDR family, so this changes every q-value — see
            ``FORGE/treatment/REPORT_PLAN.md`` §10b.
        ``include_hypotheses``
            Allowlist of hypothesis names to keep from the design spec.
        ``include_analyses``
            Allowlist of analyses this profile runs (see ``get_paradigm_analyses``).

        Keys consumed elsewhere (``title``, ``dvs``, ``connectivity_metrics``,
        ``emphasis``, ``circos``, ``delta_reference``) are ignored here.
        """
        block = self.raw.get(name)
        if not isinstance(block, dict):
            raise ValueError(
                f"No profile block '{name}:' in the study config "
                f"(expected a top-level mapping)."
            )

        bands = self.bands
        if "bands" in block:
            bands = {
                bname: tuple(limits) for bname, limits in block["bands"].items()
            }
            if not bands:
                raise ValueError(f"Profile '{name}': bands: is empty.")

        roi_categories = self.roi_categories
        rois = self.rois
        if "roi_categories" in block:
            roi_categories = {
                cat: list(members) for cat, members in block["roi_categories"].items()
            }
            rois = [r for members in roi_categories.values() for r in members]
            if not rois:
                raise ValueError(f"Profile '{name}': roi_categories: names no ROIs.")
            dupes = sorted({r for r in rois if rois.count(r) > 1})
            if dupes:
                raise ValueError(
                    f"Profile '{name}': ROI(s) in more than one category: "
                    f"{', '.join(dupes)}"
                )
            # The parent map is the atlas partition when the study doesn't override
            # it, so it's the authority on which ROI names exist.
            known = {r for members in self.roi_categories.values() for r in members}
            if known:
                unknown = sorted(set(rois) - known)
                if unknown:
                    raise ValueError(
                        f"Profile '{name}': unknown ROI(s) {', '.join(unknown)}. "
                        f"Known: {', '.join(sorted(known))}"
                    )

        design_spec = self.design_spec
        include_hyps = block.get("include_hypotheses")
        if include_hyps is not None:
            if design_spec is None:
                raise ValueError(
                    f"Profile '{name}': include_hypotheses set but the study "
                    f"declares no design/hypotheses block."
                )
            defined = {h.name for h in design_spec.hypotheses}
            unknown = sorted(set(include_hyps) - defined)
            if unknown:
                raise ValueError(
                    f"Profile '{name}': unknown hypothesis/hypotheses "
                    f"{', '.join(unknown)}. Defined: {', '.join(sorted(defined))}"
                )
            keep = set(include_hyps)
            design_spec = replace(
                design_spec,
                hypotheses=[h for h in design_spec.hypotheses if h.name in keep],
            )

        include_analyses = block.get("include_analyses")
        if include_analyses is not None:
            include_analyses = list(include_analyses)
            if not include_analyses:
                raise ValueError(f"Profile '{name}': include_analyses: is empty.")

        return replace(
            self,
            name=f"{self.name} — {name}",
            # Profile segment sits above the paradigm in both trees. The default
            # profile never calls this, so its paths are unchanged.
            output_dir=self.output_dir / name,
            results_dir=self.results_dir / name,
            bands=bands,
            roi_categories=roi_categories,
            rois=rois,
            include_analyses=include_analyses,
            design_spec=design_spec,
            profile_name=name,
        )

    def for_paradigm(self, name: str) -> StudyConfig:
        """Return a paradigm-scoped config suitable for StudyAnalyzer.

        Merges paradigm-specific fields (discovery, evoked, vertex,
        output_dir) over the shared top-level fields.  The returned config
        has ``paradigms={}`` so downstream code sees a plain single-paradigm
        config.
        """
        if name not in self.paradigms:
            available = ", ".join(self.paradigms.keys())
            raise ValueError(
                f"Unknown paradigm '{name}'. Available: {available}"
            )

        pdata = self.paradigms[name]

        # Build discovery from paradigm fields
        discovery: dict[str, Any] = {}
        if "data_dir" in pdata:
            discovery["root_dir"] = pdata["data_dir"]
        if "data_subdir" in pdata:
            discovery["data_subdir"] = pdata["data_subdir"]
        elif "data_subdir" in self.discovery:
            discovery["data_subdir"] = self.discovery["data_subdir"]
        if "subjects" in pdata:
            discovery["subject_groups"] = pdata["subjects"]
        elif "subject_groups" in self.discovery:
            # Inherit global subject list when paradigm doesn't override it
            discovery["subject_groups"] = self.discovery["subject_groups"]
        if "required_files" in pdata:
            discovery["required_files"] = pdata["required_files"]

        return StudyConfig(
            name=f"{self.name} — {name}",
            output_dir=self.output_dir / name,
            results_dir=self.results_dir,
            groups=self.groups,
            group_order=self.group_order,
            group_colors=self.group_colors,
            group_linetypes=self.group_linetypes,
            bands=self.bands,
            roi_categories=self.roi_categories,
            discovery=discovery,
            vertex_filter=pdata.get("vertex_filter", self.vertex_filter),
            vertex=_resolve_vertex_config(pdata, self.vertex),
            electrode=pdata.get("electrode", self.electrode),
            evoked=pdata.get("evoked", self.evoked),
            paradigm_name=name,
            design_spec=self.design_spec,
            # Carry profile narrowing through — for_profile() is applied to the root
            # config, so these must survive paradigm scoping.
            rois=self.rois,
            include_analyses=self.include_analyses,
            profile_name=self.profile_name,
            raw={**self.raw, **pdata},
        )

    # Keys that control subject discovery, not analysis-specific config
    _DISCOVERY_KEYS = {"data_dir", "required_files", "subjects", "vertex_filter",
                       "data_subdir"}

    def for_paradigm_analysis(
        self, paradigm: str, analysis: str,
    ) -> StudyConfig:
        """Return a config scoped to a paradigm + specific analysis.

        When ``analyses`` is a dict, each analysis entry can declare its own
        ``data_dir``, ``required_files``, ``vertex_filter``, etc.  These
        override paradigm-level values.  Non-discovery keys in the analysis
        entry are placed into ``raw[analysis]`` so the analysis class finds
        them via ``config.raw.get(analysis, {})``.
        """
        if paradigm not in self.paradigms:
            available = ", ".join(self.paradigms.keys())
            raise ValueError(
                f"Unknown paradigm '{paradigm}'. Available: {available}"
            )

        pdata = self.paradigms[paradigm]
        analyses_raw = pdata.get("analyses", {})

        # Get the per-analysis config dict
        if isinstance(analyses_raw, dict):
            analysis_cfg = dict(analyses_raw.get(analysis, {}))
        else:
            # List format — analysis-specific config at paradigm level
            analysis_cfg = dict(pdata.get(analysis, {}))

        # Separate discovery keys from analysis-specific config
        a_data_dir = analysis_cfg.pop("data_dir", None)
        a_required = analysis_cfg.pop("required_files", None)
        a_subjects = analysis_cfg.pop("subjects", None)
        a_vfilter = analysis_cfg.pop("vertex_filter", None)
        a_subdir = analysis_cfg.pop("data_subdir", None)

        # Build discovery: analysis > paradigm > top-level
        discovery: dict[str, Any] = {}

        data_dir = a_data_dir or pdata.get("data_dir")
        if data_dir:
            discovery["root_dir"] = data_dir

        data_subdir = a_subdir or pdata.get("data_subdir")
        if not data_subdir:
            data_subdir = self.discovery.get("data_subdir")
        if data_subdir:
            discovery["data_subdir"] = data_subdir

        subjects = a_subjects or pdata.get("subjects") or self.discovery.get("subject_groups")
        if subjects:
            discovery["subject_groups"] = subjects

        required_files = a_required or pdata.get("required_files")
        if required_files:
            discovery["required_files"] = required_files

        # vertex_filter: analysis > paradigm > top-level
        vertex_filter = a_vfilter or pdata.get("vertex_filter", self.vertex_filter)
        if not isinstance(vertex_filter, dict):
            vertex_filter = {}

        # Build raw dict — merge analysis config under the analysis name
        raw = {**self.raw, **pdata}
        if analysis_cfg:
            raw[analysis] = analysis_cfg

        # Dedicated config attributes: read from raw (which now has
        # analysis config merged in) falling back to paradigm then top-level
        wholebrain = _resolve_vertex_config(raw, self.vertex)
        evoked = raw.get("evoked", self.evoked)
        electrode = raw.get("electrode", self.electrode)

        return StudyConfig(
            name=f"{self.name} — {paradigm}",
            output_dir=self.output_dir / paradigm,
            results_dir=self.results_dir,
            groups=self.groups,
            group_order=self.group_order,
            group_colors=self.group_colors,
            group_linetypes=self.group_linetypes,
            bands=self.bands,
            roi_categories=self.roi_categories,
            discovery=discovery,
            vertex_filter=vertex_filter,
            vertex=wholebrain if isinstance(wholebrain, dict) else {},
            electrode=electrode if isinstance(electrode, dict) else {},
            evoked=evoked if isinstance(evoked, dict) else {},
            paradigm_name=paradigm,
            design_spec=self.design_spec,
            # Carry profile narrowing through (see for_paradigm).
            rois=self.rois,
            include_analyses=self.include_analyses,
            profile_name=self.profile_name,
            raw=raw,
        )

    @property
    def contrasts(self) -> list[Contrast]:
        """Pairwise contrasts derived on demand from the design spec.

        Replaces the former stored ``contrasts`` field (the legacy bridge). The
        single source of truth is ``design_spec`` — a legacy ``contrasts:`` block is
        lifted into it by :meth:`DesignSpec.from_dict`, so this covers both modern and
        unmigrated configs. Returns the pairwise contrast/equivalence set via
        :func:`_contrasts_from_design_spec`.
        """
        return _contrasts_from_design_spec(self.design_spec) if self.design_spec else []

    def referenced_groups(self) -> set[str]:
        """Every group named by a contrast or hypothesis (for subject discovery).

        Unions the legacy ``contrasts`` (group_a/group_b) with the ``design_spec``
        hypotheses (weights keys, group sets). A 4-group omnibus thus pulls in all
        four groups, which a pairwise-only scan would miss.
        """
        g: set[str] = set()
        for c in self.contrasts:
            g.add(c.group_a)
            g.add(c.group_b)
        if self.design_spec:
            for h in self.design_spec.hypotheses:
                g |= h.referenced_groups()
        return g

    def get_group_label(self, group_id: str) -> str:
        """Return the display label for a group, falling back to the ID."""
        return self.groups.get(group_id, group_id)

    def get_band_limits(self, band_name: str) -> tuple[float, float]:
        """Return (fmin, fmax) for a named band."""
        return self.bands[band_name]

    def get_vertex_mask(self, coords: np.ndarray) -> np.ndarray:
        """Return boolean mask for vertices passing the vertex_filter.

        Parameters
        ----------
        coords : ndarray, shape (n_vertices, 3)
            Vertex coordinates (x, y, z) in mm.

        Returns
        -------
        mask : ndarray of bool, shape (n_vertices,)
            True for vertices that pass all filter criteria.
        """
        mask = np.ones(len(coords), dtype=bool)
        vf = self.vertex_filter
        if not vf:
            return mask

        for axis, idx in [("x", 0), ("y", 1), ("z", 2)]:
            if f"{axis}_min" in vf:
                mask &= coords[:, idx] >= vf[f"{axis}_min"]
            if f"{axis}_max" in vf:
                mask &= coords[:, idx] <= vf[f"{axis}_max"]

        return mask

    @property
    def has_vertex_filter(self) -> bool:
        """True if any vertex filter is configured."""
        return bool(self.vertex_filter)

    @property
    def wholebrain(self) -> dict[str, Any]:
        """Backward-compatible alias for ``vertex``."""
        _warnings.warn(
            "StudyConfig.wholebrain is deprecated, use .vertex instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.vertex

    def validate(self) -> list[str]:
        """Check configuration for common errors. Returns list of warnings."""
        warnings = []
        if not self.groups:
            warnings.append("No groups defined")
        if not self.contrasts:
            warnings.append("No contrasts defined")
        if not self.bands:
            warnings.append("No frequency bands defined")
        for c in self.contrasts:
            if c.group_a not in self.groups:
                warnings.append(f"Contrast '{c.name}': group_a '{c.group_a}' not in groups")
            if c.group_b not in self.groups:
                warnings.append(f"Contrast '{c.name}': group_b '{c.group_b}' not in groups")
        discovery_root = self.discovery.get("root_dir")
        if discovery_root and not Path(discovery_root).exists():
            warnings.append(f"Discovery root_dir does not exist: {discovery_root}")
        return warnings


def _resolve_vertex_config(
    data: dict, fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the vertex config section, accepting both 'vertex' and legacy 'wholebrain' keys."""
    if "vertex" in data:
        return data["vertex"]
    if "wholebrain" in data:
        return data["wholebrain"]
    return fallback if fallback is not None else {}

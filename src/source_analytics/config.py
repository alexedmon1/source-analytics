"""YAML-driven study configuration loader."""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
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


@dataclass
class Contrast:
    """A between-group contrast for statistical testing."""

    name: str
    group_a: str
    group_b: str


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
    contrasts: list[Contrast]
    bands: dict[str, tuple[float, float]]
    roi_categories: dict[str, list[str]]
    discovery: dict[str, Any]
    vertex_filter: dict[str, Any] = field(default_factory=dict)
    vertex: dict[str, Any] = field(default_factory=dict)
    electrode: dict[str, Any] = field(default_factory=dict)
    evoked: dict[str, Any] = field(default_factory=dict)
    paradigms: dict[str, dict] = field(default_factory=dict, repr=False)
    paradigm_name: str | None = None
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

        contrasts = [
            Contrast(name=c["name"], group_a=c["group_a"], group_b=c["group_b"])
            for c in data.get("contrasts", [])
        ]

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
            contrasts=contrasts,
            bands=bands,
            roi_categories=roi_categories,
            discovery=discovery,
            vertex_filter=data.get("vertex_filter", {}),
            vertex=_resolve_vertex_config(data),
            electrode=data.get("electrode", {}),
            evoked=data.get("evoked", {}),
            paradigms=paradigms,
            raw=data,
        )

    @classmethod
    def _from_legacy_yaml(cls, data: dict, config_dir: Path) -> StudyConfig:
        """Parse legacy analytics YAML format."""
        # Resolve output_dir: explicit or default to config file's directory
        output_dir = Path(data["output_dir"]) if "output_dir" in data else config_dir

        # Resolve results_dir: explicit or default to sibling results/
        if "results_dir" in data:
            results_dir = Path(data["results_dir"])
        else:
            results_dir = output_dir.parent / "results"

        # Resolve discovery.root_dir: explicit or default to sibling derivatives/
        discovery = data.get("discovery", {})
        if "root_dir" not in discovery:
            discovery["root_dir"] = str(config_dir.parent / "derivatives")

        contrasts = [
            Contrast(name=c["name"], group_a=c["group_a"], group_b=c["group_b"])
            for c in data.get("contrasts", [])
        ]

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
            contrasts=contrasts,
            bands=bands,
            roi_categories=data.get("roi_categories", {}),
            discovery=discovery,
            vertex_filter=data.get("vertex_filter", {}),
            vertex=_resolve_vertex_config(data),
            electrode=data.get("electrode", {}),
            evoked=data.get("evoked", {}),
            paradigms=paradigms,
            raw=data,
        )

    @property
    def has_paradigms(self) -> bool:
        """True if this config defines multiple paradigms."""
        return bool(self.paradigms)

    def get_paradigm_analyses(self, name: str) -> list[str] | None:
        """Return the analyses list for a paradigm, or None."""
        pdata = self.paradigms.get(name)
        if pdata is None:
            return None
        analyses = pdata.get("analyses")
        if isinstance(analyses, dict):
            return list(analyses.keys())
        return analyses

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
        if "required_files" in pdata:
            discovery["required_files"] = pdata["required_files"]

        return StudyConfig(
            name=f"{self.name} — {name}",
            output_dir=self.output_dir / name,
            results_dir=self.results_dir,
            groups=self.groups,
            group_order=self.group_order,
            group_colors=self.group_colors,
            contrasts=self.contrasts,
            bands=self.bands,
            roi_categories=self.roi_categories,
            discovery=discovery,
            vertex_filter=pdata.get("vertex_filter", self.vertex_filter),
            vertex=_resolve_vertex_config(pdata, self.vertex),
            electrode=pdata.get("electrode", self.electrode),
            evoked=pdata.get("evoked", self.evoked),
            paradigm_name=name,
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

        subjects = a_subjects or pdata.get("subjects")
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
            contrasts=self.contrasts,
            bands=self.bands,
            roi_categories=self.roi_categories,
            discovery=discovery,
            vertex_filter=vertex_filter,
            vertex=wholebrain if isinstance(wholebrain, dict) else {},
            electrode=electrode if isinstance(electrode, dict) else {},
            evoked=evoked if isinstance(evoked, dict) else {},
            paradigm_name=paradigm,
            raw=raw,
        )

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

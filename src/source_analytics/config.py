"""YAML-driven study configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


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
        Root directory for analysis outputs.
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
    groups: dict[str, str]
    group_order: list[str]
    group_colors: dict[str, str]
    contrasts: list[Contrast]
    bands: dict[str, tuple[float, float]]
    roi_categories: dict[str, list[str]]
    discovery: dict[str, Any]
    vertex_filter: dict[str, Any] = field(default_factory=dict)
    wholebrain: dict[str, Any] = field(default_factory=dict)
    electrode: dict[str, Any] = field(default_factory=dict)
    evoked: dict[str, Any] = field(default_factory=dict)
    paradigms: dict[str, dict] = field(default_factory=dict, repr=False)
    paradigm_name: str | None = None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        """Load a study config from a YAML file.

        Path defaults (when keys are absent from the YAML):
        - ``output_dir`` → directory containing the YAML file
        - ``discovery.root_dir`` → ``../derivatives`` relative to the YAML file
        """
        path = Path(path).resolve()
        config_dir = path.parent

        with open(path) as f:
            data = yaml.safe_load(f)

        # Resolve output_dir: explicit or default to config file's directory
        output_dir = Path(data["output_dir"]) if "output_dir" in data else config_dir

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
            paradigms[pname] = pcopy

        return cls(
            name=data["name"],
            output_dir=output_dir,
            groups=data.get("groups", {}),
            group_order=data.get("group_order", list(data.get("groups", {}).keys())),
            group_colors=data.get("group_colors", {}),
            contrasts=contrasts,
            bands=bands,
            roi_categories=data.get("roi_categories", {}),
            discovery=discovery,
            vertex_filter=data.get("vertex_filter", {}),
            wholebrain=data.get("wholebrain", {}),
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
        return pdata.get("analyses")

    def for_paradigm(self, name: str) -> StudyConfig:
        """Return a paradigm-scoped config suitable for StudyAnalyzer.

        Merges paradigm-specific fields (discovery, evoked, wholebrain,
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
            groups=self.groups,
            group_order=self.group_order,
            group_colors=self.group_colors,
            contrasts=self.contrasts,
            bands=self.bands,
            roi_categories=self.roi_categories,
            discovery=discovery,
            vertex_filter=pdata.get("vertex_filter", self.vertex_filter),
            wholebrain=pdata.get("wholebrain", self.wholebrain),
            electrode=pdata.get("electrode", self.electrode),
            evoked=pdata.get("evoked", self.evoked),
            paradigm_name=name,
            raw=self.raw,
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

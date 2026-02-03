"""YAML-driven study configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        """Load a study config from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        contrasts = [
            Contrast(name=c["name"], group_a=c["group_a"], group_b=c["group_b"])
            for c in data.get("contrasts", [])
        ]

        bands = {
            name: tuple(limits) for name, limits in data.get("bands", {}).items()
        }

        return cls(
            name=data["name"],
            output_dir=Path(data["output_dir"]),
            groups=data.get("groups", {}),
            group_order=data.get("group_order", list(data.get("groups", {}).keys())),
            group_colors=data.get("group_colors", {}),
            contrasts=contrasts,
            bands=bands,
            roi_categories=data.get("roi_categories", {}),
            discovery=data.get("discovery", {}),
            raw=data,
        )

    def get_group_label(self, group_id: str) -> str:
        """Return the display label for a group, falling back to the ID."""
        return self.groups.get(group_id, group_id)

    def get_band_limits(self, band_name: str) -> tuple[float, float]:
        """Return (fmin, fmax) for a named band."""
        return self.bands[band_name]

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

"""StudyAnalyzer: orchestrates analysis modules on discovered subjects."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import StudyConfig
from .io.discovery import SubjectInfo, discover_subjects
from .analyses.base import BaseAnalysis
from .analyses.psd_analysis import PSDAnalysis
from .analyses.aperiodic_analysis import AperiodicAnalysis
from .analyses.connectivity_analysis import ConnectivityAnalysis
from .analyses.pac_analysis import PACAnalysis
from .analyses.wholebrain_analysis import WholebrainAnalysis
from .analyses.electrode_analysis import ElectrodeAnalysis
from .analyses.electrode_comparison_analysis import ElectrodeComparisonAnalysis

logger = logging.getLogger(__name__)

# Registry of available analyses
ANALYSIS_REGISTRY: dict[str, type[BaseAnalysis]] = {
    "psd": PSDAnalysis,
    "aperiodic": AperiodicAnalysis,
    "connectivity": ConnectivityAnalysis,
    "pac": PACAnalysis,
    "wholebrain": WholebrainAnalysis,
    "electrode": ElectrodeAnalysis,
    "electrode_comparison": ElectrodeComparisonAnalysis,
}


class StudyAnalyzer:
    """Orchestrates analysis modules for a study.

    Parameters
    ----------
    config : StudyConfig
        Study configuration.
    subjects : list[SubjectInfo], optional
        Pre-discovered subjects. If None, auto-discovers from config.
    """

    def __init__(
        self,
        config: StudyConfig,
        subjects: list[SubjectInfo] | None = None,
    ):
        self.config = config
        self.subjects = subjects or self._discover()

    def _discover(self) -> list[SubjectInfo]:
        root_dir = self.config.discovery.get("root_dir")
        if not root_dir:
            raise ValueError("No discovery.root_dir in study config")

        group_mapping = self.config.discovery.get("group_mapping", {})
        required_files = self.config.discovery.get("required_files")
        return discover_subjects(
            root_dir,
            group_mapping=group_mapping,
            required_files=required_files,
        )

    def get_subjects_for_groups(self, groups: list[str]) -> list[SubjectInfo]:
        """Filter subjects to only those in the specified groups."""
        return [s for s in self.subjects if s.group in groups]

    def run_analysis(self, analysis_name: str) -> None:
        """Run a single named analysis."""
        if analysis_name not in ANALYSIS_REGISTRY:
            available = ", ".join(ANALYSIS_REGISTRY.keys())
            raise ValueError(f"Unknown analysis '{analysis_name}'. Available: {available}")

        cls = ANALYSIS_REGISTRY[analysis_name]
        output_dir = self.config.output_dir / analysis_name
        analysis = cls(self.config, output_dir)

        # Filter subjects to only groups referenced in contrasts
        contrast_groups = set()
        for c in self.config.contrasts:
            contrast_groups.add(c.group_a)
            contrast_groups.add(c.group_b)

        if contrast_groups:
            subjects = self.get_subjects_for_groups(list(contrast_groups))
        else:
            subjects = self.subjects

        logger.info(
            "Running '%s' on %d subjects (%d groups)",
            analysis_name, len(subjects), len(set(s.group for s in subjects)),
        )
        analysis.run(subjects)

    def validate(self) -> list[str]:
        """Validate the study configuration and subject discovery."""
        issues = self.config.validate()

        if not self.subjects:
            issues.append("No subjects discovered")
        else:
            # Check group coverage
            discovered_groups = set(s.group for s in self.subjects)
            for c in self.config.contrasts:
                if c.group_a not in discovered_groups:
                    issues.append(
                        f"Contrast '{c.name}': no subjects found for group '{c.group_a}'"
                    )
                if c.group_b not in discovered_groups:
                    issues.append(
                        f"Contrast '{c.name}': no subjects found for group '{c.group_b}'"
                    )

        return issues

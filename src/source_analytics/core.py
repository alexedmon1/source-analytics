"""StudyAnalyzer: orchestrates analysis modules on discovered subjects."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import StudyConfig
from .io.discovery import SubjectInfo, discover_subjects
from .analyses.base import BaseAnalysis
from .analyses.psd_analysis import PSDAnalysis
from .analyses.aperiodic_analysis import AperiodicAnalysis
from .analyses.roi_connectivity_analysis import ConnectivityAnalysis
from .analyses.pac_analysis import PACAnalysis
from .analyses.wholebrain_analysis import WholebrainAnalysis
from .analyses.electrode_analysis import ElectrodeAnalysis
from .analyses.electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .analyses.vertex_connectivity_analysis import VertexConnectivityAnalysis
from .analyses.specparam_vertex_analysis import SpecparamVertexAnalysis
from .analyses.mvpa_analysis import MVPAAnalysis
from .analyses.roi_network_analysis import ROINetworkAnalysis
from .analyses.vertex_network_analysis import VertexNetworkAnalysis
from .analyses.spatial_lmm_analysis import SpatialLMMAnalysis
from .analyses.transfer_entropy_analysis import TransferEntropyAnalysis
from .analyses.evoked_analysis import EvokedAnalysis

logger = logging.getLogger(__name__)

# Registry of available analyses
ANALYSIS_REGISTRY: dict[str, type[BaseAnalysis]] = {
    "psd": PSDAnalysis,
    "aperiodic": AperiodicAnalysis,
    "roi_connectivity": ConnectivityAnalysis,
    "pac": PACAnalysis,
    "wholebrain": WholebrainAnalysis,
    "electrode": ElectrodeAnalysis,
    "electrode_comparison": ElectrodeComparisonAnalysis,
    "vertex_connectivity": VertexConnectivityAnalysis,
    "specparam_vertex": SpecparamVertexAnalysis,
    "mvpa": MVPAAnalysis,
    "roi_network": ROINetworkAnalysis,
    "vertex_network": VertexNetworkAnalysis,
    "spatial_lmm": SpatialLMMAnalysis,
    "transfer_entropy": TransferEntropyAnalysis,
    "evoked": EvokedAnalysis,
}

# Metadata for grouping and display
ANALYSIS_METADATA: dict[str, dict[str, str]] = {
    "psd":                  {"category": "resting", "level": "roi",        "description": "Power spectral density"},
    "aperiodic":            {"category": "resting", "level": "roi",        "description": "1/f aperiodic decomposition"},
    "roi_connectivity":     {"category": "resting", "level": "roi",        "description": "ROI pairwise connectivity"},
    "pac":                  {"category": "resting", "level": "roi",        "description": "Phase-amplitude coupling"},
    "roi_network":          {"category": "resting", "level": "roi",        "description": "ROI-level graph theory network metrics"},
    "vertex_network":       {"category": "resting", "level": "wholebrain", "description": "Vertex-level graph theory network metrics"},
    "transfer_entropy":     {"category": "resting", "level": "roi",        "description": "Directed information flow"},
    "mvpa":                 {"category": "resting", "level": "wholebrain", "description": "SVM pattern classification"},
    "wholebrain":           {"category": "resting", "level": "wholebrain", "description": "Vertex-level cluster permutation"},
    "spatial_lmm":          {"category": "resting", "level": "wholebrain", "description": "Vertex-level LMM statistics"},
    "specparam_vertex":     {"category": "resting", "level": "wholebrain", "description": "Vertex-level spectral parameterization"},
    "vertex_connectivity":  {"category": "resting", "level": "wholebrain", "description": "Vertex pairwise connectivity"},
    "electrode":            {"category": "resting", "level": "electrode",  "description": "Sensor-level PSD analysis"},
    "electrode_comparison": {"category": "resting", "level": "electrode",  "description": "Source vs electrode comparison"},
    "evoked":               {"category": "evoked",  "level": "roi",        "description": "ITC, ERSP, STP for trial-based paradigms"},
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
        data_subdir = self.config.discovery.get("data_subdir", "data")
        subject_groups = self.config.discovery.get("subject_groups")
        return discover_subjects(
            root_dir,
            group_mapping=group_mapping,
            required_files=required_files,
            data_subdir=data_subdir,
            subject_groups=subject_groups,
        )

    def get_subjects_for_groups(self, groups: list[str]) -> list[SubjectInfo]:
        """Filter subjects to only those in the specified groups."""
        return [s for s in self.subjects if s.group in groups]

    def run_analysis(self, analysis_name: str, steps: set[str] | None = None) -> None:
        """Run a single named analysis.

        Parameters
        ----------
        analysis_name : str
            Name of the analysis to run (must be in ANALYSIS_REGISTRY).
        steps : set[str] | None
            If provided, only run these lifecycle steps.
        """
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
        analysis.run(subjects, steps=steps)

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

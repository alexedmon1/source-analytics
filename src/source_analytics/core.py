"""StudyAnalyzer: orchestrates analysis modules on discovered subjects."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from .config import StudyConfig
from .io.discovery import SubjectInfo, discover_subjects
from .analyses.base import BaseAnalysis
from .analyses.roi_psd_analysis import ROIPsdAnalysis
from .analyses.roi_aperiodic_analysis import ROIAperiodicAnalysis
from .analyses.roi_connectivity_analysis import ConnectivityAnalysis
from .analyses.roi_cross_freq_analysis import ROICrossFreqAnalysis
from .analyses.vertex_cross_freq_analysis import VertexCrossFreqAnalysis
from .analyses.vertex_cluster_analysis import VertexClusterAnalysis
from .analyses.electrode_analysis import ElectrodeAnalysis
from .analyses.electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .analyses.electrode_connectivity_analysis import ElectrodeConnectivityAnalysis
from .analyses.vertex_connectivity_analysis import VertexConnectivityAnalysis
from .analyses.vertex_specparam_analysis import VertexSpecparamAnalysis
from .analyses.vertex_mvpa_analysis import VertexMVPAAnalysis
from .analyses.roi_network_analysis import (
    ROINetworkAnalysis,
    ROIGraphAnalysis,
    ROINBSAnalysis,
)
from .analyses.vertex_network_analysis import (
    VertexNetworkAnalysis,
    VertexGraphAnalysis,
    VertexNBSAnalysis,
)
from .analyses.vertex_spatial_analysis import VertexSpatialAnalysis
from .analyses.roi_directed_analysis import ROIDirectedAnalysis
from .analyses.vertex_directed_analysis import VertexDirectedAnalysis
from .analyses.roi_evoked_analysis import ROIEvokedAnalysis
from .analyses.vertex_evoked_analysis import VertexEvokedAnalysis
from .analyses.electrode_evoked_analysis import ElectrodeEvokedAnalysis
from .analyses.electrode_aperiodic_analysis import ElectrodeAperiodicAnalysis

logger = logging.getLogger(__name__)

# Registry of available analyses
ANALYSIS_REGISTRY: dict[str, type[BaseAnalysis]] = {
    "roi_psd": ROIPsdAnalysis,
    "roi_aperiodic": ROIAperiodicAnalysis,
    "roi_connectivity": ConnectivityAnalysis,
    "roi_cross_freq": ROICrossFreqAnalysis,
    "vertex_cluster": VertexClusterAnalysis,
    "electrode_psd": ElectrodeAnalysis,
    "electrode_aperiodic": ElectrodeAperiodicAnalysis,
    "electrode_comparison": ElectrodeComparisonAnalysis,
    "electrode_connectivity": ElectrodeConnectivityAnalysis,
    "vertex_connectivity": VertexConnectivityAnalysis,
    "vertex_cross_freq": VertexCrossFreqAnalysis,
    "vertex_specparam": VertexSpecparamAnalysis,
    "vertex_mvpa": VertexMVPAAnalysis,
    "roi_graph": ROIGraphAnalysis,
    "roi_nbs": ROINBSAnalysis,
    "vertex_graph": VertexGraphAnalysis,
    "vertex_nbs": VertexNBSAnalysis,
    "roi_network": ROINetworkAnalysis,      # combined alias (graph + NBS)
    "vertex_network": VertexNetworkAnalysis,  # combined alias (graph + NBS)
    "vertex_spatial": VertexSpatialAnalysis,
    "roi_directed": ROIDirectedAnalysis,
    "vertex_directed": VertexDirectedAnalysis,
    "roi_evoked": ROIEvokedAnalysis,
    "vertex_evoked": VertexEvokedAnalysis,
    "electrode_evoked": ElectrodeEvokedAnalysis,
}

# Backward-compatibility aliases (old name -> new name)
_DEPRECATED_NAMES: dict[str, str] = {
    "psd": "roi_psd",
    "aperiodic": "roi_aperiodic",
    "pac": "roi_cross_freq",
    "roi_pac": "roi_cross_freq",
    "wholebrain": "vertex_cluster",
    "spatial_lmm": "vertex_spatial",
    "specparam_vertex": "vertex_specparam",
    "mvpa": "vertex_mvpa",
    "transfer_entropy": "roi_directed",
    "roi_transfer_entropy": "roi_directed",
    "evoked": "roi_evoked",
    "electrode": "electrode_psd",
}

# Register aliases so old YAML configs still work
for _old, _new in _DEPRECATED_NAMES.items():
    ANALYSIS_REGISTRY[_old] = ANALYSIS_REGISTRY[_new]


def resolve_analysis_name(name: str) -> str:
    """Resolve a possibly-deprecated analysis name to the canonical name.

    Emits a deprecation warning if an old name is used.
    """
    if name in _DEPRECATED_NAMES:
        canonical = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"Analysis name '{name}' is deprecated, use '{canonical}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    return name


# Metadata for grouping and display.
#   domain      = how analyses are grouped/listed (by the data they use)
#   supplements = a SECONDARY analysis that can only run after the named primary
#                 (it consumes the primary's output). Absent = primary.
ANALYSIS_METADATA: dict[str, dict[str, str]] = {
    "roi_psd":              {"category": "resting", "level": "roi",        "domain": "Spectral",        "description": "PSD (power spectral density)"},
    "roi_aperiodic":        {"category": "resting", "level": "roi",        "domain": "Spectral",        "description": "1/f aperiodic decomposition"},
    "roi_connectivity":     {"category": "resting", "level": "roi",        "domain": "Connectivity",    "description": "ROI pairwise connectivity"},
    "roi_cross_freq":       {"category": "resting", "level": "roi",        "domain": "Cross-frequency", "description": "Cross-frequency coupling (PAC, AAC, n:m PPC)"},
    "roi_graph":            {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level graph-theoretic metrics (degree/clustering/betweenness)"},
    "roi_nbs":              {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level Network-Based Statistic (sub-network test)"},
    "vertex_graph":         {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level multi-density AUC graph metrics"},
    "vertex_nbs":           {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level Network-Based Statistic (sub-network test)"},
    "roi_network":          {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level graph theory + NBS (combined alias of roi_graph + roi_nbs)"},
    "vertex_network":       {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level graph theory + NBS (combined alias of vertex_graph + vertex_nbs)"},
    "roi_directed":         {"category": "resting", "level": "roi",        "domain": "Directed",        "description": "Directed connectivity (transfer entropy + DTF)"},
    "vertex_directed":      {"category": "resting", "level": "vertex",     "domain": "Directed",        "description": "Vertex DTF outflow/inflow/netflow (ridge-MVAR, cluster-corrected)"},
    "vertex_mvpa":          {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "MVPA (SVM pattern classification)"},
    "vertex_cluster":       {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "Vertex-level cluster permutation"},
    "vertex_spatial":       {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "Spatial GLS (vertex-level generalized least squares)"},
    "vertex_specparam":     {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "Vertex-level spectral parameterization"},
    "vertex_connectivity":  {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "description": "Vertex pairwise connectivity"},
    "vertex_cross_freq":    {"category": "resting", "level": "vertex",     "domain": "Cross-frequency", "description": "Vertex cross-frequency coupling (local PAC, AAC, n:m PPC)"},
    "electrode_psd":        {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "description": "Sensor-level PSD analysis"},
    "electrode_aperiodic":  {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "description": "Sensor-level aperiodic (1/f) analysis"},
    "electrode_comparison": {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "supplements": "electrode_psd",       "description": "Source vs electrode comparison"},
    "electrode_connectivity": {"category": "resting", "level": "electrode",  "domain": "Connectivity",    "description": "Sensor pairwise connectivity + FCD (source-vs-sensor comparator)"},
    "roi_evoked":           {"category": "evoked",  "level": "roi",        "domain": "Evoked",          "description": "ITC, ERSP, STP for trial-based paradigms"},
    "vertex_evoked":        {"category": "evoked",  "level": "vertex",     "domain": "Evoked",          "description": "Vertex-level ITC, ERSP, STP (cluster-corrected) for trial-based paradigms"},
    "electrode_evoked":     {"category": "evoked",  "level": "electrode",  "domain": "Evoked",          "description": "Electrode-level ITC, ERSP, STP for trial-based paradigms"},
}

# Add metadata entries for deprecated aliases (point to same metadata)
for _old, _new in _DEPRECATED_NAMES.items():
    if _new in ANALYSIS_METADATA:
        ANALYSIS_METADATA[_old] = ANALYSIS_METADATA[_new]


def analysis_meta(include_aliases: bool = False) -> dict[str, dict[str, str]]:
    """Return a copy of the analysis metadata (domain / supplements / etc.).

    Stable accessor for external tools (e.g. the gallery builder reads this via
    the source-analytics interpreter to group analyses by domain and nest each
    secondary under the primary it ``supplements``). Canonical names only unless
    ``include_aliases`` is set.
    """
    return {
        name: dict(meta)
        for name, meta in ANALYSIS_METADATA.items()
        if include_aliases or name not in _DEPRECATED_NAMES
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

    def run_analysis(
        self,
        analysis_name: str,
        steps: set[str] | None = None,
        select: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Run a single named analysis.

        Parameters
        ----------
        analysis_name : str
            Name of the analysis to run (must be in ANALYSIS_REGISTRY).
        steps : set[str] | None
            If provided, only run these lifecycle steps.
        select : dict[str, frozenset[str]] | None
            If provided, restrict each module's sub-outputs (metric/band/...) to
            the requested members. Forwarded to ``BaseAnalysis.run``.
        """
        if analysis_name not in ANALYSIS_REGISTRY:
            available = ", ".join(
                k for k in ANALYSIS_REGISTRY.keys() if k not in _DEPRECATED_NAMES
            )
            raise ValueError(f"Unknown analysis '{analysis_name}'. Available: {available}")

        # Resolve deprecated name (with warning) and use canonical output dir
        canonical_name = resolve_analysis_name(analysis_name)

        cls = ANALYSIS_REGISTRY[analysis_name]
        output_dir = self.config.output_dir / canonical_name
        analysis = cls(self.config, output_dir)

        # Filter subjects to only groups referenced in contrasts/hypotheses
        contrast_groups = self.config.referenced_groups()

        if contrast_groups:
            subjects = self.get_subjects_for_groups(list(contrast_groups))
        else:
            subjects = self.subjects

        logger.info(
            "Running '%s' on %d subjects (%d groups)",
            canonical_name, len(subjects), len(set(s.group for s in subjects)),
        )
        analysis.run(subjects, steps=steps, select=select)

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

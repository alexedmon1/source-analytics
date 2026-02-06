"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis
from .psd_analysis import PSDAnalysis
from .aperiodic_analysis import AperiodicAnalysis
from .connectivity_analysis import ConnectivityAnalysis
from .pac_analysis import PACAnalysis
from .wholebrain_analysis import WholebrainAnalysis
from .electrode_analysis import ElectrodeAnalysis
from .electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .tfce_analysis import TFCEAnalysis
from .vertex_connectivity_analysis import VertexConnectivityAnalysis
from .specparam_vertex_analysis import SpecparamVertexAnalysis
from .mvpa_analysis import MVPAAnalysis
from .network_analysis import NetworkAnalysis
from .spatial_lmm_analysis import SpatialLMMAnalysis

__all__ = [
    "BaseAnalysis",
    "PSDAnalysis",
    "AperiodicAnalysis",
    "ConnectivityAnalysis",
    "PACAnalysis",
    "WholebrainAnalysis",
    "ElectrodeAnalysis",
    "ElectrodeComparisonAnalysis",
    "TFCEAnalysis",
    "VertexConnectivityAnalysis",
    "SpecparamVertexAnalysis",
    "MVPAAnalysis",
    "NetworkAnalysis",
    "SpatialLMMAnalysis",
]

"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis, find_r_script_dir
from .psd_analysis import PSDAnalysis
from .aperiodic_analysis import AperiodicAnalysis
from .roi_connectivity_analysis import ConnectivityAnalysis
from .pac_analysis import PACAnalysis
from .wholebrain_analysis import WholebrainAnalysis
from .electrode_analysis import ElectrodeAnalysis
from .electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .vertex_connectivity_analysis import VertexConnectivityAnalysis
from .specparam_vertex_analysis import SpecparamVertexAnalysis
from .mvpa_analysis import MVPAAnalysis
from .roi_network_analysis import ROINetworkAnalysis
from .vertex_network_analysis import VertexNetworkAnalysis
from .spatial_lmm_analysis import SpatialLMMAnalysis
from .evoked_analysis import EvokedAnalysis

__all__ = [
    "BaseAnalysis",
    "find_r_script_dir",
    "PSDAnalysis",
    "AperiodicAnalysis",
    "ConnectivityAnalysis",
    "PACAnalysis",
    "WholebrainAnalysis",
    "ElectrodeAnalysis",
    "ElectrodeComparisonAnalysis",
    "VertexConnectivityAnalysis",
    "SpecparamVertexAnalysis",
    "MVPAAnalysis",
    "ROINetworkAnalysis",
    "VertexNetworkAnalysis",
    "SpatialLMMAnalysis",
    "EvokedAnalysis",
]

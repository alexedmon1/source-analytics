"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis, find_r_script_dir
from .roi_psd_analysis import ROIPsdAnalysis
from .roi_aperiodic_analysis import ROIAperiodicAnalysis
from .roi_connectivity_analysis import ConnectivityAnalysis
from .roi_cross_freq_analysis import ROICrossFreqAnalysis
from .vertex_cross_freq_analysis import VertexCrossFreqAnalysis
from .vertex_cluster_analysis import VertexClusterAnalysis
from .electrode_analysis import ElectrodeAnalysis
from .electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .electrode_connectivity_analysis import ElectrodeConnectivityAnalysis
from .vertex_connectivity_analysis import VertexConnectivityAnalysis
from .vertex_specparam_analysis import VertexSpecparamAnalysis
from .vertex_mvpa_analysis import VertexMVPAAnalysis
from .roi_network_analysis import ROINetworkAnalysis
from .vertex_network_analysis import VertexNetworkAnalysis
from .vertex_spatial_analysis import VertexSpatialAnalysis
from .roi_evoked_analysis import ROIEvokedAnalysis
from .vertex_evoked_analysis import VertexEvokedAnalysis
from .roi_directed_analysis import ROIDirectedAnalysis
from .vertex_directed_analysis import VertexDirectedAnalysis

# Backward-compatible aliases
PSDAnalysis = ROIPsdAnalysis
AperiodicAnalysis = ROIAperiodicAnalysis
ROIPacAnalysis = ROICrossFreqAnalysis  # renamed -> roi_cross_freq
PACAnalysis = ROICrossFreqAnalysis
WholebrainAnalysis = VertexClusterAnalysis
MVPAAnalysis = VertexMVPAAnalysis
SpecparamVertexAnalysis = VertexSpecparamAnalysis
SpatialLMMAnalysis = VertexSpatialAnalysis
EvokedAnalysis = ROIEvokedAnalysis
ROITransferEntropyAnalysis = ROIDirectedAnalysis  # renamed -> roi_directed
TransferEntropyAnalysis = ROIDirectedAnalysis

__all__ = [
    "BaseAnalysis",
    "find_r_script_dir",
    "ROIPsdAnalysis",
    "ROIAperiodicAnalysis",
    "ConnectivityAnalysis",
    "ROICrossFreqAnalysis",
    "VertexCrossFreqAnalysis",
    "ROIPacAnalysis",
    "VertexClusterAnalysis",
    "ElectrodeAnalysis",
    "ElectrodeComparisonAnalysis",
    "ElectrodeConnectivityAnalysis",
    "VertexConnectivityAnalysis",
    "VertexSpecparamAnalysis",
    "VertexMVPAAnalysis",
    "ROINetworkAnalysis",
    "VertexNetworkAnalysis",
    "VertexSpatialAnalysis",
    "ROIEvokedAnalysis",
    "VertexEvokedAnalysis",
    "ROIDirectedAnalysis",
    "VertexDirectedAnalysis",
    "ROITransferEntropyAnalysis",
    # Backward-compatible aliases
    "PSDAnalysis",
    "AperiodicAnalysis",
    "PACAnalysis",
    "WholebrainAnalysis",
    "MVPAAnalysis",
    "SpecparamVertexAnalysis",
    "SpatialLMMAnalysis",
    "EvokedAnalysis",
    "TransferEntropyAnalysis",
]

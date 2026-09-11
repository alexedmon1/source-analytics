"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis, find_r_script_dir
from .roi_psd_analysis import ROIPsdAnalysis
from .roi_aperiodic_analysis import ROIAperiodicAnalysis
from .roi_connectivity_analysis import ConnectivityAnalysis
from .roi_cross_freq_analysis import ROICrossFreqAnalysis
from .electrode_analysis import ElectrodeAnalysis
from .electrode_comparison_analysis import ElectrodeComparisonAnalysis
from .electrode_connectivity_analysis import ElectrodeConnectivityAnalysis
from .roi_network_analysis import ROINetworkAnalysis
from .roi_evoked_analysis import ROIEvokedAnalysis
from .roi_directed_analysis import ROIDirectedAnalysis

# Backward-compatible aliases
PSDAnalysis = ROIPsdAnalysis
AperiodicAnalysis = ROIAperiodicAnalysis
ROIPacAnalysis = ROICrossFreqAnalysis  # renamed -> roi_cross_freq
PACAnalysis = ROICrossFreqAnalysis
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
    "ROIPacAnalysis",
    "ElectrodeAnalysis",
    "ElectrodeComparisonAnalysis",
    "ElectrodeConnectivityAnalysis",
    "ROINetworkAnalysis",
    "ROIEvokedAnalysis",
    "ROIDirectedAnalysis",
    "ROITransferEntropyAnalysis",
    # Backward-compatible aliases
    "PSDAnalysis",
    "AperiodicAnalysis",
    "PACAnalysis",
    "EvokedAnalysis",
    "TransferEntropyAnalysis",
]

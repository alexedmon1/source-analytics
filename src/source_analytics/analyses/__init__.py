"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis
from .psd_analysis import PSDAnalysis
from .aperiodic_analysis import AperiodicAnalysis
from .connectivity_analysis import ConnectivityAnalysis
from .pac_analysis import PACAnalysis

__all__ = ["BaseAnalysis", "PSDAnalysis", "AperiodicAnalysis", "ConnectivityAnalysis", "PACAnalysis"]

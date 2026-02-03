"""Analysis modules: pluggable analysis pipelines."""

from .base import BaseAnalysis
from .psd_analysis import PSDAnalysis

__all__ = ["BaseAnalysis", "PSDAnalysis"]

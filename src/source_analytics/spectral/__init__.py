"""Spectral analysis: PSD, band power extraction, aperiodic fitting, connectivity, and PAC."""

from .psd import compute_psd
from .band_power import extract_band_power
from .aperiodic import fit_aperiodic
from .connectivity import compute_connectivity_matrix
from .pac import compute_pac, compute_pac_zscore, compute_pac_multiroi, get_valid_pac_pairs
from .vertex import (
    compute_psd_vertices,
    extract_band_power_vertices,
    compute_falff,
    compute_spectral_slope,
    compute_peak_frequency,
)
from .epoch_sampler import sample_epochs, get_epoch_config
from .vertex_aperiodic import fit_aperiodic_vertices
from .vertex_connectivity import (
    compute_vertex_connectivity_matrix,
    compute_vertex_connectivity_matrix_epochs,
    compute_fcd,
    compute_seed_connectivity,
)

__all__ = [
    "compute_psd",
    "extract_band_power",
    "fit_aperiodic",
    "compute_connectivity_matrix",
    "compute_pac",
    "compute_pac_zscore",
    "compute_pac_multiroi",
    "get_valid_pac_pairs",
    "compute_psd_vertices",
    "extract_band_power_vertices",
    "compute_falff",
    "compute_spectral_slope",
    "compute_peak_frequency",
    "sample_epochs",
    "get_epoch_config",
    "fit_aperiodic_vertices",
    "compute_vertex_connectivity_matrix",
    "compute_vertex_connectivity_matrix_epochs",
    "compute_fcd",
    "compute_seed_connectivity",
]

"""I/O utilities for loading pipeline outputs."""

from .loader import MonteCarloRunError, SubjectLoader
from .discovery import discover_subjects
from .electrode_loader import load_eeglab_set
from .run_manifest import (
    RunManifest,
    parcel_caveats,
    read_monte_carlo_report,
    read_run_manifest,
    undersampled_parcels,
    unreliable_parcels,
)

__all__ = [
    "SubjectLoader",
    "MonteCarloRunError",
    "discover_subjects",
    "load_eeglab_set",
    "RunManifest",
    "read_run_manifest",
    "read_monte_carlo_report",
    "parcel_caveats",
    "undersampled_parcels",
    "unreliable_parcels",
]

"""Shared visualization constants for consistent figures across all analyses."""

# Frequency bands in ascending Hz order (standard for all plots)
BAND_ORDER = ["Delta", "Theta", "Alpha", "Beta", "Low Gamma", "High Gamma"]

BAND_FREQ_RANGES = {
    "Delta": (1, 4),
    "Theta": (4, 10),
    "Alpha": (10, 13),
    "Beta": (13, 30),
    "Low Gamma": (30, 55),
    "High Gamma": (65, 100),
}

BAND_COLORS = {
    "Delta": "#1f77b4",
    "Theta": "#ff7f0e",
    "Alpha": "#2ca02c",
    "Beta": "#9467bd",
    "Low Gamma": "#d62728",
    "High Gamma": "#e377c2",
}

# Corpus Callosum ROIs — white matter tracts, excluded from all analyses
CC_ROIS = [
    "Corpus_Callosum_Genu_L",
    "Corpus_Callosum_Genu_R",
    "Corpus_Callosum_Body_L",
    "Corpus_Callosum_Body_R",
    "Corpus_Callosum_Splenium_L",
    "Corpus_Callosum_Splenium_R",
]

# Connectivity metric display labels
METRIC_LABELS = {
    "imag_coherence": "Imaginary Coherence",
    "coherence": "Coherence",
    "pli": "Phase Lag Index",
    "dwpli": "Debiased Weighted PLI",
    "aec": "Orthogonalized AEC",
    "partial_corr": "Partial Correlation",
}

# Compact display for figure titles/labels (proper acronym case), keyed by the
# lowercase metric token. Prefer this in plots over the raw column name.
METRIC_DISPLAY = {
    "imag_coherence": "Imag. coherence", "coherence": "Coherence",
    "pli": "PLI", "wpli": "wPLI", "dwpli": "dwPLI", "dpli": "dPLI",
    "aec": "AEC", "partial_corr": "Partial corr.", "partial_correlation": "Partial corr.",
    "pac": "PAC", "aac": "AAC", "ppc": "PPC", "dtf": "DTF", "te": "TE",
}


def metric_display(m) -> str:
    """Compact, properly-capitalized metric label for plots (AEC, dwPLI, …)."""
    if m is None:
        return ""
    return METRIC_DISPLAY.get(str(m).strip().lower(), str(m))

# Group display settings
GROUP_COLORS = {
    "KO_VEH": "#E74C3C",
    "WT_VEH": "#3498DB",
}

GROUP_LABELS = {
    "KO_VEH": "KO Vehicle",
    "WT_VEH": "WT Vehicle",
}

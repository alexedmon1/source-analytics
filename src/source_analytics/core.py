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
from .analyses.fcd_comparison_analysis import FCDComparisonAnalysis
from .analyses.vertex_connectivity_analysis import VertexConnectivityAnalysis
from .analyses.vertex_specparam_analysis import VertexSpecparamAnalysis
from .analyses.vertex_signature_analysis import VertexSignatureAnalysis
from .analyses.electrode_signature_analysis import ElectrodeSignatureAnalysis
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
    "fcd_comparison": FCDComparisonAnalysis,
    "vertex_connectivity": VertexConnectivityAnalysis,
    "vertex_cross_freq": VertexCrossFreqAnalysis,
    "vertex_specparam": VertexSpecparamAnalysis,
    "vertex_signature": VertexSignatureAnalysis,
    "electrode_signature": ElectrodeSignatureAnalysis,
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
    "mvpa": "vertex_signature",
    "vertex_mvpa": "vertex_signature",
    "transfer_entropy": "roi_directed",
    "roi_transfer_entropy": "roi_directed",
    "evoked": "roi_evoked",
    "electrode": "electrode_psd",
}

# Register aliases so old YAML configs still work
for _old, _new in _DEPRECATED_NAMES.items():
    ANALYSIS_REGISTRY[_old] = ANALYSIS_REGISTRY[_new]


def canonical_analysis_name(name: str) -> str:
    """Canonical registry name for ``name`` (alias-resolved), without warning."""
    return _DEPRECATED_NAMES.get(name, name)


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
#   requires    = every upstream module whose output the analysis reads (a
#                 superset of `supplements` for modules with >1 primary). The
#                 lightbox groups by `supplements`; `requires` is the honest
#                 dependency list for run ordering.
ANALYSIS_METADATA: dict[str, dict] = {
    "roi_psd":              {"category": "resting", "level": "roi",        "domain": "Spectral",        "description": "PSD (power spectral density)",
                             "about": "Resting-state oscillatory power in each source-localized ROI. Per subject, ROI, and band, power is integrated from the ROI power spectrum and reported two ways: absolute power as the mean power density over the band in dB/Hz (10*log10 of the band's integrated power divided by its bandwidth, so the across-band 1/f shape is visible rather than width-confounded) and relative power (the band's fraction of total power across the analyzed spectrum). Groups are compared per ROI and band with a linear mixed model (subject as a random effect, via lmerTest) and contrasts estimated with emmeans; effect sizes are Hedges' g, with p-values FDR-corrected (Benjamini-Hochberg) within each band. Read it as: which ROIs differ in oscillatory power, in which bands, and in which direction (an up arrow means the first-listed group of the pair is higher). Relative power controls for overall spectral amplitude and isolates spectral shape; absolute (dB/Hz) is the band's log power density."},
    "roi_aperiodic":        {"category": "resting", "level": "roi",        "domain": "Spectral",        "description": "1/f aperiodic decomposition",
                             "about": "The aperiodic (1/f) component of each ROI's resting power spectrum, fit with specparam/FOOOF over 12-45 Hz by default (spectral.aperiodic.DEFAULT_FREQ_RANGE; see docs/methods/APERIODIC_FIT_WINDOW.md): the exponent (steepness of the 1/f decay, a proxy for excitation/inhibition balance -- steeper usually read as more inhibition) and the offset (broadband power level). Groups are compared per ROI with a linear mixed model (dv ~ group * ROI, subject as a random effect; Type-III ANOVA, Satterthwaite df) followed by emmeans pairwise contrasts gated on the group-by-ROI omnibus; effect sizes are Hedges' g and per-ROI post-hoc p-values are Holm-corrected. There is no frequency-band axis -- exponent and offset are single broadband parameters. Read it as: which ROIs show a group shift in spectral slope (E/I tilt) or broadband power, and in which direction (an up arrow means the first-listed group is higher)."},
    "roi_connectivity":     {"category": "resting", "level": "roi",        "domain": "Connectivity",    "description": "ROI pairwise connectivity matrices (descriptive; inference in roi_nbs / roi_graph)",
                             "about": "Resting functional connectivity between every pair of source-localized ROIs, per band. For each pair it estimates one or more coupling measures from the cross-spectrum: coherence, imaginary coherence (Nolte et al. 2004), the phase-lag index and its weighted/debiased forms (PLI Stam et al. 2007; wPLI and dwPLI Vinck et al. 2011), the directed phase-lag index (dPLI Stam & van Straaten 2012), amplitude-envelope correlation with leakage orthogonalization (AEC Hipp et al. 2012), and shrinkage partial correlation (Ledoit-Wolf). Corpus-callosum tracts are excluded, leaving the cortical ROI pairs. This module is descriptive: it builds the group-mean ROI-by-ROI connectivity matrix per band and metric and renders it (circos + heatmap, with the between-group difference), but carries no per-edge inferential test -- the legacy per-pair Welch t-test and region-pair linear mixed model were retired because edge-by-edge testing is underpowered at this ROI count. Group-level inference for the connectivity family is provided by roi_nbs (the Network-Based Statistic, for connected sub-networks that differ between groups) and roi_graph (graph-theoretic network organization). Read it as: the connectivity structure per band and its group differences (a warm difference / up arrow means the first-listed group is higher); imaginary-coherence/PLI-family metrics suppress volume-conduction/zero-lag artifacts."},
    "roi_cross_freq":       {"category": "resting", "level": "roi",        "domain": "Cross-frequency", "description": "Cross-frequency coupling (PAC, AAC, n:m PPC)",
                             "about": "Coupling between a slow and a fast rhythm in the source-localized ROIs, for every valid slow-phase x fast-amplitude band pair (the fast band must sit above the slow band, center-frequency ratio > 2.5). It computes three complementary measures: phase-amplitude coupling (PAC) as Tort et al. (2010) modulation index -- the KL divergence of the phase-binned amplitude distribution from uniform, z-scored against circular-time-shift surrogates; amplitude-amplitude coupling (AAC) as the Pearson correlation of the two bands' power envelopes (Bruns 2000; Masimore 2004); and n:m phase-phase coupling (PPC) as the phase-locking factor |mean exp(i(n*phase_slow - m*phase_fast))| (Palva et al. 2005), also surrogate z-scored. PAC is per-ROI (within-region), AAC and PPC are ROI-pair edges; callosal tracts are excluded. Groups are compared with a linear mixed model for PAC (Hedges' g post-hocs per region) and edge-level tests for AAC/PPC. Read it as: which regions/pairs show a group difference in how one rhythm's phase organizes another's amplitude (PAC), or how two rhythms co-vary (AAC) or phase-lock at an n:m ratio (PPC), and in which direction (an up arrow means the first-listed group is higher)."},
    "roi_graph":            {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level graph-theoretic metrics (degree/clustering/betweenness)",
                             "about": "Graph-theoretic summaries of the ROI connectivity network, per band and connectivity metric. Each subject's ROI-by-ROI matrix is thresholded to a fixed connection density (proportional, default 15% of edges kept) and turned into a graph, from which it computes nodal metrics per ROI (degree, clustering coefficient, betweenness centrality) and whole-network metrics (global efficiency, modularity, small-worldness). Groups are compared per ROI/metric with a Welch t-test, effect sizes are Hedges' g, and p-values are FDR-corrected (Benjamini-Hochberg, q < 0.05). Read it as: which ROIs act as more/less connected hubs (degree, betweenness) or more clustered (clustering), and whether whole-network integration/segregation shifts between groups (an up arrow means the first-listed group is higher). It asks how the connectivity is organized as a network, beyond individual edge strengths."},
    "roi_nbs":              {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level Network-Based Statistic (sub-network test)",
                             "about": "The Network-Based Statistic (Zalesky et al. 2010) applied to the ROI connectivity network, per band and metric -- a connected-subnetwork test that has more power than edge-by-edge correction when a group effect is distributed across many connected edges. Every ROI-pair edge gets a group Welch t-statistic; edges exceeding a primary threshold (default t = 2.5) are retained and grouped into connected components; each component's size (edge count) is compared to a permutation null built by relabeling groups and tracking the largest component per permutation, giving component-level family-wise (FWE) control. Read it as: significant sub-networks -- clusters of ROI connections that jointly differ between groups (a component with p < 0.05), rather than any single edge. It complements roi_graph (network topology) and roi_connectivity (individual edges)."},
    "vertex_graph":         {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level multi-density AUC graph metrics",
                             "about": "Whole-brain graph-theoretic organization of the vertex connectivity network, per band and connectivity metric. To avoid picking one arbitrary connection density, each subject's vertex network is thresholded across a range of densities (default 0.05-0.40 in 0.01 steps) and each global metric is integrated over that range as an area-under-the-curve (AUC) value: global efficiency, characteristic path length, mean clustering, transitivity, modularity, assortativity, mean local efficiency, and small-worldness. Groups are compared on each AUC metric with a permutation test (5000 permutations of group labels), with Hedges' g effect sizes; permutation p-values are reported per metric (raw, marked significant at p < 0.05). Read it as: whole-brain shifts in network integration (efficiency, path length), segregation (clustering, modularity), or small-world balance between groups (an up arrow means the first-listed group is higher). It is the unparcellated, density-integrated counterpart of roi_graph."},
    "vertex_nbs":           {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level Network-Based Statistic (sub-network test)",
                             "about": "The Network-Based Statistic (Zalesky et al. 2010) applied to the whole-brain vertex connectivity network, per band and metric -- the unparcellated counterpart of roi_nbs. Every vertex-pair edge gets a group Welch t-statistic; edges above a primary threshold (default t = 3.0) are grouped into connected components, and each component's size is compared to a permutation null (5000 relabelings, largest-component-per-permutation) for component-level family-wise (FWE) control. Read it as: significant sub-networks -- spatially distributed sets of vertex connections that jointly differ between groups (a component with p < 0.05), rather than any single vertex pair. It complements vertex_graph (network topology) and vertex_connectivity (per-vertex FCD)."},
    "roi_network":          {"category": "resting", "level": "roi",        "domain": "Connectivity",    "supplements": "roi_connectivity",    "description": "ROI-level graph theory + NBS (combined alias of roi_graph + roi_nbs)"},
    "vertex_network":       {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "supplements": "vertex_connectivity", "description": "Vertex-level graph theory + NBS (combined alias of vertex_graph + vertex_nbs)"},
    "roi_directed":         {"category": "resting", "level": "roi",        "domain": "Directed",        "description": "Directed connectivity (transfer entropy + DTF)",
                             "about": "Directional (who-drives-whom) connectivity between ROIs, in two flavors. Transfer entropy (TE, Schreiber 2000) is a model-free information-theoretic measure -- a binned lag-1 estimator of how much one ROI's past reduces uncertainty about another's future -- computed for every directed ROI pair (callosal tracts excluded); its net asymmetry (te - te-transpose) gives the dominant direction. The optional Directed Transfer Function (DTF, Kaminski & Blinowska 1991) derives directed influence from a multivariate autoregressive (MVAR) model fit with ridge regularization (order 8) -- a deviation from ordinary-least-squares DTF used to stabilize the fit against collinear channels. Groups are compared on TE three ways: a global per-pair Welch t-test (Hedges' g), a within-group one-sample test of net TE against zero (is the driving direction consistent within a group), and a region-pair linear mixed model (te ~ group * region_pair + (1|subject)). Read it as: which ROI pairs show a group difference in directed influence, and which region drives which (an up arrow means the first-listed group is higher). DTF, when selected, currently emits directed edges without the R group stats."},
    "vertex_directed":      {"category": "resting", "level": "vertex",     "domain": "Directed",        "description": "Vertex DTF outflow/inflow/netflow (ridge-MVAR, cluster-corrected)",
                             "about": "Whole-brain directed connectivity on the dorsal source surface via the Directed Transfer Function (DTF, Kaminski & Blinowska 1991) from a multivariate autoregressive model. Because source vertices are strongly collinear (mean inter-vertex |r| ~ 0.64), the MVAR is fit with ridge regularization (order 8) rather than ordinary least squares, and the fit's stability (spectral radius) is checked. The full all-to-all directed DTF matrix is reduced to three per-vertex maps: outflow (mean directed influence a vertex sends to all others), inflow (mean it receives), and netflow (outflow minus inflow -- net source vs sink). Group differences in each map are tested with a cluster-based permutation test (per-vertex t-statistics clustered by spatial adjacency, cluster-extent FWE from a permutation null; Maris & Oostenveld 2007), with per-vertex Hedges' g. Read it as: spatially-contiguous clusters where the groups differ in how strongly a region drives (outflow), is driven by (inflow), or net-drives (netflow) the rest of the brain -- a cluster with p_corrected < 0.05 marks a region of difference, the sign of its t-values gives direction."},
    "electrode_signature": {"category": "resting", "level": "electrode", "domain": "Source vs Sensor", "display_name": "Neural signature", "description": "Sensor-level neural signature (classification/decoding on electrode band power) — the source-vs-sensor counterpart of vertex_signature"},
    "vertex_signature":     {"category": "resting", "level": "vertex",     "domain": "Multivariate",    "display_name": "Neural signature", "description": "Multivariate/ML neural signature (classification, decoding; PCA-reduced with back-projection)"},
    "vertex_cluster":       {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "Vertex-level cluster permutation",
                             "about": "Whole-brain resting spectral maps on the dorsal source surface: per vertex it computes band power (absolute as mean density in dB/Hz, the same definition as roi_psd, and relative), the 1/f spectral slope, and the peak alpha frequency, then tests where the groups differ. Inference is a cluster-based permutation test -- per-vertex t-statistics are threshold-clustered over neighbouring vertices and each cluster's extent is compared to a permutation null, giving family-wise (FWE) control (Maris & Oostenveld 2007); a threshold-free TFCE variant (Smith & Nichols 2009) is available. Effect sizes are per-vertex Hedges' g. Read it as: spatially-contiguous clusters where the groups differ in a spectral measure -- a cluster with p_corrected < 0.05 marks a region of difference, and the sign of its t-values gives the direction. This is the whole-brain, unparcellated counterpart to the ROI spectral analyses."},
    "vertex_spatial":       {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "RETIRED — was: spatial GLS robustness check; exits with empty tables (use vertex_cluster / vertex_nbs)"},
    "vertex_specparam":     {"category": "resting", "level": "vertex",     "domain": "Spectral",        "description": "Vertex-level spectral parameterization",
                             "about": "The aperiodic (1/f) spectrum fit per vertex across the dorsal source surface with specparam/FOOOF -- exponent, offset, and per-band oscillatory peaks (presence, frequency, power). Group differences in the exponent and offset maps, and in per-band peak power, are tested with a cluster-based permutation test (threshold-clustered vertex t-statistics with cluster-extent FWE correction by permutation); band peak presence is compared with a per-vertex chi-square test. Effect sizes are per-vertex Hedges' g. Read it as: spatially-contiguous clusters of vertices where the groups differ in spectral slope, broadband power, or an oscillatory peak -- a cluster with p_corrected < 0.05 marks a region of difference, the sign of its t-values gives direction. This is the whole-brain, unparcellated version of roi_aperiodic (plus peaks)."},
    "vertex_connectivity":  {"category": "resting", "level": "vertex",     "domain": "Connectivity",    "description": "Vertex pairwise connectivity",
                             "about": "Whole-brain resting connectivity between source vertices -- the unparcellated counterpart of roi_connectivity. Per band it computes all-to-all vertex coupling with the same kernels (coherence, imaginary coherence, PLI/wPLI/dwPLI, dPLI, AEC, partial correlation) and condenses each vertex's connectivity to a functional connectivity density (FCD) map: the fraction of other vertices it couples to above a threshold (degree/(n-1), Tomasi & Volkow 2010). Group differences in the FCD maps are tested with a cluster-based permutation test -- per-vertex t-statistics are clustered over neighbouring vertices (adjacency by distance) and cluster extents compared to a permutation null for family-wise (FWE) control (Maris & Oostenveld 2007). Effect sizes are per-vertex Hedges' g. Read it as: spatially-contiguous clusters where the groups differ in how densely a region is functionally connected -- a cluster with p_corrected < 0.05 marks a region of difference, the sign of its t-values gives direction. This whole-brain FCD map is the source-side input to the source-vs-sensor comparison (fcd_comparison)."},
    "vertex_cross_freq":    {"category": "resting", "level": "vertex",     "domain": "Cross-frequency", "description": "Vertex cross-frequency coupling (local PAC, AAC, n:m PPC)",
                             "about": "Whole-brain cross-frequency coupling on the dorsal source surface -- the unparcellated counterpart of roi_cross_freq, using the same kernels for each valid slow-phase x fast-amplitude band pair. Phase-amplitude coupling (PAC, Tort et al. 2010 modulation index, surrogate z-scored) is computed locally -- the slow phase and fast amplitude come from the same vertex -- yielding a whole-brain coupling map. Amplitude-amplitude coupling (AAC, power-envelope correlation) and n:m phase-phase coupling (PPC, Palva et al. 2005 phase-locking factor) are computed all-to-all across vertices and summarized to a per-vertex node strength (mean off-diagonal coupling). Group differences in these maps are tested with a cluster-based permutation test (per-vertex t-statistics clustered by spatial adjacency, cluster-extent FWE from a permutation null; Maris & Oostenveld 2007), with per-vertex Hedges' g. Read it as: spatially-contiguous clusters where the groups differ in cross-frequency coupling -- a cluster with p_corrected < 0.05 marks a region of difference, the sign of its t-values gives direction. PAC here is the primary source-spatial-advantage measure (local, no leakage between nodes)."},
    "electrode_psd":        {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "description": "Sensor-level PSD analysis",
                             "about": "Resting band power at each scalp electrode -- the sensor-space counterpart of roi_psd. Per channel and band, power is reported as absolute power density in dB/Hz (10*log10 of the band's integrated power divided by its bandwidth, as in roi_psd) and relative power (the band's fraction of total 1-100 Hz power). Groups are compared per channel and band with a linear mixed model (dv ~ group * channel, subject as a random effect), with an optional region-nested model over the configured electrode regions (channels as replicates); per-contrast effects come from the hypothesis layer as Hedges' g with band-wise Benjamini-Hochberg FDR. Read it as: which electrodes/regions differ in band power, in which bands and direction (up = the first-listed group is higher) -- the sensor-level check against the source (ROI) result."},
    "electrode_aperiodic":  {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "description": "Sensor-level aperiodic (1/f) analysis",
                             "about": "The aperiodic (1/f) exponent and offset at each scalp electrode, fit with specparam/FOOOF over 12-45 Hz by default (spectral.aperiodic.DEFAULT_FREQ_RANGE) -- the sensor-space counterpart of roi_aperiodic. Groups are compared per channel with a linear mixed model (dv ~ group * channel, subject as a random effect) plus a region-nested model (dv ~ group * region with (1|subject/channel), treating channels as replicates within scalp regions); effect sizes are Hedges' g and post-hoc p-values are Benjamini-Hochberg FDR-corrected across channels. Read it as: which electrodes/regions show a group shift in spectral slope (E/I proxy) or broadband offset, and in which direction (up = the first-listed group is higher)."},
    "electrode_comparison": {"category": "resting", "level": "electrode",  "domain": "Source vs Sensor", "display_name": "PSD",         "supplements": "electrode_psd", "requires": ["electrode_psd", "roi_psd"], "description": "Source vs electrode comparison (needs electrode_psd AND roi_psd)",
                             "about": "A source-versus-sensor check on resting band power: for each subject and band, electrode power (averaged over channels) is compared to source power (averaged over ROIs). It reports (1) the cross-subject concordance between sensor and source power (Pearson r per band) and (2) whether the group effect agrees at both levels -- per contrast, Hedges' g with 95% CIs at the electrode level and the source (ROI/region) level, plus an 'exceeds_electrode' flag where a region's effect is larger than the global sensor effect. There is no cluster/FWE correction here; significance is read from whether the 95% CI excludes zero. Read it as: does the source reconstruction recover the same spectral group effect the scalp shows, and does it localize it more sharply than the sensor average?"},
    "electrode_connectivity": {"category": "resting", "level": "electrode",  "domain": "Sensor-level",    "description": "Sensor pairwise connectivity + FCD (source-vs-sensor comparator)",
                             "about": "Resting connectivity between the 30 scalp electrodes -- the sensor-space comparator for vertex_connectivity. Per band it computes all-to-all channel coupling with the leakage/volume-conduction-robust subset of the same kernels (AEC, imaginary coherence, PLI, wPLI, dwPLI, dPLI) and the per-channel functional connectivity density (FCD; degree/(n-1) above threshold, Tomasi & Volkow 2010). Groups are compared per channel with a Welch t-test and Benjamini-Hochberg FDR across the 30 channels (effect sizes Hedges' g), plus a hypothesis-layer cluster-permutation test over the sensor montage (adjacency from channel coordinates). Read it as: which electrodes differ in connectivity/FCD, in which band and direction (up = the first-listed group is higher) -- the scalp-level check on whether the source FCD effect is also visible without source reconstruction. Because sensor space is blurred by volume conduction, the volume-conduction-sensitive metrics are omitted here."},
    "fcd_comparison":       {"category": "resting", "level": "electrode",  "domain": "Source vs Sensor", "display_name": "Connectivity", "supplements": "electrode_connectivity", "requires": ["electrode_connectivity", "vertex_connectivity"], "description": "Source vs sensor FCD comparison (mean + spatial CV; needs electrode_connectivity AND vertex_connectivity, cross-paradigm)",
                             "about": "A source-versus-sensor check on functional connectivity density (FCD), pairing the whole-brain vertex FCD maps (vertex_connectivity) against the scalp channel FCD (electrode_connectivity). FCD is each node's fraction of supra-threshold connections (degree/(n-1), threshold 0.05; for dPLI the deviation from its 0.5 no-lag center; Tomasi & Volkow 2010). Per subject and band it summarizes each map two ways -- mean FCD (overall coupling density) and spatial coefficient of variation (CV = SD/mean, how heterogeneous the map is) -- then reports (1) cross-subject concordance between source and sensor summaries (Pearson r per band) and (2) whether the group effect agrees at both levels: per contrast, Hedges' g with 95% CIs at each level plus a sign-concordance flag. There is no cluster/FWE correction here; significance is read from whether a 95% CI excludes zero. Read it as: does source-space recover the same connectivity-density group effect the scalp shows, and is the spatial pattern preserved?"},
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
        jobs: int | None = None,
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
        analysis.run(subjects, steps=steps, select=select, jobs=jobs)

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

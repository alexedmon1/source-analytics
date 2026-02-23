"""Generate Quarto (.qmd) results reports from completed analyses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StudyConfig

logger = logging.getLogger(__name__)


# ── Color mode ───────────────────────────────────────────────────────────────

class ColorMode(str, Enum):
    """Report color scheme mode."""

    ANALYSIS = "analysis"       # Per-analysis color themes (default for reports)
    PUBLICATION = "publication"  # Uniform colors from config (for manuscript figures)


# Per-analysis group color palettes for radar charts (report mode only).
# Vehicle is always gray; treatment hues shift per analysis for visual variety.
REPORT_GROUP_COLORS: dict[str, dict[str, str]] = {
    "psd":              {"Vehicle": "#8e8e8e", "30mgkg": "#c0392b", "6mgkg": "#27ae60"},
    "aperiodic":        {"Vehicle": "#8e8e8e", "30mgkg": "#d35400", "6mgkg": "#2980b9"},
    "roi_connectivity": {"Vehicle": "#8e8e8e", "30mgkg": "#8e44ad", "6mgkg": "#16a085"},
    "pac":              {"Vehicle": "#8e8e8e", "30mgkg": "#c0392b", "6mgkg": "#2c3e50"},
    "roi_network":      {"Vehicle": "#8e8e8e", "30mgkg": "#e67e22", "6mgkg": "#3498db"},
    "evoked":           {"Vehicle": "#8e8e8e", "30mgkg": "#e74c3c", "6mgkg": "#1abc9c"},
}


def _get_report_colors(
    analysis: str, config: StudyConfig, color_mode: ColorMode,
) -> dict[str, str]:
    """Get group colors for an analysis based on color mode."""
    if color_mode == ColorMode.PUBLICATION:
        return config.group_colors
    return REPORT_GROUP_COLORS.get(analysis, config.group_colors)


def _get_analysis_cmap(analysis: str, color_mode: ColorMode) -> str:
    """Get sequential colormap for non-group-colored plots."""
    if color_mode == ColorMode.PUBLICATION:
        return "Blues"  # uniform publication cmap
    from .viz import ANALYSIS_CMAPS

    key = {"roi_connectivity": "connectivity", "roi_network": "network"}.get(
        analysis, analysis,
    )
    return ANALYSIS_CMAPS.get(key, {}).get("sequential", "Blues")

# ── Display names ────────────────────────────────────────────────────────────

ANALYSIS_DISPLAY_NAMES: dict[str, str] = {
    "psd": "Power Spectral Density",
    "aperiodic": "Aperiodic (1/f) Decomposition",
    "roi_connectivity": "ROI Functional Connectivity",
    "pac": "Phase-Amplitude Coupling",
    "roi_network": "ROI Network Analysis",
    "wholebrain": "Whole-Brain Vertex-Level Analysis",
    "mvpa": "Multivariate Pattern Analysis (MVPA)",
    "specparam_vertex": "Vertex-Level Spectral Parameterization",
    "vertex_connectivity": "Vertex Connectivity & FCD",
    "vertex_network": "Vertex Network Analysis",
    "evoked": "Evoked Responses (ITC, ERSP, STP)",
}

PARADIGM_DISPLAY_NAMES: dict[str, str] = {
    "resting": "Resting State",
    "chirp": "Chirp",
    "40hz_assr": "40 Hz ASSR",
    "80hz_assr": "80 Hz ASSR",
}

# ── Figure selection priority per analysis ──────────────────────────────────

FIGURE_PRIORITY: dict[str, list[str]] = {
    # ROI analyses: prefer on-demand radar + brain_roi (generated into report/ subdir)
    "psd": [
        "report/radar_*.png",
        "report/brain_roi_*.png",
        "band_power_dB.png",
        "brain_roi_*.png",
    ],
    "aperiodic": [
        "report/radar_*.png",
        "report/brain_roi_*.png",
        "brain_roi_*.png",
    ],
    "roi_connectivity": [
        "report/circos_comparison_*.png",
        "report/connectivity_*.png",
        "report/circos_*.png",
        "circos_coherence_high_gamma.png",
        "heatmap_coherence_high_gamma.png",
        "circos_coherence_*.png",
        "heatmap_coherence_*.png",
    ],
    "pac": [
        "report/radar_*.png",
        "report/brain_roi_*.png",
        "pac_comodulogram_*.png",
        "brain_roi_*.png",
    ],
    "roi_network": [
        "report/radar_*.png",
        "report/circos_*.png",
        "roi_network_*.png",
    ],
    # Wholebrain: select pre-existing Python-generated glass brains
    "wholebrain": [
        "wholebrain_*high_gamma*.png",
        "wholebrain_*low_gamma*.png",
        "wholebrain_*beta*.png",
        "wholebrain_summary.png",
    ],
    "mvpa": [
        "mvpa_importance_*high_gamma*.png",
        "mvpa_null_*high_gamma*.png",
        "mvpa_importance_*low_gamma*.png",
        "mvpa_null_*low_gamma*.png",
    ],
    "specparam_vertex": [
        "specparam_exponent*.png",
        "specparam_offset*.png",
        "gamma_peak_presence.png",
    ],
    "vertex_connectivity": [
        "fcd_*high_gamma*.png",
        "fcd_*low_gamma*.png",
    ],
    "vertex_network": [
        "vertex_nbs_edges_*high_gamma*.png",
        "vertex_network_degree_*high_gamma*.png",
        "vertex_nbs_edges_*low_gamma*.png",
    ],
    "evoked": [
        "report/radar_*.png",
        "report/brain_roi_*.png",
        "evoked_itc_*_group.png",
        "evoked_ersp_*_group.png",
        "evoked_stp_*_group.png",
    ],
}

# Figure patterns that should never appear in reports
_BANNED_FIGURE_PATTERNS = [
    "*region_significance_heatmap*",
    "*significance_heatmap*",
    "*forest*",
]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class KeyFinding:
    """A parsed significant result from an analysis summary."""

    paradigm: str
    analysis: str
    measure: str
    contrast: str
    statistic: str  # e.g. "F=3.84, q=0.0000"
    details: str  # e.g. "group x region interaction"
    q_value: float | None = None


@dataclass
class AnalysisSummary:
    """Parsed content from an ANALYSIS_SUMMARY.md file."""

    title: str
    paradigm: str
    analysis: str
    methods_text: str
    sections: dict[str, str]  # heading -> raw text
    key_findings: list[KeyFinding]
    figure_refs: list[str]  # relative figure paths from the summary


@dataclass
class ReportBuilder:
    """Accumulates Quarto markdown content."""

    _parts: list[str] = field(default_factory=list)

    def add_raw(self, text: str) -> None:
        self._parts.append(text)

    def add_heading(self, text: str, level: int = 1) -> None:
        self._parts.append(f"\n{'#' * level} {text}\n")

    def add_text(self, text: str) -> None:
        self._parts.append(f"\n{text}\n")

    def add_figure(self, path: str, caption: str = "", width: str = "100%") -> None:
        cap = f' fig-cap="{caption}"' if caption else ""
        self._parts.append(f'\n![{caption}]({path}){{width="{width}"{cap}}}\n')

    def add_figure_grid(
        self, figures: list[tuple[str, str]], ncol: int = 2,
    ) -> None:
        """Add figures in a Quarto grid layout.

        Parameters
        ----------
        figures : list of (path, caption) tuples
        ncol : int
            Columns per row (default 2).
        """
        self._parts.append(f"\n::: {{layout-ncol={ncol}}}")
        for path, caption in figures:
            self._parts.append(f"![{caption}]({path})")
            self._parts.append("")
        self._parts.append(":::\n")

    def add_callout(self, text: str, kind: str = "note") -> None:
        self._parts.append(f"\n::: {{.callout-{kind}}}\n{text}\n:::\n")

    def build(self) -> str:
        return "\n".join(self._parts)


# ── Parsing ──────────────────────────────────────────────────────────────────

def _split_sections(md_text: str) -> dict[str, str]:
    """Split markdown text into sections by ## headings."""
    sections: dict[str, str] = {}
    current_heading = "_preamble"
    current_lines: list[str] = []

    for line in md_text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_heading] = "\n".join(current_lines).strip()
    return sections


# Regex for R-generated Key Findings bullets:
# - **measure** [contrast, level]: description (F=x, q=y)
_KEY_FINDING_RE = re.compile(
    r"^- \*\*(.+?)\*\*\s*\[([^\]]+)\]:\s*(.+?)$",
    re.MULTILINE,
)

# Regex to extract F= and q= or p= values from a finding line
_STAT_RE = re.compile(r"\(([^)]*(?:F|t|p|q)[^)]*)\)")

# Regex for MVPA accuracy lines
_MVPA_ACCURACY_RE = re.compile(
    r"\|\s*(\w+)\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|"
)

# Regex for NBS significant subnetworks
_NBS_SIG_RE = re.compile(
    r"^- (.+?):\s*(\d+)\s*edges?,\s*p\s*=\s*([\d.]+)",
    re.MULTILINE,
)


def _parse_key_findings_r(
    text: str, paradigm: str, analysis: str,
) -> list[KeyFinding]:
    """Parse Key Findings from R-generated summaries."""
    findings = []
    for m in _KEY_FINDING_RE.finditer(text):
        measure = m.group(1).strip()
        bracket = m.group(2).strip()
        details = m.group(3).strip()

        # Parse contrast from bracket (e.g. "30mgkg_vs_Vehicle, region-level")
        parts = [p.strip() for p in bracket.split(",")]
        contrast = parts[0] if parts else ""

        # Extract statistic
        stat_m = _STAT_RE.search(details)
        statistic = stat_m.group(1) if stat_m else ""

        # Extract q-value
        q_val = None
        q_m = re.search(r"q=([\d.]+)", statistic)
        if q_m:
            try:
                q_val = float(q_m.group(1))
            except ValueError:
                pass

        findings.append(KeyFinding(
            paradigm=paradigm,
            analysis=analysis,
            measure=measure,
            contrast=contrast,
            statistic=statistic,
            details=details,
            q_value=q_val,
        ))
    return findings


def _parse_key_findings_python(
    sections: dict[str, str], paradigm: str, analysis: str,
) -> list[KeyFinding]:
    """Extract key findings from Python-generated summaries."""
    findings: list[KeyFinding] = []

    # Check for NBS results
    for key, text in sections.items():
        if "NBS" in key:
            for m in _NBS_SIG_RE.finditer(text):
                name = m.group(1).strip()
                n_edges = m.group(2)
                p_val = float(m.group(3))
                if p_val < 0.05:
                    # Parse contrast from NBS name
                    # (e.g. "30mgkg_vs_Vehicle_low_gamma_coherence")
                    contrast = ""
                    for token in ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle",
                                  "30mgkg_vs_6mgkg"]:
                        if token in name:
                            contrast = token
                            break
                    findings.append(KeyFinding(
                        paradigm=paradigm,
                        analysis=analysis,
                        measure=name,
                        contrast=contrast,
                        statistic=f"{n_edges} edges, p={p_val:.4f}",
                        details=f"NBS subnetwork: {name}",
                        q_value=p_val,
                    ))

    # Check for MVPA results
    results_text = sections.get("Results", "")
    if analysis == "mvpa":
        # Table rows: | Band | Accuracy | p-value | ...
        rows = results_text.splitlines()
        contrast_idx = 0
        contrasts = ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle", "30mgkg_vs_6mgkg"]
        bands_seen: set[str] = set()
        for row in rows:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            if len(cells) >= 3 and cells[0] not in ("Band", "---", "------"):
                band = cells[0]
                if band in bands_seen:
                    bands_seen.clear()
                    contrast_idx += 1
                bands_seen.add(band)
                try:
                    acc = float(cells[1].rstrip("%"))
                    p_val = float(cells[2])
                except (ValueError, IndexError):
                    continue
                if p_val < 0.05:
                    contrast = contrasts[contrast_idx] if contrast_idx < len(contrasts) else ""
                    findings.append(KeyFinding(
                        paradigm=paradigm,
                        analysis=analysis,
                        measure=f"{band} band",
                        contrast=contrast,
                        statistic=f"accuracy={acc:.1f}%, p={p_val:.4f}",
                        details=f"MVPA classification: {acc:.1f}% accuracy",
                        q_value=p_val,
                    ))

    # Check for cluster results in wholebrain / specparam_vertex
    for key, text in sections.items():
        if "cluster" in key.lower() or key == "Results":
            # Look for "No significant clusters"
            if "No significant clusters" in text:
                continue
            # Look for cluster table rows with p < 0.05
            for line in text.splitlines():
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) >= 4:
                    try:
                        p_corr = float(cells[-2]) if "." in cells[-2] else None
                        if p_corr is not None and p_corr < 0.05:
                            findings.append(KeyFinding(
                                paradigm=paradigm,
                                analysis=analysis,
                                measure=cells[0],
                                contrast="",
                                statistic=f"cluster p={p_corr:.4f}",
                                details=f"Significant cluster: {cells[0]}",
                                q_value=p_corr,
                            ))
                    except (ValueError, IndexError):
                        continue

    return findings


def _parse_figure_refs(md_text: str) -> list[str]:
    """Extract figure paths from markdown image references."""
    return re.findall(r"!\[.*?\]\(figures/(.+?)\)", md_text)


def parse_analysis_summary(
    path: Path, paradigm: str, analysis: str,
) -> AnalysisSummary:
    """Parse an ANALYSIS_SUMMARY.md into structured data."""
    text = path.read_text(encoding="utf-8")
    sections = _split_sections(text)

    # Title from first heading
    title_m = re.match(r"^#\s+(.+)", text)
    title = title_m.group(1).strip() if title_m else f"{analysis} Analysis"

    # Methods section
    methods_text = sections.get("Methods", "")

    # Key findings — try R-style first, then Python-style
    kf_text = sections.get("Key Findings", "")
    if kf_text:
        key_findings = _parse_key_findings_r(kf_text, paradigm, analysis)
    else:
        key_findings = _parse_key_findings_python(sections, paradigm, analysis)

    figure_refs = _parse_figure_refs(text)

    return AnalysisSummary(
        title=title,
        paradigm=paradigm,
        analysis=analysis,
        methods_text=methods_text,
        sections=sections,
        key_findings=key_findings,
        figure_refs=figure_refs,
    )


# ── Discovery ───────────────────────────────────────────────────────────────

def discover_completed_analyses(
    config: StudyConfig,
) -> list[tuple[str, str, Path]]:
    """Find all completed analyses by scanning for ANALYSIS_SUMMARY.md files.

    Returns list of (paradigm, analysis, summary_path) tuples.
    """
    found = []
    for pname in config.paradigms:
        analyses = config.get_paradigm_analyses(pname) or []
        for aname in analyses:
            summary_path = config.output_dir / pname / aname / "ANALYSIS_SUMMARY.md"
            if summary_path.exists():
                found.append((pname, aname, summary_path))
                logger.debug("Found: %s/%s", pname, aname)
            else:
                logger.debug("Missing: %s/%s", pname, aname)
    return found


# ── Figure selection ─────────────────────────────────────────────────────────

def _is_banned(path: Path) -> bool:
    """Check if a figure matches any banned pattern."""
    name = path.name
    for pattern in _BANNED_FIGURE_PATTERNS:
        # Convert glob pattern to simple check
        parts = pattern.strip("*").split("*")
        if all(p in name for p in parts if p):
            return True
    return False


def select_figures(
    analysis: str,
    fig_dir: Path,
    max_figures: int = 4,
) -> list[Path]:
    """Select top figures for an analysis based on priority patterns.

    Searches both ``fig_dir`` and ``fig_dir/report/`` (for on-demand figures).
    Figures matching banned patterns are excluded.
    """
    if not fig_dir.is_dir():
        logger.warning("Figure directory not found: %s", fig_dir)
        return []

    # Collect all PNGs from fig_dir and report/ subdir
    all_figs = sorted(fig_dir.glob("*.png"))
    report_subdir = fig_dir / "report"
    if report_subdir.is_dir():
        all_figs.extend(sorted(report_subdir.glob("*.png")))

    # Filter banned figures
    all_figs = [f for f in all_figs if not _is_banned(f)]
    if not all_figs:
        return []

    patterns = FIGURE_PRIORITY.get(analysis, ["*.png"])
    selected: list[Path] = []
    used: set[Path] = set()

    for pattern in patterns:
        if len(selected) >= max_figures:
            break
        matches = sorted(fig_dir.glob(pattern))
        for m in matches:
            if m not in used and not _is_banned(m) and len(selected) < max_figures:
                selected.append(m)
                used.add(m)

    # Fill remaining slots with unused, non-banned figures
    if len(selected) < max_figures:
        for f in all_figs:
            if f not in used and len(selected) < max_figures:
                selected.append(f)
                used.add(f)

    return selected


# ── Report generation ────────────────────────────────────────────────────────

def _make_relative(fig_path: Path, report_dir: Path) -> str:
    """Return path relative to the report output directory."""
    import os
    try:
        return os.path.relpath(fig_path, report_dir)
    except ValueError:
        # Different drives on Windows
        return str(fig_path)


def _format_finding_row(f: KeyFinding) -> str:
    """Format a KeyFinding as a markdown table row."""
    paradigm = PARADIGM_DISPLAY_NAMES.get(f.paradigm, f.paradigm)
    analysis = ANALYSIS_DISPLAY_NAMES.get(f.analysis, f.analysis)
    return f"| {paradigm} | {analysis} | {f.measure} | {f.contrast} | {f.statistic} |"


def _methods_brief(summary: AnalysisSummary) -> str:
    """Extract a brief methods description from the summary."""
    text = summary.methods_text
    # Take lines that start with **Analysis:** or the first ~3 lines
    lines = text.strip().splitlines()
    brief_lines = []
    for line in lines[:6]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("**"):
            brief_lines.append(line)
    return " ".join(brief_lines) if brief_lines else lines[0] if lines else ""


# ── Figure caption helpers ───────────────────────────────────────────────────

def _format_contrast(s: str) -> str:
    """Convert contrast strings to readable form.

    ``"30mgkg_vs_Vehicle"`` → ``"30 mg/kg vs Vehicle"``
    ``"30mgkg_vs_6mgkg"`` → ``"30 mg/kg vs 6 mg/kg"``
    """
    s = s.replace("_vs_", " vs ")
    s = s.replace("30mgkg", "30 mg/kg").replace("6mgkg", "6 mg/kg")
    # Capitalize "vehicle" if it appears lowercase
    s = s.replace("vehicle", "Vehicle")
    return s


def _format_band(s: str) -> str:
    """Convert band slug to readable form.

    ``"high_gamma"`` → ``"High Gamma"``
    ``"alpha_1"`` → ``"Alpha 1"``
    """
    return s.replace("_", " ").title()


def _figure_caption(fig_path: Path, analysis: str, paradigm: str) -> str:
    """Generate a descriptive caption from figure filename and context."""
    stem = fig_path.stem

    # Radar charts
    if stem.startswith("radar_"):
        suffix = stem[len("radar_"):]
        if suffix == "dB" or suffix == "db":
            return "Regional power profile (dB) by treatment group"
        if suffix == "aperiodic":
            return "Aperiodic parameter profile by treatment group"
        if suffix == "pac":
            return "Phase-amplitude coupling z-score profile by treatment group"
        if suffix.startswith("network_"):
            metric = suffix[len("network_"):].replace("_", " ").title()
            return f"Network {metric} profile by treatment group"
        if suffix.startswith("evoked_"):
            measure = suffix[len("evoked_"):].upper()
            return f"{measure} regional profile by treatment group"
        # Fallback for other radar types
        return f"{_format_band(suffix)} profile by treatment group"

    # Brain ROI mosaics
    if stem.startswith("brain_roi_"):
        rest = stem[len("brain_roi_"):]
        # Pattern: brain_roi_{contrast}_{band}
        # Try to split on known contrast patterns
        for marker in ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle",
                       "30mgkg_vs_6mgkg", "30mgkg_vs_vehicle",
                       "6mgkg_vs_vehicle"]:
            if rest.startswith(marker):
                band = rest[len(marker):].lstrip("_")
                return f"Effect sizes for {_format_band(band)}: {_format_contrast(marker)}"
        # Fallback
        return f"ROI effect sizes: {rest.replace('_', ' ').title()}"

    # Circos comparison (3-panel: Group A | Group B | Difference)
    if stem.startswith("circos_comparison_"):
        rest = stem[len("circos_comparison_"):]
        # Pattern: circos_comparison_{band}_{contrast}
        # Try to split on known contrast patterns
        for marker in ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle",
                        "30mgkg_vs_6mgkg", "30mgkg_vs_vehicle",
                        "6mgkg_vs_vehicle"]:
            if rest.endswith(marker):
                band = rest[:-(len(marker) + 1)]  # strip _contrast
                return (f"Connectivity comparison — "
                        f"{_format_band(band)}: {_format_contrast(marker)}")
        return f"Connectivity comparison — {rest.replace('_', ' ').title()}"

    # Connectivity heatmaps
    if stem.startswith("connectivity_heatmap_"):
        band = stem[len("connectivity_heatmap_"):]
        return f"Mean coherence matrix — {_format_band(band)}"

    # Circos significance plots
    if stem.startswith("circos_sig_"):
        rest = stem[len("circos_sig_"):]
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            band, metric = parts
            return f"Significant {metric} differences — {_format_band(band)}"
        return f"Significant connectivity differences — {_format_band(rest)}"

    # Wholebrain glass brains
    if stem.startswith("wholebrain_"):
        band = stem[len("wholebrain_"):]
        if band == "summary":
            return "Whole-brain vertex-level power summary"
        return f"Whole-brain vertex-level power — {_format_band(band)}"

    # MVPA importance maps
    if stem.startswith("mvpa_importance_"):
        rest = stem[len("mvpa_importance_"):]
        for marker in ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle",
                       "30mgkg_vs_6mgkg", "30mgkg_vs_vehicle",
                       "6mgkg_vs_vehicle"]:
            if rest.startswith(marker):
                band = rest[len(marker):].lstrip("_")
                return (f"SVM feature importance — {_format_band(band)} "
                        f"({_format_contrast(marker)})")
        return f"SVM feature importance — {_format_band(rest)}"

    if stem.startswith("mvpa_null_"):
        rest = stem[len("mvpa_null_"):]
        # Try to extract contrast from the name
        for marker in ["30mgkg_vs_Vehicle", "6mgkg_vs_Vehicle",
                       "30mgkg_vs_6mgkg", "30mgkg_vs_vehicle",
                       "6mgkg_vs_vehicle"]:
            if rest.startswith(marker):
                band = rest[len(marker):].lstrip("_")
                return (f"MVPA null distribution — {_format_band(band)} "
                        f"({_format_contrast(marker)})")
        return f"MVPA null distribution — {_format_band(rest)}"

    # Specparam vertex
    if stem.startswith("specparam_"):
        param = stem[len("specparam_"):]
        return f"Vertex-level {param.replace('_', ' ')} distribution"

    # FCD maps
    if stem.startswith("fcd_"):
        rest = stem[len("fcd_"):]
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            band, metric = parts
            return f"Functional connectivity density — {_format_band(band)} ({metric})"
        return f"Functional connectivity density — {_format_band(rest)}"

    # Evoked time-series group plots
    if stem.startswith("evoked_"):
        rest = stem[len("evoked_"):]
        # Pattern: evoked_{type}_{measure}_group
        if rest.endswith("_group"):
            rest = rest[:-len("_group")]
        parts = rest.split("_", 1)
        if len(parts) == 2:
            etype, measure = parts
            return f"{etype.upper()} {measure.replace('_', ' ')} by treatment group"
        return f"Evoked {rest.replace('_', ' ')} by treatment group"

    # Vertex NBS edges
    if stem.startswith("vertex_nbs_edges_"):
        band = stem[len("vertex_nbs_edges_"):]
        return f"NBS significant edges — {_format_band(band)}"

    # Vertex network degree
    if stem.startswith("vertex_network_degree_"):
        band = stem[len("vertex_network_degree_"):]
        return f"Vertex network degree — {_format_band(band)}"

    # Gamma peak presence
    if stem == "gamma_peak_presence":
        return "Gamma peak presence across vertices"

    # PAC comodulogram
    if stem.startswith("pac_comodulogram_"):
        rest = stem[len("pac_comodulogram_"):]
        return f"PAC comodulogram — {rest.replace('_', ' ').title()}"

    # Fallback: clean stem
    return stem.replace("_", " ").title()


# ── On-demand figure generation ──────────────────────────────────────────────

def _roi_to_region_map(roi_categories: dict[str, list[str]]) -> dict[str, str]:
    """Build reverse mapping: ROI abbreviation -> region name."""
    mapping = {}
    for region, rois in roi_categories.items():
        for roi in rois:
            mapping[roi] = region
    return mapping


def _aggregate_to_regions(
    df: pd.DataFrame,
    roi_categories: dict[str, list[str]],
    value_cols: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    """Aggregate ROI-level data to region-level by averaging."""
    roi_map = _roi_to_region_map(roi_categories)
    df = df.copy()
    df["region"] = df["roi"].map(roi_map)
    df = df.dropna(subset=["region"])
    return df.groupby(group_cols + ["region"])[value_cols].mean().reset_index()


def _filter_significant_posthoc(
    posthoc_csv: Path,
    sig_col: str = "significant",
) -> pd.DataFrame | None:
    """Read a posthoc CSV and return only significant rows."""
    if not posthoc_csv.exists():
        return None
    df = pd.read_csv(posthoc_csv)
    if df.empty or sig_col not in df.columns:
        return None
    # Handle R-style TRUE/FALSE and Python True/False
    if df[sig_col].dtype == object:
        sig = df[df[sig_col].str.upper() == "TRUE"]
    else:
        sig = df[df[sig_col].astype(bool)]
    return sig if not sig.empty else None


def _load_sig_data(posthoc_csv: Path) -> pd.DataFrame | None:
    """Load posthoc CSV and extract significant rows as sig_data for radar.

    Returns a DataFrame with columns ``band``, ``region``, ``contrast``,
    ``sig_label`` or *None* if no significant results.
    """
    if not posthoc_csv.exists():
        return None
    try:
        ph = pd.read_csv(posthoc_csv)
    except Exception:
        return None
    if ph.empty:
        return None
    # Determine significance column
    sig_col = "significant" if "significant" in ph.columns else None
    if sig_col is None:
        return None
    if ph[sig_col].dtype == object:
        sig = ph[ph[sig_col].str.upper() == "TRUE"]
    else:
        sig = ph[ph[sig_col].astype(bool)]
    if sig.empty:
        return None
    # Build sig_data with required columns
    result = pd.DataFrame()
    result["band"] = sig["band"] if "band" in sig.columns else ""
    result["region"] = sig["region"] if "region" in sig.columns else ""
    result["contrast"] = sig["contrast"] if "contrast" in sig.columns else ""
    # Build sig_label from q_value or sig_label column
    if "sig_label" in sig.columns:
        result["sig_label"] = sig["sig_label"].values
    elif "q_value" in sig.columns:
        def _stars(q):
            try:
                q = float(q)
            except (TypeError, ValueError):
                return ""
            if q < 0.001:
                return "***"
            if q < 0.01:
                return "**"
            if q < 0.05:
                return "*"
            return ""
        result["sig_label"] = sig["q_value"].apply(_stars).values
    else:
        result["sig_label"] = "*"
    result = result[result["sig_label"].astype(str).str.strip() != ""]
    return result if not result.empty else None


def _generate_psd_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate PSD report figures (radar chart + brain ROI mosaics)."""
    from .viz import plot_radar, render_posthoc_mosaics

    saved: list[Path] = []
    band_power_csv = data_dir / "band_power.csv"
    colors = _get_report_colors("psd", config, color_mode)

    # Load significance data for radar markers
    sig_data = _load_sig_data(tbl_dir / "psd_posthoc_region.csv")

    # 1. Radar chart from band_power.csv
    if band_power_csv.exists():
        df = pd.read_csv(band_power_csv)
        region_df = _aggregate_to_regions(
            df, config.roi_categories,
            value_cols=["dB", "relative", "absolute"],
            group_cols=["subject", "group", "band"],
        )
        for value_col in ["dB"]:
            out = fig_dir / f"radar_{value_col}.png"
            try:
                plot_radar(
                    region_df, out, value_col=value_col,
                    group_colors=colors,
                    group_labels=config.groups,
                    title=f"PSD Regional Profile ({value_col})",
                    sig_data=sig_data,
                )
                saved.append(out)
                logger.info("Generated: %s", out)
            except Exception as e:
                logger.warning("Failed to generate PSD radar: %s", e)

    # 2. Brain ROI mosaics from posthoc_roi.csv (significant only)
    posthoc_csv = tbl_dir / "psd_posthoc_roi.csv"
    sig = _filter_significant_posthoc(posthoc_csv)
    if sig is not None and len(saved) < max_figures:
        # Render mosaics for significant contrast×band×power_type combos
        # Write filtered CSV to temp location
        filtered_csv = fig_dir / "_sig_posthoc.csv"
        sig.to_csv(filtered_csv, index=False)
        try:
            facet_cols = ["contrast", "band"]
            if "power_type" in sig.columns:
                # Only use dB power type for mosaics
                sig_db = sig[sig["power_type"] == "dB"]
                if not sig_db.empty:
                    sig_db.to_csv(filtered_csv, index=False)
                    facet_cols = ["contrast", "band"]
            mosaic_paths = render_posthoc_mosaics(
                filtered_csv, config.roi_categories, fig_dir,
                analysis_name="psd",
                roi_col="roi",
                facet_cols=facet_cols,
            )
            saved.extend(mosaic_paths[:max_figures - len(saved)])
        except Exception as e:
            logger.warning("Failed to generate PSD brain mosaics: %s", e)
        finally:
            filtered_csv.unlink(missing_ok=True)

    return saved


def _generate_aperiodic_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate aperiodic report figures (radar charts + brain ROI mosaics)."""
    from .viz import plot_radar, render_posthoc_mosaics

    saved: list[Path] = []
    params_csv = data_dir / "aperiodic_params.csv"
    colors = _get_report_colors("aperiodic", config, color_mode)

    # 1. Radar chart per parameter
    if params_csv.exists():
        df = pd.read_csv(params_csv)
        region_df = _aggregate_to_regions(
            df, config.roi_categories,
            value_cols=["exponent", "offset"],
            group_cols=["subject", "group"],
        )
        # Add a dummy band column for radar (aperiodic has no bands)
        # Instead, use exponent and offset as separate "bands" on a single chart
        # Reshape: pivot params to long form with "band" = param name
        melted = region_df.melt(
            id_vars=["subject", "group", "region"],
            value_vars=["exponent", "offset"],
            var_name="band", value_name="value",
        )
        out = fig_dir / "radar_aperiodic.png"
        try:
            plot_radar(
                melted, out, value_col="value",
                group_colors=colors,
                group_labels=config.groups,
                title="Aperiodic Parameters by Region",
            )
            saved.append(out)
            logger.info("Generated: %s", out)
        except Exception as e:
            logger.warning("Failed to generate aperiodic radar: %s", e)

    # 2. Brain ROI mosaics from posthoc
    posthoc_csv = tbl_dir / "aperiodic_posthoc_roi.csv"
    sig = _filter_significant_posthoc(posthoc_csv)
    if sig is not None and len(saved) < max_figures:
        filtered_csv = fig_dir / "_sig_posthoc.csv"
        sig.to_csv(filtered_csv, index=False)
        try:
            mosaic_paths = render_posthoc_mosaics(
                filtered_csv, config.roi_categories, fig_dir,
                analysis_name="aperiodic",
                roi_col="roi",
                facet_cols=["contrast", "dv"],
            )
            saved.extend(mosaic_paths[:max_figures - len(saved)])
        except Exception as e:
            logger.warning("Failed to generate aperiodic brain mosaics: %s", e)
        finally:
            filtered_csv.unlink(missing_ok=True)

    return saved


def _generate_roi_connectivity_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate ROI connectivity figures (heatmap + significance circos)."""
    from .viz import (
        build_roi_matrix, plot_connectivity_heatmap,
    )
    import matplotlib.pyplot as plt

    saved: list[Path] = []
    edges_csv = data_dir / "roi_connectivity_edges.csv"
    heatmap_cmap = _get_analysis_cmap("roi_connectivity", color_mode)

    if not edges_csv.exists():
        return saved

    df = pd.read_csv(edges_csv)

    # Preferred band/metric for connectivity display
    preferred_bands = ["high_gamma", "low_gamma", "beta"]
    metric = "coherence"

    for band in preferred_bands:
        if len(saved) >= max_figures:
            break
        band_df = df[df["band"] == band]
        if band_df.empty:
            continue

        # 1. Group mean heatmap
        try:
            matrix, roi_labels, region_names, region_sizes = build_roi_matrix(
                band_df, config.roi_categories, metric,
            )
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            plot_connectivity_heatmap(
                matrix, roi_labels, region_names, region_sizes, ax,
                cmap=heatmap_cmap,
            )
            ax.set_title(f"Mean {metric.title()} — {band.replace('_', ' ').title()}")
            out = fig_dir / f"connectivity_heatmap_{band}.png"
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            saved.append(out)
            logger.info("Generated: %s", out)
        except Exception as e:
            logger.warning("Failed to generate connectivity heatmap (%s): %s", band, e)

    # 2. Per-condition circos comparison (Group A | Group B | Difference)
    from .viz import plot_connectivity_comparison

    for band in preferred_bands:
        if len(saved) >= max_figures:
            break
        band_df = df[df["band"] == band]
        if band_df.empty:
            continue

        for contrast in config.contrasts[:2]:  # Top 2 contrasts
            if len(saved) >= max_figures:
                break
            try:
                mat_a, roi_labels, region_names, region_sizes = build_roi_matrix(
                    band_df, config.roi_categories, metric, group=contrast.group_a,
                )
                mat_b, _, _, _ = build_roi_matrix(
                    band_df, config.roi_categories, metric, group=contrast.group_b,
                )
                out = fig_dir / f"circos_comparison_{band}_{contrast.name}.png"
                plot_connectivity_comparison(
                    mat_a, mat_b, roi_labels, region_names, region_sizes, out,
                    plot_type="circos",
                    group_labels=(
                        config.get_group_label(contrast.group_a),
                        config.get_group_label(contrast.group_b),
                    ),
                    title=f"Coherence — {band.replace('_', ' ').title()}",
                    show_roi_labels=False,
                )
                saved.append(out)
                logger.info("Generated: %s", out)
            except Exception as e:
                logger.warning(
                    "Failed to generate circos comparison (%s/%s): %s",
                    band, contrast.name, e,
                )

    return saved


def _generate_pac_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate PAC report figures (radar chart + brain ROI mosaics)."""
    from .viz import plot_radar, render_posthoc_mosaics

    saved: list[Path] = []
    pac_csv = data_dir / "pac_values.csv"
    colors = _get_report_colors("pac", config, color_mode)

    # Load significance data for radar markers
    sig_data = _load_sig_data(tbl_dir / "pac_posthoc_region.csv")

    # 1. Radar chart from pac_values.csv
    if pac_csv.exists():
        df = pd.read_csv(pac_csv)
        # Aggregate z_score to region level using freq_pair as "band"
        region_df = _aggregate_to_regions(
            df, config.roi_categories,
            value_cols=["z_score"],
            group_cols=["subject", "group", "freq_pair"],
        )
        region_df = region_df.rename(columns={"freq_pair": "band"})
        out = fig_dir / "radar_pac.png"
        try:
            plot_radar(
                region_df, out, value_col="z_score",
                group_colors=colors,
                group_labels=config.groups,
                title="PAC Z-Score by Region",
                sig_data=sig_data,
            )
            saved.append(out)
            logger.info("Generated: %s", out)
        except Exception as e:
            logger.warning("Failed to generate PAC radar: %s", e)

    # 2. Brain ROI mosaics from posthoc
    posthoc_csv = tbl_dir / "pac_posthoc_region.csv"
    sig = _filter_significant_posthoc(posthoc_csv)
    if sig is not None and len(saved) < max_figures:
        filtered_csv = fig_dir / "_sig_posthoc.csv"
        sig.to_csv(filtered_csv, index=False)
        try:
            mosaic_paths = render_posthoc_mosaics(
                filtered_csv, config.roi_categories, fig_dir,
                analysis_name="pac",
                roi_col="region",
                facet_cols=["contrast", "freq_pair"],
            )
            saved.extend(mosaic_paths[:max_figures - len(saved)])
        except Exception as e:
            logger.warning("Failed to generate PAC brain mosaics: %s", e)
        finally:
            filtered_csv.unlink(missing_ok=True)

    return saved


def _generate_roi_network_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate ROI network figures (radar chart of nodal metrics)."""
    from .viz import plot_radar

    saved: list[Path] = []
    nodal_csv = data_dir / "roi_network_nodal_metrics.csv"
    colors = _get_report_colors("roi_network", config, color_mode)

    if not nodal_csv.exists():
        return saved

    df = pd.read_csv(nodal_csv)

    # Radar chart of degree by band
    for metric_col in ["degree", "clustering"]:
        if len(saved) >= max_figures:
            break
        region_df = _aggregate_to_regions(
            df, config.roi_categories,
            value_cols=[metric_col],
            group_cols=["subject", "group", "band"],
        )
        out = fig_dir / f"radar_network_{metric_col}.png"
        try:
            plot_radar(
                region_df, out, value_col=metric_col,
                group_colors=colors,
                group_labels=config.groups,
                title=f"Network {metric_col.title()} by Region",
            )
            saved.append(out)
            logger.info("Generated: %s", out)
        except Exception as e:
            logger.warning("Failed to generate network radar (%s): %s", metric_col, e)

    return saved


def _generate_evoked_figures(
    config: StudyConfig,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate evoked report figures (radar chart + brain ROI mosaics)."""
    from .viz import plot_radar, render_posthoc_mosaics

    saved: list[Path] = []
    evoked_csv = data_dir / "evoked_measures.csv"
    colors = _get_report_colors("evoked", config, color_mode)

    # 1. Radar chart from evoked_measures.csv
    if evoked_csv.exists():
        df = pd.read_csv(evoked_csv)
        # Use measure_name as "band" for radar chart, value as the measure
        region_df = _aggregate_to_regions(
            df, config.roi_categories,
            value_cols=["value"],
            group_cols=["subject", "group", "measure_name"],
        )
        region_df = region_df.rename(columns={"measure_name": "band"})

        # Only show ITC measures (most relevant for FXS)
        itc_df = region_df[region_df["band"].str.startswith("itc")]
        if not itc_df.empty:
            out = fig_dir / "radar_evoked_itc.png"
            try:
                plot_radar(
                    itc_df, out, value_col="value",
                    group_colors=colors,
                    group_labels=config.groups,
                    title="ITC Regional Profile",
                )
                saved.append(out)
                logger.info("Generated: %s", out)
            except Exception as e:
                logger.warning("Failed to generate evoked ITC radar: %s", e)

    # 2. Brain ROI mosaics from posthoc
    posthoc_csv = tbl_dir / "evoked_posthoc_roi.csv"
    sig = _filter_significant_posthoc(posthoc_csv)
    if sig is not None and len(saved) < max_figures:
        filtered_csv = fig_dir / "_sig_posthoc.csv"
        sig.to_csv(filtered_csv, index=False)
        try:
            mosaic_paths = render_posthoc_mosaics(
                filtered_csv, config.roi_categories, fig_dir,
                analysis_name="evoked",
                roi_col="roi",
                facet_cols=["contrast", "measure"],
            )
            saved.extend(mosaic_paths[:max_figures - len(saved)])
        except Exception as e:
            logger.warning("Failed to generate evoked brain mosaics: %s", e)
        finally:
            filtered_csv.unlink(missing_ok=True)

    return saved


# Dispatcher: analysis name -> generator function
_FIGURE_GENERATORS: dict[str, callable] = {
    "psd": _generate_psd_figures,
    "aperiodic": _generate_aperiodic_figures,
    "roi_connectivity": _generate_roi_connectivity_figures,
    "pac": _generate_pac_figures,
    "roi_network": _generate_roi_network_figures,
    "evoked": _generate_evoked_figures,
}


def generate_figures(
    config: StudyConfig,
    paradigm: str,
    analysis: str,
    fig_dir: Path,
    data_dir: Path,
    tbl_dir: Path,
    max_figures: int = 4,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> list[Path]:
    """Generate on-demand figures for one analysis.

    Dispatches to analysis-specific generators that read intermediate CSVs
    and statistical tables to produce radar charts and brain ROI mosaics.

    Parameters
    ----------
    config : StudyConfig
        Study configuration (for roi_categories, group info).
    paradigm : str
        Paradigm name (e.g. "resting", "chirp").
    analysis : str
        Analysis name (e.g. "psd", "evoked").
    fig_dir : Path
        Output directory for generated figures (report/ subdir is created).
    data_dir : Path
        Directory containing intermediate CSVs.
    tbl_dir : Path
        Directory containing statistical tables.
    max_figures : int
        Maximum number of figures to generate.
    color_mode : ColorMode
        Color scheme mode for generated figures.

    Returns
    -------
    list[Path]
        Paths to generated figures.
    """
    generator = _FIGURE_GENERATORS.get(analysis)
    if generator is None:
        logger.debug("No on-demand figure generator for %s/%s", paradigm, analysis)
        return []

    # Create report/ subdirectory
    report_fig_dir = fig_dir / "report"
    report_fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        paths = generator(
            config, report_fig_dir, data_dir, tbl_dir, max_figures,
            color_mode=color_mode,
        )
        if paths:
            logger.info("Generated %d on-demand figures for %s/%s",
                        len(paths), paradigm, analysis)
        return paths
    except Exception as e:
        logger.warning("On-demand figure generation failed for %s/%s: %s",
                       paradigm, analysis, e)
        return []


# ── Report section building ─────────────────────────────────────────────────

def _build_analysis_section(
    rb: ReportBuilder,
    summary: AnalysisSummary,
    config: StudyConfig,
    paradigm: str,
    fig_dir: Path,
    report_dir: Path,
    max_figures: int,
    no_figures: bool = False,
    heading_level: int = 3,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> None:
    """Add one analysis subsection to the report."""
    display_name = ANALYSIS_DISPLAY_NAMES.get(
        summary.analysis, summary.analysis,
    )
    rb.add_heading(display_name, level=heading_level)

    # Brief methods
    brief = _methods_brief(summary)
    if brief:
        rb.add_text(brief)

    # Key findings
    if summary.key_findings:
        rb.add_text(f"**Significant results ({len(summary.key_findings)}):**\n")
        for f in summary.key_findings[:10]:  # Cap at 10 to keep report manageable
            rb.add_text(f"- **{f.measure}** [{f.contrast}]: {f.details}")
    else:
        rb.add_text("No significant effects after correction for multiple comparisons.")

    # On-demand figure generation
    if not no_figures:
        data_dir = config.output_dir / paradigm / summary.analysis / "data"
        tbl_dir = config.results_dir / "tables" / paradigm / summary.analysis
        generate_figures(
            config, paradigm, summary.analysis,
            fig_dir, data_dir, tbl_dir, max_figures,
            color_mode=color_mode,
        )

    # Select figures (from report/ subdir + pre-existing)
    figures = select_figures(summary.analysis, fig_dir, max_figures)
    if figures:
        for fig in figures:
            rel = _make_relative(fig, report_dir)
            caption = _figure_caption(fig, summary.analysis, paradigm)
            rb.add_figure(rel, caption=caption, width="90%")


def build_synthesis(all_findings: list[KeyFinding]) -> str:
    """Build a cross-paradigm synthesis narrative from all findings."""
    if not all_findings:
        return "No statistically significant effects were observed across any analysis after correction for multiple comparisons."

    # Count by contrast
    contrast_counts: dict[str, int] = {}
    band_counts: dict[str, int] = {}
    paradigm_counts: dict[str, int] = {}

    for f in all_findings:
        contrast_counts[f.contrast] = contrast_counts.get(f.contrast, 0) + 1
        paradigm_counts[f.paradigm] = paradigm_counts.get(f.paradigm, 0) + 1
        # Extract band from measure
        for band in ["delta", "theta", "alpha", "beta", "low_gamma",
                      "high_gamma", "gamma", "epsilon"]:
            if band in f.measure.lower():
                band_counts[band] = band_counts.get(band, 0) + 1
                break

    lines = []
    lines.append(f"Across all analyses, **{len(all_findings)} significant effects** "
                 f"were identified.")

    # Most affected contrast
    if contrast_counts:
        top_contrast = max(contrast_counts, key=contrast_counts.get)
        lines.append(f"The **{top_contrast}** contrast showed the most findings "
                     f"({contrast_counts[top_contrast]} effects).")

    # Most affected band
    if band_counts:
        top_band = max(band_counts, key=band_counts.get)
        lines.append(f"**{top_band.replace('_', ' ').title()}** was the most "
                     f"frequently implicated frequency band "
                     f"({band_counts[top_band]} effects).")

    # Per-paradigm breakdown
    if len(paradigm_counts) > 1:
        breakdown = ", ".join(
            f"{PARADIGM_DISPLAY_NAMES.get(p, p)} ({n})"
            for p, n in sorted(paradigm_counts.items(), key=lambda x: -x[1])
        )
        lines.append(f"Effects by paradigm: {breakdown}.")

    return "\n\n".join(lines)


def generate_report(
    config: StudyConfig,
    output_path: Path | None = None,
    max_figures: int = 4,
    no_figures: bool = False,
    color_mode: ColorMode = ColorMode.ANALYSIS,
) -> Path:
    """Generate a Quarto results report from completed analyses.

    Parameters
    ----------
    config : StudyConfig
        Study configuration (must have paradigms).
    output_path : Path or None
        Output .qmd path. Defaults to results_dir/reports/results_report.qmd.
    max_figures : int
        Maximum figures per analysis section.
    no_figures : bool
        If True, skip on-demand figure generation and use only pre-existing
        figures. Default False.
    color_mode : ColorMode
        Color scheme for generated figures. ``"analysis"`` uses per-analysis
        color themes; ``"publication"`` uses uniform config colors.

    Returns
    -------
    Path
        Path to the generated .qmd file.
    """
    if output_path is None:
        output_path = config.results_dir / "reports" / "results_report.qmd"
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output_path.parent

    # Discover completed analyses
    completed = discover_completed_analyses(config)
    if not completed:
        raise RuntimeError("No completed analyses found (no ANALYSIS_SUMMARY.md files)")

    logger.info("Found %d completed analyses", len(completed))

    # Parse all summaries
    summaries: list[AnalysisSummary] = []
    for paradigm, analysis, summary_path in completed:
        try:
            s = parse_analysis_summary(summary_path, paradigm, analysis)
            summaries.append(s)
            logger.info("Parsed: %s/%s (%d findings)", paradigm, analysis,
                        len(s.key_findings))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", summary_path, e)

    # Organize by paradigm
    resting = [s for s in summaries if s.paradigm == "resting"]
    evoked = [s for s in summaries if s.paradigm != "resting"]

    # Collect all findings
    all_findings = [f for s in summaries for f in s.key_findings]

    # Build report
    rb = ReportBuilder()

    # ── YAML front matter ──
    rb.add_raw("---")
    rb.add_raw(f'title: "{config.name} — Results Report"')
    rb.add_raw('format:')
    rb.add_raw('  html:')
    rb.add_raw('    toc: true')
    rb.add_raw('    toc-depth: 3')
    rb.add_raw('    number-sections: true')
    rb.add_raw('    embed-resources: true')
    rb.add_raw('  pdf:')
    rb.add_raw('    toc: true')
    rb.add_raw('    number-sections: true')
    rb.add_raw('    documentclass: article')
    rb.add_raw('    geometry: margin=1in')
    rb.add_raw("---")

    # ── Study Overview ──
    rb.add_heading("Study Overview")

    # Groups table
    group_lines = ["| Group | Label | N |", "| --- | --- | --- |"]
    for gid in config.group_order:
        label = config.get_group_label(gid)
        # Count subjects from paradigms
        n = "—"
        for pname, pdata in config.paradigms.items():
            subjects = pdata.get("subjects", {})
            count = sum(1 for v in subjects.values() if v == gid)
            if count > 0:
                n = str(count)
                break
        group_lines.append(f"| {gid} | {label} | {n} |")
    rb.add_text("\n".join(group_lines))

    # Paradigms
    rb.add_text("**Paradigms analyzed:**\n")
    for pname in config.paradigms:
        analyses = config.get_paradigm_analyses(pname) or []
        display = PARADIGM_DISPLAY_NAMES.get(pname, pname)
        completed_names = [a for _, a, _ in completed if _ == pname]
        # Fix: need paradigm match
        completed_names = [a for p, a, _ in completed if p == pname]
        rb.add_text(f"- **{display}**: {', '.join(completed_names)} "
                    f"({len(completed_names)}/{len(analyses)} complete)")

    rb.add_text(f"\n**Contrasts:** " + ", ".join(
        f"{c.name} ({config.get_group_label(c.group_a)} vs "
        f"{config.get_group_label(c.group_b)})"
        for c in config.contrasts
    ))

    # ── Resting State Results ──
    if resting:
        rb.add_heading("Resting State Results", level=2)

        # Order: ROI-level first, then wholebrain
        roi_order = ["psd", "aperiodic", "roi_connectivity", "pac", "roi_network"]
        wb_order = ["wholebrain", "mvpa", "specparam_vertex",
                     "vertex_connectivity", "vertex_network"]

        resting_by_name = {s.analysis: s for s in resting}

        rb.add_heading("ROI-Level Analyses", level=3)
        for aname in roi_order:
            if aname in resting_by_name:
                s = resting_by_name[aname]
                fig_dir = config.results_dir / "figures" / "resting" / aname
                _build_analysis_section(
                    rb, s, config, "resting", fig_dir, report_dir,
                    max_figures, no_figures=no_figures, heading_level=4,
                    color_mode=color_mode,
                )

        rb.add_heading("Whole-Brain Analyses", level=3)
        for aname in wb_order:
            if aname in resting_by_name:
                s = resting_by_name[aname]
                fig_dir = config.results_dir / "figures" / "resting" / aname
                _build_analysis_section(
                    rb, s, config, "resting", fig_dir, report_dir,
                    max_figures, no_figures=no_figures, heading_level=4,
                    color_mode=color_mode,
                )

    # ── Evoked Response Results ──
    if evoked:
        rb.add_heading("Evoked Response Results", level=2)

        evoked_order = ["chirp", "40hz_assr", "80hz_assr"]
        for pname in evoked_order:
            paradigm_summaries = [s for s in evoked if s.paradigm == pname]
            if not paradigm_summaries:
                continue

            display = PARADIGM_DISPLAY_NAMES.get(pname, pname)
            rb.add_heading(display, level=3)

            for s in paradigm_summaries:
                fig_dir = config.results_dir / "figures" / pname / s.analysis
                _build_analysis_section(
                    rb, s, config, pname, fig_dir, report_dir,
                    max_figures, no_figures=no_figures, heading_level=4,
                    color_mode=color_mode,
                )

    # ── Cross-Paradigm Summary ──
    rb.add_heading("Cross-Paradigm Summary", level=2)

    # Synthesis narrative
    synthesis = build_synthesis(all_findings)
    rb.add_text(synthesis)

    # Summary table
    if all_findings:
        rb.add_heading("All Significant Findings", level=3)
        table_lines = [
            "| Paradigm | Analysis | Measure | Contrast | Statistic |",
            "| --- | --- | --- | --- | --- |",
        ]
        for f in all_findings:
            table_lines.append(_format_finding_row(f))
        rb.add_text("\n".join(table_lines))

    # Write
    content = rb.build()
    output_path.write_text(content, encoding="utf-8")
    logger.info("Report written: %s", output_path)

    # Validate figure paths
    fig_pattern = re.compile(r'!\[.*?\]\((.+?)\)')
    missing = 0
    for match in fig_pattern.finditer(content):
        fig_rel = match.group(1).split("){")[0]  # strip Quarto attributes
        fig_abs = (report_dir / fig_rel).resolve()
        if not fig_abs.exists():
            logger.warning("Missing figure: %s", fig_abs)
            missing += 1
    if missing:
        logger.warning("%d figure path(s) could not be resolved", missing)
    else:
        logger.info("All figure paths verified")

    return output_path

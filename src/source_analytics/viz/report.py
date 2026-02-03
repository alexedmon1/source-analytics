"""Markdown summary report writer."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ReportWriter:
    """Builds a markdown analysis summary."""

    def __init__(self, title: str):
        self.title = title
        self.sections: list[str] = []
        self._add_header()

    def _add_header(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.sections.append(f"# {self.title}\n")
        self.sections.append(f"**Generated:** {now}\n")

    def add_section(self, heading: str, content: str):
        self.sections.append(f"\n## {heading}\n")
        self.sections.append(content)

    def add_methods(
        self,
        n_subjects: dict[str, int],
        bands: dict[str, tuple[float, float]],
        sfreq: float,
        analysis_name: str,
    ):
        """Add a methods section."""
        group_str = ", ".join(f"{g} (n={n})" for g, n in n_subjects.items())
        band_str = ", ".join(f"{b}: {lo}-{hi} Hz" for b, (lo, hi) in bands.items())

        text = (
            f"**Analysis:** {analysis_name}\n\n"
            f"**Groups:** {group_str}\n\n"
            f"**Sampling Rate:** {sfreq} Hz\n\n"
            f"**Frequency Bands:** {band_str}\n\n"
            f"**PSD Method:** Welch's method (2-second Hann windows, 50% overlap)\n\n"
            f"**Statistics:** Independent samples Welch's t-test and linear mixed models "
            f"(LMM: power ~ group + (1|subject)). "
            f"Effect sizes reported as Hedges' g. "
            f"Multiple comparison correction via Benjamini-Hochberg FDR.\n"
        )
        self.add_section("Methods", text)

    def add_statistics_table(self, stats_df: pd.DataFrame, heading: str = "Results"):
        """Add a statistics table as markdown."""
        if stats_df.empty:
            self.add_section(heading, "*No results available.*\n")
            return

        # Format for display
        display_cols = [c for c in [
            "contrast", "band", "t_stat", "p_value", "q_value",
            "hedges_g", "lmm_z", "lmm_p", "lmm_q",
            "group_a_mean", "group_b_mean", "significant",
        ] if c in stats_df.columns]

        table = stats_df[display_cols].to_markdown(index=False, floatfmt=".4f")
        self.add_section(heading, table + "\n")

    def add_key_findings(self, findings: list[str]):
        """Add a bullet list of key findings."""
        if not findings:
            return
        content = "\n".join(f"- {f}" for f in findings)
        self.add_section("Key Findings", content + "\n")

    def add_figure_reference(self, fig_path: Path, caption: str):
        """Add a figure reference."""
        self.sections.append(f"\n![{caption}]({fig_path.name})\n")

    def write(self, output_path: Path):
        """Write the report to a file."""
        text = "\n".join(self.sections)
        output_path.write_text(text)
        logger.info("Report written: %s", output_path)

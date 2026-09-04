"""Vertex-level spatial analysis — RETIRED.

This module used to fit a per-contrast spatial-covariance GLS (nlme::gls,
corExp + nugget) on per-vertex band power as a robustness check on the vertex
group difference. It iterated the legacy ``config$contrasts`` block, which the
declarative ``design:``/``hypotheses:`` spec no longer populates, and the
spatial-covariance table was never a manuscript result. It was retired rather
than migrated (2026-06): spatially-resolved vertex inference is delivered by
``vertex_cluster`` (cluster-based permutation glass-brain maps) and
``vertex_nbs`` (network-based statistic).

The module is kept in the registry so old configs and ``--analysis
vertex_spatial`` (and its ``spatial_lmm`` alias) still resolve, but it does no
work: it neither loads source estimates nor calls R. It writes well-formed
empty result tables and a retirement note so downstream consumers (the
gallery) find the expected files, then exits cleanly. ``R/vertex_spatial_analysis.R``
is retained for reference only and is not invoked.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import StudyConfig
from ..io.discovery import SubjectInfo
from .base import BaseAnalysis

logger = logging.getLogger(__name__)

RETIRE_NOTE = (
    "vertex_spatial is RETIRED (design-spec migration, 2026-06). The per-contrast "
    "GLS spatial-covariance robustness model iterated config$contrasts, which the "
    "declarative design:/hypotheses: spec no longer populates. Spatially-resolved "
    "vertex inference is provided by vertex_cluster (cluster-permutation glass-brain "
    "maps) and vertex_nbs (network-based statistic). No data was processed."
)


class VertexSpatialAnalysis(BaseAnalysis):
    """RETIRED — writes empty result tables + a note; processes no subjects."""

    name = "vertex_spatial"
    SELECTABLE = {"band": "frequency band"}

    def __init__(self, config: StudyConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self._warned = False

    def _warn_once(self) -> None:
        if not self._warned:
            logger.warning(RETIRE_NOTE)
            self._warned = True

    def setup(self) -> None:
        self._warn_once()

    def process_subject(self, subject: SubjectInfo) -> None:  # noqa: D401
        """No-op: the retired module loads nothing."""

    def aggregate(self) -> None:  # noqa: D401
        """No-op: nothing was computed."""

    def statistics(self) -> None:
        """Write the well-formed empty tables downstream consumers expect."""
        tbl_dir = self.tbl_dir
        tbl_dir.mkdir(parents=True, exist_ok=True)
        for name in ("vertex_spatial_results.csv", "vertex_spatial_residuals.csv"):
            pd.DataFrame().to_csv(tbl_dir / name, index=False)
        logger.info("vertex_spatial (retired): wrote empty result tables to %s", tbl_dir)

    def figures(self) -> None:  # noqa: D401
        """No figures: there are no results."""

    def summary(self) -> None:
        lines = [
            "# Vertex Spatial Analysis — RETIRED",
            "",
            f"**Study**: {self.config.name}",
            "",
            RETIRE_NOTE,
            "",
            "## Output Files",
            "",
            "- `tables/vertex_spatial_results.csv` — empty (retired)",
            "- `tables/vertex_spatial_residuals.csv` — empty (retired)",
            "",
        ]
        path = self.output_dir / "ANALYSIS_SUMMARY.md"
        path.write_text("\n".join(lines))
        logger.info("Wrote %s", path)

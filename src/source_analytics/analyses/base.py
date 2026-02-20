"""BaseAnalysis: abstract base class for all analysis modules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import StudyConfig
from ..io.discovery import SubjectInfo

logger = logging.getLogger(__name__)

VALID_STEPS = {"setup", "process", "aggregate", "statistics", "figures", "summary"}


def find_r_script_dir() -> Path:
    """Locate the R/ directory relative to this package.

    Searches upward from the analyses/ directory to find the R/ scripts
    directory that lives at the package root (sibling to src/).
    """
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent  # src/../..
    r_dir = pkg_root / "R"
    if r_dir.is_dir():
        return r_dir
    for candidate in [Path.cwd() / "R", Path(__file__).parent.parent.parent / "R"]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find R/ scripts directory. Expected at: " + str(pkg_root / "R")
    )


class BaseAnalysis(ABC):
    """Abstract base for analysis modules.

    Lifecycle: setup → process_subject (per subject) → aggregate → statistics → figures → summary
    """

    name: str = "base"

    def __init__(self, config: StudyConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        # figures and tables go under results_dir
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tbl_dir.mkdir(parents=True, exist_ok=True)

    @property
    def fig_dir(self) -> Path:
        """Directory for figures (under results_dir)."""
        paradigm = self.config.paradigm_name or ""
        return self.config.results_dir / "figures" / paradigm / self.name

    @property
    def tbl_dir(self) -> Path:
        """Directory for tables (under results_dir)."""
        paradigm = self.config.paradigm_name or ""
        return self.config.results_dir / "tables" / paradigm / self.name

    @abstractmethod
    def setup(self) -> None:
        """Initialize analysis-specific data structures."""
        ...

    @abstractmethod
    def process_subject(self, subject: SubjectInfo) -> None:
        """Process a single subject's data. Called once per subject."""
        ...

    @abstractmethod
    def aggregate(self) -> None:
        """Aggregate subject-level results into group-level summaries."""
        ...

    @abstractmethod
    def statistics(self) -> None:
        """Run statistical tests on aggregated data."""
        ...

    @abstractmethod
    def figures(self) -> None:
        """Generate publication-quality figures."""
        ...

    @abstractmethod
    def summary(self) -> None:
        """Write markdown summary report."""
        ...

    def run(self, subjects: list[SubjectInfo], steps: set[str] | None = None) -> None:
        """Execute the analysis lifecycle.

        Parameters
        ----------
        subjects : list[SubjectInfo]
            Subjects to process.
        steps : set[str] | None
            If provided, only run these steps (from VALID_STEPS).
            ``setup`` always runs. If None, run all steps.
        """
        logger.info("=== %s Analysis ===", self.name)
        if steps is not None:
            logger.info("Running steps: %s", ", ".join(sorted(steps)))

        def _should_run(step: str) -> bool:
            return steps is None or step in steps

        logger.info("Step 1/6: Setup")
        self.setup()

        if _should_run("process"):
            logger.info("Step 2/6: Processing %d subjects", len(subjects))
            for i, subject in enumerate(subjects, 1):
                logger.info("  [%d/%d] %s (%s)", i, len(subjects), subject.subject_id, subject.group)
                try:
                    self.process_subject(subject)
                except Exception as e:
                    logger.error("  Failed to process %s: %s", subject.subject_id, e)
        else:
            logger.info("Step 2/6: Processing — skipped")

        if _should_run("aggregate"):
            logger.info("Step 3/6: Aggregating")
            self.aggregate()
        else:
            logger.info("Step 3/6: Aggregating — skipped")

        if _should_run("statistics"):
            logger.info("Step 4/6: Statistics")
            self.statistics()
        else:
            logger.info("Step 4/6: Statistics — skipped")

        if _should_run("figures"):
            logger.info("Step 5/6: Figures")
            self.figures()
        else:
            logger.info("Step 5/6: Figures — skipped")

        if _should_run("summary"):
            logger.info("Step 6/6: Summary")
            self.summary()
        else:
            logger.info("Step 6/6: Summary — skipped")

        logger.info("=== %s complete ===", self.name)

"""BaseAnalysis: abstract base class for all analysis modules."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import StudyConfig
from ..io.discovery import SubjectInfo

logger = logging.getLogger(__name__)


class BaseAnalysis(ABC):
    """Abstract base for analysis modules.

    Lifecycle: setup → process_subject (per subject) → aggregate → statistics → figures → summary
    """

    name: str = "base"

    def __init__(self, config: StudyConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

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

    def run(self, subjects: list[SubjectInfo]) -> None:
        """Execute the full analysis lifecycle."""
        logger.info("=== %s Analysis ===", self.name)

        logger.info("Step 1/6: Setup")
        self.setup()

        logger.info("Step 2/6: Processing %d subjects", len(subjects))
        for i, subject in enumerate(subjects, 1):
            logger.info("  [%d/%d] %s (%s)", i, len(subjects), subject.subject_id, subject.group)
            try:
                self.process_subject(subject)
            except Exception as e:
                logger.error("  Failed to process %s: %s", subject.subject_id, e)

        logger.info("Step 3/6: Aggregating")
        self.aggregate()

        logger.info("Step 4/6: Statistics")
        self.statistics()

        logger.info("Step 5/6: Figures")
        self.figures()

        logger.info("Step 6/6: Summary")
        self.summary()

        logger.info("=== %s complete ===", self.name)

"""BaseAnalysis: abstract base class for all analysis modules."""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ..config import StudyConfig
from ..io.discovery import SubjectInfo

logger = logging.getLogger(__name__)

VALID_STEPS = {"setup", "process", "aggregate", "statistics", "figures", "summary"}
DEFAULT_RUN_STEPS = VALID_STEPS - {"figures"}


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
        self._generate_figures = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        # figures and tables go under results_dir
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tbl_dir.mkdir(parents=True, exist_ok=True)

        # Resolve atlas directory for on-the-fly ROI extraction
        from ..atlas.atlas_utils import find_atlas_dir

        atlas_name = config.raw.get("pipeline", {}).get("atlas")
        atlas_dir_cfg = config.raw.get("atlas_dir")
        try:
            self._atlas_dir: Path | None = find_atlas_dir(
                atlas_dir_cfg, atlas_name=atlas_name,
            )
        except FileNotFoundError:
            self._atlas_dir = None

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

    def _call_r_figures_only(self, r_script_name: str, data_csv: str) -> bool:
        """Call an R analysis script with --figures-only to regenerate figures.

        Parameters
        ----------
        r_script_name : str
            Name of the R script (e.g. "psd_analysis.R").
        data_csv : str
            Name of the required data CSV (e.g. "band_power.csv").
            Used to verify that data exists before calling R.

        Returns
        -------
        bool
            True if the R script ran successfully.
        """
        data_dir = self.output_dir / "data"
        if not (data_dir / data_csv).exists():
            logger.warning(
                "%s not found — skipping R figure regeneration", data_csv,
            )
            return False

        try:
            r_dir = find_r_script_dir()
        except FileNotFoundError as e:
            logger.error(str(e))
            return False

        r_script = r_dir / r_script_name
        if not r_script.exists():
            logger.error("R script not found: %s", r_script)
            return False

        # Ensure study config YAML exists in data dir
        config_path = data_dir / "study_config.yaml"
        if not config_path.exists():
            import yaml
            config_data = dict(self.config.raw)
            if hasattr(self, "_sfreq") and self._sfreq is not None:
                config_data["sfreq"] = self._sfreq
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)

        cmd = [
            "Rscript", str(r_script),
            "--data-dir", str(data_dir),
            "--config", str(config_path),
            "--output-dir", str(self.output_dir),
            "--fig-dir", str(self.fig_dir),
            "--tbl-dir", str(self.tbl_dir),
            "--figures-only",
        ]

        logger.info("Calling R (figures-only): %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    if line.strip():
                        logger.info("[R] %s", line)
            if result.returncode != 0:
                logger.error(
                    "R figures-only failed (exit %d)", result.returncode,
                )
                return False
            return True
        except FileNotFoundError:
            logger.error("Rscript not found — install R for figure generation")
            return False
        except subprocess.TimeoutExpired:
            logger.error("R figures-only timed out after 300s")
            return False

    def _r_no_figures_flags(self) -> list[str]:
        """Return ['--no-figures'] if figure generation is disabled, else []."""
        if not self._generate_figures:
            return ["--no-figures"]
        return []

    def run(self, subjects: list[SubjectInfo], steps: set[str] | None = None) -> None:
        """Execute the analysis lifecycle.

        Parameters
        ----------
        subjects : list[SubjectInfo]
            Subjects to process.
        steps : set[str] | None
            If provided, only run these steps (from VALID_STEPS).
            ``setup`` always runs. If None, run DEFAULT_RUN_STEPS
            (all steps except ``figures``).
        """
        logger.info("=== %s Analysis ===", self.name)
        if steps is None:
            steps = DEFAULT_RUN_STEPS
        self._generate_figures = "figures" in steps
        logger.info("Running steps: %s", ", ".join(sorted(steps)))

        def _should_run(step: str) -> bool:
            return step in steps

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

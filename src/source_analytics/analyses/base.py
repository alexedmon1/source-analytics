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

    # Sub-output dimensions this analysis can filter via ``--select`` (and the
    # ``--metric`` / ``--band`` shorthands). Maps dimension name -> short label
    # used in ``--list`` / help. Empty = the whole module is the unit (no
    # per-output selection). Declaring a dim is a promise that ``setup`` (or
    # earlier) routes the corresponding configured list through ``self._select``.
    SELECTABLE: dict[str, str] = {}

    def __init__(self, config: StudyConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self._generate_figures = True
        # Active sub-output selection (dim -> normalized allowed values), set by
        # run(select=...). Empty = run every configured sub-output. See _select.
        self._selection: dict[str, frozenset[str]] = {}
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

        # Epoch sampling config (applies to both ROI and vertex analyses)
        # Per-analysis override: analyses can set epoch_sampling in their
        # paradigm config block to override the global settings.
        analysis_cfg = config.raw.get(self.name, {})
        analysis_epoch = analysis_cfg.get("epoch_sampling", {})
        global_epoch = config.raw.get("epoch_sampling", {})
        epoch_cfg = {**global_epoch, **analysis_epoch}
        self._epoch_equalize: bool = epoch_cfg.get("enabled", False)
        self._epoch_duration_sec: float = epoch_cfg.get("epoch_duration_sec", 2.0)
        self._epoch_n_epochs: int = epoch_cfg.get("n_epochs", 80)
        self._epoch_seed: int | None = epoch_cfg.get("seed", None)
        self._epoch_n_bootstrap: int = epoch_cfg.get("n_bootstrap", 1)

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

    def _equalize_roi_timeseries(
        self, roi_ts: dict[str, "np.ndarray"], sfreq: float,
    ) -> list[dict[str, "np.ndarray"]]:
        """Apply epoch equalization to ROI timeseries if configured.

        Randomly samples ``n_epochs`` non-overlapping windows per
        bootstrap draw.  Returns a list of dicts (one per draw).
        Analyses should compute their metric on each draw independently,
        then average across draws.

        No-op if ``epoch_sampling.enabled`` is False in config, in
        which case a single-element list containing the original
        timeseries is returned.
        """
        if not self._epoch_equalize:
            return [roi_ts]

        from ..spectral.epoch_sampler import sample_roi_epochs

        return sample_roi_epochs(
            roi_ts, sfreq,
            epoch_duration_sec=self._epoch_duration_sec,
            n_epochs=self._epoch_n_epochs,
            seed=self._epoch_seed,
            n_bootstrap=self._epoch_n_bootstrap,
        )

    @staticmethod
    def _select_norm(value: str) -> str:
        """Normalize a metric/band token for selection matching.

        Lowercases and collapses spaces/hyphens to underscores so that
        ``--band "Low Gamma"``, ``low-gamma`` and ``low_gamma`` all match the
        same configured band, and metric names are case-insensitive.
        """
        return str(value).strip().lower().replace(" ", "_").replace("-", "_")

    def _select(self, dim: str, available: "list") -> "list":
        """Filter a module's configured sub-outputs by an active ``--select``.

        ``available`` is the list the module would otherwise process for this
        dimension (e.g. its configured connectivity metrics, or its bands). If
        the user requested ``--select {dim}=...`` (or ``--metric``/``--band``),
        only the requested members are kept — matched via :meth:`_select_norm`,
        order preserved from ``available``. With no active selection for ``dim``
        the list is returned unchanged, so the default behaviour (run all) is
        untouched. Raises ``ValueError`` if a selection is active but matches
        nothing (a typo'd request should fail loudly, not silently run zero
        work). Requested-but-absent members are warned and ignored.
        """
        wanted = self._selection.get(dim)
        if not wanted:
            return list(available)
        norm_map = {self._select_norm(a): a for a in available}
        keep = [a for a in available if self._select_norm(a) in wanted]
        unknown = set(wanted) - set(norm_map)
        if unknown:
            logger.warning(
                "%s: --select %s requested %s not in available set %s; ignoring",
                self.name, dim, sorted(unknown), sorted(norm_map),
            )
        if not keep:
            raise ValueError(
                f"{self.name}: --select {dim}={sorted(wanted)} matched none of "
                f"the available {dim}(s) {sorted(norm_map)}"
            )
        logger.info("%s: --select %s -> %s", self.name, dim, keep)
        return keep

    def _selected_bands(self) -> dict:
        """``config.bands`` filtered by an active ``--select band=...``.

        Convenience for the many modules that iterate frequency bands: use
        ``self._selected_bands().items()`` in place of ``self.config.bands``.
        Returns all bands when no band selection is active. Cached per run so
        repeated per-subject calls don't re-log the active filter.
        """
        cache = getattr(self, "_bands_cache", None)
        if cache is None:
            names = self._select("band", list(self.config.bands.keys()))
            cache = {n: self.config.bands[n] for n in names}
            self._bands_cache = cache
        return cache

    def _pairwise_contrasts(self) -> list:
        """Pairwise contrast records derived from the declared hypotheses.

        The per-vertex / per-edge map modules iterate these to drive their
        two-sample cluster / NBS tests directly from ``design:``/``hypotheses:``,
        instead of the legacy ``config.contrasts`` bridge. Each record exposes
        ``.name`` / ``.group_a`` / ``.group_b`` / ``.label`` / ``.role`` — a drop-in
        for the old ``config.contrasts`` iteration. Returns the same pairwise
        contrast/equivalence set the bridge produced (omnibus/regression/multi-
        group hypotheses have no two-sample analogue and are handled only by the
        hypothesis layer's cluster table).
        """
        from ..config import _contrasts_from_design_spec
        spec = self.config.design_spec
        return _contrasts_from_design_spec(spec) if spec is not None else []

    # ---- Figure-state persistence (regenerable figures) ------------------
    # Standard: figures() must be regenerable from persisted data via
    # `--steps figures` alone — never dependent on in-memory state from an
    # earlier step in the same process. Map/cluster modules use these to
    # persist their cluster-test results in statistics() and reload them in
    # figures(); the good model is vertex_cluster.

    def _save_cluster_state(self, **extra) -> None:
        """Pickle this module's cluster-test results + coords so figures() can
        regenerate without re-running statistics. Writes data/<name>_results.pkl."""
        import pickle
        path = self.output_dir / "data" / f"{self.name}_results.pkl"
        state = {
            "cluster_results": getattr(self, "_cluster_results", {}),
            "source_coords": getattr(self, "_source_coords", None),
            **extra,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved %s", path.name)

    def _load_cluster_state(self) -> bool:
        """Reload cluster-test results from data/<name>_results.pkl (for
        `--steps figures`). Returns True if loaded."""
        import pickle
        path = self.output_dir / "data" / f"{self.name}_results.pkl"
        if not path.exists():
            logger.warning("No saved figure state at %s", path)
            return False
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self._cluster_results = saved.get("cluster_results", {})
        if saved.get("source_coords") is not None:
            self._source_coords = saved["source_coords"]
        logger.info("Loaded %s figure state from %s", self.name, path.name)
        return True

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
        cmd.extend(self._r_roi_categories_flags())

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

    def _r_roi_categories_flags(self) -> list[str]:
        """Return ['--roi-categories', path] if atlas roi_categories.yaml exists."""
        if self._atlas_dir is not None:
            cat_path = self._atlas_dir / "roi_categories.yaml"
            if cat_path.exists():
                return ["--roi-categories", str(cat_path)]
        return []

    def _r_no_figures_flags(self) -> list[str]:
        """Return ['--no-figures'] if figure generation is disabled, else []."""
        if not self._generate_figures:
            return ["--no-figures"]
        return []

    # ---- Anatomical labeling of vertex clusters --------------------------- #
    def _label_vertex_regions(self, coords_mm) -> list[str | None]:
        """One atlas-ROI name per vertex, for describing where clusters sit.

        Returns an all-``None`` list when no atlas or no coords are available, so
        callers can add a (blank) ``region`` column unconditionally.
        """
        import numpy as np

        n = 0 if coords_mm is None else len(coords_mm)
        if self._atlas_dir is None or n == 0:
            return [None] * n
        try:
            from ..atlas.atlas_utils import label_vertices_to_rois
            return label_vertices_to_rois(np.asarray(coords_mm, dtype=float), self._atlas_dir)
        except Exception as e:  # noqa: BLE001 — labeling is descriptive, never fatal
            logger.warning("Vertex ROI labeling failed (%s); regions omitted", e)
            return [None] * n

    @staticmethod
    def _cluster_region(vertex_rois: list[str | None], mask) -> str:
        """Anatomical-coverage string ('ROI x%, … (+N ROIs ≤5%)') for the cluster
        selected by ``mask`` (a boolean array over vertices)."""
        import numpy as np
        from ..atlas.atlas_utils import format_region_coverage

        idx = np.where(mask)[0]
        labels = [vertex_rois[i] for i in idx if i < len(vertex_rois)]
        return format_region_coverage(labels)

    def run(
        self,
        subjects: list[SubjectInfo],
        steps: set[str] | None = None,
        select: dict[str, frozenset[str]] | None = None,
    ) -> None:
        """Execute the analysis lifecycle.

        Parameters
        ----------
        subjects : list[SubjectInfo]
            Subjects to process.
        steps : set[str] | None
            If provided, only run these steps (from VALID_STEPS).
            ``setup`` always runs. If None, run DEFAULT_RUN_STEPS
            (all steps except ``figures``).
        select : dict[str, frozenset[str]] | None
            Sub-output selection (dim -> normalized allowed values). Applied via
            :meth:`_select` inside each module's ``setup``. Set before ``setup``
            runs so the configured metric/band lists can be filtered. None = run
            every configured sub-output.
        """
        logger.info("=== %s Analysis ===", self.name)
        if select:
            self._selection = select
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

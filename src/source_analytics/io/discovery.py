"""Auto-discover subjects from a pipeline output directory tree.

Supports two directory layouts:

**Grouped layout** (default):
    root_dir/
    ├── Group Name 1/
    │   ├── Subject_A/
    │   │   └── data/
    │   └── Subject_B/
    │       └── data/
    └── Group Name 2/
        └── ...

**Flat layout** (source_localization output, enabled by ``subject_groups``):
    root_dir/
    ├── sub-Subject_A/
    │   └── pipeline/data/
    ├── sub-Subject_B/
    │   └── pipeline/data/
    └── ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SubjectInfo:
    """Metadata for a discovered subject."""

    subject_id: str
    group: str
    data_dir: Path
    pipeline_dir: Path


def _check_data_dir(
    data_dir: Path,
    required_files: list[str] | None,
) -> bool:
    """Return True if *data_dir* passes file-presence checks."""
    if required_files is not None:
        missing = [f for f in required_files if not (data_dir / f).exists()]
        if missing:
            logger.warning(
                "Missing required files in %s: %s, skipping",
                data_dir, ", ".join(missing),
            )
            return False
    else:
        has_roi = any(
            (data_dir / f).exists()
            for f in [
                "step6_roi_timeseries_magnitude.pkl",
                "step6_roi_timeseries.pkl",
            ]
        )
        has_stc = (
            (data_dir / "step5_stc.pkl").exists()
            and (data_dir / "step3_source_coords_mm.npy").exists()
        )
        if not has_roi and not has_stc:
            logger.warning(
                "No ROI timeseries or STC data in %s, skipping", data_dir,
            )
            return False
    return True


def discover_subjects(
    root_dir: str | Path,
    group_mapping: dict[str, str] | None = None,
    required_files: list[str] | None = None,
    data_subdir: str = "data",
    subject_groups: dict[str, str] | None = None,
) -> list[SubjectInfo]:
    """Walk a pipeline output tree and discover all subjects.

    Parameters
    ----------
    root_dir : Path
        Top-level directory containing group or subject subdirectories.
    group_mapping : dict, optional
        Maps directory group names to canonical group IDs (grouped mode).
        E.g. ``{"KO ICV": "KO_VEH", "WT ICV": "WT_VEH"}``.
    required_files : list[str], optional
        Files that must exist in the data directory for a subject to be
        included.  When *None* (default), checks for ROI timeseries files
        (``step6_roi_timeseries_magnitude.pkl`` or
        ``step6_roi_timeseries.pkl``).  When provided, *all* listed files
        must be present.
    data_subdir : str, default ``"data"``
        Relative path from each subject directory to its data directory.
        Use ``"pipeline/data"`` for source-localization output.
    subject_groups : dict, optional
        Maps subject directory names to group IDs.  When provided, enables
        **flat mode**: iterates subject directories directly under
        *root_dir* (no group subdirectories).  Subjects not present in this
        dict are skipped.

    Returns
    -------
    list[SubjectInfo]
        All discovered subjects with their group assignments.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Discovery root not found: {root_dir}")

    subjects: list[SubjectInfo] = []

    if subject_groups is not None:
        # --- Flat mode: subject dirs directly under root_dir ---
        for subj_dir in sorted(root_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            group_id = subject_groups.get(subj_dir.name)
            if group_id is None:
                logger.debug("Skipping %s (not in subject_groups)", subj_dir.name)
                continue

            data_dir = subj_dir / data_subdir
            if not data_dir.is_dir():
                logger.warning("No %s directory in %s, skipping", data_subdir, subj_dir)
                continue

            if not _check_data_dir(data_dir, required_files):
                continue

            subjects.append(
                SubjectInfo(
                    subject_id=subj_dir.name,
                    group=group_id,
                    data_dir=data_dir,
                    pipeline_dir=subj_dir,
                )
            )
    else:
        # --- Grouped mode: root_dir / Group / Subject ---
        group_mapping = group_mapping or {}

        for group_dir in sorted(root_dir.iterdir()):
            if not group_dir.is_dir():
                continue

            dir_name = group_dir.name
            group_id = group_mapping.get(dir_name, dir_name)

            for subj_dir in sorted(group_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue

                data_dir = subj_dir / data_subdir
                if not data_dir.is_dir():
                    logger.warning("No %s directory in %s, skipping", data_subdir, subj_dir)
                    continue

                if not _check_data_dir(data_dir, required_files):
                    continue

                subjects.append(
                    SubjectInfo(
                        subject_id=subj_dir.name,
                        group=group_id,
                        data_dir=data_dir,
                        pipeline_dir=subj_dir,
                    )
                )

    logger.info("Discovered %d subjects across %d groups", len(subjects), len(set(s.group for s in subjects)))
    return subjects

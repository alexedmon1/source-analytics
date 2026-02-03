"""Auto-discover subjects from a pipeline output directory tree.

Expected layout (from source_localization batch processing):

    root_dir/
    ├── Group Name 1/
    │   ├── Subject_A/
    │   │   └── data/
    │   │       ├── step6_roi_timeseries_magnitude.pkl
    │   │       └── step1_info.pkl
    │   └── Subject_B/
    │       └── data/
    │           └── ...
    └── Group Name 2/
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


def discover_subjects(
    root_dir: str | Path,
    group_mapping: dict[str, str] | None = None,
    required_files: list[str] | None = None,
) -> list[SubjectInfo]:
    """Walk a pipeline output tree and discover all subjects.

    Parameters
    ----------
    root_dir : Path
        Top-level directory containing group subdirectories.
    group_mapping : dict, optional
        Maps directory group names to canonical group IDs.
        E.g. ``{"KO ICV": "KO_VEH", "WT ICV": "WT_VEH"}``.
    required_files : list[str], optional
        Files that must exist in ``data/`` for a subject to be included.
        When *None* (default), checks for ROI timeseries files
        (``step6_roi_timeseries_magnitude.pkl`` or ``step6_roi_timeseries.pkl``).
        When provided, *all* listed files must be present.

    Returns
    -------
    list[SubjectInfo]
        All discovered subjects with their group assignments.
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Discovery root not found: {root_dir}")

    group_mapping = group_mapping or {}
    subjects = []

    for group_dir in sorted(root_dir.iterdir()):
        if not group_dir.is_dir():
            continue

        dir_name = group_dir.name
        group_id = group_mapping.get(dir_name, dir_name)

        for subj_dir in sorted(group_dir.iterdir()):
            if not subj_dir.is_dir():
                continue

            data_dir = subj_dir / "data"
            if not data_dir.is_dir():
                logger.warning("No data/ directory in %s, skipping", subj_dir)
                continue

            if required_files is not None:
                # Check that all required files are present
                missing = [f for f in required_files if not (data_dir / f).exists()]
                if missing:
                    logger.warning(
                        "Missing required files in %s: %s, skipping",
                        data_dir, ", ".join(missing),
                    )
                    continue
            else:
                # Default: verify at least one ROI timeseries file exists
                has_roi = any(
                    (data_dir / f).exists()
                    for f in [
                        "step6_roi_timeseries_magnitude.pkl",
                        "step6_roi_timeseries.pkl",
                    ]
                )
                if not has_roi:
                    logger.warning("No ROI timeseries in %s, skipping", data_dir)
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

"""What produced a result: written beside the tables, as ``provenance.json``.

A stats table on its own cannot say what made it. The numbers depend on the
source-analytics version that computed them, on the plugin that provided the
analysis if it was not built in, and — most of all — on how the recordings were
localized: a different atlas, inverse or source-sampling mode is a different
measurement, not a different view of the same one.

source-localization already records its side, in the ``config_resolved.yaml`` it
writes beside each subject's outputs. This carries that forward into the
published tree, so one file next to the CSVs answers "what is this?" without the
reader having to find the study config, the localization tree and the installed
package versions separately.

Monte Carlo parcel caveats are copied in for the same reason. They are logged
when an analysis runs, but a log is not what someone reading a table six months
later has.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                                  # pragma: no cover
    from .io.discovery import SubjectInfo
    from .io.run_manifest import RunManifest

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "provenance.json"

#: Bumped when a consumer would have to read the file differently.
SCHEMA_VERSION = 1

#: Localization settings copied into the record. The ones that change the
#: numbers, matching ``BaseAnalysis._COHORT_FIELDS``.
_MANIFEST_FIELDS = (
    "version", "preset", "atlas", "bem_type", "source_type", "surface_method",
    "spacing_mm", "source_sampling", "inverse_method", "orientation",
)


def _package_versions(analysis: str) -> tuple[dict, dict]:
    """(this package's version, {plugin name: version}) — best effort.

    Version resolution is import-time work that must not take a run down, and a
    plugin that fails to import is already logged and skipped by the loader.
    """
    from . import __version__
    from ._version import git_describe

    core = {"version": __version__, "git_describe": git_describe()}

    plugins: dict[str, Any] = {}
    try:
        from .plugins import load_plugins

        for name, module in load_plugins():
            entry = {"version": getattr(module, "__version__", None)}
            if analysis in (getattr(module, "ANALYSES", {}) or {}):
                entry["provides_this_analysis"] = True
            plugins[name] = entry
    except Exception as exc:                       # pragma: no cover - loader edge cases
        logger.debug("Could not record plugin versions: %s", exc)
    return core, plugins


def _localization(manifests: "dict[str, RunManifest]", n_subjects: int) -> dict:
    """The localization settings this cohort shares.

    ``BaseAnalysis.run`` has already refused the cohort if they disagree, so one
    manifest describes all of them. Subjects with no manifest are counted, not
    guessed at: a record that quietly claimed settings for them would be worse
    than one that says how many are unaccounted for.
    """
    if not manifests:
        return {"n_with_manifest": 0, "n_unrecorded": n_subjects}

    sample = next(iter(manifests.values()))
    record: dict[str, Any] = {
        "n_with_manifest": len(manifests),
        "n_unrecorded": n_subjects - len(manifests),
        "description": sample.describe(),
    }
    for field in _MANIFEST_FIELDS:
        record[field] = getattr(sample, field, None)
    if sample.is_monte_carlo:
        record["monte_carlo"] = dict(sample.monte_carlo)
    return record


def _parcel_caveats(manifests: "dict[str, RunManifest]") -> dict:
    """Monte Carlo caveats, unioned over the cohort. Empty for a fixed grid."""
    if not any(m.is_monte_carlo for m in manifests.values()):
        return {}

    from .io.run_manifest import parcel_caveats, read_monte_carlo_report

    out: dict[str, str] = {}
    for data_dir in {m.path.parent for m in manifests.values() if m.path}:
        for parcel, why in parcel_caveats(read_monte_carlo_report(data_dir)).items():
            out.setdefault(parcel, why)
    return out


def build_provenance(
    *,
    analysis: str,
    paradigm: str | None = None,
    profile: str | None = None,
    study_config: Path | str | None = None,
    subjects: "list[SubjectInfo] | None" = None,
    manifests: "dict[str, RunManifest] | None" = None,
    steps: "set[str] | list[str] | None" = None,
) -> dict:
    """Assemble the record. Pure — nothing here touches the output tree."""
    subjects = subjects or []
    manifests = manifests or {}

    groups: dict[str, int] = {}
    for s in subjects:
        groups[s.group] = groups.get(s.group, 0) + 1

    core, plugins = _package_versions(analysis)
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "written": datetime.now().astimezone().isoformat(timespec="seconds"),
        "analysis": analysis,
        "paradigm": paradigm,
        "profile": profile,
        "study_config": str(study_config) if study_config else None,
        "source_analytics": core,
        "steps": sorted(steps) if steps else [],
        "subjects": {
            "n": len(subjects),
            "groups": dict(sorted(groups.items())),
            "ids": sorted(s.subject_id for s in subjects),
        },
        "localization": _localization(manifests, len(subjects)),
    }
    if plugins:
        record["plugins"] = plugins
    caveats = _parcel_caveats(manifests)
    if caveats:
        record["parcel_caveats"] = caveats
    return record


def write_provenance(tbl_dir: Path | str, record: dict) -> Path | None:
    """Write *record* to ``<tbl_dir>/provenance.json``.

    Returns the path, or None if it could not be written — provenance is a
    record *about* a run, so failing to write it must not fail the run that
    already produced its numbers.
    """
    path = Path(tbl_dir) / PROVENANCE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
        return None
    return path


def read_provenance(tbl_dir: Path | str) -> dict | None:
    """Read ``<tbl_dir>/provenance.json``, or None when absent/unreadable."""
    path = Path(tbl_dir) / PROVENANCE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

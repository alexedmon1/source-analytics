"""What each figure shows, recorded where it is drawn (Esprit's results plan, 5b.4).

The results specification lets a figure name its test (``measure``, ``contrast``,
``facets``), its ``kind`` and the table it ``shows``, so a gallery groups and filters
figures by fields rather than by file names. Only the code that draws a figure knows
these, so it records them as it saves: one JSON line per figure in ``figures.jsonl``
beside the images (R does the same through ``record_figure()`` in R/stats_utils.R).
``source-analytics gallery build`` reads the file when it lists a module's figures,
mapping each measure from the module's name to the specification's.

Tags: ``measure`` (the module's name, e.g. a ``dv``), ``contrast`` (the declared
hypothesis), ``facets`` (``{FacetName: value}``), ``kind``, ``shows`` (a table file
name), ``caption``. A figure with no record is listed as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RECORDS = "figures.jsonl"
TAGS = ("measure", "contrast", "facets", "kind", "shows", "caption")


def record_figure(fig_dir: Path | str, file: Path | str, **tags: Any) -> None:
    """Append what ``file`` (a figure in ``fig_dir``) shows; empty tags are left out."""
    unknown = set(tags) - set(TAGS)
    if unknown:
        raise ValueError(f"unknown figure tags {sorted(unknown)}; known: {', '.join(TAGS)}")
    entry: dict[str, Any] = {"file": Path(file).name}
    for k in TAGS:
        v = tags.get(k)
        if v in (None, "", {}):
            continue
        entry[k] = {str(a): str(b) for a, b in v.items()} if k == "facets" else str(v)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    with open(fig_dir / RECORDS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_records(fig_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Each recorded figure's tags, by file name (the latest record of a file wins)."""
    path = Path(fig_dir) / RECORDS
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        name = entry.pop("file", None)
        if name:
            out[name] = entry
    return out


def clear_records(fig_dir: Path | str) -> None:
    """Forget the records, with the images, before a module redraws its figures."""
    path = Path(fig_dir) / RECORDS
    if path.exists():
        path.unlink()

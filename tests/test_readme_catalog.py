"""The README must not advertise analyses this package does not have.

PR #5 removed the twelve vertex analyses and left the README describing all of
them as available — a catalog, a study-config example, and a run-in-order script
naming modules that now fail at `--analysis`. Documentation drift of that shape
is the reason `--method` in source-localization offered half its inverse methods
for several releases, so it is checked rather than reviewed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from source_analytics.core import ANALYSIS_REGISTRY
from source_analytics.plugins import MOVED_ALIASES, MOVED_CANONICAL

README = Path(__file__).resolve().parent.parent / "README.md"

#: Where a retired name is legitimate: the section documenting the retirement,
#: the note on what the shared hypothesis declaration used to enable, and the
#: deprecated-alias list. Matched against the enclosing `##`/`###` heading.
_RETIREMENT_CONTEXTS = ("Retired: the vertex level", "Core concepts",
                        "Analysis catalog")

# Names that look like analyses but are config keys.
_NOT_ANALYSES = {"roi_categories"}


def _readme() -> str:
    return README.read_text()


def _named_analyses(text: str) -> set[str]:
    return {n for n in re.findall(r"`((?:roi|vertex|electrode|fcd)_[a-z_]+)`", text)
            if n not in _NOT_ANALYSES}


def _sections(text: str) -> list[tuple[str, str]]:
    """(heading, body) for each ## / ### section."""
    parts = re.split(r"^(#{2,3} .+)$", text, flags=re.M)
    out = []
    for i in range(1, len(parts), 2):
        out.append((parts[i].lstrip("# ").strip(), parts[i + 1]))
    return out


def test_every_analysis_named_is_registered_or_documented_as_gone():
    """No README name is simply unknown."""
    known = set(ANALYSIS_REGISTRY) | set(MOVED_CANONICAL) | set(MOVED_ALIASES)
    unknown = sorted(n for n in _named_analyses(_readme()) if n not in known)
    assert not unknown, f"README names analyses that do not exist: {unknown}"


def test_retired_analyses_appear_only_where_the_retirement_is_explained():
    """Not in the catalog tables, the config example, or the run-in-order script."""
    retired = set(MOVED_CANONICAL) | set(MOVED_ALIASES)
    offenders = {}
    for heading, body in _sections(_readme()):
        if any(ctx in heading for ctx in _RETIREMENT_CONTEXTS):
            continue
        found = sorted(_named_analyses(body) & retired)
        if found:
            offenders[heading] = found
    assert not offenders, (
        "retired analyses are still presented as available:\n"
        + "\n".join(f"  under {h!r}: {', '.join(n)}" for h, n in offenders.items()))


def test_the_retirement_section_exists_and_names_the_plugin():
    text = _readme()
    assert "### Retired: the vertex level" in text
    assert "source-analytics-vertex" in text
    assert "v0.7.1" in text, "the README must say which version reproduces them"


@pytest.mark.parametrize("name", sorted(MOVED_CANONICAL))
def test_each_retired_analysis_is_listed_in_the_retirement_section(name):
    """So the section is a complete inventory, not a sample."""
    section = next(body for heading, body in _sections(_readme())
                   if "Retired: the vertex level" in heading)
    assert f"`{name}`" in section


@pytest.mark.parametrize("name", sorted(ANALYSIS_REGISTRY))
def test_each_registered_analysis_is_mentioned(name):
    """The catalog must cover what the package actually offers."""
    from source_analytics.core import _DEPRECATED_NAMES

    if name in _DEPRECATED_NAMES:
        pytest.skip("deprecated alias")
    assert f"`{name}`" in _readme(), f"{name} is runnable but undocumented"

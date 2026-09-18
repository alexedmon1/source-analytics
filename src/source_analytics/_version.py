"""Runtime version resolution for provenance stamping.

The dev workflow runs from a git checkout with ``uv run --no-sync``, so the
installed package metadata is frozen at whatever was last ``pip install``-ed and
goes stale the moment new commits land (it reported ``0.3.0`` while HEAD was 26
commits past ``v0.6.0``). ``git describe`` is the only truthful source in that
workflow, so we prefer it and fall back to installed metadata (for a real
installed package with no git tree) and finally a sentinel.

The preference is only safe because the git tree is checked to be *this*
package's. ``git -C`` walks up out of a directory that is not a repository, so
without that check an installed copy reported whatever repository happened to
enclose its virtualenv. See :func:`_describe_at`.

This is the single source of truth for ``source_analytics.__version__`` and for
the ``git describe`` string that run manifests / compute keys stamp into outputs.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_FALLBACK = "0.0.0+unknown"


def _git(repo_root: Path, *args: str) -> str | None:
    """Run git in *repo_root*, or None if it fails or git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    result = out.stdout.strip()
    return result if out.returncode == 0 and result else None


def _describe_at(repo_root: Path) -> str | None:
    """Describe *repo_root* only if it is itself the root of a git checkout.

    The containment check is the point. ``git -C DIR`` does not fail on a
    directory that is not a repository -- it walks *up* until it finds one. For
    an installed (non-editable) package ``repo_root`` is ``<venv>/lib/pythonX.Y``,
    so any venv that happens to live inside a git repository made this report
    that repository's version as source-analytics'. A checkout of a consumer
    package is exactly where that happens: the plugin's venv described the plugin.

    Comparing ``rev-parse --show-toplevel`` against ``repo_root`` accepts a real
    source checkout (including a git worktree, whose toplevel is the worktree
    root) and rejects an enclosing repository, which then falls through to
    installed metadata -- the correct answer for an installed package.
    """
    toplevel = _git(repo_root, "rev-parse", "--show-toplevel")
    if toplevel is None:
        return None
    try:
        if Path(toplevel).resolve() != repo_root:
            return None
    except OSError:                                # unresolvable path
        return None
    return _git(repo_root, "describe", "--tags", "--dirty", "--always")


@lru_cache(maxsize=1)
def git_describe() -> str | None:
    """``git describe --tags --dirty --always`` for the source tree, or None.

    Returns None when not run from a git checkout (e.g. an installed wheel) or
    when git is unavailable. Cached so import-time and run-manifest callers share
    one subprocess.
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/source_analytics/_version.py -> repo root
    return _describe_at(repo_root)


@lru_cache(maxsize=1)
def get_version() -> str:
    """Resolve the package version, preferring the truthful git-describe string."""
    described = git_describe()
    if described:
        return described
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("source-analytics")
    except PackageNotFoundError:
        return _FALLBACK
    except Exception:  # pragma: no cover - metadata backend edge cases
        return _FALLBACK

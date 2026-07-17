"""Runtime version resolution for provenance stamping.

The dev workflow runs from a git checkout with ``uv run --no-sync``, so the
installed package metadata is frozen at whatever was last ``pip install``-ed and
goes stale the moment new commits land (it reported ``0.3.0`` while HEAD was 26
commits past ``v0.6.0``). ``git describe`` is the only truthful source in that
workflow, so we prefer it and fall back to installed metadata (for a real
installed package with no git tree) and finally a sentinel.

This is the single source of truth for ``source_analytics.__version__`` and for
the ``git describe`` string that run manifests / compute keys stamp into outputs.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_FALLBACK = "0.0.0+unknown"


@lru_cache(maxsize=1)
def git_describe() -> str | None:
    """``git describe --tags --dirty --always`` for the source tree, or None.

    Returns None when not run from a git checkout (e.g. an installed wheel) or
    when git is unavailable. Cached so import-time and run-manifest callers share
    one subprocess.
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/source_analytics/_version.py -> repo root
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "describe", "--tags", "--dirty", "--always"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    described = out.stdout.strip()
    return described if out.returncode == 0 and described else None


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

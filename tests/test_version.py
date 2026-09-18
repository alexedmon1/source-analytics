"""Version resolution must describe *this* package's checkout, or nothing.

`git -C DIR` does not fail on a directory that is not a repository — it walks up
until it finds one. For an installed (non-editable) package the derived repo root
is `<venv>/lib/pythonX.Y`, so any virtualenv inside a git repository made
source-analytics report that repository's version as its own. Observed while
verifying the v0.8.0 re-pin: the vertex plugin's venv reported the plugin's
commit (`76d899d-dirty`) as the source-analytics version.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from source_analytics._version import _FALLBACK, _describe_at, get_version, git_describe


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True,
        env={"PATH": __import__("os").environ.get("PATH", ""),
             "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "HOME": str(cwd)},
    )


@pytest.fixture
def repo(tmp_path):
    """A git repo with one commit and a tag."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", ".")
    _git(root, "commit", "-q", "--allow-empty", "-m", "init")
    _git(root, "tag", "-a", "v9.9.9", "-m", "x")
    return root


class TestDescribeAt:
    def test_a_real_checkout_is_described(self, repo):
        assert _describe_at(repo) == "v9.9.9"

    def test_an_enclosing_repo_is_not_borrowed(self, repo):
        """The bug: a venv inside someone else's repo reported their version."""
        installed = repo / ".venv" / "lib" / "python3.12"
        installed.mkdir(parents=True)
        assert _describe_at(installed) is None

    def test_a_plain_directory_is_not_described(self, tmp_path):
        d = tmp_path / "nowhere"
        d.mkdir()
        assert _describe_at(d) is None

    def test_a_subdirectory_of_a_checkout_is_not_the_checkout(self, repo):
        """Only the toplevel is this package's root; `src/` is not."""
        sub = repo / "src"
        sub.mkdir()
        assert _describe_at(sub) is None

    def test_a_missing_directory_is_not_described(self, tmp_path):
        assert _describe_at(tmp_path / "does-not-exist") is None

    def test_a_worktree_root_is_described(self, repo, tmp_path):
        """Worktrees are a real dev workflow here (sa-worktrees/); keep them working."""
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "HEAD")
        described = _describe_at(wt)
        assert described is not None and described.startswith("v9.9.9")

    def test_dirty_checkouts_are_marked(self, repo):
        (repo / "changed.txt").write_text("x")
        _git(repo, "add", "changed.txt")
        assert _describe_at(repo).endswith("-dirty")


class TestGetVersion:
    def test_falls_back_to_installed_metadata_when_not_a_checkout(self, monkeypatch):
        """An installed package must read its metadata, not a neighbouring repo."""
        import source_analytics._version as v

        monkeypatch.setattr(v, "_describe_at", lambda root: None)
        v.git_describe.cache_clear()
        v.get_version.cache_clear()
        try:
            resolved = v.get_version()
        finally:
            v.git_describe.cache_clear()
            v.get_version.cache_clear()
        assert resolved != _FALLBACK
        assert resolved[0].isdigit(), f"expected a metadata version, got {resolved!r}"

    def test_this_checkout_still_describes_itself(self):
        """The dev workflow the git-describe preference exists for."""
        described = git_describe()
        if described is None:
            pytest.skip("not running from a git checkout")
        assert described.startswith("v0."), described
        assert get_version() == described

    def test_the_described_root_is_this_package(self):
        """Guards the parents[2] arithmetic, not just the containment check."""
        root = Path(__file__).resolve().parent.parent
        assert (root / "src" / "source_analytics" / "_version.py").exists()

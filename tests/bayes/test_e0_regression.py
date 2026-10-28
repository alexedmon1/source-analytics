"""Regression against the engine plan's E0 fixtures (FORGE G1 fits and gate verdicts, frozen 2026-10-28).

The fixtures are study results and live outside this repository (bayes-sl ``fixtures/e0/``, made by its
``R/e0_freeze.R``). Set ``SA_BAYES_E0`` to that directory to run; ``SA_BAYES_SCRATCH`` moves the gate replay's
working copy off pytest's tmp_path. No sampling: saved fits are read out again and the gate summarized again.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from source_analytics.bayes import r_dir

E0 = os.environ.get("SA_BAYES_E0")
pytestmark = [pytest.mark.skipif(not E0, reason="SA_BAYES_E0 not set (fixtures live outside the repository)"),
              pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")]


def test_e0_fixtures_reproduce(tmp_path):
    scratch = Path(os.environ.get("SA_BAYES_SCRATCH", tmp_path)) / "e0_replay"
    cmd = ["Rscript", str(Path(__file__).with_name("e0_replay.R")), "--fixtures", E0,
           "--engine", str(r_dir() / "bayes" / "engine.R"), "--mode", "check", "--scratch", str(scratch)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    assert r.returncode == 0 and "E0 PASS" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]

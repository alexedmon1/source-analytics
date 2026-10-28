"""Slow engine tests (compile Stan and sample: minutes each) are marked ``@pytest.mark.slow`` and run only with
``SA_BAYES_SLOW=1``. Write their scratch under ``SA_BAYES_SCRATCH`` when pytest's temp dir is on a drive that
refuses symlinks (WSL /mnt), and set no TMPDIR there."""
from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiles Stan and samples (minutes); runs only with SA_BAYES_SLOW=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("SA_BAYES_SLOW"):
        return
    skip = pytest.mark.skip(reason="slow (compiles and samples); set SA_BAYES_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

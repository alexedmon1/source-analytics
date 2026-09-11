"""The plugin hook: installed packages add analyses through an entry point."""

from __future__ import annotations

import logging
import types

import pytest

import source_analytics.analyses
from source_analytics import plugins
from source_analytics.core import ANALYSIS_REGISTRY, StudyAnalyzer
from source_analytics.viz import figure_registry


class _Toy:
    """Stand-in analysis class; install_plugins only files it."""


def _plugin(**attrs):
    return types.SimpleNamespace(**attrs)


def test_install_adds_analyses_metadata_and_aliases():
    registry, metadata, aliases = {"roi_psd": object}, {}, {"psd": "roi_psd"}
    toy = _plugin(ANALYSES={"toy": _Toy}, METADATA={"toy": {"domain": "Toy"}},
                  ALIASES={"old_toy": "toy"})
    plugins.install_plugins(registry, metadata, aliases, plugins=[("toy", toy)])
    assert registry["toy"] is _Toy
    assert metadata["toy"] == {"domain": "Toy"}
    assert aliases == {"psd": "roi_psd", "old_toy": "toy"}


@pytest.mark.parametrize("attrs", [
    {"ANALYSES": {"roi_psd": _Toy}},     # an analysis name core already has
    {"ALIASES": {"psd": "toy"}},         # a deprecated alias core already has
])
def test_a_plugin_cannot_reuse_a_name(attrs):
    registry, aliases = {"roi_psd": object}, {"psd": "roi_psd"}
    with pytest.raises(ValueError, match="reuses existing analysis names"):
        plugins.install_plugins(registry, {}, aliases, plugins=[("bad", _plugin(**attrs))])


def test_register_figures_receives_the_figure_registry():
    seen = []
    plugins.install_plugins({}, {}, {}, plugins=[("toy", _plugin(register_figures=seen.append))])
    assert seen == [figure_registry]


def test_a_plugin_that_fails_to_import_is_logged_and_skipped(monkeypatch, caplog):
    class _BrokenEntryPoint:
        name, value = "broken", "not_a_real_module"

        def load(self):
            raise ImportError("no module named not_a_real_module")

    monkeypatch.setattr(plugins, "entry_points", lambda group: [_BrokenEntryPoint()])
    plugins.load_plugins.cache_clear()
    try:
        with caplog.at_level(logging.ERROR, logger="source_analytics.plugins"):
            assert plugins.load_plugins() == ()
        assert "'broken'" in caplog.text and "failed to import" in caplog.text
    finally:
        monkeypatch.undo()
        plugins.load_plugins.cache_clear()


def test_the_vertex_analyses_left_core():
    assert not hasattr(source_analytics.analyses, "VertexClusterAnalysis")
    assert "source-analytics-vertex" in plugins.missing_analysis_hint("vertex_cluster")
    assert "source-analytics-vertex" in plugins.missing_analysis_hint("wholebrain")
    assert plugins.missing_analysis_hint("roi_psd") == ""


def test_unknown_analysis_error_names_the_plugin():
    if "vertex_cluster" in ANALYSIS_REGISTRY:
        pytest.skip("source-analytics-vertex is installed in this environment")
    analyzer = StudyAnalyzer(config=None, subjects=[object()])
    with pytest.raises(ValueError, match="install source-analytics-vertex"):
        analyzer.run_analysis("vertex_cluster")

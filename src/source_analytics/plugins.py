"""Analysis plugins: installed packages that add analyses to source-analytics.

A plugin declares an entry point in the ``source_analytics.plugins`` group, e.g. in
its pyproject.toml::

    [project.entry-points."source_analytics.plugins"]
    vertex = "source_analytics_vertex"

The entry point names an object (usually the plugin's top-level module) that may
define any of:

``ANALYSES``
    ``dict[str, type[BaseAnalysis]]``: the analyses it adds, by registry name.
``METADATA``
    ``dict[str, dict]``: their ``ANALYSIS_METADATA`` entries (domain, level, ...).
``ALIASES``
    ``dict[str, str]``: deprecated name -> canonical name, so old configs still run.
``register_figures(registry)``
    Called with the ``viz.figure_registry`` module, to add ``TABLE_SCHEMAS``
    entries and ``register()`` figure types for the plugin's analyses.

``core`` installs every plugin when it is imported. A plugin that fails to import
is logged and skipped, so a broken install cannot stop the core analyses. A plugin
that reuses an existing analysis name raises: that is a packaging bug, and letting
it shadow a built-in would silently change what a config runs.
"""

from __future__ import annotations

import logging
from functools import cache
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "source_analytics.plugins"

# Analyses that left the core package, and the plugin that now provides them.
# Only used to turn "unknown analysis" into an instruction.
#: Canonical names of the analyses that left core, by plugin. Listed to a user.
MOVED_CANONICAL: dict[str, str] = {
    name: "source-analytics-vertex"
    for name in (
        "vertex_cluster", "vertex_connectivity", "vertex_cross_freq", "vertex_directed",
        "vertex_evoked", "vertex_graph", "vertex_nbs", "vertex_network",
        "vertex_signature", "vertex_spatial", "vertex_specparam", "fcd_comparison",
    )
}

#: Their deprecated aliases. Resolved for hints so an old config gets the same
#: instruction, but kept out of listings, where they would read as extra
#: analyses rather than as old spellings of the ones above.
MOVED_ALIASES: dict[str, str] = {
    name: "source-analytics-vertex"
    for name in ("wholebrain", "spatial_lmm", "specparam_vertex", "mvpa", "vertex_mvpa")
}

MOVED_TO_PLUGIN: dict[str, str] = {**MOVED_CANONICAL, **MOVED_ALIASES}


@cache
def load_plugins() -> tuple[tuple[str, object], ...]:
    """Import every installed plugin once; return ``(entry point name, object)`` pairs."""
    loaded = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        try:
            loaded.append((ep.name, ep.load()))
        except Exception:
            logger.exception(
                "source-analytics plugin %r (%s) failed to import; skipping it",
                ep.name, ep.value,
            )
    return tuple(loaded)


def install_plugins(
    registry: dict, metadata: dict, aliases: dict, plugins=None,
) -> None:
    """Add each plugin's analyses, metadata, aliases and figure types, in place.

    ``plugins`` defaults to :func:`load_plugins`; tests pass their own.
    """
    plugins = load_plugins() if plugins is None else plugins
    for name, plugin in plugins:
        added = dict(getattr(plugin, "ANALYSES", None) or {})
        new_aliases = dict(getattr(plugin, "ALIASES", None) or {})
        clash = sorted((set(added) | set(new_aliases)) & (set(registry) | set(aliases)))
        if clash:
            raise ValueError(
                f"source-analytics plugin {name!r} reuses existing analysis names: "
                f"{', '.join(clash)}"
            )
        registry.update(added)
        metadata.update(getattr(plugin, "METADATA", None) or {})
        aliases.update(new_aliases)

    hooks = [h for _, p in plugins if (h := getattr(p, "register_figures", None))]
    if hooks:
        from .viz import figure_registry

        for hook in hooks:
            hook(figure_registry)


def missing_analysis_hint(name: str) -> str:
    """A sentence naming the plugin that provides ``name``, or ``""``."""
    plugin = MOVED_TO_PLUGIN.get(name)
    if plugin is None:
        return ""
    return (
        f" '{name}' moved out of source-analytics in v0.8.0 into the {plugin} "
        f"plugin; install {plugin} to run it."
    )

"""``source-analytics gallery build``: the study's gallery, through the shared shell.

For every module of the study that describes its tables by the results
specification (a ``RESULTS_SPEC``; see results_spec.py) and has written its
``analysis.json``, this runs the module's own figures step (its R figures and
brain mosaics, from the tables and data already computed), copies
the module's figures into the analysis folder and lists them in its
``analysis.json``, and then builds one gallery for the study with neurobox-dev's
shell (``neurobox.gallery.build``). The study's localization summaries
(source-localization's ``study qc`` with its ``results`` extra) are included from
the study config's ``paths.localizations``.

Settings come from ``gallery.yaml`` beside the study config, if there is one
(the format every tool shares; see ``neurobox.gallery``), and otherwise from the
study config: its ``gallery_title`` or ``name``, ``groups`` and ``group_order``.
Needs the optional ``results`` extra.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: A figure's kind, from its file name (first match wins).
_KINDS = (("effect_size_", "brain"), ("mosaic", "brain"), ("brain", "brain"), ("glass", "brain"), ("circos", "matrix"),
          ("matrix", "matrix"), ("heatmap", "summary"), ("by_region", "summary"), ("boxplot", "distribution"),
          ("violin", "distribution"), ("psd", "spectrum"), ("spectrum", "spectrum"), ("trajector", "trajectory"))
FIGURES = "figures"


def figure_kind(name: str) -> str:
    low = name.lower()
    return next((kind for key, kind in _KINDS if key in low), "other")


def caption(name: str, module: str) -> str:
    stem = re.sub(rf"^{re.escape(module)}_?", "", Path(name).stem)
    return stem.replace("_", " ").strip().capitalize() or Path(name).stem


def attach_figures(analysis: Any, drawn_now: set[str] | None = None) -> int:
    """Copy the module's figures into its analysis folder and list them in analysis.json.

    Replaces the figures listed under ``figures/`` by a previous build; keeps any
    other figure entries. The folder is checked afterwards, and the previous
    analysis.json restored if it no longer conforms. Returns the figures listed.
    """
    from neurobox import check_analysis

    folder = Path(analysis.tbl_dir)
    path = folder / "analysis.json"
    before = path.read_text()
    a = json.loads(before)
    pngs = sorted(Path(analysis.fig_dir).glob("*.png")) if Path(analysis.fig_dir).is_dir() else []
    target = folder / FIGURES
    if target.exists():
        shutil.rmtree(target)
    entries = []
    if pngs:
        target.mkdir()
    for png in pngs:
        shutil.copy2(png, target / png.name)
        entries.append({"path": f"{FIGURES}/{png.name}", "caption": caption(png.name, analysis.name),
                        "kind": figure_kind(png.name),
                        "drawn_by": "gallery" if drawn_now and png.name in drawn_now else "analysis"})
    kept = [f for f in a.get("figures") or [] if not str(f.get("path", "")).startswith(f"{FIGURES}/")]
    a["figures"] = kept + entries
    if not a["figures"]:
        a.pop("figures")
    path.write_text(json.dumps(a, indent=2) + "\n")
    rep = check_analysis(folder)
    if not rep.ok:
        path.write_text(before)
        logger.warning("Figures not listed for %s: the folder would not conform: %s", analysis.name,
                       "; ".join(rep.errors))
        return 0
    return len(entries)


def settings_for(config: Any, study_yaml: Path) -> dict:
    """The gallery settings: ``gallery.yaml`` beside the study config, else from the config."""
    import yaml

    own = Path(study_yaml).parent / "gallery.yaml"
    if own.is_file():
        return yaml.safe_load(own.read_text()) or {}
    raw = config.raw
    study_dir = Path(study_yaml).parent
    results = [Path(config.results_dir) / "tables"]
    for loc in (raw.get("paths") or {}).get("localizations") or []:
        p = loc.get("path") if isinstance(loc, dict) else loc
        if p:
            qc = (study_dir / p).resolve() / "qc"
            if (qc / "analysis.json").is_file():
                results.append(qc)
    groups = raw.get("groups") or {}
    return {
        "name": raw.get("gallery_title") or config.name,
        "paths": {"results": [str(p) for p in results],
                  "gallery": (raw.get("paths") or {}).get("gallery") or "gallery"},
        "groups": groups if isinstance(groups, dict) else {},
        "group_order": raw.get("group_order") or [],
    }


def build(study_yaml: Path, out: Path | None = None, paradigms: list[str] | None = None,
          draw: bool = True) -> Path:
    """Build the study's gallery; the path of its index.html."""
    try:
        from neurobox import gallery as shell
    except ImportError as exc:
        raise RuntimeError("the gallery needs the optional `results` extra: "
                           "pip install 'source-analytics[results]'") from exc
    from . import __version__
    from .config import StudyConfig
    from .core import ANALYSIS_REGISTRY

    study_yaml = Path(study_yaml).resolve()
    config = StudyConfig.from_yaml(study_yaml)
    names = list(config.paradigms) if config.has_paradigms else [None]
    for pname in names:
        if paradigms and pname not in paradigms:
            continue
        analyses = (config.paradigms[pname].get("analyses") or {}) if pname else (config.raw.get("analyses") or {})
        for aname in analyses:
            cls = ANALYSIS_REGISTRY.get(aname)
            if cls is None or getattr(cls, "RESULTS_SPEC", None) is None:
                continue
            aconfig = config.for_paradigm_analysis(pname, aname) if pname else config
            module = cls(aconfig, aconfig.output_dir / aname)
            if not (Path(module.tbl_dir) / "analysis.json").is_file():
                logger.info("Gallery: %s/%s has no analysis.json yet (run its statistics with the results "
                            "extra installed)", pname, aname)
                continue
            drawn: set[str] = set()
            if draw:
                started = time.time()
                try:
                    module.figures()                 # the module's figures step, from its tables
                except Exception as exc:  # noqa: BLE001 - one figure set must not stop the gallery
                    logger.warning("Gallery: figures for %s/%s not drawn: %s", pname, aname, exc)
                drawn = {p.name for p in Path(module.fig_dir).glob("*.png") if p.stat().st_mtime >= started - 1}
            n = attach_figures(module, drawn)
            logger.info("Gallery: %s/%s: %d figure(s) listed", pname, aname, n)
    settings = settings_for(config, study_yaml)
    settings["generator"] = f"source-analytics {__version__}"
    return shell.build(study_yaml.parent, settings, out)

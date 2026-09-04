"""CLI entry point for source-analytics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import StudyConfig
from .core import StudyAnalyzer, ANALYSIS_REGISTRY, ANALYSIS_METADATA, canonical_analysis_name
from .analyses.base import VALID_STEPS, BaseAnalysis


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_study_summary(config: StudyConfig, analyzer: StudyAnalyzer):
    """Print study name, subject counts, and group breakdown."""
    print(f"Study: {config.name}")
    print(f"Subjects discovered: {len(analyzer.subjects)}")
    groups = {}
    for s in analyzer.subjects:
        groups.setdefault(s.group, []).append(s.subject_id)
    for g in config.group_order:
        if g in groups:
            print(f"  {config.get_group_label(g)} ({g}): n={len(groups[g])}")
    print()


def _run_single(
    config: StudyConfig,
    analysis_name: str,
    steps: set[str] | None = None,
    select: dict[str, frozenset[str]] | None = None,
    jobs: int | None = None,
):
    """Run one analysis on a (possibly paradigm-scoped) config."""
    analyzer = StudyAnalyzer(config)
    _print_study_summary(config, analyzer)
    analyzer.run_analysis(analysis_name, steps=steps, select=select, jobs=jobs)
    # run_analysis writes under the CANONICAL name (deprecated aliases resolve).
    print(f"\nDone. Output: {config.output_dir / canonical_analysis_name(analysis_name)}")


def _parse_selection(args) -> dict[str, frozenset[str]] | None:
    """Build the ``--select`` map (dim -> normalized values) from run args.

    Accepts the ergonomic ``--metric``/``--band`` shorthands plus the generic
    repeatable ``--select dim=v1,v2``. Values are normalized via
    :meth:`BaseAnalysis._select_norm` so matching is case/format-insensitive.
    Returns None when no selection was requested.
    """
    sel: dict[str, set[str]] = {}

    def _add(dim: str, raw: str):
        vals = {BaseAnalysis._select_norm(v) for v in raw.split(",") if v.strip()}
        if vals:
            sel.setdefault(dim, set()).update(vals)

    if getattr(args, "metric", None):
        _add("metric", args.metric)
    if getattr(args, "band", None):
        _add("band", args.band)
    if getattr(args, "hypothesis", None):
        _add("hypothesis", args.hypothesis)
    for item in getattr(args, "select", None) or []:
        if "=" not in item:
            print(f"ERROR: --select expects DIM=val[,val...] (got '{item}')", file=sys.stderr)
            sys.exit(1)
        dim, raw = item.split("=", 1)
        _add(dim.strip(), raw)

    if not sel:
        return None

    # Validate requested dims against what modules can actually filter. A dim no
    # analysis declares is almost certainly a typo — fail loudly. When a single
    # analysis is targeted, validate against that analysis specifically.
    known_dims: set[str] = set()
    for cls in ANALYSIS_REGISTRY.values():
        known_dims |= set(getattr(cls, "SELECTABLE", {}) or {})
    target_cls = ANALYSIS_REGISTRY.get(getattr(args, "analysis", None) or "")
    target_dims = set(getattr(target_cls, "SELECTABLE", {}) or {}) if target_cls else None
    for dim in sel:
        if target_dims is not None:
            if dim not in target_dims:
                allowed = ", ".join(sorted(target_dims)) or "(none)"
                print(
                    f"ERROR: analysis '{args.analysis}' has no selectable dimension "
                    f"'{dim}'. Selectable: {allowed}",
                    file=sys.stderr,
                )
                sys.exit(1)
        elif dim not in known_dims:
            print(
                f"ERROR: unknown --select dimension '{dim}'. "
                f"Known: {', '.join(sorted(known_dims))}",
                file=sys.stderr,
            )
            sys.exit(1)

    return {d: frozenset(v) for d, v in sel.items()}


def _prepare_output(config, analysis_name, *, strict, force, steps=None):
    """Apply --strict-output / --force to the analysis's output directories.

    All paths use the CANONICAL analysis name (a deprecated alias such as
    ``psd`` writes to ``roi_psd/``), matching what ``run_analysis`` does.

    --strict-output: error if the working dir (``analytics/<paradigm>/<name>``)
    already holds output, unless --force.

    --force: actually remove the previous output before running — the
    published ``tables/`` and ``figures/`` dirs always, and the working dir
    (``data/`` + summary) only when the ``process`` step will re-create it.
    With ``--steps`` that excludes ``process`` the persisted data is kept, since
    the requested steps read it back.
    """
    import shutil

    canonical = canonical_analysis_name(analysis_name)
    work = config.output_dir / canonical
    paradigm = config.paradigm_name or ""
    published = [
        config.results_dir / "tables" / paradigm / canonical,
        config.results_dir / "figures" / paradigm / canonical,
    ]
    has_output = work.exists() and any(work.iterdir())

    if strict and has_output and not force:
        print(
            f"ERROR: --strict-output set and analysis output already exists: {work}",
            file=sys.stderr,
        )
        print("Pass --force to overwrite, or remove the directory first.", file=sys.stderr)
        sys.exit(1)

    if force:
        reprocess = steps is None or "process" in steps
        targets = list(published) + ([work] if reprocess else [])
        for t in targets:
            if t.exists():
                shutil.rmtree(t)
                print(f"--force: removed {t}")
        if not reprocess and work.exists():
            print(f"--force: kept {work} (data/) because --steps excludes 'process'")
    return work


def cmd_run(args):
    """Run an analysis module."""
    config = StudyConfig.from_yaml(args.study)

    # Apply the profile narrowing to the ROOT config, before any paradigm scoping,
    # so the profile segment lands above the paradigm in both output trees and
    # for_paradigm*() carries the narrowed bands/rois/hypotheses through.
    profile = getattr(args, "profile", None)
    if profile:
        try:
            config = config.for_profile(profile)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        if (
            args.analysis
            and config.include_analyses is not None
            and args.analysis not in config.include_analyses
        ):
            print(
                f"ERROR: analysis '{args.analysis}' is not in profile "
                f"'{profile}' (include_analyses: "
                f"{', '.join(config.include_analyses)})."
            )
            sys.exit(1)
        print(f"Profile: {profile}")
        print(f"  bands:      {', '.join(config.bands)}")
        if config.rois:
            print(f"  ROIs:       {len(config.rois)} (FDR family size)")
        if config.design_spec:
            print(
                f"  hypotheses: "
                f"{', '.join(h.name for h in config.design_spec.hypotheses)}"
            )
        print(f"  results ->  {config.results_dir}")
        print()

    # Parse --steps
    steps = None
    if args.steps:
        steps = {s.strip() for s in args.steps.split(",")}
        invalid = steps - VALID_STEPS
        if invalid:
            print(f"ERROR: Invalid step(s): {', '.join(sorted(invalid))}")
            print(f"Valid steps: {', '.join(sorted(VALID_STEPS))}")
            sys.exit(1)

    # Parse --metric / --band / --select sub-output selection
    select = _parse_selection(args)
    jobs = getattr(args, "jobs", None)  # None → YAML `jobs:` → serial; 0/-1 → auto

    strict = getattr(args, "strict_output", False)
    force = getattr(args, "force", False)

    if config.has_paradigms:
        # Multi-paradigm config
        if args.paradigm:
            if args.analysis:
                # Scope to one paradigm + one analysis
                aconfig = config.for_paradigm_analysis(args.paradigm, args.analysis)
                _prepare_output(aconfig, args.analysis, strict=strict, force=force, steps=steps)
                _run_single(aconfig, args.analysis, steps=steps, select=select, jobs=jobs)
            else:
                # Run all analyses listed for this paradigm
                analyses = config.get_paradigm_analyses(args.paradigm)
                if not analyses:
                    print(f"No analyses listed for paradigm '{args.paradigm}'")
                    sys.exit(1)
                for analysis_name in analyses:
                    print(f"{'='*60}")
                    print(f"Paradigm: {args.paradigm}  |  Analysis: {analysis_name}")
                    print(f"{'='*60}")
                    aconfig = config.for_paradigm_analysis(args.paradigm, analysis_name)
                    _prepare_output(aconfig, analysis_name, strict=strict, force=force, steps=steps)
                    _run_single(aconfig, analysis_name, steps=steps, select=select, jobs=jobs)
                    print()
        else:
            if args.analysis:
                print("ERROR: --analysis without --paradigm is ambiguous in multi-paradigm mode.")
                print("Specify --paradigm or omit --analysis to run everything.")
                sys.exit(1)
            # Run all paradigms, all their analyses
            for pname in config.paradigms:
                analyses = config.get_paradigm_analyses(pname) or []
                if not analyses:
                    print(f"Skipping paradigm '{pname}' (no analyses listed)")
                    continue
                for analysis_name in analyses:
                    print(f"{'='*60}")
                    print(f"Paradigm: {pname}  |  Analysis: {analysis_name}")
                    print(f"{'='*60}")
                    aconfig = config.for_paradigm_analysis(pname, analysis_name)
                    _prepare_output(aconfig, analysis_name, strict=strict, force=force, steps=steps)
                    _run_single(aconfig, analysis_name, steps=steps, select=select, jobs=jobs)
                    print()
    else:
        # Legacy single-paradigm config
        if not args.analysis:
            print("ERROR: --analysis is required for single-paradigm configs.")
            sys.exit(1)
        _prepare_output(config, args.analysis, strict=strict, force=force, steps=steps)
        analyzer = StudyAnalyzer(config)
        _print_study_summary(config, analyzer)
        analyzer.run_analysis(args.analysis, steps=steps, select=select, jobs=jobs)
        print(f"\nDone. Output: {config.output_dir / canonical_analysis_name(args.analysis)}")


def _validate_single(config: StudyConfig, paradigm_name: str | None = None):
    """Validate one config and print results. Returns list of issues."""
    try:
        analyzer = StudyAnalyzer(config)
    except Exception as e:
        prefix = f"[{paradigm_name}] " if paradigm_name else ""
        print(f"{prefix}ERROR: Failed to initialize: {e}")
        return [str(e)]

    issues = analyzer.validate()

    print(f"Study: {config.name}")
    print(f"Subjects discovered: {len(analyzer.subjects)}")

    groups = {}
    for s in analyzer.subjects:
        groups.setdefault(s.group, []).append(s.subject_id)
    for g in config.group_order:
        if g in groups:
            label = config.get_group_label(g)
            print(f"  {label} ({g}): n={len(groups[g])}")

    if issues:
        print(f"\n  Warnings ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")

    return issues


def cmd_validate(args):
    """Validate a study configuration."""
    config = StudyConfig.from_yaml(args.study)

    if config.has_paradigms:
        paradigm_names = [args.paradigm] if args.paradigm else list(config.paradigms.keys())
        all_issues = []

        print(f"Study: {config.name}")
        print(f"Config: {args.study}")
        print(f"Paradigms: {len(config.paradigms)}")
        print()

        # Shared validation
        print("Shared configuration:")
        print(f"  Groups: {len(config.groups)}")
        for gid, label in config.groups.items():
            print(f"    {gid}: {label}")
        print(f"  Contrasts: {len(config.contrasts)}")
        for c in config.contrasts:
            print(f"    {c.name}: {c.group_a} vs {c.group_b}")
        print(f"  Bands: {len(config.bands)}")
        for name, (lo, hi) in config.bands.items():
            print(f"    {name}: {lo}-{hi} Hz")
        print()

        for pname in paradigm_names:
            print(f"--- Paradigm: {pname} ---")
            analyses = config.get_paradigm_analyses(pname) or []
            print(f"  Analyses: {', '.join(analyses) if analyses else '(none)'}")
            for aname in analyses:
                aconfig = config.for_paradigm_analysis(pname, aname)
                print(f"  [{aname}]")
                issues = _validate_single(aconfig, f"{pname}/{aname}")
                all_issues.extend(issues)
            print()

        if all_issues:
            print(f"Total warnings: {len(all_issues)}")
            sys.exit(1)
        else:
            print("Validation passed (all paradigms).")
    else:
        # Legacy single-paradigm validation
        config_issues = config.validate()

        try:
            analyzer = StudyAnalyzer(config)
        except Exception as e:
            print(f"ERROR: Failed to initialize: {e}")
            sys.exit(1)

        issues = analyzer.validate()

        print(f"Study: {config.name}")
        print(f"Config: {args.study}")
        print(f"Subjects discovered: {len(analyzer.subjects)}")

        groups = {}
        for s in analyzer.subjects:
            groups.setdefault(s.group, []).append(s.subject_id)
        for g, subs in sorted(groups.items()):
            label = config.get_group_label(g)
            print(f"  {label} ({g}): n={len(subs)}")

        print(f"\nContrasts: {len(config.contrasts)}")
        for c in config.contrasts:
            print(f"  {c.name}: {c.group_a} vs {c.group_b}")

        print(f"\nBands: {len(config.bands)}")
        for name, (lo, hi) in config.bands.items():
            print(f"  {name}: {lo}-{hi} Hz")

        if issues:
            print(f"\nWarnings ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print("\nValidation passed.")


def cmd_list(args):
    """List available analyses."""
    # If --study provided with paradigms, show paradigm-aware listing
    if hasattr(args, "study") and args.study:
        config = StudyConfig.from_yaml(args.study)
        if config.has_paradigms:
            print(f"Study: {config.name}\n")
            for pname, pdata in config.paradigms.items():
                analyses = pdata.get("analyses", [])
                print(f"  {pname}:")
                for aname in analyses:
                    meta = ANALYSIS_METADATA.get(aname, {})
                    desc = meta.get("description", "")
                    if not desc:
                        cls = ANALYSIS_REGISTRY.get(aname)
                        desc = cls.__doc__.strip().splitlines()[0] if cls and cls.__doc__ else ""
                    print(f"    {aname:<24s} {desc}")
                print()
            return

    # Default: group analyses by category and level
    groups: dict[str, list[tuple[str, str]]] = {}
    from .core import _DEPRECATED_NAMES
    for name in sorted(ANALYSIS_REGISTRY.keys()):
        if name in _DEPRECATED_NAMES:
            continue  # hide deprecated aliases from listing
        meta = ANALYSIS_METADATA.get(name, {})
        category = meta.get("category", "other")
        level = meta.get("level", "")
        desc = meta.get("description", "")
        if not desc:
            cls = ANALYSIS_REGISTRY[name]
            desc = cls.__doc__.strip().splitlines()[0] if cls.__doc__ else "No description"
        key = f"{category}|{level}"
        groups.setdefault(key, []).append((name, desc))

    # Display headers
    category_labels = {
        "resting|roi": "Resting State (ROI Level)",
        "resting|vertex": "Resting State (Vertex Level)",
        "resting|electrode": "Resting State (Electrode Level)",
        "evoked|roi": "Evoked Response (ROI Level)",
        "evoked|vertex": "Evoked Response (Vertex Level)",
        "evoked|electrode": "Evoked Response (Electrode Level)",
    }

    print("Available analyses:\n")
    for key, label in category_labels.items():
        if key in groups:
            print(f"  {label}:")
            for name, desc in groups[key]:
                dims = list(getattr(ANALYSIS_REGISTRY[name], "SELECTABLE", {}) or {})
                tag = f"  [--select: {', '.join(dims)}]" if dims else ""
                print(f"    {name:<24s} {desc}{tag}")
            print()

    # Any uncategorized
    shown_keys = set(category_labels.keys())
    for key, items in groups.items():
        if key not in shown_keys:
            print(f"  Other:")
            for name, desc in items:
                print(f"    {name:<24s} {desc}")
            print()


def cmd_figure(args):
    """Generate on-demand summary figures from pre-computed stats tables."""
    from .viz.figure_registry import generate_figure, list_figure_types, TABLE_SCHEMAS

    config = StudyConfig.from_yaml(args.study)

    # --list mode
    if args.list:
        fig_types = list_figure_types()
        if config.has_paradigms:
            print(f"Study: {config.name}\n")
            for pname, pdata in config.paradigms.items():
                analyses = pdata.get("analyses", [])
                print(f"  {pname}:")
                for aname in analyses:
                    types = fig_types.get(aname, [])
                    if types:
                        print(f"    {aname:<24s} {', '.join(types)}")
                    else:
                        print(f"    {aname:<24s} (no figure types)")
                print()
        else:
            print("Available figure types:\n")
            for aname, types in sorted(fig_types.items()):
                print(f"  {aname:<24s} {', '.join(types)}")
        return

    # Generate mode — require paradigm + analysis + type
    if not args.paradigm:
        print("ERROR: --paradigm is required (or use --list to see options)")
        sys.exit(1)
    if not args.analysis:
        print("ERROR: --analysis is required")
        sys.exit(1)
    if not args.type:
        fig_types = list_figure_types(args.analysis)
        available = fig_types.get(args.analysis, [])
        print(f"ERROR: --type is required. Available for '{args.analysis}': {', '.join(available)}")
        sys.exit(1)

    # Resolve directories from config
    if config.has_paradigms:
        aconfig = config.for_paradigm_analysis(args.paradigm, args.analysis)
    else:
        aconfig = config

    tbl_dir = aconfig.results_dir / "tables" / args.paradigm / args.analysis
    fig_dir = Path(args.output) if args.output else (
        aconfig.results_dir / "figures" / args.paradigm / args.analysis
    )

    if not tbl_dir.exists():
        print(f"ERROR: Stats table directory not found: {tbl_dir}")
        sys.exit(1)

    # Build kwargs (analysis passed separately to generate_figure)
    gen_kwargs = {}
    if args.contrast:
        gen_kwargs["contrast"] = args.contrast
    if args.band:
        gen_kwargs["band"] = args.band
    if hasattr(aconfig, "roi_categories"):
        gen_kwargs["config"] = aconfig
    # Pass data_dir for glass_brain
    data_dir = aconfig.output_dir / args.analysis / "data"
    if data_dir.exists():
        gen_kwargs["data_dir"] = data_dir

    print(f"Generating {args.type} for {args.paradigm}/{args.analysis}...")
    print(f"  Tables: {tbl_dir}")
    print(f"  Output: {fig_dir}")

    paths = generate_figure(args.analysis, args.type, tbl_dir, fig_dir, **gen_kwargs)

    if paths:
        print(f"\nGenerated {len(paths)} figure(s):")
        for p in paths:
            print(f"  {p}")
    else:
        print("\nNo figures generated (no data matched filters or no significant results).")


def _discover_init_subjects(derivatives_dir: Path, data_subdir: str):
    """Find subjects under a reconstruction root, in either supported layout.

    Returns ``(layout, subject_groups)`` where ``layout`` is ``"flat"`` (BIDS
    ``sub-*`` dirs directly under ``derivatives/``; groups unknown unless
    ``--groups-from`` supplies them) or ``"grouped"`` (``<Group>/<Subject>/``
    dirs; the group is the folder name).
    """
    flat = sorted(
        d.name for d in derivatives_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    if flat:
        return "flat", {name: "UNKNOWN" for name in flat}

    grouped: dict[str, str] = {}
    for group_dir in sorted(d for d in derivatives_dir.iterdir() if d.is_dir()):
        for subj_dir in sorted(d for d in group_dir.iterdir() if d.is_dir()):
            if (subj_dir / data_subdir).is_dir() or (subj_dir / "data").is_dir():
                grouped[subj_dir.name] = group_dir.name
    if grouped:
        return "grouped", grouped
    return "flat", {}


# Starter bands for a scaffolded config — the README's canonical set.
_INIT_BANDS = {
    "Delta": [1, 4],
    "Theta": [4, 10],
    "Alpha": [10, 13],
    "Beta": [13, 30],
    "Low Gamma": [30, 55],
    "High Gamma": [65, 80],
}

_INIT_DEFAULT_ANALYSES = ["roi_psd", "roi_aperiodic", "roi_connectivity"]


def cmd_init(args):
    """Scaffold a study YAML (design:/hypotheses: + paradigms:) from a reconstruction dir.

    Writes ``{paradigm_dir}/analysis/{name}.yaml`` by default; ``--output -``
    prints the YAML to stdout (status goes to stderr) so it can be redirected.
    The file parses with ``StudyConfig.from_yaml`` as-is; edit groups/hypotheses
    and the analysis list, then ``validate`` it.
    """
    paradigm_dir = Path(args.paradigm_dir).resolve()
    derivatives_dir = paradigm_dir / "derivatives"
    data_subdir = args.data_subdir
    err = sys.stderr

    if not derivatives_dir.is_dir():
        print(f"ERROR: derivatives directory not found: {derivatives_dir}", file=err)
        sys.exit(1)

    layout, subject_groups = _discover_init_subjects(derivatives_dir, data_subdir)
    if not subject_groups:
        print(
            f"ERROR: no subjects found in {derivatives_dir} (expected sub-* dirs, "
            "or <Group>/<Subject>/ dirs).", file=err,
        )
        sys.exit(1)

    # --groups-from: source-localization study_config has subjects[].id/.group
    if args.groups_from:
        with open(Path(args.groups_from).resolve()) as f:
            src_config = yaml.safe_load(f) or {}
        id_to_group = {}
        for s_ in src_config.get("subjects", []) or []:
            sid, group = str(s_.get("id", "")), s_.get("group")
            if sid and group:
                id_to_group[sid] = group
                id_to_group[f"sub-{sid}"] = group
        for subj in subject_groups:
            if subj in id_to_group:
                subject_groups[subj] = id_to_group[subj]
            elif subj.startswith("sub-") and subj[4:] in id_to_group:
                subject_groups[subj] = id_to_group[subj[4:]]

    unique_groups = sorted({g for g in subject_groups.values() if g != "UNKNOWN"})
    levels = unique_groups or ["Group1", "Group2"]

    # Declared hypotheses: an omnibus when there are >2 groups, plus every
    # pairwise contrast (weights: later level minus earlier level).
    hypotheses: list[dict] = []
    if len(levels) > 2:
        hypotheses.append({"name": "group_omnibus", "kind": "omnibus", "role": "phenotype"})
    for i, ga in enumerate(levels):
        for gb in levels[i + 1:]:
            hypotheses.append({
                "name": f"{gb}_vs_{ga}",
                "kind": "contrast",
                "label": f"{gb} vs {ga}",
                "weights": {gb: 1, ga: -1},
                "role": "phenotype",
            })

    config_name = args.name or paradigm_dir.name
    out_dir = paradigm_dir / "analysis"
    analyses = [a.strip() for a in args.analyses.split(",") if a.strip()]
    unknown = [a for a in analyses if a not in ANALYSIS_REGISTRY]
    if unknown:
        print(f"ERROR: unknown analyses: {', '.join(unknown)} (see `source-analytics list`)", file=err)
        sys.exit(1)

    paradigm_block: dict = {
        "data_dir": str(derivatives_dir),
        "data_subdir": data_subdir,
        "analyses": {a: {} for a in analyses},
    }
    if layout == "flat":
        # Flat sub-* layout needs the explicit subject → group map.
        paradigm_block["subjects"] = dict(subject_groups)

    config = {
        "name": config_name,
        "groups": {g: g for g in levels},
        "group_order": list(levels),
        "group_colors": {},
        "design": {"factor": "group", "reference": levels[0], "levels": list(levels)},
        "hypotheses": hypotheses,
        "bands": dict(_INIT_BANDS),
        "epoch_sampling": {
            "enabled": False, "epoch_duration_sec": 2.0, "n_epochs": 80, "n_bootstrap": 1,
        },
        # Absolute so the file works from any working directory.
        "output_dir": str(out_dir / "analytics"),
        "results_dir": str(out_dir / "results"),
        "paradigms": {args.paradigm: paradigm_block},
    }

    header = (
        f"# source-analytics study config, scaffolded by `source-analytics init` from\n"
        f"# {paradigm_dir}\n"
        f"# Layout: {layout}. Edit groups / hypotheses / analyses, then:\n"
        f"#   source-analytics validate --study <this file>\n"
        f"#   source-analytics run --study <this file> --paradigm {args.paradigm} "
        f"--analysis {analyses[0] if analyses else 'roi_psd'}\n"
    )
    text = header + yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if args.output == "-":
        sys.stdout.write(text)
        out_path = None
    else:
        out_path = Path(args.output).resolve() if args.output else out_dir / f"{config_name}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"Config written: {out_path}", file=err)

    print(f"Layout: {layout}; subjects: {len(subject_groups)}", file=err)
    for g in unique_groups:
        n = sum(1 for v in subject_groups.values() if v == g)
        print(f"  {g}: n={n}", file=err)
    n_unknown = sum(1 for v in subject_groups.values() if v == "UNKNOWN")
    if n_unknown:
        print(
            f"  UNKNOWN: n={n_unknown} (pass --groups-from, or edit "
            f"paradigms.{args.paradigm}.subjects to assign groups)", file=err,
        )
    if out_path is not None:
        # Prove the scaffold parses before the user edits it.
        try:
            StudyConfig.from_yaml(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: scaffolded config failed to parse: {exc}", file=err)


def main():
    parser = argparse.ArgumentParser(
        prog="source-analytics",
        description="Statistical analysis toolkit for source-localized EEG data",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Run an analysis")
    p_run.add_argument("--study", required=True, type=Path, help="Path to study YAML config")
    p_run.add_argument("--paradigm", help="Paradigm name (multi-paradigm configs)")
    p_run.add_argument("--analysis", choices=list(ANALYSIS_REGISTRY.keys()), help="Analysis to run")
    p_run.add_argument(
        "--profile", metavar="NAME",
        help="Run under the top-level '<NAME>:' profile block, which narrows bands, "
        "ROIs, hypotheses and the analysis set, and writes to a separate tree "
        "(results/<NAME>/, analytics/<NAME>/). Omit for the default/exploratory "
        "profile. NOTE: narrowing ROIs changes the FDR family, so a profile's "
        "q-values are NOT comparable to the default profile's.",
    )
    p_run.add_argument(
        "--steps",
        help="Comma-separated lifecycle steps to run (default: all). "
        f"Valid: {', '.join(sorted(VALID_STEPS))}",
    )
    p_run.add_argument(
        "--jobs", "-j", type=int, default=None, metavar="N",
        help="Parallel worker processes for the per-subject process step. "
        "An explicit N wins over the study YAML's top-level `jobs:`; when omitted "
        "that YAML value is used, else serial. N<=0 uses all-but-one core. Only "
        "parallel-capable modules use it (the vertex analyses, roi_connectivity, "
        "electrode_connectivity); others run serially. Results are identical to "
        "serial regardless of N.",
    )
    p_run.add_argument(
        "--metric",
        help="Comma-separated connectivity metric(s) to compute, restricting a "
        "module's configured set (e.g. --metric dwpli,wpli). Shared compute "
        "passes are preserved; only the requested metrics are emitted. "
        "Shorthand for --select metric=...",
    )
    p_run.add_argument(
        "--band",
        help="Comma-separated band(s) to compute, restricting a module's "
        "configured bands (e.g. --band low_gamma). Case/format-insensitive. "
        "Shorthand for --select band=...",
    )
    p_run.add_argument(
        "--hypothesis",
        help="Comma-separated declared hypothesis(es) to test (e.g. --hypothesis "
        "disease_effect). Runs ONE (or a few) by name from the design spec; "
        "the rest are skipped. Manual control — no auto-gating. "
        "Shorthand for --select hypothesis=...",
    )
    p_run.add_argument(
        "--select",
        action="append",
        metavar="DIM=val[,val...]",
        help="Generic sub-output selection (repeatable). DIM is a module's "
        "selectable dimension (see `list`); values restrict that dimension. "
        "e.g. --select metric=pli --select band=beta,low_gamma",
    )
    p_run.add_argument(
        "--strict-output",
        action="store_true",
        help="Error if the analysis output directory already exists; "
             "--force overrides",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Remove the analysis's previous output before running: its published "
             "tables/ and figures/ dirs always, and its analytics working dir "
             "(data/ + summary) when the process step runs. Also overrides "
             "--strict-output.",
    )
    p_run.set_defaults(func=cmd_run)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate study config")
    p_val.add_argument("--study", required=True, type=Path, help="Path to study YAML config")
    p_val.add_argument("--paradigm", help="Validate a specific paradigm only")
    p_val.set_defaults(func=cmd_validate)

    # list
    p_list = subparsers.add_parser("list", help="List available analyses")
    p_list.add_argument("--study", type=Path, help="Study YAML (shows paradigm-aware listing)")
    p_list.set_defaults(func=cmd_list)

    # figure
    p_fig = subparsers.add_parser("figure", help="Generate on-demand summary figures")
    p_fig.add_argument("--study", required=True, type=Path, help="Path to study YAML config")
    p_fig.add_argument("--paradigm", help="Paradigm name")
    p_fig.add_argument("--analysis", help="Analysis name")
    p_fig.add_argument("--type", help="Figure type (effect_heatmap, volcano, circos, glass_brain)")
    p_fig.add_argument("--list", action="store_true", help="List available figure types")
    p_fig.add_argument("--contrast", help="Filter to specific contrast")
    p_fig.add_argument("--band", help="Filter to specific band")
    p_fig.add_argument("--output", type=Path, help="Custom output directory")
    p_fig.set_defaults(func=cmd_figure)

    # init
    p_init = subparsers.add_parser(
        "init",
        help="Scaffold a study YAML (design/hypotheses + one paradigm) from a "
             "reconstruction directory; writes <dir>/analysis/<name>.yaml",
    )
    p_init.add_argument("paradigm_dir", type=Path, help="Reconstruction directory (contains derivatives/)")
    p_init.add_argument("--name", help="Study name (default: directory name)")
    p_init.add_argument("--groups-from", type=Path, help="source-localization study_config.yaml for group mappings")
    p_init.add_argument("--paradigm", default="resting", help="Name of the paradigm block to emit (default: resting)")
    p_init.add_argument(
        "--analyses", default=",".join(_INIT_DEFAULT_ANALYSES), metavar="a,b,...",
        help=f"Analyses to list under the paradigm (default: {','.join(_INIT_DEFAULT_ANALYSES)})",
    )
    p_init.add_argument("--data-subdir", default="pipeline/data", help="Per-subject data subdir (default: pipeline/data)")
    p_init.add_argument(
        "--output", "-o", metavar="PATH",
        help="Where to write the YAML (default: <paradigm_dir>/analysis/<name>.yaml); '-' prints to stdout",
    )
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()

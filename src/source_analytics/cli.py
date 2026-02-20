"""CLI entry point for source-analytics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import StudyConfig
from .core import StudyAnalyzer, ANALYSIS_REGISTRY, ANALYSIS_METADATA


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


def _run_single(config: StudyConfig, analysis_name: str):
    """Run one analysis on a (possibly paradigm-scoped) config."""
    analyzer = StudyAnalyzer(config)
    _print_study_summary(config, analyzer)
    analyzer.run_analysis(analysis_name)
    print(f"\nDone. Output: {config.output_dir / analysis_name}")


def cmd_run(args):
    """Run an analysis module."""
    config = StudyConfig.from_yaml(args.study)

    if config.has_paradigms:
        # Multi-paradigm config
        if args.paradigm:
            if args.analysis:
                # Scope to one paradigm + one analysis
                aconfig = config.for_paradigm_analysis(args.paradigm, args.analysis)
                _run_single(aconfig, args.analysis)
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
                    _run_single(aconfig, analysis_name)
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
                    _run_single(aconfig, analysis_name)
                    print()
    else:
        # Legacy single-paradigm config
        if not args.analysis:
            print("ERROR: --analysis is required for single-paradigm configs.")
            sys.exit(1)
        analyzer = StudyAnalyzer(config)
        _print_study_summary(config, analyzer)
        analyzer.run_analysis(args.analysis)
        print(f"\nDone. Output: {config.output_dir / args.analysis}")


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
    for name in sorted(ANALYSIS_REGISTRY.keys()):
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
        "resting|roi": "Resting State (ROI level)",
        "resting|wholebrain": "Resting State (Wholebrain level)",
        "resting|electrode": "Resting State (Electrode level)",
        "evoked|roi": "Evoked Response",
    }

    print("Available analyses:\n")
    for key, label in category_labels.items():
        if key in groups:
            print(f"  {label}:")
            for name, desc in groups[key]:
                print(f"    {name:<24s} {desc}")
            print()

    # Any uncategorized
    shown_keys = set(category_labels.keys())
    for key, items in groups.items():
        if key not in shown_keys:
            print(f"  Other:")
            for name, desc in items:
                print(f"    {name:<24s} {desc}")
            print()


def cmd_init(args):
    """Scaffold a source-analytics YAML config from a paradigm directory."""
    paradigm_dir = Path(args.paradigm_dir).resolve()
    derivatives_dir = paradigm_dir / "derivatives"

    if not derivatives_dir.is_dir():
        print(f"ERROR: derivatives directory not found: {derivatives_dir}")
        sys.exit(1)

    # Discover sub-* directories
    subject_dirs = sorted(
        d.name for d in derivatives_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    if not subject_dirs:
        print(f"ERROR: no sub-* directories found in {derivatives_dir}")
        sys.exit(1)

    # Build subject_groups mapping
    subject_groups = {}
    if args.groups_from:
        groups_path = Path(args.groups_from).resolve()
        with open(groups_path) as f:
            src_config = yaml.safe_load(f)
        # source-localization study_config has subjects[].id and subjects[].group
        id_to_group = {}
        for s in src_config.get("subjects", []):
            sid = s.get("id", "")
            group = s.get("group")
            if sid and group:
                id_to_group[f"sub-{sid}"] = group
        for subj in subject_dirs:
            if subj in id_to_group:
                subject_groups[subj] = id_to_group[subj]
            else:
                subject_groups[subj] = "UNKNOWN"
    else:
        for subj in subject_dirs:
            subject_groups[subj] = "UNKNOWN"

    # Determine unique groups (excluding UNKNOWN)
    unique_groups = sorted(set(
        g for g in subject_groups.values() if g != "UNKNOWN"
    ))

    # Build YAML structure
    config_name = args.name or paradigm_dir.name
    config = {
        "name": config_name,
        "groups": {g: g for g in unique_groups} if unique_groups else {"Group1": "Group 1"},
        "group_order": unique_groups if unique_groups else ["Group1"],
        "group_colors": {},
        "contrasts": [],
        "bands": {
            "delta": [2, 3.5],
            "theta": [3.5, 7.5],
            "alpha_1": [8, 10],
            "alpha_2": [10.5, 12.5],
            "beta": [13, 30],
            "gamma_1": [30, 55],
            "gamma_2": [65, 80],
            "epsilon": [81, 120],
        },
        "roi_categories": {},
        "discovery": {
            "data_subdir": "pipeline/data",
            "subject_groups": subject_groups,
        },
    }

    # Generate pairwise contrasts
    if len(unique_groups) >= 2:
        for i, ga in enumerate(unique_groups):
            for gb in unique_groups[i + 1:]:
                config["contrasts"].append({
                    "name": f"{ga}_vs_{gb}",
                    "group_a": ga,
                    "group_b": gb,
                })

    # Write config
    analysis_dir = paradigm_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    out_path = analysis_dir / f"{config_name}.yaml"

    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written: {out_path}")
    print(f"Subjects: {len(subject_dirs)}")
    if unique_groups:
        for g in unique_groups:
            n = sum(1 for v in subject_groups.values() if v == g)
            print(f"  {g}: n={n}")
    n_unknown = sum(1 for v in subject_groups.values() if v == "UNKNOWN")
    if n_unknown:
        print(f"  UNKNOWN: n={n_unknown} (edit config to assign groups)")


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

    # init
    p_init = subparsers.add_parser("init", help="Scaffold analysis config from paradigm directory")
    p_init.add_argument("paradigm_dir", type=Path, help="Paradigm directory (contains derivatives/)")
    p_init.add_argument("--name", help="Study name (default: directory name)")
    p_init.add_argument("--groups-from", type=Path, help="source-localization study_config.yaml for group mappings")
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()

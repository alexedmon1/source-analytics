"""CLI entry point for source-analytics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import StudyConfig
from .core import StudyAnalyzer, ANALYSIS_REGISTRY


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args):
    """Run an analysis module."""
    config = StudyConfig.from_yaml(args.study)
    analyzer = StudyAnalyzer(config)

    print(f"Study: {config.name}")
    print(f"Subjects discovered: {len(analyzer.subjects)}")
    groups = {}
    for s in analyzer.subjects:
        groups.setdefault(s.group, []).append(s.subject_id)
    for g, subs in groups.items():
        print(f"  {config.get_group_label(g)} ({g}): n={len(subs)}")
    print()

    analyzer.run_analysis(args.analysis)
    print(f"\nDone. Output: {config.output_dir / args.analysis}")


def cmd_validate(args):
    """Validate a study configuration."""
    config = StudyConfig.from_yaml(args.study)

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
    print("Available analyses:")
    for name in sorted(ANALYSIS_REGISTRY.keys()):
        cls = ANALYSIS_REGISTRY[name]
        print(f"  {name}: {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else 'No description'}")


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
    p_run.add_argument("--analysis", required=True, choices=list(ANALYSIS_REGISTRY.keys()), help="Analysis to run")
    p_run.set_defaults(func=cmd_run)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate study config")
    p_val.add_argument("--study", required=True, type=Path, help="Path to study YAML config")
    p_val.set_defaults(func=cmd_validate)

    # list
    p_list = subparsers.add_parser("list", help="List available analyses")
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

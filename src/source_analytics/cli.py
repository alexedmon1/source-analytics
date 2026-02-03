"""CLI entry point for source-analytics."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

    args = parser.parse_args()
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()

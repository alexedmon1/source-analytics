"""Declared outcome tables (``bayes.tables``): assembling the engine's long table from any per-subject table.

The engine reads one long table: subject, the design factor, an outcome label (the engine's "band"), optionally a
spatial column (the engine's ROI), and the value. Built-in analyses (``tables.BAYES_TABLES``) write that table
themselves. A declared table builds it from rows the config selects:

- **source** tables: an analysis's per-subject table (``<output_dir>/<paradigm>/<analysis>/data/*measures*.csv``, or
  the built-in table's file) in one or more paradigms; **csv** tables: any file, for example source-localization's
  ``field region-values`` output.
- **outcomes**: label -> selection. Each selection maps a column to a value or a list of values (``paradigm`` selects
  among the paradigms read). ``combine: mean`` averages the selected rows per subject (for example the mean over a
  set of parcels); otherwise a selection must give at most one row per subject (and spatial unit).
- **groups**: maps the table's group labels onto the design factor's levels; rows of unmapped groups are dropped.
  This is how groups are pooled (several KO groups into one level) without changing the design spec.
- **spatial**: ``none`` (no ROI in the model: one row per subject and outcome) or a column kept as the engine's ROI.

The assembled table, its inputs and their sha256 are recorded in the job's provenance.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from .config import BayesConfigError, TableSpec
from .tables import BAYES_TABLES

logger = logging.getLogger(__name__)

__all__ = ["source_files", "assemble"]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(spec: TableSpec, output_dir: Path, default_paradigm: str | None = None) -> dict[str, Path]:
    """The input file per paradigm ("" for a csv table)."""
    if spec.csv is not None:
        return {"": spec.csv}
    paradigms = spec.paradigms or ((default_paradigm,) if default_paradigm else ("",))
    out = {}
    for p in paradigms:
        d = Path(output_dir) / p / spec.source / "data" if p else Path(output_dir) / spec.source / "data"
        if spec.source in BAYES_TABLES:
            f = d / BAYES_TABLES[spec.source].csv
        else:
            hits = sorted(d.glob("*measures*.csv"))
            if len(hits) != 1:
                raise BayesConfigError(f"bayes.tables.{spec.name}: expected one *measures*.csv in {d}, "
                                       f"found {len(hits)}")
            f = hits[0]
        if not f.exists():
            raise BayesConfigError(f"bayes.tables.{spec.name}: {f} not found (run {spec.source} first)")
        out[p] = f
    return out


def assemble(spec: TableSpec, output_dir: Path, factor: str, default_paradigm: str | None = None):
    """The engine's long table for ``spec`` and the record of its inputs.

    Returns (DataFrame with columns subject, ``factor``, outcome, [spatial], ``spec.value``; provenance dict).
    """
    files = source_files(spec, spec.output_dir or output_dir, default_paradigm)
    frames = []
    for p, f in files.items():
        df = pd.read_csv(f)
        df["paradigm"] = p
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for col in (spec.subject, spec.value):
        if col not in df.columns:
            raise BayesConfigError(f"bayes.tables.{spec.name}: column {col!r} not in its input")
    if spec.group not in df.columns:
        if spec.subject_groups is None:
            raise BayesConfigError(f"bayes.tables.{spec.name}: no {spec.group!r} column; give subject_groups")
        sg = pd.read_csv(spec.subject_groups)
        if not {"subject", "group"} <= set(sg.columns):
            raise BayesConfigError(f"bayes.tables.{spec.name}.subject_groups: needs columns subject, group")
        df = df.merge(sg[["subject", "group"]].rename(columns={"subject": spec.subject, "group": spec.group}),
                      on=spec.subject, how="left")
    df[spec.group] = df[spec.group].astype(str)
    if spec.groups:
        n0 = df[spec.subject].nunique()
        df = df[df[spec.group].isin(spec.groups)].copy()
        df[spec.group] = df[spec.group].map(spec.groups)
        logger.info("bayes.tables.%s: %d of %d subjects in mapped groups", spec.name, df[spec.subject].nunique(), n0)
    rows = []
    keys = [spec.subject] + ([spec.spatial] if spec.spatial else [])
    for label, sel in spec.outcomes.items():
        sel = dict(sel)
        combine = sel.pop("combine", "none")
        mask = pd.Series(True, index=df.index)
        for col, want in sel.items():
            if col not in df.columns:
                raise BayesConfigError(f"bayes.tables.{spec.name}.outcomes.{label}: no column {col!r}")
            vals = [str(v) for v in want] if isinstance(want, list) else [str(want)]
            mask &= df[col].astype(str).isin(vals)
        part = df[mask]
        if part.empty:
            raise BayesConfigError(f"bayes.tables.{spec.name}.outcomes.{label}: selects no rows")
        if combine == "mean":
            part = (part.groupby(keys + [spec.group], as_index=False)[spec.value].mean())
        elif part.duplicated(keys).any():
            raise BayesConfigError(f"bayes.tables.{spec.name}.outcomes.{label}: more than one row per subject"
                                   f"{' and ' + spec.spatial if spec.spatial else ''}; narrow it or set combine: mean")
        part = part[keys + [spec.group, spec.value]].copy()
        part["outcome"] = label
        rows.append(part)
    out = pd.concat(rows, ignore_index=True).rename(columns={spec.subject: "subject", spec.group: factor})
    if spec.spatial and spec.spatial != "roi":
        out = out.rename(columns={spec.spatial: "roi"})
    if out[spec.value].isna().any():
        raise BayesConfigError(f"bayes.tables.{spec.name}: missing values in {spec.value}")
    cols = ["subject", factor, "outcome"] + (["roi"] if spec.spatial else []) + [spec.value]
    out = out[cols]
    prov = {"name": spec.name, "inputs": {p or "csv": {"path": str(f), "sha256": _sha(f)} for p, f in files.items()},
            "definition": {"source": spec.source, "csv": str(spec.csv) if spec.csv else None,
                           "output_dir": str(spec.output_dir) if spec.output_dir else None,
                           "paradigms": list(spec.paradigms) if spec.paradigms else None, "value": spec.value,
                           "spatial": spec.spatial or "none", "outcomes": spec.outcomes, "groups": spec.groups,
                           "subject_groups": str(spec.subject_groups) if spec.subject_groups else None},
            "n_rows": int(len(out)), "n_subjects": int(out["subject"].nunique()),
            "outcomes": sorted(out["outcome"].unique().tolist())}
    return out, prov

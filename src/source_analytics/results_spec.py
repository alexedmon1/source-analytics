"""The results specification: ``analysis.json`` beside a module's tables.

So that any reader -- a study gallery, a manuscript script, Esprit -- can read a
module's results without this package, a module that declares a
:class:`ResultsSpec` describes its hypothesis tables in the shared format of
Lumière (``docs/RESULTS_SPEC.md`` there, schema 0.4): one ``analysis.json``,
a column dictionary per table (``<table>.json``), and the specification's fields
added to this package's own ``provenance.json``, whose other fields stay as they
are for source-lightbox.

The folder is the module's published tables folder (``BaseAnalysis.tbl_dir``),
adopted in place, as the specification allows for existing output trees.

Optional: Lumière is the ``results`` extra. Without it nothing is written
and the run goes on; with it, a folder that does not conform is reported in the
log but never fails a run that already produced its numbers.
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:                                               # the optional `results` extra
    import lumiere
    from lumiere.spec import analysis_id as _analysis_id
    from lumiere.spec import default_run_id as _default_run_id
except ImportError:                                # pragma: no cover - depends on install
    lumiere = None

INSTALL_HINT = "install the optional extra: pip install 'source-analytics[results]'"

#: BIDS task label of each paradigm category (ANALYSIS_METADATA ``category``).
TASK_OF_CATEGORY = {"resting": "rest"}
#: The specification's analysis_type for each ANALYSIS_METADATA ``level``.
ANALYSIS_TYPE_OF_LEVEL = {"roi": "roi", "electrode": "sensor", "vertex": "vertex"}

#: p.adjust's method -> (the spec's p_kind, its method name).
_METHODS = {"BH": ("fdr", "BH"), "fdr": ("fdr", "BH"), "BY": ("fdr", "BY"),
            "holm": ("fwe", "holm"), "hochberg": ("fwe", "hochberg"), "hommel": ("fwe", "other"),
            "bonferroni": ("fwe", "bonferroni"), "none": ("uncorrected", "none")}
_FAMILY = re.compile(r"^scope=(?P<scope>\S+) method=(?P<method>\S+) key=(?P<key>.*) "
                     r"members=(?P<axis>\S+)\[(?P<n>\d+)\] hash=(?P<hash>\S+)$")
#: R writes q < 0.05 as `significant` (hypothesis.R .apply_fdr).
SIGNIFICANCE_ALPHA = 0.05


@dataclass(frozen=True)
class SpecTable:
    """One hypothesis table of a module, as the specification lists it."""

    suffix: str          # after the module name: "_global_hypotheses.csv"
    role: str            # tests | elements
    rows: str            # what one row is
    description: str
    element: bool        # `spatial` names the element (ROI, region, channel, edge)
    headline: bool = False
    file: str | None = None      # the table's whole file name, when it is not module name + suffix
    spatial: str | None = None   # what `spatial` holds, when the module's default wording does not fit
    #: the column naming this table's measure, when not the module's ``measure_col``
    measure_col: str | None = None
    #: what one element of this table is (spec 0.4.2 ElementName), when not the module's ``element``
    element_name: str | None = None

    def filename(self, module: str) -> str:
        return self.file or f"{module}{self.suffix}"


@dataclass(frozen=True)
class ResultsSpec:
    """What a module tells the writer that its tables cannot."""

    domain: str
    measures: dict[str, str]           # the module's dv name -> canonical measure name
    units: dict[str, str]              # dv -> the estimate's units, in words
    tables: tuple[SpecTable, ...]
    method: str                        # inference.method, in a sentence
    input_file: str | None = None      # the per-subject table in the module's data/
    caveats: tuple[str, ...] = ()
    name: str | None = None            # the id's name; default: module name minus its level
    facet: str = "band"                # FacetName of the `band` column (spec §5.2): band, freq_pair
    facet_description: str = "frequency band"
    element: str = "ROI"               # what one spatial cell is, in words: ROI, electrode
    #: the column naming each row's measure, in the module's words (``measures`` maps it)
    measure_col: str = "dv"
    #: further facet columns: (column, FacetName, description), e.g. a graph's connectivity metric
    extra_facets: tuple[tuple[str, str, str], ...] = ()
    #: dictionary entries for the module's own columns, or replacing the shared ones (permutation
    #: tables: component, mass, the permutation p)
    extra_columns: tuple[tuple[str, dict], ...] = ()
    #: ``inference.correction`` for tables with no ``fdr_family`` (a permutation test's FWE)
    correction: dict | None = None
    #: ``effect`` in place of the per-row effect size, e.g. ``{"measure": None, "reason": ...}``
    effect: dict | None = None
    #: the analysis's role: exploratory (declared hypotheses, tested) or descriptive (no tests)
    role: str = "exploratory"


def permutation_columns(unit: str, units: str, p_col: str, p_scope: str) -> tuple[tuple[str, dict], ...]:
    """Dictionary entries for the permutation tables (hypothesis/edge.py: NBS components;
    hypothesis/permutation.py: clusters of a map). ``unit`` is what one row's element is
    (component, cluster), ``units`` what it is made of (edges, electrodes); ``p_col`` holds
    the permutation p, corrected over the whole map by the permutation maximum (FWE)."""
    return (
        (p_col, {"Description": f"the {unit}'s permutation p: {p_scope}", "Standard": "p_value",
                 "PKind": "fwe", "PScope": p_scope}),
        ("n_perms", {"Description": "label permutations in the null distribution"}),
        ("threshold", {"Description": "the per-element statistic's threshold that forms the " + unit + "s"}),
        ("mass", {"Description": f"the {unit}'s mass: the sum of |statistic| over its supra-threshold {units}"}),
        ("peak_stat", {"Description": f"the largest |statistic| in the {unit}", "Standard": "stat",
                       "StatName": "per row, as stat_type says: t or F", "StatScope": "peak"}),
        ("stat_type", {"Description": "the per-element statistic's kind",
                       "Levels": {"t": "t (two-group contrasts)", "F": "F (omnibus and weighted contrasts)",
                                  "tost": f"a per-element equivalence summary, not a {unit}"}}),
        ("significant", {"Description": f"the {unit}'s permutation p < 0.05", "Standard": "significant",
                         "Alpha": SIGNIFICANCE_ALPHA}),
        ("region", {"Description": f"the regions the {unit}'s nodes cover"}),
        ("n_equivalent", {"Description": f"equivalence rows: the {units} whose two one-sided tests conclude "
                                         "equivalence"}),
        ("frac_equivalent", {"Description": f"equivalence rows: n_equivalent as a fraction of the {units}"}),
        ("margin_sd", {"Description": f"equivalence rows: the margin each of the {units}' two one-sided tests "
                                      "used, in that element's pooled SDs (the declared margin's value, "
                                      "read as SDs)", "Standard": "margin",
                       "Units": f"pooled SDs of each of the {units}"}),
    )


#: Band power, as roi_psd and electrode_psd name it (spectral/band_power.py).
POWER_MEASURES = {"absolute": "band_power", "relative": "relative_power",
                  "delta_ref": "delta_referenced_power"}
POWER_UNITS = {"absolute": "dB/Hz (10*log10 of the band's mean power density)",
               "relative": "a fraction of the total power over the relative-power range (unitless)",
               "delta_ref": "dB relative to the power density of the delta_reference anchor window"}
POWER_CAVEATS = (
    ("Relative power is not computed for bands above the relative-power range (relative_power.fmax "
     "in the study config), so those bands have no relative rows."),
    ("delta_ref, when the study config has a delta_reference block, is each band's power density "
     "referenced to the anchor window, in dB; bands in delta_reference.exclude_bands (the anchor's "
     "own) are not tested on it."),
)

#: Caveats every hypothesis-layer analysis carries (docs/KNOWN_ISSUES.md).
COMMON_CAVEATS = (
    ("effect_size_type 'hedges_g' is the estimate divided by the model's residual SD, with no "
     "small-sample correction: an uncorrected standardized difference, despite its name "
     "(docs/KNOWN_ISSUES.md 1)."),
    ("The interval of a standardized difference is the estimate's interval divided by the "
     "residual SD, which it treats as known, so it is slightly too narrow."),
    ("mde is at power 0.80 and two-sided alpha 0.05, nominal: no multiple-comparison "
     "correction, so it is optimistic for rows corrected within a family (KNOWN_ISSUES 6); "
     "TOST rows carry the MDE of the difference test (KNOWN_ISSUES 7)."),
    ("The module's legacy omnibus and post-hoc tables in this folder are not part of this "
     "analysis; its declared hypotheses are."),
)


def available() -> bool:
    return lumiere is not None


# --------------------------------------------------------------------- tables ---
def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def prepare_table(path: Path, measures: dict[str, str],
                  measure_col: str = "dv") -> tuple[list[str], list[dict[str, str]]]:
    """Make a hypothesis table read as the specification reads tables, in place.

    R writes a missing value as ``NA``; the specification's missing value is an
    empty cell (which R's readr and pandas also read as missing). And the
    specification names measures by its vocabulary (``aperiodic_exponent``), so a
    ``measure`` column with the canonical name is added after ``dv`` (or the module's
    ``measure_col``). Idempotent.
    """
    cols, rows = _read(path)
    for r in rows:
        for c in cols:
            if r.get(c) == "NA":
                r[c] = ""
        if measure_col in r:
            r["measure"] = measures.get(r[measure_col], r[measure_col])
    if measure_col in cols and "measure" not in cols:
        cols.insert(cols.index(measure_col) + 1, "measure")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return cols, rows


def families(rows: list[dict[str, str]]) -> dict[str, dict]:
    """Correction families, by their ``fdr_family`` label (the family's id), with
    each family's rows counted."""
    out: dict[str, dict] = {}
    for r in rows:
        label = r.get("fdr_family", "")
        if not label:
            continue
        if label in out:
            out[label]["_rows"] += 1
            continue
        m = _FAMILY.match(label)
        if not m:
            raise ValueError(f"fdr_family label not understood: {label!r}")
        p_kind, method = _METHODS.get(m["method"], ("fdr", "other"))
        out[label] = {"id": label, "p_kind": p_kind, "method": method, "scope": m["scope"],
                      "members": f"{m['axis']}[{m['n']}]", "members_hash": m["hash"],
                      "alpha": SIGNIFICANCE_ALPHA, "_key": m["key"], "_method": m["method"], "_rows": 1}
    for f in out.values():
        f["statement"] = (f"{f.pop('_method')} within the family {f.pop('_key')} "
                          f"({f['_rows']} test(s); scope {f['scope']}, {f['members']}), q < 0.05")
    return out


def columns(cols: list[str], rows: list[dict[str, str]], table: SpecTable, spec: ResultsSpec,
            fams: dict[str, dict]) -> dict[str, dict]:
    """The column dictionary for a hypothesis table: every column, or KeyError."""
    kinds = {fams[r["fdr_family"]]["p_kind"] for r in rows if r.get("fdr_family") in fams}
    units = "; ".join(f"{dv}: {u}" for dv, u in spec.units.items())
    has_band = any(r.get("band") for r in rows)
    docs: dict[str, dict[str, Any]] = {
        "hypothesis": {"Description": "the declared hypothesis tested (design.hypotheses in the "
                                      "study config)", "Standard": "contrast"},
        "label": {"Description": "the hypothesis in words", "Standard": "contrast_label"},
        "kind": {"Description": "the hypothesis's kind", "Standard": "test_kind",
                 "Levels": {"omnibus": "F test that any fitted group differs",
                            "contrast": "a weighted contrast of the groups' model means",
                            "equivalence": "two one-sided tests (TOST) that the contrast lies "
                                           "within the margin",
                            "regression": "the slope of the measure on a continuous predictor"}},
        "role": {"Description": "the hypothesis's role tag in the study config (display and "
                                "grouping only)"},
        "test": {"Description": "what the row tests",
                 "Levels": {"difference": "a difference (or omnibus F)",
                            "equivalence": "equivalence within a margin"}},
        "band": ({"Description": spec.facet_description, "Standard": "facet", "FacetName": spec.facet}
                 if has_band else {"Description": f"{spec.facet_description}; empty, as the measure is "
                                                  "broadband"}),
        "spatial": ({"Description": table.spatial or f"the {spec.element} tested (empty on omnibus rows)",
                     "Standard": "element", "ElementName": table.element_name or spec.element}
                    if table.element else
                    {"Description": table.spatial or "empty: these tests are on group means marginalized "
                                                     f"over every {spec.element}"}),
        "region": {"Description": f"the region the row's {spec.element}s belong to"},
        "source": {"Description": f"the edge's source {spec.element} (for coupling, the one whose slower "
                                  "band is used)"},
        "target": {"Description": f"the edge's target {spec.element} (for coupling, the one whose faster "
                                  "band is used)"},
        "estimate": {"Description": "the contrast's estimate (the weighted difference of the groups' "
                                    "model means) or the slope, in the measure's units",
                     "Standard": "estimate", "Units": units},
        "SE": {"Description": "standard error of the estimate from the fitted model", "Standard": "se"},
        "df": {"Description": "error degrees of freedom (Satterthwaite; Inf when emmeans fell back "
                              "to asymptotic z)", "Standard": "df"},
        "df_num": {"Description": "numerator degrees of freedom (F rows; 1 otherwise)"},
        "stat": {"Description": "the test statistic", "Standard": "stat",
                 "StatName": "per row, as stat_type says: t, F, or the TOST t"},
        "stat_type": {"Description": "the statistic's kind",
                      "Levels": {"t": "t (or z)", "F": "F", "tost": "TOST t"}},
        "p_value": {"Description": "the test's p before correction", "Standard": "p_uncorrected"},
        "q_value": {"Description": "p corrected within the row's family (fdr_family), by its method",
                    "Standard": "p_value",
                    "PKind": next(iter(kinds)) if len(kinds) == 1 else "per_family"},
        "fdr_family": {"Description": "the row's correction family: scope, method, key, members "
                                      "and a hash of the member set", "Standard": "correction_family"},
        "significant": {"Description": "q_value < 0.05", "Standard": "significant",
                        "Alpha": SIGNIFICANCE_ALPHA},
        "sig_label": {"Description": "significance stars for figures"},
        "estimate_lcl": {"Description": "lower bound of the estimate's 95% interval"},
        "estimate_ucl": {"Description": "upper bound of the estimate's 95% interval"},
        "effect_size": {"Description": "the effect size; its measure is effect_size_type",
                        "Standard": "effect_size", "EffectMeasure": "per_row"},
        "effect_size_type": {"Description": "the row's effect measure", "Standard": "effect_measure",
                             "Levels": {"hedges_g": "the estimate divided by the model's residual SD, "
                                                    "uncorrected (an uncorrected standardized "
                                                    "difference, despite the name)",
                                        "omega2_partial": "partial omega squared of the omnibus F",
                                        "std_beta": "the standardized slope"}},
        "effect_size_lcl": {"Description": "lower bound of the effect's 95% interval",
                            "Standard": "effect_ci_low", "CILevel": 0.95},
        "effect_size_ucl": {"Description": "upper bound of the effect's 95% interval",
                            "Standard": "effect_ci_high", "CILevel": 0.95},
        "n": {"Description": "subjects in the fitted model", "Standard": "n"},
        "n_a": {"Description": "subjects in group_a (two-group contrasts)", "Standard": "n_a"},
        "n_b": {"Description": "subjects in group_b (two-group contrasts)", "Standard": "n_b"},
        "group_a": {"Description": "the contrast's positively weighted group, higher when the "
                                  "estimate is positive (two-group contrasts)", "Standard": "group_a"},
        "group_b": {"Description": "the contrast's negatively weighted group", "Standard": "group_b"},
        (table.measure_col or spec.measure_col): {
            "Description": "the module's name for the measure (measure gives the specification's)"},
        **{col: {"Description": desc, "Standard": "facet", "FacetName": name}
           for col, name, desc in spec.extra_facets},
        **dict(spec.extra_columns),
        "measure": {"Description": "the measure, by its name in the specification's vocabulary",
                    "Standard": "measure"},
        "margin_used": {"Description": "the equivalence margin, in the estimate's units",
                        "Standard": "margin", "Units": units},
        "equivalent": {"Description": "the TOST concludes equivalence within the margin",
                       "Standard": "equivalent"},
        "mde": {"Description": "minimum detectable effect, in the estimate's units: "
                               "(t(1 - alpha/2, df) + t(power, df)) * SE", "Standard": "mde",
                "Power": 0.8, "Alpha": 0.05},
        "mde_percent": {"Description": "mde as a percent of the reference mean", "Standard": "mde_percent",
                        "Units": "%",
                        "PercentOf": "the design.reference group's model mean in the same cell, or the "
                                     "contrast's negatively weighted side when it leaves the reference "
                                     "out; blank when that mean is within one residual SD of zero. For a "
                                     "log-scale measure (the aperiodic offset, dB power) it is a percent "
                                     "of a log value and means little (docs/KNOWN_ISSUES.md 3)"},
        "mde_note": {"Description": "empty when mde is given; otherwise why not", "Standard": "mde_note"},
        "mde_power": {"Description": "the power the mde is computed at"},
        "mde_alpha": {"Description": "the two-sided alpha the mde is computed at (nominal)"},
    }
    missing = [c for c in cols if c not in docs]
    if missing:
        raise KeyError(f"no description for columns {missing}")
    return {c: docs[c] for c in cols}


# ------------------------------------------- contrasts and variants (0.4.1) ---
#: What varies between paradigms that share a display group, unless the study config says
#: (``variant_axis``): in a source-analytics study they are the same recordings localized
#: different ways.
DEFAULT_VARIANT_AXIS = "reconstruction"


def contrasts(config, rows: list[dict[str, str]]) -> list[dict]:
    """The declared hypotheses the tables test (spec 0.4.1 §3.6), in the study config's order.

    A hypothesis's study ``role`` (phenotype, rescue, normalization) is the spec's ``tier``.
    """
    spec = getattr(config, "design_spec", None)
    tested = {r.get("hypothesis") for r in rows}
    out = []
    for h in getattr(spec, "hypotheses", None) or []:
        if h.name not in tested:
            continue
        c: dict[str, Any] = {"id": h.name, "kind": h.kind}
        if h.label:
            c["label"] = h.label
        ends = h.pairwise_endpoints()
        if ends:
            c["group_a"], c["group_b"] = ends
        if h.role:
            c["tier"] = h.role
        out.append(c)
    return out


def variant(config, analysis_id: str, paradigm: str | None, profile: str | None) -> dict | None:
    """This paradigm as one variant of the analysis (spec 0.4.1 §3.7), or None.

    Paradigms that share a ``display.group`` in the study config are variants of each other
    (the same recordings localized several ways); a lone paradigm is not. Only the default
    profile is a variant: a profile's run is its own analysis.
    """
    raw = getattr(config, "raw", None) or {}
    display = raw.get("display") if isinstance(raw.get("display"), dict) else {}
    group = display.get("group")
    if not paradigm or profile or not group:
        return None
    siblings = [n for n, p in (raw.get("paradigms") or {}).items()
                if isinstance(p, dict) and (p.get("display") or {}).get("group") == group]
    if len(siblings) < 2 or not analysis_id.endswith("/" + paradigm):
        return None
    v = {"of": analysis_id[: -len(paradigm) - 1], "axis": str(raw.get("variant_axis") or DEFAULT_VARIANT_AXIS),
         "value": analysis_id.rsplit("/", 1)[1]}
    if display.get("label"):
        v["label"] = str(display["label"])
    return v


# ------------------------------------------------------------------- writing ---
def _r_version() -> str | None:
    try:
        out = subprocess.run(["Rscript", "-e", "cat(paste(R.version$major, R.version$minor, sep='.'))"],
                             capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _test_kinds(rows: list[dict[str, str]]) -> list[str]:
    kinds = set()
    for r in rows:
        k = r.get("kind")
        if k == "contrast":
            kinds.add("two_group" if r.get("group_a") and r.get("group_b") else "custom")
        elif k in ("equivalence", "regression"):
            kinds.add(k)
        elif k == "omnibus":
            kinds.add("custom")
    order = ["two_group", "regression", "equivalence", "custom"]
    return [k for k in order if k in kinds]


def write(analysis, record: dict | None, started: str) -> Any:
    """Describe ``analysis``'s tables by the specification; the checker's report, or None.

    ``analysis`` is a BaseAnalysis whose class declares ``RESULTS_SPEC``;
    ``record`` is the provenance record this package just wrote beside the
    tables; ``started`` is when the run began (ISO 8601). Never raises.
    """
    spec: ResultsSpec | None = getattr(analysis, "RESULTS_SPEC", None)
    if spec is None:
        return None
    if lumiere is None:
        logger.info("Results specification not written for %s: %s", analysis.name, INSTALL_HINT)
        return None
    try:
        return _write(analysis, spec, record or {}, started)
    except Exception as exc:  # noqa: BLE001 - a description must never fail a finished run
        logger.warning("Results specification not written for %s: %s", analysis.name, exc)
        return None


def _write(analysis, spec: ResultsSpec, record: dict, started: str) -> Any:
    from .core import ANALYSIS_METADATA

    folder = Path(analysis.tbl_dir)
    meta = ANALYSIS_METADATA.get(analysis.name, {})
    level = meta.get("level", "roi")
    atype = ANALYSIS_TYPE_OF_LEVEL.get(level, "other")
    name = spec.name or re.sub(rf"^{level}_", "", analysis.name)
    paradigm = analysis.config.paradigm_name
    profile = analysis.config.profile_name

    tables, all_rows, fams = [], [], {}
    found = [(t, folder / t.filename(analysis.name)) for t in spec.tables]
    found = [(t, p) for t, p in found if p.exists()]
    if not found:
        logger.info("Results specification: %s has no hypothesis tables to describe", analysis.name)
        return None
    prepared = []
    for t, path in found:
        cols, rows = prepare_table(path, spec.measures, t.measure_col or spec.measure_col)
        prepared.append((t, path, cols, rows))
        fams.update(families(rows))
        all_rows += rows
    for t, path, cols, rows in prepared:
        lumiere.write_columns(path, columns(cols, rows, t, spec, fams))
        tables.append({"path": path.name, "role": t.role, "rows": t.rows, "description": t.description,
                       "n_rows": len(rows), **({"headline": True} if t.headline else {})})

    subjects = record.get("subjects") or {}
    seen = {r.get(t.measure_col or spec.measure_col) for t, _p, _c, rows in prepared for r in rows}
    measures = list(dict.fromkeys(m for dv, m in spec.measures.items() if dv in seen))
    run_id = _default_run_id()
    analysis_json: dict[str, Any] = {
        "id": _analysis_id("eeg", atype, name, *(p for p in (paradigm, profile) if p)),
        "title": f"{meta.get('description', analysis.name)} ({paradigm})" if paradigm
        else meta.get("description", analysis.name),
        "description": meta.get("about") or meta.get("description") or analysis.name,
        "analysis_type": atype, "modality": "eeg", "domain": spec.domain,
        "measures": measures, "role": spec.role,
        "design": {"n": int(subjects.get("n") or 0) or max((int(r["n"]) for r in all_rows if r.get("n")),
                                                           default=0),
                   "groups": {g: int(n) for g, n in (subjects.get("groups") or {}).items()},
                   "test_kinds": _test_kinds(all_rows)},
        "inference": ({"method": spec.method, "correction": spec.correction}
                      if spec.correction and not fams else
                      {"method": spec.method,
                       "families": [{k: v for k, v in f.items() if not k.startswith("_")}
                                    for f in fams.values()]}),
        "effect": spec.effect or {"measure": "per_row", "ci_level": 0.95,
                                  "definition": "per row, as effect_size_type says: the estimate over the "
                                                "model's residual SD for contrasts, partial omega squared "
                                                "for omnibus rows, the standardized slope for regressions"},
        "tables": tables,
        "caveats": [*spec.caveats, *COMMON_CAVEATS],
        "run": {"id": run_id, "label": f"{analysis.name} on {paradigm}" if paradigm else analysis.name},
    }
    declared = contrasts(analysis.config, all_rows)
    if declared:
        analysis_json["contrasts"] = declared
    var = variant(analysis.config, analysis_json["id"], paradigm, profile)
    if var:
        analysis_json["variant"] = var
    task = TASK_OF_CATEGORY.get(meta.get("category", ""))
    if task:
        analysis_json["task"] = task
    atlas = (record.get("localization") or {}).get("atlas") or \
        (analysis.config.raw.get("pipeline") or {}).get("atlas")
    if atlas and atype in ("roi", "vertex"):
        analysis_json["atlas"] = {"name": str(atlas)}

    core = record.get("source_analytics") or {}
    version, describe = str(core.get("version") or "unknown"), core.get("git_describe")
    generated_by = [{"Name": "source-analytics", "Version": version,
                     **({"Description": f"git {describe}"} if describe and describe != version else {})}]
    r_version = _r_version()
    if r_version:
        generated_by.append({"Name": "R", "Version": r_version})
    inputs = []
    if spec.input_file:
        inputs.append({"path": str(Path(analysis.output_dir) / "data" / spec.input_file),
                       "role": "per-subject values the statistics were fitted to"})
    spec_prov = lumiere.provenance(
        generated_by, status="completed", start=started,
        end=datetime.now(timezone.utc).isoformat(timespec="seconds"), run_id=run_id, inputs=inputs,
        settings={"paradigm": paradigm, "profile": profile, "analysis": analysis.name})
    merged = {**record, **spec_prov}                # this package's fields stay for source-lightbox
    report = lumiere.write_analysis(folder, analysis_json, merged, strict=False)
    if report.ok:
        logger.info("Results specification: %s conforms (%s; %d warning(s))", folder, analysis_json["id"],
                    len(report.warnings))
    else:
        logger.warning("Results specification: %s does not conform: %s", folder, "; ".join(report.errors))
    for w in report.warnings:
        logger.info("  results spec warning: %s", w)
    return report

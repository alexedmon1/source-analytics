"""The Bayesian engine's catalogue: every model, prior and decision rule, with an id and a version.

The engine's priors and rules are code in ``R/bayes/`` (``model.R``'s constants and ``bayes_priors()``,
``readout.R``'s decision rules, ``provenance.R``'s gate thresholds). This module declares each of them once, with a
stable id, a version, its values and where it holds, so a result can name what produced it and a reader can find
every assumption (bayes-sl ``reports/2026-11-09_field_method_library_plan.md``, "Identification and
documentation"; source-localization keeps the same catalogue for its location posteriors).

Rules:

- ``id`` never changes once released; ``version`` goes up whenever what the entry computes changes (a value, a
  formula, a rule), not for docs or refactors. ``BAYES_ENGINE_VERSION`` still versions the engine as a whole.
- ``r_prior`` ties a prior to the row ``bayes_priors()`` writes for it, and ``r_constants`` to the R constants that
  hold its values; ``tests/bayes/test_catalogue.py`` builds the real prior tables in R and checks both, so this
  catalogue cannot drift from the code.
- ``docs/methods/BAYES_CATALOGUE.md`` is generated from it
  (``python -m source_analytics.bayes.catalogue > docs/methods/BAYES_CATALOGUE.md``) and linked from the README and
  the docs site; the test fails when the page is stale.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple


class ModelSpec(NamedTuple):
    """One catalogued model, prior or rule: what it is, its values, where it holds."""

    id: str
    version: int
    summary: str
    #: The model, prior or rule, stated in full.
    model: str
    #: Its values, as text (each a number with its unit).
    values: Tuple[str, ...] = ()
    #: Ids of the priors a model rests on, or the rules a decision uses.
    rests_on: Tuple[str, ...] = ()
    validity: str = ""
    #: ``R/bayes/<file>.R:<name>`` or ``source_analytics.<module>:<name>``.
    implementation: str = ""
    tests: str = ""
    reference: str = ""
    caveat: str = ""
    #: The prior row ``bayes_priors()`` writes: (class, group, dpar, coef prefix, prior string, design), with design
    #: one of 'roi' (ROI, subject term and a covariate) or 'noroi' (no ROI, one row per subject and band).
    r_prior: Optional[Tuple[str, str, str, str, str, str]] = None
    #: R constants holding its values, name -> value (numbers, or strings for ids).
    r_constants: Tuple[Tuple[str, object], ...] = ()


_W_LEVEL = 'normal(0, 0.707107)'

MODELS: Dict[str, ModelSpec] = {
    'model.group_levels': ModelSpec(
        'model.group_levels', 1,
        'hierarchical group model over bands and ROIs',
        'The outcome is standardised per band by the pooled within-group between-subject SD within ROI x band (so '
        'effects are in the unit of a per-cell Hedges g): z = (y - band mean) / that SD. Then z ~ Normal(mu, '
        'sigma_band) with mu = one level per band x group + each band\'s group effect deviating by ROI (partial '
        'pooling) + a level per ROI x band + a subject level per band; log sigma per band. A set-I entry re-expresses '
        'that group\'s level as reference level + dx.',
        rests_on=('prior.w.group_level', 'prior.w.covariate', 'prior.w.roi_deviation_sd', 'prior.w.roi_band_sd',
                  'prior.w.subject_sd', 'prior.w.log_sigma', 'prior.i.contrast_entry'),
        validity='Gaussian outcomes per band. Each study design must pass the gate (gate.rules) before a real fit.',
        implementation='R/bayes/model.R:bayes_formula',
        tests='tests/bayes/test_bayes_job.py, tests/bayes/test_e5_slow.py',
        reference='docs/methods/BAYES_ENGINE.md section 3',
        r_constants=(('BAYES_CODING', 'group-levels-2026-10-19'),)),
    'model.group_levels_noroi': ModelSpec(
        'model.group_levels_noroi', 1,
        'the same model with no ROI: one value per subject, e.g. along a located field',
        'model.group_levels with spatial_col = NULL: the two ROI terms are dropped; the subject term is kept only '
        'when some subject has more than one row in a band (with one row it would be the residual under another '
        'name). The readout has one spatial cell, "all".',
        rests_on=('prior.w.group_level', 'prior.w.covariate', 'prior.w.log_sigma_noroi', 'prior.i.contrast_entry'),
        validity='As model.group_levels. Reachable from R only until the config path exists (field plan, phase F6).',
        implementation='R/bayes/model.R:bayes_prepare',
        tests='tests/bayes/test_no_roi.py',
        reference='docs/methods/BAYES_ENGINE.md section 3'),
}

PRIORS: Dict[str, ModelSpec] = {
    'prior.w.group_level': ModelSpec(
        'prior.w.group_level', 1,
        'set W: each group level per band',
        'Each band x group level ~ Normal(0, 1/sqrt(2)), independently, so every pairwise contrast is a priori '
        'Normal(0, 1) and no group is privileged.',
        values=('mean 0', 'SD 1/sqrt(2) = 0.707 between-subject SD (contrast SD 1)'),
        reference='docs/methods/BAYES_ENGINE.md section 5; group-level coding the Conductor, 2026-10-19',
        implementation='R/bayes/model.R:bayes_priors',
        r_prior=('b', '', '', 'lv_', _W_LEVEL, 'roi'),
        r_constants=(('BAYES_W_EFFECT_SD', 1.0), ('BAYES_W_LEVEL_SD', 0.7071067811865476))),
    'prior.w.covariate': ModelSpec(
        'prior.w.covariate', 1,
        'set W: covariate slopes',
        'Each centred covariate\'s slope ~ Normal(0, 1) in SD units of the outcome per unit of the covariate.',
        values=('mean 0', 'SD 1'),
        implementation='R/bayes/model.R:bayes_priors',
        r_prior=('b', '', '', '', 'normal(0, 1)', 'roi')),
    'prior.w.roi_deviation_sd': ModelSpec(
        'prior.w.roi_deviation_sd', 1,
        'set W: how much a group effect varies by ROI',
        'SD of each band\'s group-effect deviations across ROIs ~ half-Normal(0, 0.25).',
        values=('half-Normal scale 0.25 between-subject SD',),
        implementation='R/bayes/model.R:bayes_priors',
        r_prior=('sd', 'roi', '', '', 'normal(0, 0.25)', 'roi')),
    'prior.w.roi_band_sd': ModelSpec(
        'prior.w.roi_band_sd', 1,
        'set W: how much ROI means vary within a band',
        'SD of the ROI x band levels ~ half-Normal(0, 3).',
        values=('half-Normal scale 3 between-subject SD',),
        reference='raised from 0.5 (the Conductor, 2026-10-15): FORGE source-power ROI means spread 0.9-3.0 SD',
        implementation='R/bayes/model.R:bayes_priors',
        r_prior=('sd', 'roi:band', '', '', 'normal(0, 3)', 'roi'),
        r_constants=(('BAYES_SD_ROI_BAND', 3.0),)),
    'prior.w.subject_sd': ModelSpec(
        'prior.w.subject_sd', 1,
        'set W: subject levels',
        'SD of the subject level per band ~ half-Normal(0, 0.5).',
        values=('half-Normal scale 0.5 between-subject SD',),
        implementation='R/bayes/model.R:bayes_priors',
        r_prior=('sd', 'subject', '', '', 'normal(0, 0.5)', 'roi')),
    'prior.w.log_sigma': ModelSpec(
        'prior.w.log_sigma', 1,
        'set W: residual SD per band, with ROI or subject terms',
        'log sigma per band ~ Normal(log 0.5, 0.5): the residual is the part of the standardised spread that the ROI '
        'and subject terms do not take.',
        values=('median sigma 0.5 between-subject SD', 'SD 0.5 on the log scale'),
        implementation='R/bayes/model.R:bayes_log_sigma_mean',
        r_prior=('b', '', 'sigma', '', 'normal(-0.693147, 0.5)', 'roi'),
        r_constants=(('BAYES_LOG_SIGMA', -0.6931471805599453),)),
    'prior.w.log_sigma_noroi': ModelSpec(
        'prior.w.log_sigma_noroi', 1,
        'set W: residual SD per band, no ROI and no subject term',
        'log sigma per band ~ Normal(log 1, 0.5). With one row per subject and band, the standardisation makes the '
        'within-group SD 1, so the residual is all of it.',
        values=('median sigma 1 between-subject SD', 'SD 0.5 on the log scale'),
        reference='engine 2026.11.09.1 (the Conductor, 2026-11-09): log 0.5 put 1 at the prior\'s 92nd percentile',
        implementation='R/bayes/model.R:bayes_log_sigma_mean',
        r_prior=('b', '', 'sigma', '', 'normal(0, 0.5)', 'noroi'),
        r_constants=(('BAYES_LOG_SIGMA_NOROI', 0.0),)),
    'prior.i.contrast_entry': ModelSpec(
        'prior.i.contrast_entry', 1,
        'set I: an informative prior on one contrast against the reference',
        'An entry x_<band>_<group> = Normal(mean, sd) on (group - reference) in between-subject SD. The group\'s level '
        'becomes reference level + dx with dx ~ the entry, so that contrast\'s prior is exactly the entry and no other '
        'group is tied to the reference. Terms without an entry stay as set W.',
        values=('mean and SD per entry, from the study\'s informative-prior file',),
        validity='Only from a logged literature search: each entry carries its source (priors.py requires it). The '
                 'file is identified in the run manifest by its path and sha256.',
        implementation='R/bayes/model.R:bayes_prepare',
        tests='tests/bayes/test_e2_config.py',
        reference='docs/methods/BAYES_ENGINE.md section 5'),
}

RULES: Dict[str, ModelSpec] = {
    'rule.contrast_decision': ModelSpec(
        'rule.contrast_decision', 1,
        'effect, null or undecided, from the HDI against the ROPE',
        '95% highest-density interval (HDI) of the contrast. "effect" when it lies wholly outside the ROPE, "null" '
        'when wholly inside, else "undecided". Reported beside it: median, P(direction), P(inside ROPE), and BF10 by '
        'Savage-Dickey (prior density at 0 over posterior density at 0, from draws).',
        values=('HDI 95%', 'ROPE +-0.1 between-subject SD by default (a study may set its own)'),
        implementation='R/bayes/readout.R:.decide',
        tests='tests/bayes/test_e5_slow.py',
        reference='docs/methods/BAYES_ENGINE.md section 4',
        caveat='BF10 from draws is unreliable far in the tails; read very large values only as decisive.',
        r_constants=(('BAYES_HDI', 0.95), ('BAYES_ROPE_SD', 0.1))),
    'rule.equivalence_decision': ModelSpec(
        'rule.equivalence_decision', 1,
        'equivalent or not shown, from the HDI against a margin',
        '"equivalent" when the 95% HDI lies inside +-margin, the margin per draw being margin x the posterior residual '
        'SD (mode: sd); else "not_shown".',
        values=('HDI 95%', 'margin set by the hypothesis'),
        implementation='R/bayes/readout.R:.decide_equivalence',
        tests='tests/bayes/test_e5_gate.py, tests/bayes/test_e5_slow.py',
        reference='docs/methods/BAYES_ENGINE.md section 4'),
    'rule.convergence': ModelSpec(
        'rule.convergence', 1,
        'what must converge before a result is read',
        'R-hat (split by chain) of every decided estimand and of every scale parameter (SDs, log sigma) below 1.05. '
        'The maximum over every parameter is reported, not gated.',
        values=('R-hat < 1.05',),
        implementation='R/bayes/readout.R:.estimand_conv',
        reference='the Conductor, 2026-10-21',
        r_constants=(('BAYES_RHAT_MAX', 1.05),)),
    'gate.rules': ModelSpec(
        'gate.rules', 1,
        'the simulation checks a design must pass before a real fit',
        'At the study\'s own design and prior set: the real outcome\'s spread (SD and MAD per band) within the 5th-95th '
        'percentile of the prior predictive; simulation-based calibration rank-uniformity chi-square p above 0.01 / '
        'number of estimands and 95% HDI coverage at least 0.85; with every group effect 0, false "effect" claims at '
        'most 5% and direction claims at most 10% per estimand; recovery coverage at least 0.85 at the real fit\'s '
        'settings; false "equivalent" at most 5% at the margin; every configured replication present.',
        values=('prior predictive 5-95th percentile', 'SBC alpha 0.01, Bonferroni over estimands',
                'coverage >= 0.85', 'null effect claims <= 0.05', 'null direction claims <= 0.10',
                'recovery coverage >= 0.85', 'R-hat < 1.05', 'false equivalent <= 0.05',
                'default replications: SBC 100, null 100, recovery 10, equivalence 100'),
        rests_on=('rule.contrast_decision', 'rule.equivalence_decision', 'rule.convergence'),
        implementation='R/bayes/gate.R:run_reps',
        tests='tests/bayes/test_e3_provenance.py, tests/bayes/test_e5_gate.py',
        reference='docs/methods/BAYES_ENGINE.md section 6',
        r_constants=(('BAYES_GATE_RULES$prior_pred_lo', 0.05), ('BAYES_GATE_RULES$prior_pred_hi', 0.95),
                     ('BAYES_GATE_RULES$sbc_alpha', 0.01), ('BAYES_GATE_RULES$sbc_min_coverage', 0.85),
                     ('BAYES_GATE_RULES$null_max_claim_effect', 0.05), ('BAYES_GATE_RULES$null_max_claim_dir', 0.10),
                     ('BAYES_GATE_RULES$recovery_min_coverage', 0.85), ('BAYES_GATE_RULES$equiv_max_false', 0.05),
                     ('BAYES_GATE_DEFAULTS$sbc', 100), ('BAYES_GATE_DEFAULTS$null', 100),
                     ('BAYES_GATE_DEFAULTS$recovery', 10), ('BAYES_GATE_DEFAULTS$equivalence', 100))),
}

#: The catalogue's sections, in page order.
CATALOGUE: Dict[str, Dict[str, ModelSpec]] = {
    'Models': MODELS,
    'Priors (set W, weakly informative; set I, informative)': PRIORS,
    'Decision rules and the gate': RULES,
}


def catalogue_entries() -> Dict[str, ModelSpec]:
    """Every catalogued entry by id, across sections."""
    return {k: v for section in CATALOGUE.values() for k, v in section.items()}


def catalogue_markdown(engine_version: str) -> str:
    """The catalogue as docs/methods/BAYES_CATALOGUE.md, stamped with the engine version it describes."""
    out = ['# Bayesian engine catalogue', '',
           'Every model, prior and decision rule of the Bayesian group engine, each with a stable **id** and a '
           f'**version**. Engine version described: `{engine_version}`. Units: between-subject SD of the outcome '
           '(the unit of a per-cell Hedges g) unless stated.', '',
           '*Generated from `source_analytics/bayes/catalogue.py` (`python -m source_analytics.bayes.catalogue > '
           'docs/methods/BAYES_CATALOGUE.md`); do not edit by hand. `tests/bayes/test_catalogue.py` checks it '
           'against the R code and checks that it is current.*', '']
    entries = catalogue_entries()
    for title, section in CATALOGUE.items():
        out += [f'## {title}', '', '| id | version | what it is |', '|---|---|---|']
        out += [f'| `{m.id}` | {m.version} | {m.summary} |' for m in section.values()]
        out.append('')
        for m in section.values():
            out += [f'### `{m.id}` (version {m.version})', '', f'{m.summary[0].upper()}{m.summary[1:]}.', '',
                    f'- **Definition:** {m.model}']
            if m.values:
                out.append('- **Values:** ' + '; '.join(m.values))
            if m.rests_on:
                out.append('- **Rests on:** ' + ', '.join(f'`{r}`' for r in m.rests_on if r in entries))
            for field, name in (('validity', 'Validity'), ('caveat', 'Caveat'), ('implementation', 'Implementation'),
                                ('tests', 'Tests'), ('reference', 'Reference')):
                text = getattr(m, field)
                if text:
                    out.append(f'- **{name}:** ' + (f'`{text}`' if field in ('implementation', 'tests') else text))
            out.append('')
    return '\n'.join(out).rstrip('\n') + '\n'


def engine_version() -> str:
    """BAYES_ENGINE_VERSION as R/bayes/provenance.R declares it."""
    import re

    from source_analytics.bayes import r_dir
    text = (r_dir() / 'bayes' / 'provenance.R').read_text()
    return re.search(r'BAYES_ENGINE_VERSION <- "([^"]+)"', text).group(1)


if __name__ == '__main__':
    print(catalogue_markdown(engine_version()), end='')

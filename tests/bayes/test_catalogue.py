"""The Bayesian engine's catalogue must match the R code and stay documented.

``source_analytics.bayes.catalogue`` declares every model, prior and rule of the engine with an id and a version.
These tests build the engine's real prior tables in R (a design with ROI, subject term and a covariate; a design with
no ROI) and check every catalogued prior row and constant against them, so the catalogue cannot drift from the code.
They also check ids, the R and Python implementations named, the generated page and its links.
"""
from __future__ import annotations

import importlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from source_analytics.bayes import r_dir
from source_analytics.bayes import catalogue as C

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'docs' / 'methods' / 'BAYES_CATALOGUE.md'
ENTRIES = C.catalogue_entries()
ID = re.compile(r'^(model|prior\.(w|i)|rule|gate)\.[a-z0-9_]+$')
needs_r = pytest.mark.skipif(shutil.which('Rscript') is None, reason='Rscript not installed')

_R = f"""
suppressMessages(source('{r_dir() / "bayes" / "engine.R"}'))
spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'A', levels = list('A', 'B'),
                                             covariates = list('age')),
                               hypotheses = list(list(name = 'effect', kind = 'contrast', weights = list(B = 1, A = -1)))))
set.seed(3)
one <- data.frame(subject = paste0('m', 1:24), group = rep(c('A', 'B'), each = 12), age = rnorm(24))
one$value <- rnorm(24)
roi <- merge(one[, c('subject', 'group', 'age')], data.frame(roi = c('r1', 'r2', 'r3')))
roi$value <- rnorm(nrow(roi))
tab <- function(p, design) {{
  pr <- as.data.frame(bayes_priors(p))
  lapply(seq_len(nrow(pr)), function(i) list(design = design, prior = pr$prior[i], class = pr$class[i],
                                             group = pr$group[i], dpar = pr$dpar[i], coef = pr$coef[i]))
}}
rows <- c(tab(bayes_prepare(roi, spec, 'value', 'roi', NULL), 'roi'),
          tab(bayes_prepare(one, spec, 'value', NULL, NULL), 'noroi'))
inf <- bayes_prepare(roi, spec, 'value', 'roi', NULL, set = 'I', informative = list(x_all_B = c(-0.3, 0.4)))
consts <- list(BAYES_CODING = BAYES_CODING, BAYES_W_EFFECT_SD = BAYES_W_EFFECT_SD, BAYES_W_LEVEL_SD = BAYES_W_LEVEL_SD,
               BAYES_SD_ROI_BAND = BAYES_SD_ROI_BAND, BAYES_LOG_SIGMA = BAYES_LOG_SIGMA,
               BAYES_LOG_SIGMA_NOROI = BAYES_LOG_SIGMA_NOROI, BAYES_HDI = BAYES_HDI, BAYES_ROPE_SD = BAYES_ROPE_SD,
               BAYES_RHAT_MAX = BAYES_RHAT_MAX)
for (k in names(BAYES_GATE_RULES)) consts[[paste0('BAYES_GATE_RULES$', k)]] <- BAYES_GATE_RULES[[k]]
for (k in c('sbc', 'null', 'recovery', 'equivalence')) consts[[paste0('BAYES_GATE_DEFAULTS$', k)]] <- BAYES_GATE_DEFAULTS[[k]]
cat(jsonlite::toJSON(list(rows = rows, consts = consts, inf = tab(inf, 'roi')), auto_unbox = TRUE, digits = NA))
"""


@pytest.fixture(scope='module')
def r_side():
    r = subprocess.run(['Rscript', '-e', _R], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_ids_unique_well_formed_and_versioned():
    seen = []
    for section in C.CATALOGUE.values():
        for key, m in section.items():
            assert key == m.id and ID.match(m.id), m.id
            assert isinstance(m.version, int) and m.version >= 1, m.id
            assert m.summary and m.model and m.implementation, m.id
            seen.append(m.id)
    assert len(seen) == len(set(seen))


def test_rests_on_names_catalogued_entries():
    for m in ENTRIES.values():
        for r in m.rests_on:
            assert r in ENTRIES, f'{m.id} rests on {r!r}, not catalogued'


@pytest.mark.parametrize('m', list(ENTRIES.values()), ids=lambda m: m.id)
def test_implementation_exists(m):
    where, _, name = m.implementation.partition(':')
    if where.startswith('R/'):
        text = (r_dir().parent / where).read_text() if (r_dir().parent / where).exists() else (r_dir() / where[2:]).read_text()
        assert re.search(rf'(^|\n){re.escape(name)} *<- *function', text), m.implementation
    else:
        obj = importlib.import_module(where)
        for part in name.split('.'):
            obj = getattr(obj, part)


@pytest.mark.parametrize('m', list(ENTRIES.values()), ids=lambda m: m.id)
def test_named_test_files_exist(m):
    for f in filter(None, (t.strip() for t in m.tests.split(','))):
        assert (ROOT / f).exists(), f'{m.id}: {f}'


@needs_r
@pytest.mark.parametrize('m', [m for m in ENTRIES.values() if m.r_prior], ids=lambda m: m.id)
def test_prior_row_matches_bayes_priors(m, r_side):
    cls, group, dpar, coef, prior, design = m.r_prior
    hits = [r for r in r_side['rows'] if r['design'] == design and r['class'] == cls and r['group'] == group
            and r['dpar'] == dpar and (r['coef'].startswith(coef) if coef else r['coef'] == '')]
    assert hits, f'{m.id}: no {cls}/{group}/{dpar}/{coef!r} row in the {design} design'
    assert {h['prior'] for h in hits} == {prior}, (m.id, {h['prior'] for h in hits})


@needs_r
@pytest.mark.parametrize('m', [m for m in ENTRIES.values() if m.r_constants], ids=lambda m: m.id)
def test_constants_match_r(m, r_side):
    for name, value in m.r_constants:
        got = r_side['consts'][name]
        assert (got == value) if isinstance(value, str) else got == pytest.approx(value, abs=1e-12), (m.id, name, got)


@needs_r
def test_every_prior_row_is_catalogued(r_side):
    """No prior reaches Stan without a catalogue entry (set-I dx rows are prior.i.contrast_entry)."""
    known = {(m.r_prior[0], m.r_prior[1], m.r_prior[2], m.r_prior[3], m.r_prior[5]) for m in ENTRIES.values()
             if m.r_prior}
    for r in r_side['rows']:
        coef = 'lv_' if r['coef'].startswith('lv_') else r['coef']
        assert (r['class'], r['group'], r['dpar'], coef, r['design']) in known or (
            r['design'] == 'noroi' and (r['class'], r['group'], r['dpar'], coef, 'roi') in known), r


@needs_r
def test_set_i_entry_becomes_its_contrast_prior(r_side):
    dx = [r for r in r_side['inf'] if r['coef'].startswith('dx_')]
    assert [r['prior'] for r in dx] == ['normal(-0.3, 0.4)']


def test_generated_page_is_current():
    assert PAGE.exists(), 'run: python -m source_analytics.bayes.catalogue > docs/methods/BAYES_CATALOGUE.md'
    assert PAGE.read_text() == C.catalogue_markdown(C.engine_version()), (
        'docs/methods/BAYES_CATALOGUE.md is stale; run: '
        'python -m source_analytics.bayes.catalogue > docs/methods/BAYES_CATALOGUE.md')


def test_page_is_linked_from_readme_and_docs_site():
    assert 'docs/methods/BAYES_CATALOGUE.md' in (ROOT / 'README.md').read_text()
    assert 'methods/BAYES_CATALOGUE.md' in repr(yaml.safe_load((ROOT / 'mkdocs.yml').read_text())['nav'])

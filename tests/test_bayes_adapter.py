"""Python side of the Bayesian hypothesis adapter (hypothesis/bayes.py): job building only, no R or Stan."""
from __future__ import annotations

import json
import subprocess

import pytest

from source_analytics.config import StudyConfig
from source_analytics.hypothesis.bayes import BAYES_TABLES, R_DIR, build_job, design_payload

YAML = """
name: t
design: {factor: group, reference: WT, levels: [WT, KO, KO_D]}
hypotheses:
  - {name: omni, kind: omnibus}
  - {name: disease, kind: contrast, weights: {KO: 1, WT: -1}}
  - {name: norm, kind: equivalence, weights: {KO_D: 1, WT: -1}, margin: {mode: sd, value: 0.25}}
paths: {analytics: ./analytics, results: ./results}
bayes: {rope: {mode: sd, value: 0.2}}
paradigms:
  arm: {data_dir: ./d, analyses: {roi_psd: {}, roi_aperiodic: {}}}
"""


@pytest.fixture
def cfg(tmp_path):
    p = tmp_path / "study.yaml"
    p.write_text(YAML)
    return StudyConfig.from_yaml(p)


def test_paths_follow_module_layout(cfg):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    job = build_job(a, "roi_psd", "absolute")
    assert job.data_csv.endswith("arm/roi_psd/data/band_power.csv")
    assert job.table_out.endswith("results/tables/arm/roi_psd/roi_psd_hypotheses_bayes.csv")
    assert job.work_dir.endswith("arm/roi_psd/data/bayes")
    assert job.band_col == "band" and job.spatial_col == "roi"


def test_out_dir_redirects_table_and_fits(cfg, tmp_path):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    job = build_job(a, "roi_psd", "absolute", out_dir=tmp_path / "elsewhere")
    assert job.table_out == str(tmp_path / "elsewhere" / "roi_psd_hypotheses_bayes.csv")
    assert job.work_dir == str(tmp_path / "elsewhere" / "bayes")


def test_aperiodic_has_no_band(cfg):
    a = cfg.for_paradigm_analysis("arm", "roi_aperiodic")
    job = build_job(a, "roi_aperiodic", "exponent")
    assert job.band_col is None and job.data_csv.endswith("aperiodic_params.csv")


def test_design_payload_round_trips(cfg):
    design, hyps = design_payload(cfg)
    assert design == {"factor": "group", "reference": "WT", "levels": ["WT", "KO", "KO_D"], "covariates": []}
    by = {h["name"]: h for h in hyps}
    assert by["disease"]["weights"] == {"KO": 1.0, "WT": -1.0}
    assert by["norm"]["margin"] == {"mode": "sd", "value": 0.25}
    assert "weights" not in by["omni"]                       # None fields are dropped for R's parser
    json.dumps(hyps)                                          # serialisable


def test_hypothesis_filter_and_unknown(cfg):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    job = build_job(a, "roi_psd", "absolute", hypotheses=["disease"])
    assert [h["name"] for h in job.hypotheses] == ["disease"]
    with pytest.raises(ValueError, match="unknown hypothesis"):
        build_job(a, "roi_psd", "absolute", hypotheses=["nope"])


def test_set_I_needs_informative(cfg, tmp_path):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    assert build_job(a, "roi_psd", "absolute").prior_sets == ["W"]
    info = tmp_path / "I.json"
    info.write_text(json.dumps({"x_Delta_KO": [-0.3, 0.3]}))
    job = build_job(a, "roi_psd", "absolute", informative=info)
    assert job.prior_sets == ["W", "I"] and job.informative == {"x_Delta_KO": [-0.3, 0.3]}


def test_rope_from_study_then_flag(cfg):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    assert build_job(a, "roi_psd", "absolute").rope_sd == 0.2
    assert build_job(a, "roi_psd", "absolute", rope_sd=0.1).rope_sd == 0.1


def test_unknown_analysis_and_dv(cfg):
    a = cfg.for_paradigm_analysis("arm", "roi_psd")
    with pytest.raises(ValueError, match="not supported"):
        build_job(a, "roi_pac", "x")
    with pytest.raises(ValueError, match="dv 'power'"):
        build_job(a, "roi_psd", "power")
    assert set(BAYES_TABLES) == {"roi_psd", "roi_aperiodic"}


def test_r_scripts_parse():
    for f in ("bayes_hypothesis.R", "bayes_analysis.R"):
        r = subprocess.run(["Rscript", "-e", f"invisible(parse(file='{R_DIR / f}'))"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_template_guard_rejects_other_prior_set():
    """brms writes priors into the compiled code, so update(recompile=FALSE) on a W template would run W priors
    whatever set is asked for; bayes_check_template must refuse it (empty brmsfit: no compiling, no sampling)."""
    code = f"""
    suppressMessages({{ source('{R_DIR / "hypothesis.R"}'); source('{R_DIR / "bayes_hypothesis.R"}') }})
    d <- expand.grid(subject = paste0('s', 1:6), roi = c('r1', 'r2'), stringsAsFactors = FALSE)
    d$group <- ifelse(d$subject %in% c('s1', 's2', 's3'), 'WT', 'KO'); set.seed(1); d$value <- rnorm(nrow(d))
    spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'WT', levels = list('WT', 'KO'))))
    inf <- list(x_all_KO = c(0.5, 0.8))
    pW <- bayes_prepare(d, spec, 'value', 'roi', NULL, 'W', inf)
    pI <- bayes_prepare(d, spec, 'value', 'roi', NULL, 'I', inf)
    tW <- suppressMessages(brm(bayes_formula(pW), data = pW$data, family = gaussian(),
                               prior = bayes_priors(pW), empty = TRUE))
    bayes_check_template(tW, pW, bayes_priors(pW))
    r <- try(bayes_check_template(tW, pI, bayes_priors(pI)), silent = TRUE)
    cat(if (inherits(r, 'try-error')) 'REJECTED' else 'ACCEPTED')
    """
    r = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("REJECTED")


_THREE_GROUPS = f"""
    suppressMessages({{ source('{R_DIR / "hypothesis.R"}'); source('{R_DIR / "bayes_hypothesis.R"}') }})
    d <- expand.grid(subject = paste0('s', 1:9), roi = c('r1', 'r2'), band = c('Alpha', 'Beta'), stringsAsFactors = FALSE)
    d$group <- c('WT', 'KO', 'TR')[(as.integer(sub('s', '', d$subject)) - 1) %/% 3 + 1]
    set.seed(1); d$value <- rnorm(nrow(d))
    spec <- parse_design_spec(list(design = list(factor = 'group', reference = 'WT', levels = list('WT', 'KO', 'TR'))))
    inf <- list(x_Alpha_KO = c(0.5, 0.3))
    con <- function(prep, band, a, b) {{ w <- setNames(c(0, 0, 0), prep$levels); w[a] <- 1; w[b] <- -1
      set.seed(2); .contrast_draws(.prior_effect_draws(prep, 200000)$b, NULL, prep, band, w) }}
"""


def _r(code):
    r = subprocess.run(["Rscript", "-e", _THREE_GROUPS + code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_w_prior_is_symmetric_over_groups():
    """Group-level coding (2026-10-19): under W every pairwise contrast, with or without the reference, is a priori
    N(0, 1); reference-coded N(0, 1) effects had given contrasts between non-reference groups SD sqrt(2)."""
    out = _r("""p <- bayes_prepare(d, spec, 'value', 'roi', 'band', 'W', inf)
    cat(sapply(list(c('KO', 'WT'), c('TR', 'WT'), c('TR', 'KO')), function(g) sd(con(p, 'Alpha', g[1], g[2]))))""")
    assert all(abs(float(x) - 1) < 0.01 for x in out)


def test_set_I_entry_is_the_contrast_prior():
    """A set-I entry x_<band>_<group> is the prior of that group - reference, exactly; other bands stay W."""
    out = _r("""p <- bayes_prepare(d, spec, 'value', 'roi', 'band', 'I', inf)
    x <- con(p, 'Alpha', 'KO', 'WT'); y <- con(p, 'Beta', 'KO', 'WT')
    cat(mean(x), sd(x), sd(y), sum(startsWith(p$fixed$col, 'dx_')))""")
    m, s, s_beta, n_dx = map(float, out)
    assert abs(m - 0.5) < 0.01 and abs(s - 0.3) < 0.01 and abs(s_beta - 1) < 0.01 and n_dx == 1


def test_design_columns_give_group_levels():
    """Each row's fixed part is its own group's level in its own band, under both codings."""
    out = _r("""ok <- sapply(c('W', 'I'), function(set) {
      p <- bayes_prepare(d, spec, 'value', 'roi', 'band', set, inf)
      b <- setNames(rnorm(nrow(p$fixed)), p$fixed$col)
      fx <- as.matrix(p$data[, p$fixed$col]) %*% b
      lev <- sapply(seq_len(nrow(p$data)), function(i) {
        M <- p$lmap[[as.character(p$data$band[i])]]; sum(M[as.character(p$data$.grp[i]), ] * b[colnames(M)]) })
      isTRUE(all.equal(as.numeric(fx), lev)) })
    cat(all(ok))""")
    assert out == ["TRUE"]

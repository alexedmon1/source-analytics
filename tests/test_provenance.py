"""`provenance.json`: what produced a result, written beside the result.

A stats CSV cannot say which source-analytics computed it, nor how the
recordings were localized — and a different atlas, inverse or sampling mode is a
different measurement. This checks the record carries both, and that assembling
or writing it can never take down a run that already produced its numbers.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from source_analytics.io.discovery import SubjectInfo
from source_analytics.io.run_manifest import read_run_manifest
from source_analytics.provenance import (
    PROVENANCE_FILE,
    SCHEMA_VERSION,
    build_provenance,
    read_provenance,
    write_provenance,
)


def _manifest_snapshot(sampling="fixed", atlas="allen32"):
    src = {"surface": {"method": "anatomical", "spacing_mm": 0.5}}
    if sampling == "monte_carlo":
        src["source_sampling"] = "monte_carlo"
        src["monte_carlo"] = {"n_sources": 160, "n_draws": 100, "seed": 20260821}
    return {"source_localization_version": "0.5.1", "config": {
        "pipeline": {"bem_type": "ellipsoid", "source_type": "surface"},
        "source_space": src,
        "inverse": {"method": "sLORETA", "orientation": "fixed"},
        "outputs": {"output_variants": "signed"},
        "provenance": {"preset": "ellipsoid_surface_anatomical", "atlas": atlas}}}


def _subject(root, sub_id, group="WT", sampling="fixed", atlas="allen32",
             mc_report=None, with_manifest=True):
    data = root / sub_id / "data"
    data.mkdir(parents=True)
    with open(data / "step6_roi_timeseries_signed.pkl", "wb") as f:
        pickle.dump({"Auditory_L": np.zeros(256)}, f)
    if with_manifest:
        (data / "config_resolved.yaml").write_text(
            yaml.safe_dump(_manifest_snapshot(sampling, atlas)))
    if sampling == "monte_carlo":
        (data / "monte_carlo_report.json").write_text(json.dumps(
            mc_report or {"n_draws": 100, "per_parcel": {}}))
    return SubjectInfo(subject_id=sub_id, group=group, data_dir=data,
                       pipeline_dir=data.parent)


def _cohort(tmp_path, **kw):
    subs = [_subject(tmp_path, "sub-1", "WT", **kw),
            _subject(tmp_path, "sub-2", "KO", **kw)]
    manifests = {s.subject_id: read_run_manifest(s.data_dir) for s in subs}
    manifests = {k: v for k, v in manifests.items() if v is not None}
    return subs, manifests


class TestRecordContents:
    def test_it_names_the_source_analytics_that_built_it(self, tmp_path):
        import source_analytics

        subs, man = _cohort(tmp_path)
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man)
        assert rec["source_analytics"]["version"] == source_analytics.__version__
        assert rec["schema"] == SCHEMA_VERSION

    def test_it_carries_the_localization_settings(self, tmp_path):
        subs, man = _cohort(tmp_path)
        loc = build_provenance(analysis="roi_psd", subjects=subs,
                               manifests=man)["localization"]
        assert loc["atlas"] == "allen32"
        assert loc["source_sampling"] == "fixed"
        assert loc["inverse_method"] == "sLORETA"
        assert loc["bem_type"] == "ellipsoid"
        assert loc["version"] == "0.5.1"
        assert loc["n_with_manifest"] == 2
        assert loc["n_unrecorded"] == 0

    def test_monte_carlo_draw_parameters_are_recorded(self, tmp_path):
        subs, man = _cohort(tmp_path, sampling="monte_carlo")
        loc = build_provenance(analysis="roi_psd", subjects=subs,
                               manifests=man)["localization"]
        assert loc["source_sampling"] == "monte_carlo"
        assert loc["monte_carlo"]["n_draws"] == 100
        assert loc["monte_carlo"]["seed"] == 20260821

    def test_monte_carlo_caveats_outlive_the_log(self, tmp_path):
        report = {"per_parcel": {
            "Thalamus_L": {"gain": 1.1, "coverage": 0.2, "collinear_with": "Thalamus_R"},
            "Auditory_L": {"gain": 1.2, "coverage": 1.0, "collinear_with": None}}}
        subs, man = _cohort(tmp_path, sampling="monte_carlo", mc_report=report)
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man)
        assert "Thalamus_L" in rec["parcel_caveats"]
        assert "Auditory_L" not in rec["parcel_caveats"]

    def test_a_fixed_grid_run_carries_no_caveats_key(self, tmp_path):
        subs, man = _cohort(tmp_path)
        assert "parcel_caveats" not in build_provenance(
            analysis="roi_psd", subjects=subs, manifests=man)

    def test_unrecorded_subjects_are_counted_not_invented(self, tmp_path):
        subs = [_subject(tmp_path, "sub-1"),
                _subject(tmp_path, "sub-legacy", with_manifest=False)]
        man = {"sub-1": read_run_manifest(subs[0].data_dir)}
        loc = build_provenance(analysis="roi_psd", subjects=subs,
                               manifests=man)["localization"]
        assert loc["n_with_manifest"] == 1
        assert loc["n_unrecorded"] == 1

    def test_a_cohort_with_no_manifests_claims_nothing(self, tmp_path):
        subs = [_subject(tmp_path, "sub-1", with_manifest=False)]
        loc = build_provenance(analysis="roi_psd", subjects=subs,
                               manifests={})["localization"]
        assert loc == {"n_with_manifest": 0, "n_unrecorded": 1}
        assert "atlas" not in loc

    def test_subjects_and_groups_are_recorded(self, tmp_path):
        subs, man = _cohort(tmp_path)
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man)
        assert rec["subjects"]["n"] == 2
        assert rec["subjects"]["groups"] == {"KO": 1, "WT": 1}
        assert rec["subjects"]["ids"] == ["sub-1", "sub-2"]

    def test_steps_record_what_actually_ran(self, tmp_path):
        subs, man = _cohort(tmp_path)
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man,
                               steps={"setup", "process"})
        assert rec["steps"] == ["process", "setup"]

    def test_paradigm_and_profile_are_recorded(self, tmp_path):
        subs, man = _cohort(tmp_path)
        rec = build_provenance(analysis="roi_psd", paradigm="resting",
                               profile="exploratory", subjects=subs, manifests=man)
        assert rec["paradigm"] == "resting"
        assert rec["profile"] == "exploratory"

    def test_the_record_is_json_serialisable(self, tmp_path):
        subs, man = _cohort(tmp_path, sampling="monte_carlo")
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man)
        json.loads(json.dumps(rec, default=str))    # no raise


class TestWriteAndRead:
    def test_round_trip(self, tmp_path):
        subs, man = _cohort(tmp_path / "loc")
        rec = build_provenance(analysis="roi_psd", subjects=subs, manifests=man)
        out = tmp_path / "tables" / "resting" / "roi_psd"
        path = write_provenance(out, rec)
        assert path == out / PROVENANCE_FILE
        assert read_provenance(out)["analysis"] == "roi_psd"

    def test_it_creates_the_directory(self, tmp_path):
        assert write_provenance(tmp_path / "a" / "b", {"x": 1}) is not None

    def test_reading_an_absent_record_is_none_not_an_error(self, tmp_path):
        assert read_provenance(tmp_path) is None

    def test_reading_a_corrupt_record_is_none_not_an_error(self, tmp_path):
        (tmp_path / PROVENANCE_FILE).write_text("{ not json")
        assert read_provenance(tmp_path) is None

    def test_a_failed_write_does_not_raise(self, tmp_path):
        """Provenance is a record *about* a run; it must not be able to fail one."""
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory")
        assert write_provenance(blocker, {"x": 1}) is None

    def test_it_is_not_mistaken_for_a_stats_table(self, tmp_path):
        """source-lightbox globs `*.csv` in the tables dir."""
        write_provenance(tmp_path, {"x": 1})
        assert list(tmp_path.glob("*.csv")) == []
        assert (tmp_path / PROVENANCE_FILE).suffix == ".json"


class TestLifecycleIntegration:
    """`BaseAnalysis.run` writes it, and cannot be taken down by it."""

    @staticmethod
    def _analysis(tmp_path):
        from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
        from source_analytics.config import StudyConfig

        cfg = StudyConfig.__new__(StudyConfig)
        cfg.name = "study"
        cfg.paradigm_name = "resting"
        cfg.profile_name = None
        cfg.results_dir = tmp_path / "results"
        a = ROIPsdAnalysis.__new__(ROIPsdAnalysis)
        a.config = cfg
        a.name = "roi_psd"
        return a

    def test_run_writes_the_record_with_the_steps_it_ran(self, tmp_path):
        a = self._analysis(tmp_path)
        subs, man = _cohort(tmp_path / "loc")
        a._write_provenance(subs, man, {"setup", "aggregate"})
        rec = read_provenance(a.tbl_dir)
        assert rec["analysis"] == "roi_psd"
        assert rec["paradigm"] == "resting"
        assert rec["steps"] == ["aggregate", "setup"]
        assert rec["localization"]["atlas"] == "allen32"

    def test_a_broken_record_does_not_fail_a_finished_run(self, tmp_path, monkeypatch):
        import source_analytics.provenance as prov

        a = self._analysis(tmp_path)
        monkeypatch.setattr(prov, "build_provenance",
                            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        a._write_provenance([], {}, {"setup"})      # no raise
        assert read_provenance(a.tbl_dir) is None

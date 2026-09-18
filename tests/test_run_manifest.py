"""Reading what built a localization run, and refusing to pool mismatched ones."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import yaml

from source_analytics.io import SubjectLoader
from source_analytics.io.loader import MonteCarloRunError
from source_analytics.io.run_manifest import (
    parcel_caveats,
    read_monte_carlo_report,
    read_run_manifest,
    undersampled_parcels,
    unreliable_parcels,
)


def _manifest(sampling="fixed", atlas="allen32", **over):
    cfg = {
        "pipeline": {"bem_type": "ellipsoid", "source_type": "surface"},
        "source_space": {"surface": {"method": "anatomical", "spacing_mm": 0.5}},
        "inverse": {"method": "sLORETA", "orientation": "fixed"},
        "outputs": {"output_variants": "signed"},
        "provenance": {"config_source": "preset", "atlas": atlas,
                       "preset": "ellipsoid_surface_anatomical"},
    }
    if sampling == "monte_carlo":
        cfg["source_space"]["source_sampling"] = "monte_carlo"
        cfg["source_space"]["monte_carlo"] = {"n_sources": 160, "n_draws": 100,
                                              "seed": 20260821}
    for key, value in over.items():
        cfg["provenance"][key] = value
    return {"source_localization_version": "0.5.1", "config": cfg}


def _write_run(root, name, sampling="fixed", atlas="allen32", mc_report=None,
               with_stc=True):
    data = root / name / "data"
    data.mkdir(parents=True)
    (data / "config_resolved.yaml").write_text(
        yaml.safe_dump(_manifest(sampling, atlas)))
    with open(data / "step6_roi_timeseries_signed.pkl", "wb") as f:
        pickle.dump({"Auditory_L": np.zeros(100)}, f)
    if sampling == "monte_carlo":
        (data / "monte_carlo_report.json").write_text(json.dumps(
            mc_report or {"n_draws": 100, "per_parcel": {}}))
    elif with_stc:
        with open(data / "step5_stc_signed.pkl", "wb") as f:
            pickle.dump(np.zeros((10, 100)), f)
        np.save(data / "step3_source_coords_mm.npy", np.zeros((10, 3)))
    return data


class TestReadManifest:
    def test_absent_manifest_is_none_not_an_error(self, tmp_path):
        (tmp_path / "data").mkdir()
        assert read_run_manifest(tmp_path / "data") is None

    def test_malformed_yaml_is_none(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "config_resolved.yaml").write_text("{ not: valid: yaml: [")
        assert read_run_manifest(d) is None

    def test_fixed_run_fields(self, tmp_path):
        m = read_run_manifest(_write_run(tmp_path, "sub-1"))
        assert m.source_sampling == "fixed"
        assert not m.is_monte_carlo
        assert m.has_vertex_output
        assert m.atlas == "allen32"
        assert m.inverse_method == "sLORETA"
        assert m.orientation == "fixed"
        assert m.output_variants == ("signed",)
        assert "fixed grid" in m.describe()

    def test_monte_carlo_run_fields(self, tmp_path):
        m = read_run_manifest(_write_run(tmp_path, "sub-1", "monte_carlo"))
        assert m.is_monte_carlo
        assert not m.has_vertex_output
        assert m.monte_carlo["n_draws"] == 100
        assert "Monte Carlo (K=100, 160 sources/draw)" in m.describe()

    def test_output_variants_both_expands(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        snap = _manifest()
        snap["config"]["outputs"]["output_variants"] = "both"
        (d / "config_resolved.yaml").write_text(yaml.safe_dump(snap))
        assert read_run_manifest(d).output_variants == ("signed", "magnitude")

    def test_signature_ignores_preset_name(self, tmp_path):
        a = read_run_manifest(_write_run(tmp_path, "a"))
        b = read_run_manifest(_write_run(tmp_path, "b"))
        assert a.signature() == b.signature()


class TestParcelCaveats:
    REPORT = {"per_parcel": {
        "Auditory_L": {"gain": 1.2, "coverage": 1.0, "collinear_with": None},
        "Thalamus_L": {"gain": 1.1, "coverage": 0.31, "collinear_with": "Thalamus_R"},
        "Cerebellum_R": {"gain": 1.0, "coverage": 0.95, "collinear_with": "Cerebellum_L"},
    }}

    def test_collinear_parcels_listed(self):
        assert unreliable_parcels(self.REPORT) == {
            "Thalamus_L": ["Thalamus_R"], "Cerebellum_R": ["Cerebellum_L"]}

    def test_undersampled_parcels_listed(self):
        assert set(undersampled_parcels(self.REPORT)) == {"Thalamus_L"}

    def test_clean_parcel_has_no_caveat(self):
        assert "Auditory_L" not in parcel_caveats(self.REPORT)

    def test_both_caveats_combine(self):
        why = parcel_caveats(self.REPORT)["Thalamus_L"]
        assert "31% of draws" in why and "near-collinear with Thalamus_R" in why

    def test_fixed_run_has_no_caveats(self):
        assert parcel_caveats(None) == {}
        assert unreliable_parcels(None) == {}
        assert undersampled_parcels(None) == {}

    def test_report_read_from_disk(self, tmp_path):
        data = _write_run(tmp_path, "sub-1", "monte_carlo", mc_report=self.REPORT)
        assert set(parcel_caveats(read_monte_carlo_report(data))) == {
            "Thalamus_L", "Cerebellum_R"}


class TestLoaderIntegration:
    def test_fixed_run_loader_flags(self, tmp_path):
        loader = SubjectLoader(_write_run(tmp_path, "sub-1"))
        assert not loader.is_monte_carlo
        assert loader.parcel_caveats() == {}
        assert loader.load_source_timecourses().shape == (10, 100)

    def test_monte_carlo_roi_series_load_normally(self, tmp_path):
        loader = SubjectLoader(_write_run(tmp_path, "sub-1", "monte_carlo"))
        assert loader.is_monte_carlo
        assert "Auditory_L" in loader.load_roi_timeseries(signed=True)

    def test_monte_carlo_source_timecourses_raise_clearly(self, tmp_path):
        loader = SubjectLoader(_write_run(tmp_path, "sub-1", "monte_carlo"))
        with pytest.raises(MonteCarloRunError, match="no single source grid"):
            loader.load_source_timecourses()

    def test_monte_carlo_source_coords_raise_clearly(self, tmp_path):
        loader = SubjectLoader(_write_run(tmp_path, "sub-1", "monte_carlo"))
        with pytest.raises(MonteCarloRunError, match="redrawn every draw"):
            loader.load_source_coords()

    def test_unmanifested_run_keeps_the_old_error(self, tmp_path):
        data = tmp_path / "sub-1" / "data"
        data.mkdir(parents=True)
        loader = SubjectLoader(data)
        assert loader.manifest is None
        with pytest.raises(FileNotFoundError):
            loader.load_source_timecourses()


class TestCohortHomogeneity:
    """`BaseAnalysis.run` refuses to pool subjects localized differently."""

    @staticmethod
    def _analysis(tmp_path):
        from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
        from source_analytics.config import StudyConfig

        cfg = StudyConfig.__new__(StudyConfig)
        analysis = ROIPsdAnalysis.__new__(ROIPsdAnalysis)
        analysis.config = cfg
        analysis.name = "roi_psd"
        return analysis

    @staticmethod
    def _subjects(tmp_path, specs):
        from source_analytics.io.discovery import SubjectInfo

        out = []
        for name, kwargs in specs:
            data = _write_run(tmp_path, name, **kwargs)
            out.append(SubjectInfo(subject_id=name, group="G",
                                   data_dir=data, pipeline_dir=data.parent))
        return out

    def test_matching_cohort_passes(self, tmp_path):
        a = self._analysis(tmp_path)
        subjects = self._subjects(tmp_path, [("sub-1", {}), ("sub-2", {})])
        a._check_cohort_homogeneous(subjects)      # no raise

    def test_mixed_sampling_is_refused(self, tmp_path):
        a = self._analysis(tmp_path)
        subjects = self._subjects(tmp_path, [
            ("sub-1", {}), ("sub-2", {"sampling": "monte_carlo"})])
        with pytest.raises(ValueError, match="source sampling"):
            a._check_cohort_homogeneous(subjects)

    def test_mixed_atlas_is_refused(self, tmp_path):
        a = self._analysis(tmp_path)
        subjects = self._subjects(tmp_path, [
            ("sub-1", {}), ("sub-2", {"atlas": "allen26"})])
        with pytest.raises(ValueError, match="atlas"):
            a._check_cohort_homogeneous(subjects)

    def test_unmanifested_subjects_are_skipped_not_guessed(self, tmp_path):
        from source_analytics.io.discovery import SubjectInfo

        a = self._analysis(tmp_path)
        subjects = self._subjects(tmp_path, [("sub-1", {})])
        bare = tmp_path / "sub-legacy" / "data"
        bare.mkdir(parents=True)
        subjects.append(SubjectInfo(subject_id="sub-legacy", group="G",
                                    data_dir=bare, pipeline_dir=bare.parent))
        a._check_cohort_homogeneous(subjects)      # no raise

    def test_all_monte_carlo_cohort_passes_and_warns(self, tmp_path, caplog):
        a = self._analysis(tmp_path)
        report = {"per_parcel": {
            "Thalamus_L": {"gain": 1.1, "coverage": 0.2,
                           "collinear_with": "Thalamus_R"}}}
        subjects = self._subjects(tmp_path, [
            ("sub-1", {"sampling": "monte_carlo", "mc_report": report}),
            ("sub-2", {"sampling": "monte_carlo", "mc_report": report})])
        with caplog.at_level("WARNING"):
            a._check_cohort_homogeneous(subjects)
        assert "Thalamus_L" in caplog.text


class TestListing:
    """`source-analytics list` answers "what can this install run?"."""

    def test_atlases_listing_covers_the_registry(self, capsys):
        from source_analytics.atlas import registered_atlases
        from source_analytics.cli import _print_atlases

        _print_atlases()
        out = capsys.readouterr().out
        for name in registered_atlases():
            assert name in out

    def test_atlas_listing_reports_parcel_counts(self, capsys):
        from source_analytics.cli import _print_atlases

        _print_atlases()
        out = capsys.readouterr().out
        assert "26" in out and "allen26" in out

    def test_plugin_listing_names_the_vertex_plugin(self, capsys):
        from source_analytics.cli import _print_plugins

        _print_plugins()
        out = capsys.readouterr().out
        assert "source-analytics-vertex" in out
        assert "vertex_specparam" in out

    def test_plugin_listing_hides_deprecated_aliases(self, capsys):
        from source_analytics.cli import _print_plugins

        _print_plugins()
        out = capsys.readouterr().out
        for alias in ("wholebrain", "spatial_lmm", "specparam_vertex"):
            assert alias not in out

    def test_moved_aliases_still_produce_a_hint(self):
        from source_analytics.plugins import missing_analysis_hint

        assert "source-analytics-vertex" in missing_analysis_hint("wholebrain")
        assert "source-analytics-vertex" in missing_analysis_hint("vertex_specparam")
        assert missing_analysis_hint("not_an_analysis") == ""

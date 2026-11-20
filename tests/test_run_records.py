"""D10 step 1b: reading source-localization's run records.

(a) the pooling check compares what the numbers now depend on, reading a record an older run lacks as that
release's only value (mirroring source-localization's study/run_status.py); (b) a parcel a run has no sources in
is reported before any subject is processed, with the choice to drop the parcel or the subject; (c) region-level
results carry the spatial-leakage caveat.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import yaml

from source_analytics.io.discovery import SubjectInfo
from source_analytics.io.loader import SubjectLoader
from source_analytics.io.run_manifest import (
    COHORT_FIELDS,
    REGION_LEAKAGE_CAVEAT,
    read_empty_parcels,
    read_run_manifest,
)

PARCELS = ["Auditory_L", "Auditory_R", "Thalamus_L"]


def _config(version="0.9.12", bem_type="ellipsoid", sampling="fixed", inverse=None, bem=None,
            source_type="surface", orientation="fixed"):
    cfg = {
        "pipeline": {"bem_type": bem_type, "source_type": source_type},
        "source_space": {"surface": {"method": "anatomical", "spacing_mm": 0.5}, "source_sampling": sampling},
        "inverse": {"method": "sLORETA", "orientation": orientation, **(inverse or {})},
        "outputs": {"output_variants": "signed"},
        "provenance": {"config_source": "preset", "atlas": "allen32", "preset": "p"},
    }
    if bem is not None:
        cfg["bem"] = bem
    if sampling == "monte_carlo":
        cfg["source_space"]["monte_carlo"] = {"n_sources": 160, "n_draws": 100}
    return {"source_localization_version": version, "config": cfg}


def _run(root, name, snapshot=None, records=None, empty=None, parcels=PARCELS):
    """A run folder: config_resolved.yaml, optional step records, the ROI series and (0.9.11+) the coverage
    record; ``empty`` parcels are left out of the series, as source-localization does from 0.9.11."""
    data = root / name / "data"
    data.mkdir(parents=True)
    if snapshot is not None:
        (data / "config_resolved.yaml").write_text(yaml.safe_dump(snapshot))
    for fname, rec in (records or {}).items():
        (data / fname).write_text(json.dumps(rec))
    if empty is not None:
        (data / "step6_roi_coverage.json").write_text(json.dumps({
            "n_parcels": len(parcels), "parcels_without_sources": list(empty),
            "sources_per_parcel": {p: 0 if p in empty else 5 for p in parcels}}))
    with open(data / "step6_roi_timeseries_signed.pkl", "wb") as f:
        pickle.dump({p: np.zeros(100) for p in parcels if p not in (empty or ())}, f)
    return data


def _subjects(root, specs):
    return [SubjectInfo(subject_id=name, group="G", data_dir=_run(root, name, **kw),
                        pipeline_dir=root / name) for name, kw in specs]


def _analysis(name="roi_psd", raw=None, rois=None, roi_categories=None):
    from source_analytics.analyses.roi_psd_analysis import ROIPsdAnalysis
    from source_analytics.config import StudyConfig

    cfg = StudyConfig(name="t", output_dir=None, results_dir=None, groups={}, group_order=[], group_colors={},
                      bands={}, roi_categories=roi_categories or {}, discovery={}, rois=rois or [],
                      raw=raw or {})
    a = ROIPsdAnalysis.__new__(ROIPsdAnalysis)
    a.config, a.name = cfg, name
    return a


# --------------------------------------------------------------- (a) pooling ---
class TestRecordedSettings:
    def test_one_list_behind_signature_and_the_check(self):
        from source_analytics.analyses.base import BaseAnalysis

        assert BaseAnalysis._COHORT_FIELDS is COHORT_FIELDS
        assert "mc_draws" in {attr for attr, _ in COHORT_FIELDS}

    def test_0911_record_reads_its_release_defaults(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config("0.9.11")))
        assert m.regularisation == "snr 3 (lambda2 0.111111)"
        assert m.bem_mesh_vertices == 642
        assert m.analytic_electrodes is None and m.electrode_centered is None

    def test_0912_record_reads_snr_1(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config("0.9.12")))
        assert m.regularisation == "snr 1 (lambda2 1)"

    def test_explicit_settings_win(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config(
            "0.9.12", inverse={"snr": 3}, bem={"ellipsoid": {"n_vertices": 2562}})))
        assert m.regularisation == "snr 3 (lambda2 0.111111)" and m.bem_mesh_vertices == 2562
        m = read_run_manifest(_run(tmp_path, "b", _config("0.9.12", inverse={"snr": 3, "lambda2": 0.05})))
        assert m.regularisation == "lambda2 0.05"

    def test_no_version_and_no_snr_is_unknown(self, tmp_path):
        snap = _config()
        del snap["source_localization_version"]
        assert read_run_manifest(_run(tmp_path, "a", snap)).regularisation is None

    def test_analytic_sphere(self, tmp_path):
        old = read_run_manifest(_run(tmp_path, "a", _config("0.9.11", "sphere", bem={"sphere": {"numerical": False}})))
        assert (old.bem_mesh_vertices, old.analytic_electrodes, old.electrode_centered) == (None, "raw", False)
        new = read_run_manifest(_run(tmp_path, "b", _config("0.9.12", "sphere", bem={"sphere": {
            "numerical": False, "analytic_electrodes": "projected", "electrode_centered": True}})))
        assert (new.analytic_electrodes, new.electrode_centered) == ("projected", True)

    def test_numerical_sphere_has_a_mesh(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config("0.9.11", "sphere", bem={"sphere": {}})))
        assert (m.bem_mesh_vertices, m.analytic_electrodes, m.electrode_centered) == (642, None, False)

    @pytest.mark.parametrize("record, expected", [
        ({"inverse_standardisation": "per-source-v1", "method": "sLORETA", "orientation": "free"}, "correct"),
        ({"inverse_standardisation": "per-component-legacy", "method": "sLORETA", "orientation": "free"}, "faulty"),
        ({"method": "eLORETA", "orientation": "free"}, "faulty"),                  # 0.9.1: scalar weight
        ({"method": "eLORETA", "orientation": "free", "eloreta_weights": "block"}, "correct"),
        ({"inverse_standardisation": "per-source-v1", "orientation": "fixed"}, "not affected"),
    ])
    def test_inverse_standardisation_record(self, tmp_path, record, expected):
        data = _run(tmp_path, "a", _config(orientation="free"),
                    records={"step5_inverse_standardisation.json": record})
        assert read_run_manifest(data).inverse_standardisation == expected

    def test_inverse_standardisation_without_record(self, tmp_path):
        free = read_run_manifest(_run(tmp_path, "a", _config(orientation="free")))
        assert free.inverse_standardisation == "faulty"
        fixed = read_run_manifest(_run(tmp_path, "b", _config(orientation="fixed")))
        assert fixed.inverse_standardisation == "not affected"

    @pytest.mark.parametrize("record, expected", [
        (None, ("dominant", "none")),                                               # before 0.9.2
        ({"mc_directions": "dominant-direction-legacy"}, ("dominant", "none")),
        ({"mc_directions": "three-direction-block-v1"}, ("three", "vector_mean")),  # 0.9.2-0.9.12
        ({"mc_directions": "three-direction-block-v1", "magnitude_readout": "mean_of_norms"},
         ("three", "mean_of_norms")),                                               # 0.9.13
        ({"mc_directions": "something-new"}, ("unknown", "unknown")),
    ])
    def test_monte_carlo_directions_and_readout(self, tmp_path, record, expected):
        records = {"step6_mc_directions.json": record} if record else None
        m = read_run_manifest(_run(tmp_path, "a", _config(sampling="monte_carlo", source_type="volume",
                                                          orientation="free"), records=records))
        assert (m.mc_directions, m.mc_readout) == expected

    def test_fixed_surface_monte_carlo_pool_has_nothing_to_collapse(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config(sampling="monte_carlo")))
        assert (m.mc_directions, m.mc_readout) == ("none", "none")

    def test_fixed_sampling_has_no_monte_carlo_fields(self, tmp_path):
        m = read_run_manifest(_run(tmp_path, "a", _config()))
        assert (m.mc_draws, m.mc_directions, m.mc_readout) == (None, None, None)


class TestPooling:
    def test_mixed_0911_0912_study_is_refused_naming_the_settings(self, tmp_path):
        # As each release writes its record: 0.9.12 writes the mesh default into the config, 0.9.11 had none.
        new = _config("0.9.12", inverse={"snr": 1.0}, bem={"ellipsoid": {"n_vertices": 2562}})
        subjects = _subjects(tmp_path, [("sub-1", {"snapshot": _config("0.9.11")}),
                                        ("sub-2", {"snapshot": new})])
        with pytest.raises(ValueError) as err:
            _analysis()._check_cohort_homogeneous(subjects)
        assert "inverse regularisation" in str(err.value)
        assert "BEM mesh vertices" in str(err.value)
        assert "sub-1" in str(err.value) and "sub-2" in str(err.value)

    def test_0912_run_made_with_0911_settings_pools_with_0911(self, tmp_path):
        repro = _config("0.9.12", inverse={"snr": 3}, bem={"ellipsoid": {"n_vertices": 642}})
        subjects = _subjects(tmp_path, [("sub-1", {"snapshot": _config("0.9.11")}),
                                        ("sub-2", {"snapshot": repro})])
        _analysis()._check_cohort_homogeneous(subjects)                 # no raise: the version is not compared

    def test_mixed_monte_carlo_readouts_are_refused(self, tmp_path):
        mc = _config(sampling="monte_carlo", source_type="volume", orientation="free")
        three = {"mc_directions": "three-direction-block-v1"}
        subjects = _subjects(tmp_path, [
            ("sub-1", {"snapshot": mc, "records": {"step6_mc_directions.json": three}}),
            ("sub-2", {"snapshot": mc, "records": {"step6_mc_directions.json": {
                **three, "magnitude_readout": "mean_of_norms"}}})])
        with pytest.raises(ValueError, match="Monte Carlo magnitude readout"):
            _analysis()._check_cohort_homogeneous(subjects)

    def test_explicit_vector_mean_pools_with_a_pre_readout_run(self, tmp_path):
        mc = _config(sampling="monte_carlo", source_type="volume", orientation="free")
        three = {"mc_directions": "three-direction-block-v1"}
        subjects = _subjects(tmp_path, [
            ("sub-1", {"snapshot": mc, "records": {"step6_mc_directions.json": three}}),
            ("sub-2", {"snapshot": mc, "records": {"step6_mc_directions.json": {
                **three, "magnitude_readout": "vector_mean"}}})])
        _analysis()._check_cohort_homogeneous(subjects)

    def test_different_monte_carlo_draws_are_refused(self, tmp_path):
        more = _config(sampling="monte_carlo")
        more["config"]["source_space"]["monte_carlo"]["n_draws"] = 200
        subjects = _subjects(tmp_path, [("sub-1", {"snapshot": _config(sampling="monte_carlo")}),
                                        ("sub-2", {"snapshot": more})])
        with pytest.raises(ValueError, match="sources per draw, draws"):
            _analysis()._check_cohort_homogeneous(subjects)

    def test_fixed_and_faulty_standardisation_are_refused(self, tmp_path):
        rec = {"method": "sLORETA", "orientation": "free"}
        subjects = _subjects(tmp_path, [
            ("sub-1", {"snapshot": _config(orientation="free"), "records": {
                "step5_inverse_standardisation.json": {**rec, "inverse_standardisation": "per-source-v1"}}}),
            ("sub-2", {"snapshot": _config(orientation="free")})])
        with pytest.raises(ValueError, match="inverse standardisation"):
            _analysis()._check_cohort_homogeneous(subjects)

    def test_provenance_records_the_new_settings(self, tmp_path):
        from source_analytics.provenance import _localization

        m = read_run_manifest(_run(tmp_path, "a", _config("0.9.12")))
        rec = _localization({"a": m}, 1)
        assert rec["regularisation"] == "snr 1 (lambda2 1)" and rec["bem_mesh_vertices"] == 642


# ------------------------------------------------------- (b) missing parcels ---
class TestMissingParcels:
    def test_reader(self, tmp_path):
        assert read_empty_parcels(_run(tmp_path, "a", empty=["Thalamus_L"])) == ["Thalamus_L"]
        assert read_empty_parcels(_run(tmp_path, "b")) is None

    def test_declared_parcel_missing_is_reported_with_both_choices(self, tmp_path):
        subjects = _subjects(tmp_path, [("sub-801", {"empty": ["Thalamus_L"]}), ("sub-802", {"empty": []})])
        with pytest.raises(ValueError) as err:
            _analysis(rois=PARCELS)._check_parcel_coverage(subjects)
        msg = str(err.value)
        assert "sub-801: Thalamus_L" in msg and "sub-802" not in msg
        assert "drop_parcels: [Thalamus_L]" in msg
        assert "exclude: true on 801" in msg

    def test_undeclared_parcel_missing_in_some_subjects_is_reported(self, tmp_path):
        subjects = _subjects(tmp_path, [("sub-1", {"empty": ["Thalamus_L"]}), ("sub-2", {"empty": []})])
        with pytest.raises(ValueError, match="sub-1: Thalamus_L"):
            _analysis()._check_parcel_coverage(subjects)

    def test_undeclared_parcel_empty_everywhere_is_consistent(self, tmp_path, caplog):
        subjects = _subjects(tmp_path, [("sub-1", {"empty": ["Thalamus_L"]}),
                                        ("sub-2", {"empty": ["Thalamus_L"]})])
        with caplog.at_level("INFO"):
            _analysis()._check_parcel_coverage(subjects)
        assert "Thalamus_L" in caplog.text

    def test_empty_everywhere_but_a_subject_without_the_record_is_reported(self, tmp_path):
        # A run from before 0.9.11 zero-filled the parcel, so it is in that subject's data.
        subjects = _subjects(tmp_path, [("sub-1", {"empty": ["Thalamus_L"]}), ("sub-old", {})])
        with pytest.raises(ValueError, match="sub-1: Thalamus_L"):
            _analysis()._check_parcel_coverage(subjects)

    def test_drop_parcels_resolves_it(self, tmp_path):
        subjects = _subjects(tmp_path, [("sub-1", {"empty": ["Thalamus_L"]}), ("sub-2", {"empty": []})])
        a = _analysis(rois=PARCELS, raw={"drop_parcels": ["Thalamus_L"]})
        a._check_parcel_coverage(subjects)
        assert a.config.rois == ["Auditory_L", "Auditory_R"]
        loader = SubjectLoader(subjects[1].data_dir)
        assert list(loader.load_or_extract_roi_timeseries(
            rois=a.config.rois, drop=a.config.drop_parcels)) == ["Auditory_L", "Auditory_R"]

    def test_electrode_modules_are_not_checked(self, tmp_path):
        subjects = _subjects(tmp_path, [("sub-1", {"empty": ["Thalamus_L"]}), ("sub-2", {"empty": []})])
        _analysis(name="electrode_psd", rois=PARCELS)._check_parcel_coverage(subjects)

    def test_loader_error_names_both_choices(self, tmp_path):
        loader = SubjectLoader(_run(tmp_path, "a", empty=["Thalamus_L"]))
        with pytest.raises(KeyError, match="drop_parcels"):
            loader.load_or_extract_roi_timeseries(rois=PARCELS)

    def test_loader_drop_without_allowlist(self, tmp_path):
        loader = SubjectLoader(_run(tmp_path, "a"))
        assert list(loader.load_or_extract_roi_timeseries(drop=["Auditory_R"])) == ["Auditory_L", "Thalamus_L"]


class TestDropParcelsConfig:
    def test_removed_from_rois_and_categories(self):
        cfg = _analysis(rois=PARCELS, roi_categories={"Sensory": ["Auditory_L", "Auditory_R"],
                                                      "Deep": ["Thalamus_L"]},
                        raw={"drop_parcels": "Thalamus_L"}).config
        assert cfg.drop_parcels == ["Thalamus_L"]
        assert cfg.rois == ["Auditory_L", "Auditory_R"]
        assert cfg.roi_categories == {"Sensory": ["Auditory_L", "Auditory_R"]}

    def test_dropping_every_declared_roi_is_refused(self):
        with pytest.raises(ValueError, match="leaves none"):
            _analysis(rois=["Thalamus_L"], raw={"drop_parcels": ["Thalamus_L"]})

    def test_survives_profile_and_paradigm(self, tmp_path):
        from source_analytics.config import StudyConfig

        study = tmp_path / "study.yaml"
        study.write_text(yaml.safe_dump({
            "name": "s", "groups": {"A": "a"}, "drop_parcels": ["Thalamus_L"],
            "roi_categories": {"Sensory": ["Auditory_L", "Auditory_R"], "Deep": ["Thalamus_L"]},
            "paradigms": {"resting": {"data_dir": ".", "analyses": {"roi_psd": {}}}},
            "external": {"roi_categories": {"Sensory": ["Auditory_L"], "Deep": ["Thalamus_L"]}},
        }))
        cfg = StudyConfig.from_yaml(study)
        assert cfg.roi_categories == {"Sensory": ["Auditory_L", "Auditory_R"]}
        prof = cfg.for_profile("external")                             # a dropped parcel is not "unknown"
        assert prof.rois == ["Auditory_L"]
        assert prof.for_paradigm("resting").drop_parcels == ["Thalamus_L"]


# --------------------------------------------------------- (c) leakage caveat ---
class TestLeakageCaveat:
    def test_appended_once_to_a_region_summary(self, tmp_path):
        a = _analysis()
        a.output_dir = tmp_path
        (tmp_path / "ANALYSIS_SUMMARY.md").write_text("# roi_psd\n\nresults\n")
        a._append_region_caveat({})
        a._append_region_caveat({})
        text = (tmp_path / "ANALYSIS_SUMMARY.md").read_text()
        assert text.count("spatial leakage") == 1 and REGION_LEAKAGE_CAVEAT in text

    def test_not_for_electrode_modules_or_field_runs(self, tmp_path):
        from source_analytics.io.run_manifest import RunManifest

        (tmp_path / "ANALYSIS_SUMMARY.md").write_text("x\n")
        a = _analysis(name="electrode_psd")
        a.output_dir = tmp_path
        a._append_region_caveat({})
        b = _analysis()
        b.output_dir = tmp_path
        b._append_region_caveat({"s": RunManifest(readout_mode="field")})
        assert (tmp_path / "ANALYSIS_SUMMARY.md").read_text() == "x\n"

    def test_results_spec_caveat(self):
        from source_analytics.results_spec import region_caveats

        assert region_caveats("roi", {}) == [REGION_LEAKAGE_CAVEAT]
        assert region_caveats("roi", {"localization": {"readout_mode": "field"}}) == []
        assert region_caveats("electrode", {}) == []

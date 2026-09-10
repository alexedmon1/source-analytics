"""roi_signature: the source-side decoding counterpart of electrode_signature.

Runs the whole lifecycle on the synthetic study (5 WT vs 5 KO; every subject in
a group is identical and KO carries an added component, so the groups are
separable) and checks the results table, that the features are per ROI, that a
separable contrast decodes, and that figures regenerate from disk alone.
"""

from __future__ import annotations

import pandas as pd

from source_analytics.config import StudyConfig
from source_analytics.core import StudyAnalyzer

LIFECYCLE = {"setup", "process", "aggregate", "statistics", "figures", "summary"}


def _config(sample_config_yaml, tmp_path) -> StudyConfig:
    text = sample_config_yaml.read_text() + (
        "\nroi_signature:\n  classifiers: [svm_linear]\n  n_permutations: 20\n")
    path = tmp_path / "signature_config.yaml"
    path.write_text(text)
    return StudyConfig.from_yaml(path)


def test_roi_signature_full_lifecycle(sample_config_yaml, tmp_path):
    config = _config(sample_config_yaml, tmp_path)
    StudyAnalyzer(config).run_analysis("roi_signature", steps=LIFECYCLE)

    results = pd.read_csv(next(config.results_dir.rglob("roi_signature_results.csv")))
    assert set(results["band"]) == set(config.bands)
    assert set(results["classifier"]) == {"svm_linear"}
    assert set(results["contrast"]) == {"disease_effect"}
    assert results["accuracy"].max() >= 0.9, results[["band", "accuracy"]]

    work = config.output_dir / "roi_signature"
    features = pd.read_csv(work / "data" / "roi_signature_features.csv")
    assert "roi" in features.columns and "channel" not in features.columns
    assert features["roi"].nunique() == 8
    assert (work / "ANALYSIS_SUMMARY.md").read_text().startswith("# ROI Neural Signature")


def test_roi_signature_figures_regenerate_from_disk(sample_config_yaml, tmp_path):
    config = _config(sample_config_yaml, tmp_path)
    analyzer = StudyAnalyzer(config)
    analyzer.run_analysis("roi_signature", steps=LIFECYCLE)
    pngs = list(config.results_dir.rglob("roi_signature_importance_*.png"))
    assert pngs, "no ROI importance figures were drawn"
    for p in pngs:
        p.unlink()
    analyzer.run_analysis("roi_signature", steps={"figures"})
    assert list(config.results_dir.rglob("roi_signature_importance_*.png"))

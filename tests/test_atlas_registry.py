"""An atlas is resolved by NAME to its own files, never by directory convention.

Locks out the bug that gave allen26 studies allen32's partition: allen32, allen26
and allen64 all live in ``allen/``, so the category file and label volume found
*in the directory* were allen32's whatever the study named. Two allen26
categories matched no parcel and vanished from the region-level tables, Deep
Subcortical was built from 4 of its 8 parcels, and the six merged parcels drew
blank in every mosaic.

Also locks the header-based 10x convention: the filename rule treated Antwerp's
true-unit label file as inflated and shrank its affine 10x.

Needs the source-localization atlas data; skipped when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from source_analytics.atlas import (
    find_atlas_dir,
    header_is_inflated,
    load_atlas,
    load_roi_categories,
    load_roi_mapping,
    registered_atlases,
    resolve_atlas,
)

try:
    _HAVE_REGISTRY = (find_atlas_dir() / "registry.yaml").exists()
except FileNotFoundError:
    _HAVE_REGISTRY = False

pytestmark = pytest.mark.skipif(not _HAVE_REGISTRY,
                                reason="source-localization atlas registry not found")

# The six bilateral pairs allen26 merges; none of these names exists in allen32.
MERGED = {"Frontal_Anterior", "Olfactory_Bulb", "Thalamus",
          "Hypothalamus", "Brainstem_Tectum", "Cerebellum"}


def _mapping_names(spec) -> set[str]:
    return {v["name"] for v in load_roi_mapping(spec)["rois"].values()}


def test_allen26_resolves_to_its_own_files():
    s = resolve_atlas(atlas_name="allen26")
    assert s.labels.name == "allen26_labels.nii.gz"
    assert s.roi_mapping.name == "roi_mapping_allen26.json"
    assert s.roi_categories.name == "roi_categories_allen26.yaml"


def test_allen26_categories_and_labels_name_its_merged_parcels():
    s = resolve_atlas(atlas_name="allen26")
    members = {r for rois in load_roi_categories(s).values() for r in rois}
    assert MERGED <= members, "category map is another atlas's partition"
    assert MERGED <= _mapping_names(s), "merged parcels would draw blank"
    labels, _ = load_atlas(s)
    assert len(np.unique(labels)) - 1 == 26


def test_a_shared_directory_no_longer_decides_the_atlas():
    a26 = resolve_atlas(atlas_name="allen26")
    a32 = resolve_atlas(atlas_name="allen32")
    assert a26.labels.parent == a32.labels.parent            # one directory...
    assert a26.labels != a32.labels                          # ...two atlases
    assert a26.roi_categories != a32.roi_categories


def test_aliases_and_atlases_without_categories():
    assert resolve_atlas(atlas_name="allen").labels == resolve_atlas(atlas_name="allen32").labels
    assert (resolve_atlas(atlas_name="coarse_22roi").labels
            == resolve_atlas(atlas_name="coarse22").labels)
    a64 = resolve_atlas(atlas_name="allen64")
    assert a64.roi_categories is None
    assert load_roi_categories(a64) == {}


def test_every_registered_atlas_resolves():
    names = registered_atlases()
    assert {"allen26", "allen32", "antwerp"} <= set(names)
    for name in names:
        s = resolve_atlas(atlas_name=name)
        assert s.labels.exists() and s.roi_mapping.exists(), name


def test_an_unknown_atlas_refuses_to_guess():
    with pytest.raises(ValueError, match="not in"):
        resolve_atlas(atlas_name="allen99")


def test_explicit_files_define_an_unregistered_atlas():
    ref = resolve_atlas(atlas_name="allen26")
    s = resolve_atlas(atlas_name="lab_atlas", files={
        "brain_labels": str(ref.labels),
        "roi_mapping": str(ref.roi_mapping),
        "roi_categories": str(ref.roi_categories),
    })
    assert s.labels == ref.labels
    assert load_roi_categories(s) == load_roi_categories(ref)


def test_unknown_atlas_files_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        resolve_atlas(atlas_name="allen26", files={"labelz": "x.nii.gz"})


def test_a_spec_passes_through_unchanged():
    s = resolve_atlas(atlas_name="allen26")
    assert resolve_atlas(s) is s


def test_the_header_decides_the_10x_correction():
    # Antwerp's label file stores true units; the filename rule shrank it 10x.
    antwerp = resolve_atlas(atlas_name="antwerp")
    assert not header_is_inflated(antwerp.labels)
    _, affine = load_atlas(antwerp)
    assert affine[0, 0] > 0.1            # 0.203 mm, not 0.0203
    # coarse22's label file IS inflated and must still be corrected.
    coarse = resolve_atlas(atlas_name="coarse22")
    assert header_is_inflated(coarse.labels)
    _, affine = load_atlas(coarse)
    assert affine[0, 0] < 0.5


def test_study_default_categories_follow_the_named_atlas():
    from source_analytics.config import _load_atlas_roi_categories

    members = {r for rois in _load_atlas_roi_categories("allen26").values() for r in rois}
    assert MERGED <= members

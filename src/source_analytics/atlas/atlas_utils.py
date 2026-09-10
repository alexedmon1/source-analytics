"""Atlas utilities: load atlas NIfTI, map vertex coordinates to ROI labels.

Some bundled NIfTI files store voxel sizes 10x larger than reality and some
store true units. Which is which is read from the header, as
source_localization.utils.atlas.header_is_inflated does, never guessed from a
filename: the guess was wrong for Atlas_3DRoisLeftRight.Labels.nii, which stores
true units and was being shrunk 10x. Inflated files get
ATLAS_VOXEL_SCALE_FACTOR = 0.1 applied to the rotation/scaling block and the
translation vector of the affine.

An atlas is resolved by NAME to its own file set (:func:`resolve_atlas`), from
source-localization's ``registry.yaml``. The registry is where an atlas is
defined; a directory is not, because several atlases share one (allen32, allen26
and allen64 all live in ``allen/``), so "the roi_mapping.json in the atlas
directory" names allen32's files whatever the study asked for. Atlases that are
not registered can name their files explicitly.

Also provides on-the-fly ROI extraction from vertex-level source time courses,
ported from source_localization/steps/roi_extraction.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import yaml

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

ATLAS_VOXEL_SCALE_FACTOR = 0.1
_ATLAS_NIFTI = "Atlas_3DRoisLeftRight.Labels.nii"
_ALLEN_NIFTI = "allen_labels.nii.gz"
_ROI_MAPPING_FILE = "roi_mapping.json"
_ROI_CATEGORIES_FILE = "roi_categories.yaml"


def find_atlas_dir(
    config_atlas_dir: str | Path | None = None,
    atlas_name: str | None = None,
) -> Path:
    """Locate the atlas data directory.

    Parameters
    ----------
    config_atlas_dir : str or Path, optional
        Explicit atlas directory from config. If None, falls back to the
        source_localization package's bundled atlas.
    atlas_name : str, optional
        Atlas name (``"antwerp"`` or ``"allen"``).  When provided and the
        resolved directory has a ``<name>/`` subdirectory, return that
        subdirectory instead.

    Returns
    -------
    Path
        Directory containing atlas NIfTI and roi_mapping.json.
    """
    if config_atlas_dir is not None:
        p = Path(config_atlas_dir)
        if p.is_dir():
            return p

    # Try source_localization package data directory
    base_atlas_dir: Path | None = None
    try:
        import source_localization

        pkg_dir = Path(source_localization.__file__).parent
        atlas_dir = pkg_dir / "data" / "atlas"
        if atlas_dir.is_dir():
            base_atlas_dir = atlas_dir
    except ImportError:
        pass

    if base_atlas_dir is None:
        # Fallback: well-known path
        fallback = Path(
            "/home/edm9fd/sandbox/source-localization"
            "/src/source_localization/data/atlas"
        )
        if fallback.is_dir():
            base_atlas_dir = fallback

    if base_atlas_dir is None:
        raise FileNotFoundError(
            "Cannot find atlas directory. Set atlas_dir in config or install source_localization."
        )

    # If atlas_name given, check for subdirectory (e.g. atlas/allen/)
    # allen32/allen64 are variants of the same allen/ directory
    if atlas_name is not None:
        candidates = [atlas_name]
        if atlas_name.startswith("allen"):
            candidates.append("allen")
        for candidate in candidates:
            sub = base_atlas_dir / candidate
            if sub.is_dir():
                return sub

    return base_atlas_dir


def _find_atlas_nifti(atlas_dir: Path) -> tuple[Path, bool]:
    """Find the atlas NIfTI file in a directory.

    Returns
    -------
    nii_path : Path
        Path to the NIfTI file.
    needs_10x_correction : bool
        Read from the header (:func:`header_is_inflated`), not the filename.
    """
    # Try Allen atlas first (correct header)
    allen_path = atlas_dir / _ALLEN_NIFTI
    if allen_path.exists():
        return allen_path, header_is_inflated(allen_path)

    # Try Antwerp atlas (needs 10x correction)
    antwerp_path = atlas_dir / _ATLAS_NIFTI
    if antwerp_path.exists():
        return antwerp_path, header_is_inflated(antwerp_path)

    raise FileNotFoundError(
        f"No atlas NIfTI found in {atlas_dir}. "
        f"Expected {_ALLEN_NIFTI} or {_ATLAS_NIFTI}"
    )



# Source-localization's threshold for the inflated-header convention: true-unit
# mouse atlases have sub-0.3 mm voxels, inflated ones 0.8-2 mm.
_INFLATED_ZOOM_THRESHOLD_MM = 0.5
_REGISTRY_FILE = "registry.yaml"
# Names source-localization still accepts for atlases that were renamed.
_LEGACY_ATLAS_ALIASES = {"full": "antwerp", "coarse_22roi": "coarse22"}
# Atlases the fixed per-directory file names genuinely describe, for when no
# registry is reachable. Anything else must be registered or named explicitly.
_LEGACY_ATLAS_DIRS = {"allen": "allen", "allen32": "allen", "antwerp": "."}
_ATLAS_FILE_KEYS = ("labels", "roi_mapping", "roi_categories", "brain_volume", "brain_mask")


def header_is_inflated(nii_path: str | Path) -> bool:
    """True if this NIfTI stores the 10x-inflated voxel sizes.

    Mirrors ``source_localization.utils.atlas.header_is_inflated``. Applying the
    correction to a true-unit file is silent -- the affine stays well formed and
    every coordinate lands 10x away -- so the convention is read, not assumed.
    """
    import nibabel as nib

    zooms = nib.load(str(nii_path)).header.get_zooms()[:3]
    return bool(max(float(z) for z in zooms) >= _INFLATED_ZOOM_THRESHOLD_MM)


@dataclass(frozen=True)
class AtlasSpec:
    """One atlas, as the set of files that define it.

    Everything that needs atlas data accepts one of these wherever it used to
    accept an atlas directory. ``roi_categories``, ``brain_volume`` and
    ``brain_mask`` are optional: allen64 and coarse22 ship no category file.
    """

    name: str | None
    labels: Path
    roi_mapping: Path
    roi_categories: Path | None = None
    brain_volume: Path | None = None
    brain_mask: Path | None = None

    @property
    def labels_inflated(self) -> bool:
        return header_is_inflated(self.labels)


@lru_cache(maxsize=None)
def _read_registry(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _registry_path(base_dir: Path) -> Path | None:
    p = Path(base_dir) / _REGISTRY_FILE
    return p if p.is_file() else None


def _registry_file(entry_path: str, registry: Path) -> Path:
    """Registry paths are relative to the source-localization package root
    (``data/atlas/...``); a hand-written registry may use paths relative to itself."""
    p = Path(entry_path)
    if p.is_absolute():
        return p
    pkg_root = registry.parent.parent.parent
    for root in (registry.parent, pkg_root):
        if (root / p).exists():
            return root / p
    return pkg_root / p


def registered_atlases(base_dir: str | Path | None = None) -> list[str]:
    """Names in the atlas registry (empty if none is reachable)."""
    reg = _registry_path(Path(base_dir) if base_dir is not None else find_atlas_dir())
    return sorted(_read_registry(str(reg))) if reg else []


def _normalise_files(files: dict | None, base: Path | None) -> dict:
    out: dict[str, Path] = {}
    for key, value in (files or {}).items():
        k = "labels" if key == "brain_labels" else key
        if k not in _ATLAS_FILE_KEYS:
            raise ValueError(
                f"atlas_files: unknown key '{key}'. Known: brain_labels (or labels), "
                "roi_mapping, roi_categories, brain_volume, brain_mask.")
        if value is None:
            continue
        p = Path(value)
        out[k] = p if (p.is_absolute() or base is None) else base / p
    return out


def _spec_from_dir(atlas_dir: Path, name: str | None) -> AtlasSpec:
    """The legacy convention: fixed file names inside one directory."""
    labels, _ = _find_atlas_nifti(atlas_dir)
    cats = atlas_dir / _ROI_CATEGORIES_FILE

    def _near(fname: str) -> Path | None:
        for d in (atlas_dir, atlas_dir.parent):
            if (d / fname).exists():
                return d / fname
        return None

    return AtlasSpec(
        name=name, labels=labels, roi_mapping=atlas_dir / _ROI_MAPPING_FILE,
        roi_categories=cats if cats.exists() else None,
        brain_volume=_near("Atlas_3DRois.nii"),
        brain_mask=_near("Atlas_3DRois_brain.nii.gz"),
    )


def resolve_atlas(
    atlas: "AtlasSpec | str | Path | None" = None,
    *,
    atlas_name: str | None = None,
    files: dict | None = None,
) -> AtlasSpec:
    """Resolve an atlas to its own file set.

    Parameters
    ----------
    atlas : AtlasSpec, directory, or None
        A spec is returned as is (with any ``files`` overrides applied). A
        directory is the atlas base: its ``registry.yaml`` is used when present,
        otherwise the legacy fixed file names inside it. None means the
        source-localization atlas data.
    atlas_name : str, optional
        A registered atlas (renamed aliases accepted).
    files : dict, optional
        Explicit files, for an atlas that is not registered or to override one
        that is: ``brain_labels`` (or ``labels``), ``roi_mapping``,
        ``roi_categories``, ``brain_volume``, ``brain_mask``. Relative paths are
        taken relative to the atlas base directory.

    Raises
    ------
    ValueError
        ``atlas_name`` cannot be resolved to its own files. Falling back to
        whatever files share a directory is how allen32's partition was applied
        to allen26 data without a word.
    FileNotFoundError
        The atlas files do not exist.
    """
    if isinstance(atlas, AtlasSpec):
        overrides = _normalise_files(files, None)
        return replace(atlas, **overrides) if overrides else atlas

    base = Path(atlas) if atlas is not None else find_atlas_dir()
    if not base.is_dir():
        raise FileNotFoundError(f"Atlas directory not found: {base}")
    overrides = _normalise_files(files, base)
    name = _LEGACY_ATLAS_ALIASES.get(atlas_name, atlas_name) if atlas_name else None
    explicit = "labels" in overrides and "roi_mapping" in overrides

    spec: AtlasSpec | None = None
    registry = _registry_path(base)
    if name is not None and registry is not None:
        entries = _read_registry(str(registry))
        if name in entries:
            ins = entries[name].get("inputs", {})
            opt = {k: _registry_file(ins[k], registry)
                   for k in ("roi_categories", "brain_volume", "brain_mask") if ins.get(k)}
            spec = AtlasSpec(
                name=atlas_name,
                labels=_registry_file(ins["brain_labels"], registry),
                roi_mapping=_registry_file(ins["roi_mapping"], registry),
                **opt)
        elif not explicit:
            raise ValueError(
                f"Atlas '{atlas_name}' is not in {registry}. Registered: "
                f"{', '.join(sorted(entries))}. Register it there, or name its files "
                "under atlas_files: (brain_labels, roi_mapping, roi_categories).")
    if spec is None and explicit:
        spec = AtlasSpec(name=atlas_name, labels=overrides["labels"],
                         roi_mapping=overrides["roi_mapping"])
    if spec is None:
        if name is None:
            spec = _spec_from_dir(base, None)
        elif (base / name).is_dir():
            spec = _spec_from_dir(base / name, atlas_name)
        elif name in _LEGACY_ATLAS_DIRS and (base / _LEGACY_ATLAS_DIRS[name]).is_dir():
            spec = _spec_from_dir(base / _LEGACY_ATLAS_DIRS[name], atlas_name)
        else:
            raise ValueError(
                f"Cannot resolve atlas '{atlas_name}' under {base}: no registry.yaml "
                "there, and its files are not identifiable by name. Name them under "
                "atlas_files: (brain_labels, roi_mapping, roi_categories).")
    if overrides:
        spec = replace(spec, **overrides)
    for f in (spec.labels, spec.roi_mapping):
        if not Path(f).exists():
            raise FileNotFoundError(f"Atlas file not found: {f}")
    return spec


def load_atlas(
    atlas_dir: str | Path,
    *,
    raw_affine: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load atlas NIfTI and return label data with appropriate affine.

    Parameters
    ----------
    atlas_dir : AtlasSpec, str or Path
        An atlas: anything :func:`resolve_atlas` accepts.
    raw_affine : bool, default False
        If True, return the raw NIfTI affine without any correction.
        Use this when source coordinates are in the same uncorrected frame
        as the atlas (e.g., coordinates from source-localization pipeline).
        If False (default), apply the 10x voxel correction when the labels
        header is inflated (read from the header; see :func:`header_is_inflated`).

    Returns
    -------
    label_data : ndarray
        3D integer array of ROI label indices.
    affine : ndarray, shape (4, 4)
        Affine matrix (corrected or raw depending on *raw_affine*).
    """
    import nibabel as nib

    spec = resolve_atlas(atlas_dir)
    nii_path = spec.labels
    needs_correction = spec.labels_inflated

    nii = nib.load(str(nii_path))
    label_data = np.asarray(nii.dataobj, dtype=np.int32)

    if raw_affine or not needs_correction:
        return label_data, nii.affine.copy()

    # Apply the 10x voxel correction (inflated header)
    true_affine = nii.affine.copy()
    true_affine[:3, :3] *= ATLAS_VOXEL_SCALE_FACTOR
    true_affine[:3, 3] *= ATLAS_VOXEL_SCALE_FACTOR

    return label_data, true_affine


def load_roi_mapping(atlas_dir: str | Path) -> dict:
    """Load ROI mapping from roi_mapping.json.

    Parameters
    ----------
    atlas_dir : str or Path
        Directory containing roi_mapping.json.

    Returns
    -------
    dict
        ROI mapping: label_id (str) -> {abbreviation, name, category, color, ...}.
    """
    mapping_path = Path(resolve_atlas(atlas_dir).roi_mapping)
    if not mapping_path.exists():
        raise FileNotFoundError(f"ROI mapping not found: {mapping_path}")

    with open(mapping_path) as f:
        return json.load(f)


def load_roi_categories(atlas_dir: str | Path) -> dict[str, list[str]]:
    """Load canonical ROI categories from the atlas directory.

    Reads the atlas's OWN category file (``AtlasSpec.roi_categories``). Returns
    an empty dict if the atlas ships none (studies can define their own via config).

    Parameters
    ----------
    atlas_dir : str or Path
        Directory containing roi_categories.yaml (same dir as roi_mapping.json).

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name -> list of ROI names.
    """
    categories_path = resolve_atlas(atlas_dir).roi_categories
    if categories_path is None or not Path(categories_path).exists():
        return {}
    with open(categories_path) as f:
        raw = yaml.safe_load(f) or {}
    # Drop reserved metadata keys (e.g., backward-compat alias maps) so they
    # are not mistaken for an extra category by downstream consumers.
    return {k: v for k, v in raw.items() if k != "deprecated_aliases"}


def load_vertex_roi_labels(
    coords_mm: np.ndarray,
    atlas_dir: str | Path,
) -> list[str]:
    """Map vertex coordinates (mm) to atlas ROI abbreviations.

    Uses nearest-neighbor lookup: converts mm → voxel indices via the inverse
    corrected affine, clips to volume bounds, and returns the ROI abbreviation
    for each vertex.

    Parameters
    ----------
    coords_mm : ndarray, shape (n_vertices, 3)
        Vertex coordinates in mm.
    atlas_dir : str or Path
        Atlas directory.

    Returns
    -------
    list[str]
        ROI abbreviation per vertex (e.g., "FrA_L", "S1_R", "Exterior").
    """
    label_data, true_affine = load_atlas(atlas_dir)
    roi_mapping = load_roi_mapping(atlas_dir)

    # Build label_id -> abbreviation lookup
    id_to_abbr = {}
    for label_id_str, info in roi_mapping.items():
        label_id = int(label_id_str)
        id_to_abbr[label_id] = info.get("abbreviation", f"ROI_{label_id}")

    # mm -> voxel via inverse affine
    inv_affine = np.linalg.inv(true_affine)
    ones = np.ones((coords_mm.shape[0], 1))
    coords_hom = np.hstack([coords_mm, ones])  # (n, 4)
    voxel_coords = (inv_affine @ coords_hom.T).T[:, :3]  # (n, 3)

    # Round to nearest voxel and clip
    voxel_idx = np.round(voxel_coords).astype(int)
    for dim in range(3):
        voxel_idx[:, dim] = np.clip(voxel_idx[:, dim], 0, label_data.shape[dim] - 1)

    # Look up labels
    labels = []
    for i in range(len(voxel_idx)):
        x, y, z = voxel_idx[i]
        label_id = int(label_data[x, y, z])
        abbr = id_to_abbr.get(label_id, f"Unknown_{label_id}")
        labels.append(abbr)

    n_labeled = sum(1 for lbl in labels if lbl != "Exterior" and not lbl.startswith("Unknown"))
    logger.info(
        "Atlas labeling: %d/%d vertices mapped to named ROIs",
        n_labeled, len(labels),
    )

    return labels


# ---------------------------------------------------------------------------
# ROI extraction from vertex-level source time courses
# Ported from source_localization/steps/roi_extraction.py
# ---------------------------------------------------------------------------


def _map_sources_to_rois_nearest(
    source_coords_mm: np.ndarray,
    label_data: np.ndarray,
    affine: np.ndarray,
    label_to_roi: dict[int, str],
) -> dict[str, list[int]]:
    """Map sources to ROIs using nearest labeled voxel assignment.

    Each source is assigned to exactly one ROI.  If the voxel at the source
    coordinate is unlabeled, falls back to the nearest labeled voxel via
    KD-tree.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
    label_data : ndarray, 3-D integer label volume
    affine : ndarray, shape (4, 4)
    label_to_roi : dict mapping label_id (int) -> ROI name (str)

    Returns
    -------
    dict[str, list[int]]
        ROI name -> list of source indices.
    """
    roi_source_mapping: dict[str, list[int]] = {}

    # Convert source coordinates to voxel indices
    affine_inv = np.linalg.inv(affine)
    source_hom = np.column_stack([source_coords_mm, np.ones(len(source_coords_mm))])
    source_voxels = (affine_inv @ source_hom.T).T[:, :3]
    source_voxels = np.round(source_voxels).astype(int)

    # Clip to volume bounds
    for i in range(3):
        source_voxels[:, i] = np.clip(source_voxels[:, i], 0, label_data.shape[i] - 1)

    # Build KD-tree of labeled voxel coordinates for fallback lookup
    labeled_voxel_indices = np.argwhere(label_data > 0)
    labeled_voxel_hom = np.column_stack([
        labeled_voxel_indices, np.ones(len(labeled_voxel_indices))
    ])
    labeled_mm = (affine @ labeled_voxel_hom.T).T[:, :3]
    labeled_labels = label_data[
        labeled_voxel_indices[:, 0],
        labeled_voxel_indices[:, 1],
        labeled_voxel_indices[:, 2],
    ].astype(int)
    tree = cKDTree(labeled_mm)

    n_fallback = 0
    for source_idx in range(len(source_coords_mm)):
        voxel = source_voxels[source_idx]
        label_id = int(label_data[voxel[0], voxel[1], voxel[2]])

        if label_id not in label_to_roi:
            _, nearest_idx = tree.query(source_coords_mm[source_idx])
            label_id = int(labeled_labels[nearest_idx])
            n_fallback += 1

        if label_id in label_to_roi:
            roi_name = label_to_roi[label_id]
            roi_source_mapping.setdefault(roi_name, []).append(source_idx)

    if n_fallback > 0:
        logger.info(
            "Nearest-labeled fallback used for %d/%d sources",
            n_fallback, len(source_coords_mm),
        )

    return roi_source_mapping


def _map_sources_to_rois_proximity(
    source_coords_mm: np.ndarray,
    label_data: np.ndarray,
    affine: np.ndarray,
    label_to_roi: dict[int, str],
    radius_mm: float,
) -> dict[str, list[int]]:
    """Map sources to ROIs using proximity-based assignment.

    Sources are assigned to **all** ROIs within *radius_mm*, so a source
    near an ROI boundary can contribute to multiple ROIs.

    Parameters
    ----------
    source_coords_mm : ndarray, shape (n_sources, 3)
    label_data : ndarray, 3-D integer label volume
    affine : ndarray, shape (4, 4)
    label_to_roi : dict mapping label_id (int) -> ROI name (str)
    radius_mm : float

    Returns
    -------
    dict[str, list[int]]
        ROI name -> list of source indices.
    """
    roi_source_mapping: dict[str, list[int]] = {}

    # Get all labeled voxels and their labels
    label_voxels = np.argwhere(label_data > 0)
    label_ids = label_data[
        label_voxels[:, 0], label_voxels[:, 1], label_voxels[:, 2]
    ].astype(int)

    # Convert voxel coordinates to mm
    label_voxels_hom = np.column_stack([label_voxels, np.ones(len(label_voxels))])
    label_coords_mm = (affine @ label_voxels_hom.T).T[:, :3]

    # Build KD-tree for fast spatial queries
    tree = cKDTree(label_coords_mm)

    for source_idx, source_coord in enumerate(source_coords_mm):
        indices = tree.query_ball_point(source_coord, r=radius_mm)
        nearby_rois: set[str] = set()
        for idx in indices:
            lid = label_ids[idx]
            if lid in label_to_roi:
                nearby_rois.add(label_to_roi[lid])
        for roi_name in nearby_rois:
            roi_source_mapping.setdefault(roi_name, []).append(source_idx)

    return roi_source_mapping


def label_vertices_to_rois(
    coords_mm: np.ndarray,
    atlas_dir: str | Path,
    *,
    include_categories: list[str] | None = None,
) -> list[str | None]:
    """Assign each vertex to one atlas ROI (name) via nearest labeled voxel.

    Same assignment path as :func:`extract_roi_timeseries` (raw affine, nested
    ``rois`` mapping, nearest-labeled fallback), but returns one ROI *name* per
    vertex — used to describe which regions a significant cluster covers.

    Parameters
    ----------
    coords_mm : ndarray, shape (n_vertices, 3)
        Vertex coordinates in mm (same frame as the atlas).
    atlas_dir : str or Path
    include_categories : list[str], optional
        Restrict assignable ROIs to these atlas categories (None = all ROIs).

    Returns
    -------
    list[str | None]
        ROI name per vertex (``None`` only if a vertex mapped to no labeled ROI).
    """
    atlas_dir = resolve_atlas(atlas_dir)
    label_data, affine = load_atlas(atlas_dir, raw_affine=True)
    roi_mapping = load_roi_mapping(atlas_dir)
    rois = roi_mapping.get("rois", roi_mapping)

    included_roi_ids: set[int] | None = None
    if include_categories:
        categories = roi_mapping.get("categories", {})
        included_roi_ids = set()
        for category in include_categories:
            included_roi_ids.update(categories.get(category, []))

    SKIP_LABEL_NAMES = {"Background", "Exterior"}
    label_to_roi: dict[int, str] = {}
    for roi_id_str, roi_info in rois.items():
        label_id = int(roi_id_str)
        roi_name = roi_info.get("name", roi_info.get("abbreviation", f"ROI_{label_id}"))
        if roi_name in SKIP_LABEL_NAMES:
            continue
        if included_roi_ids is not None and label_id not in included_roi_ids:
            continue
        label_to_roi[label_id] = roi_name

    coords_mm = np.asarray(coords_mm, dtype=float)
    mapping = _map_sources_to_rois_nearest(coords_mm, label_data, affine, label_to_roi)
    per_vertex: list[str | None] = [None] * len(coords_mm)
    for roi_name, idxs in mapping.items():
        for i in idxs:
            per_vertex[i] = roi_name
    return per_vertex


def format_region_coverage(roi_labels: list[str | None], threshold: float = 0.05) -> str:
    """Describe a cluster's regional composition as ``'ROI x%, ROI y% (+N ROIs ≤5%)'``.

    Lists every ROI whose share of the cluster's vertices exceeds *threshold*
    (default 5%), ordered by share, and tallies the remaining minor ROIs. Shares
    are of the total cluster vertex count. Returns '' for an empty cluster.
    """
    from collections import Counter

    n = len(roi_labels)
    if n == 0:
        return ""
    counts = Counter(r for r in roi_labels if r)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    listed = [(roi, c) for roi, c in ranked if c / n > threshold]
    n_minor = len(ranked) - len(listed)
    pct = int(round(threshold * 100))
    parts = [f"{roi} {round(100 * c / n)}%" for roi, c in listed]
    text = ", ".join(parts)
    if n_minor > 0:
        tail = f"+{n_minor} ROIs ≤{pct}%" if text else f"{n_minor} ROIs ≤{pct}%"
        text = f"{text} ({tail})" if text else tail
    return text


def extract_roi_timeseries(
    stc_data: np.ndarray,
    coords_mm: np.ndarray,
    atlas_dir: str | Path,
    *,
    method: str = "nearest",
    proximity_radius_mm: float = 2.0,
    include_categories: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract ROI time series from vertex-level source time courses.

    This replicates the core logic of source-localization's step 6
    (``roi_extraction.py``) so that ROI analyses can run from any source
    model that produces step5 data, without requiring pre-extracted step6
    files.

    Parameters
    ----------
    stc_data : ndarray, shape (n_vertices, n_times)
        Source-space time courses (signed or magnitude).
    coords_mm : ndarray, shape (n_vertices, 3)
        Vertex coordinates in mm (same coordinate frame as atlas).
    atlas_dir : str or Path
        Directory containing the atlas NIfTI and ``roi_mapping.json``.
    method : ``"nearest"`` | ``"proximity"``, default ``"nearest"``
        Source-to-ROI assignment strategy.
    proximity_radius_mm : float, default 2.0
        Radius for proximity-based assignment (ignored when
        *method* = ``"nearest"``).
    include_categories : list[str], optional
        If provided, only include ROIs belonging to these atlas categories.

    Returns
    -------
    dict[str, ndarray]
        Mapping of ROI name -> 1-D time course (mean across assigned
        sources).  Only ROIs with at least one assigned source are
        included.
    """
    atlas_dir = resolve_atlas(atlas_dir)

    # Load atlas with RAW affine — source coordinates from the pipeline
    # are in the same uncorrected frame as the original NIfTI headers.
    label_data, affine = load_atlas(atlas_dir, raw_affine=True)
    roi_mapping = load_roi_mapping(atlas_dir)

    # Build label_id -> ROI name mapping
    rois = roi_mapping.get("rois", roi_mapping)

    # Optional category filtering
    included_roi_ids: set[int] | None = None
    if include_categories:
        categories = roi_mapping.get("categories", {})
        included_roi_ids = set()
        for category in include_categories:
            if category in categories:
                included_roi_ids.update(categories[category])

    SKIP_LABEL_NAMES = {"Background", "Exterior"}
    label_to_roi: dict[int, str] = {}
    for roi_id_str, roi_info in rois.items():
        label_id = int(roi_id_str)
        roi_name = roi_info.get("name", roi_info.get("abbreviation", f"ROI_{label_id}"))
        if roi_name in SKIP_LABEL_NAMES:
            continue
        if included_roi_ids is not None and label_id not in included_roi_ids:
            continue
        label_to_roi[label_id] = roi_name

    # Truncate if length mismatch
    n_stc = len(stc_data)
    n_coords = len(coords_mm)
    if n_stc != n_coords:
        logger.warning(
            "Source count mismatch: stc=%d, coords=%d — truncating to %d",
            n_stc, n_coords, min(n_stc, n_coords),
        )
        n = min(n_stc, n_coords)
        stc_data = stc_data[:n]
        coords_mm = coords_mm[:n]

    # Map sources to ROIs
    if method == "proximity":
        roi_source_mapping = _map_sources_to_rois_proximity(
            coords_mm, label_data, affine, label_to_roi, proximity_radius_mm,
        )
    else:
        roi_source_mapping = _map_sources_to_rois_nearest(
            coords_mm, label_data, affine, label_to_roi,
        )

    # Average source activity within each ROI
    roi_ts: dict[str, np.ndarray] = {}
    for roi_name, source_indices in roi_source_mapping.items():
        if source_indices:
            roi_ts[roi_name] = stc_data[source_indices, :].mean(axis=0)

    logger.info(
        "ROI extraction: %d ROIs with sources out of %d atlas ROIs",
        len(roi_ts), len(set(label_to_roi.values())),
    )

    return roi_ts

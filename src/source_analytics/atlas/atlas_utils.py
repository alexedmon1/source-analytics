"""Atlas utilities: load atlas NIfTI, map vertex coordinates to ROI labels.

Replicates the 10x voxel-size correction from source_localization/utils/atlas.py.
The Atlas_3DRoisLeftRight.Labels.nii header has voxel sizes 10x larger than reality,
so we apply ATLAS_VOXEL_SCALE_FACTOR = 0.1 to both the rotation/scaling block
and the translation vector of the affine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ATLAS_VOXEL_SCALE_FACTOR = 0.1
_ATLAS_NIFTI = "Atlas_3DRoisLeftRight.Labels.nii"
_ROI_MAPPING_FILE = "roi_mapping.json"


def find_atlas_dir(config_atlas_dir: str | Path | None = None) -> Path:
    """Locate the atlas data directory.

    Parameters
    ----------
    config_atlas_dir : str or Path, optional
        Explicit atlas directory from config. If None, falls back to the
        source_localization package's bundled atlas.

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
    try:
        import source_localization

        pkg_dir = Path(source_localization.__file__).parent
        atlas_dir = pkg_dir / "data" / "atlas"
        if atlas_dir.is_dir():
            return atlas_dir
    except ImportError:
        pass

    # Fallback: well-known path
    fallback = Path(
        "/home/edm9fd/sandbox/AlexProjects/mouse-eeg-source-localization"
        "/source_localization/src/source_localization/data/atlas"
    )
    if fallback.is_dir():
        return fallback

    raise FileNotFoundError(
        "Cannot find atlas directory. Set atlas_dir in config or install source_localization."
    )


def load_atlas(atlas_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load atlas NIfTI and return label data with corrected affine.

    Parameters
    ----------
    atlas_dir : str or Path
        Directory containing the atlas NIfTI file.

    Returns
    -------
    label_data : ndarray
        3D integer array of ROI label indices.
    true_affine : ndarray, shape (4, 4)
        Corrected affine matrix (10x voxel scaling applied).
    """
    import nibabel as nib

    atlas_dir = Path(atlas_dir)
    nii_path = atlas_dir / _ATLAS_NIFTI
    if not nii_path.exists():
        raise FileNotFoundError(f"Atlas NIfTI not found: {nii_path}")

    nii = nib.load(str(nii_path))
    label_data = np.asarray(nii.dataobj, dtype=np.int32)

    # Apply 10x voxel correction
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
    atlas_dir = Path(atlas_dir)
    mapping_path = atlas_dir / _ROI_MAPPING_FILE
    if not mapping_path.exists():
        raise FileNotFoundError(f"ROI mapping not found: {mapping_path}")

    with open(mapping_path) as f:
        return json.load(f)


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

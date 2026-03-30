# Figure Library

Reference for all visualization functions in `source_analytics.viz`.

## Brain ROI Mosaics (`brain_roi.py`)

### `plot_significance_mosaic`
- **Purpose**: ROI-level p-value map on atlas slices (coronal, axial, sagittal)
- **Colormap**: Sequential (YlOrRd), internally 1-p, labeled as p
- **Layout**: 1 row (all ROIs) or 2 rows (+ FDR-corrected threshold row)
- **Input**: DataFrame with `roi`, `p_value` columns
- **Correction**: Benjamini-Hochberg FDR applied internally across ROIs
- **Typical filename**: `significance_{band}_{power_type}.png`, `significance_aperiodic_{dv}.png`

### `plot_effect_size_mosaic`
- **Purpose**: ROI-level Hedges' g map on atlas slices with diverging colormap
- **Colormap**: Diverging (RdBu_r), blue = KO < WT, red = KO > WT
- **Layout**: 1 row (all ROIs) or 2 rows (+ FDR-corrected threshold row)
- **Input**: DataFrame with `roi`, `hedges_g`, `p_value` columns
- **Correction**: Benjamini-Hochberg FDR applied internally across ROIs
- **Typical filename**: `effect_size_{band}_{power_type}.png`, `effect_size_aperiodic_{dv}.png`

### `plot_brain_roi_mosaic`
- **Purpose**: 3x3 slice mosaic (3 coronal, 3 axial, 3 sagittal) colored by region-level scalar
- **Colormap**: Diverging (configurable), symmetric around vcenter
- **Layout**: 3 rows x 3 cols + colorbar
- **Input**: `region_values` dict + `roi_categories` dict
- **Typical filename**: `brain_roi_{facet}.png`

### `render_posthoc_mosaics`
- **Purpose**: Batch wrapper — reads a posthoc CSV and generates one `plot_brain_roi_mosaic` per facet group
- **Input**: Posthoc CSV path + facet columns

### `plot_brain_roi`
- **Purpose**: 3D rendered views (dorsal, lateral, posterior) using PyVista
- **Requires**: PyVista (optional dependency)

## Connectivity (`connectivity_plots.py`)

### `plot_circos` / `plot_significance_circos`
- **Purpose**: Circular connectivity diagrams showing ROI-to-ROI connections
- **Typical filename**: `circos_{metric}_{band}.png`

### `plot_connectivity_heatmap` / `plot_connectivity_comparison`
- **Purpose**: Matrix heatmaps of connectivity metrics

## Glass Brain (`glass_brain.py`)

### `plot_glass_brain`
- **Purpose**: Vertex-level maps projected onto transparent brain outline

### `plot_vertex_cluster_summary`
- **Purpose**: Cluster-corrected vertex results

## Other

### `plot_radar` (`radar.py`)
- **Purpose**: Radar/spider plots for multi-band comparisons

## Shared Utilities

### `fdr_bh` (`brain_roi.py`)
- **Purpose**: Benjamini-Hochberg FDR correction
- **Input**: Array of p-values
- **Output**: Array of q-values

### Palettes (`palettes.py`)
- `ANALYSIS_CMAPS`: per-analysis colormap defaults
- `get_diverging_cmap`, `get_sequential_cmap`: colormap accessors

### Figure Registry (`figure_registry.py`)
- `generate_figure(analysis, fig_type, ...)`: dispatch to registered generators
- `list_figure_types()`: list available (analysis, fig_type) combinations
- `TABLE_SCHEMAS`: column schema per analysis for reading posthoc CSVs

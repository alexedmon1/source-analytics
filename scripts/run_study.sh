#!/usr/bin/env bash
# Run a full source-analytics study in dependency order (primaries before the
# supplements that read their output). This is the canonical recipe the README
# refers to; trim the lists to the analyses your config declares.
#
#   scripts/run_study.sh study.yaml [extra `run` flags, e.g. --jobs -1 --steps ...]
#
# Paradigm names (resting / vertex / evoked) must match the `paradigms:` keys in
# your study YAML. fcd_comparison reads electrode_connectivity (resting) AND
# vertex_connectivity (vertex), so it runs last. Figures are OFF by default; add
# --steps setup,process,aggregate,statistics,figures,summary (or run
# `source-analytics figure ...`) to render them.
set -euo pipefail

STUDY="${1:?usage: $0 study.yaml [run flags...]}"; shift || true
SA=(source-analytics run --study "$STUDY" "$@")

run() { local paradigm="$1" analysis="$2"; echo "=== $paradigm / $analysis ==="; "${SA[@]}" --paradigm "$paradigm" --analysis "$analysis"; }

# Resting paradigm — ROI + electrode
run resting roi_psd
run resting roi_aperiodic
run resting roi_connectivity        # PRIMARY
run resting roi_graph               #   after roi_connectivity
run resting roi_nbs                 #   after roi_connectivity
run resting roi_cross_freq          # PAC + AAC + PPC   (--metric to pick one)
run resting roi_directed            # transfer entropy + DTF (--metric te|dtf)
run resting electrode_psd           # PRIMARY
run resting electrode_aperiodic
run resting electrode_comparison    #   after electrode_psd AND roi_psd
run resting electrode_connectivity  # sensor FC comparator
run resting electrode_signature     #   after electrode_psd

# Vertex paradigm — whole-brain
run vertex vertex_connectivity      # PRIMARY (slow; computes matrices)
run vertex vertex_graph             #   after vertex_connectivity
run vertex vertex_nbs               #   after vertex_connectivity
run vertex vertex_cluster
run vertex vertex_specparam
run vertex vertex_signature
run vertex vertex_cross_freq        # local PAC + AAC + PPC
run vertex vertex_directed          # vertex DTF (outflow/inflow/netflow)

# Source-vs-sensor FCD comparison (cross-paradigm: reads resting + vertex output)
run resting fcd_comparison

# Evoked paradigm (trial-based data only) — uncomment if your study has one
# run evoked roi_evoked
# run evoked vertex_evoked
# run evoked electrode_evoked

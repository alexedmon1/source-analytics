"""Outcome tables the Bayesian engine knows: where a module's per-subject long table lives and how it is shaped."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BayesTable:
    """Where a module's per-subject long table lives and how it is shaped."""

    csv: str
    dvs: tuple[str, ...]
    spatial_col: str = "roi"
    band_col: str | None = "band"


# Modules the adapter knows (G1: FORGE rest). Extend as outcome families arrive (G4).
BAYES_TABLES: dict[str, BayesTable] = {
    "roi_psd": BayesTable("band_power.csv", ("absolute", "relative", "delta_ref")),
    "roi_aperiodic": BayesTable("aperiodic_params.csv", ("exponent", "offset"), band_col=None),
}

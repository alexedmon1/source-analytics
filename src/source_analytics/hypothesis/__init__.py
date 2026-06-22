"""The ``hypothesis`` inference layer — permutation adapter (Python side).

This package is the Python half of the shared ``hypothesis`` layer (the R half is
``R/hypothesis.R``). It does NOT define a registry analysis module; vertex/network
modules call into it from their ``statistics()`` step.

The spec dataclasses (``Hypothesis`` / ``DesignSpec``) live in ``config.py`` and are
re-exported here for convenience. The permutation adapter — group-contrast and omnibus
tests over per-subject vertex maps, with a map+cluster result contract — lives in
``permutation.py``. See ``DESIGN_SPEC.md`` and ``HYPOTHESIS.md``.
"""

from __future__ import annotations

from ..config import DesignSpec, Hypothesis
from .permutation import (
    run_hypothesis_permutation,
    write_module_hypotheses_perm,
)

__all__ = [
    "DesignSpec",
    "Hypothesis",
    "run_hypothesis_permutation",
    "write_module_hypotheses_perm",
]

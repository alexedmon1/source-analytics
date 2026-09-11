"""Guard the standard: figures() must be regenerable from PERSISTED data via
`--steps figures` alone — never dependent on in-memory state carried within one
process. These are structural guards (cheap, catch regressions) complementing the
end-to-end reload checks; the vertex modules' guards moved to the
source-analytics-vertex plugin.
"""

import inspect

from source_analytics.analyses.base import BaseAnalysis
def test_base_has_cluster_state_helpers():
    assert callable(getattr(BaseAnalysis, "_save_cluster_state", None))
    assert callable(getattr(BaseAnalysis, "_load_cluster_state", None))

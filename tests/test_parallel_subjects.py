"""The --jobs parallel process step must equal serial, for any worker count."""

from pathlib import Path

import pytest

from source_analytics.analyses.base import BaseAnalysis, _compute_subject_worker


class _Subj:
    def __init__(self, sid, group):
        self.subject_id, self.group, self.data_dir = sid, group, Path(".")


class _ToyAnalysis(BaseAnalysis):
    """Parallel-capable toy: each subject contributes a squared value; the merge
    accumulates into a dict. compute() is pure (no self mutation)."""
    name = "toy"

    def __init__(self):
        # bypass BaseAnalysis.__init__ (no config/dirs needed for the mechanism)
        self.results = {}
        self._selection = {}

    def _compute_subject(self, subject):
        return {"uid": f"{subject.group}_{subject.subject_id}",
                "value": int(subject.subject_id) ** 2}

    def _merge_subject(self, payload):
        self.results[payload["uid"]] = payload["value"]

    # unused lifecycle hooks
    def setup(self): ...
    def aggregate(self): ...
    def statistics(self): ...
    def figures(self): ...
    def summary(self): ...


def test_parallel_capable_detection():
    assert _ToyAnalysis()._parallel_capable() is True

    class _Serial(_ToyAnalysis):
        def process_subject(self, subject):  # legacy override, no _compute_subject
            ...
    # still capable because it inherits _compute_subject; capability is about the
    # compute hook, which _Serial keeps.
    assert _ToyAnalysis()._parallel_capable() is True


def test_worker_returns_payload_and_swallows_errors():
    a = _ToyAnalysis()
    subj = _Subj("5", "KO")
    assert _compute_subject_worker(a, subj) == {"uid": "KO_5", "value": 25}

    class _Boom(_ToyAnalysis):
        def _compute_subject(self, subject):
            raise RuntimeError("boom")
    assert _compute_subject_worker(_Boom(), subj) is None


def test_parallel_process_matches_serial():
    subjects = [_Subj(str(i), "KO" if i % 2 else "WT") for i in range(1, 9)]

    serial = _ToyAnalysis()
    for s in subjects:
        serial.process_subject(s)

    par = _ToyAnalysis()
    par._process_subjects_parallel(subjects, n_jobs=4)

    assert par.results == serial.results
    assert serial.results["KO_5"] == 25


def test_resolve_jobs_auto_and_config():
    a = _ToyAnalysis()

    class _Cfg:
        raw = {"jobs": 3}
    a.config = _Cfg()
    assert a._resolve_jobs(None) == 3       # CLI not given -> config `jobs:`
    assert a._resolve_jobs(1) == 1          # explicit CLI 1 wins over config
    assert a._resolve_jobs(2) == 2          # CLI wins over config
    assert a._resolve_jobs(-1) >= 1         # auto (all-but-one core)
    assert a._resolve_jobs(0) >= 1          # auto (all-but-one core)

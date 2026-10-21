# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

"""Pin the suspicious_leakage guard behavior in runner.py.

Regression test for the Stage 3 diagnosis: run_results.id=24 (Telco B1,
test_score=1.0) escaped the guard because the guard commit landed after
id=24 was created. Guard is correct as written; this test freezes it so
the next refactor cannot silently re-introduce a boundary bug.

Stubbed via monkeypatch: no DB, no subprocess, no fixtures.
"""

from types import SimpleNamespace

from src.contracts import ErrorCategory, GeneratedPipeline, TaskType
from src.execution import runner


def _pipeline() -> GeneratedPipeline:
    return GeneratedPipeline(
        code="print('SCORE=stub')",
        condition="B1",
        llm_backend="stub",
        dataset_id="stub-ds",
        seed=42,
        iteration=0,
    )


def _install_stubs(monkeypatch, *, score: float | None) -> None:
    """Bypass subprocess + score-parsing so tests exercise only the guard."""

    def fake_run(*_a, **_kw):
        return SimpleNamespace(stdout="SCORE=stub", stderr="", returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "extract_score", lambda *_a, **_kw: score)


def test_binary_score_at_one_is_flagged_as_leakage(monkeypatch):
    _install_stubs(monkeypatch, score=1.0)
    result = runner.execute_pipeline(_pipeline(), task_type=TaskType.BINARY_CLASSIFICATION)
    assert result.success is False
    assert result.error_category == ErrorCategory.SUSPICIOUS_LEAKAGE.value
    assert result.test_score == 1.0


def test_binary_score_at_threshold_is_flagged(monkeypatch):
    _install_stubs(monkeypatch, score=0.995)
    result = runner.execute_pipeline(_pipeline(), task_type=TaskType.BINARY_CLASSIFICATION)
    assert result.success is False
    assert result.error_category == ErrorCategory.SUSPICIOUS_LEAKAGE.value


def test_binary_score_just_below_threshold_succeeds(monkeypatch):
    _install_stubs(monkeypatch, score=0.9949)
    result = runner.execute_pipeline(_pipeline(), task_type=TaskType.BINARY_CLASSIFICATION)
    assert result.success is True
    assert result.error_category is None
    assert result.test_score == 0.9949


def test_regression_score_never_flagged(monkeypatch):
    # neg_rmse can be arbitrarily close to 0; the guard is classification-only.
    _install_stubs(monkeypatch, score=-0.0001)
    result = runner.execute_pipeline(_pipeline(), task_type=TaskType.REGRESSION)
    assert result.success is True
    assert result.error_category is None

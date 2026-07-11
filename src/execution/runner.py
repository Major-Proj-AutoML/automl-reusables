from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.contracts import ErrorCategory, GeneratedPipeline, RunResult, TaskType
from src.execution.error_taxonomy import classify_error
from src.execution.metrics import extract_score

_LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "runs"


def _safe(token: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in token)


def execute_pipeline(
    pipeline: GeneratedPipeline,
    task_type: TaskType,
    timeout_seconds: int = 600,
    python_executable: str = sys.executable,
) -> RunResult:
    """Run a GeneratedPipeline as an isolated subprocess and return a RunResult.

    The pipeline code is written to a persisted file under logs/runs/ so it can
    be inspected after the experiment. The subprocess has no shared memory with
    this process; the only channel back is stdout/stderr/returncode.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{_safe(pipeline.dataset_id)}_{pipeline.condition}"
        f"_seed{pipeline.seed}_iter{pipeline.iteration}_"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=prefix,
        delete=False,
        dir=str(_LOGS_DIR),
        encoding="utf-8",
    ) as f:
        f.write(pipeline.code)
        code_path = f.name

    start = time.monotonic()
    timed_out = False
    stdout = ""
    stderr = ""
    returncode = 0
    try:
        proc = subprocess.run(
            [python_executable, code_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(_LOGS_DIR),
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    runtime = time.monotonic() - start

    # Persist stdout and stderr as sidecars so downstream verification (trace
    # extraction, ablation, etc.) can read them without re-executing the code.
    try:
        Path(code_path + ".stdout.txt").write_text(stdout, encoding="utf-8")
        Path(code_path + ".stderr.txt").write_text(stderr, encoding="utf-8")
    except OSError:
        pass  # sidecar failures should not break the run

    common = dict(
        dataset_id=pipeline.dataset_id,
        condition=pipeline.condition,
        llm_backend=pipeline.llm_backend,
        seed=pipeline.seed,
        iteration=pipeline.iteration,
        runtime_seconds=runtime,
        generated_code_path=code_path,
    )

    if timed_out:
        return RunResult(
            success=False,
            error_category=ErrorCategory.TIMEOUT.value,
            error_message=f"Exceeded {timeout_seconds}s wall-clock limit",
            test_score=None,
            **common,
        )

    if returncode != 0:
        category, message = classify_error(stderr, returncode)
        return RunResult(
            success=False,
            error_category=category.value,
            error_message=message,
            test_score=None,
            **common,
        )

    score = extract_score(stdout, task_type)
    if score is None:
        return RunResult(
            success=False,
            error_category=ErrorCategory.RUNTIME_OTHER.value,
            error_message="Pipeline ran but did not print a valid SCORE line",
            test_score=None,
            **common,
        )

    # Sanity check: near-perfect balanced accuracy on real datasets is
    # almost always a leakage bug (target column not dropped, train/test
    # confusion, etc.). Flag it instead of treating it as a valid result.
    if task_type != TaskType.REGRESSION and score >= 0.995:
        return RunResult(
            success=False,
            error_category=ErrorCategory.SUSPICIOUS_LEAKAGE.value,
            error_message=(
                f"SCORE={score:.4f} exceeds sanity threshold (>=0.995). "
                "Likely leakage in generated code (e.g. target column not "
                "dropped from features, test data used during training)."
            ),
            test_score=score,
            **common,
        )

    return RunResult(
        success=True,
        error_category=None,
        error_message=None,
        test_score=score,
        **common,
    )

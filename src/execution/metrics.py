from __future__ import annotations

import json
import math
import re
from typing import Optional

from src.contracts import ReasoningTrace, TaskType

_SCORE_RE = re.compile(r"SCORE:\s*([-\d.]+(?:e[-+]?\d+)?)")
_REASONING_RE = re.compile(r"REASONING:\s*(\{.*\})\s*$", re.MULTILINE)


def extract_score(stdout: str, task_type: TaskType) -> Optional[float]:
    """Extract the final SCORE printed by a generated pipeline.

    Returns None if no SCORE line is found, the value is unparseable, or it
    fails a sanity check (classification: [0, 1]; regression: finite).
    """
    match = _SCORE_RE.search(stdout)
    if match is None:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None

    if task_type == TaskType.REGRESSION:
        return score if math.isfinite(score) else None
    if 0.0 <= score <= 1.0:
        return score
    return None


def extract_reasoning(stdout: str) -> Optional[ReasoningTrace]:
    """Extract the LLM's structured reasoning trace from stdout.

    Returns None if the ``REASONING: {json}`` line is missing, empty, or
    malformed. Never raises — a broken trace should propagate as ``None`` so
    the executor can flag the run rather than crash.
    """
    match = _REASONING_RE.search(stdout)
    if match is None:
        return None
    raw = match.group(1).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        return ReasoningTrace.model_validate(payload)
    except Exception:  # noqa: BLE001 pydantic ValidationError etc.
        return None

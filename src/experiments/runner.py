from __future__ import annotations

import os
import re

import requests

from src.contracts import ErrorCategory

_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_llm(prompt: str, backend: str, seed: int) -> str:
    """POST to a local Ollama server and return the generated text (fences stripped)."""
    body = {
        "model": backend,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": seed, "temperature": 0.7},
    }
    try:
        response = requests.post(_OLLAMA_URL, json=body, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to reach Ollama at {_OLLAMA_URL}. Is `ollama serve` running? Original: {exc}"
        ) from exc

    payload = response.json()
    text = payload.get("response", "")
    return _strip_fences(text)


def build_error_feedback(
    error_message: str,
    error_category: str | None = None,
    prev_code: str | None = None,
) -> str:
    """Build a feedback section to append to the base prompt after a failed attempt."""
    parts = [
        "\n\n## Previous Attempt Failed\n\n"
        "Your previous code produced the following error:\n\n"
        f"```\n{error_message}\n```\n"
    ]
    if prev_code:
        trimmed = "\n".join(prev_code.splitlines()[:40])
        parts.append(
            f"\nHere is the code that failed (first 40 lines):\n\n"
            f"```python\n{trimmed}\n```\n"
        )
    if error_category == ErrorCategory.MISSING_NAME.value:
        parts.append(
            "\nThis error is caused by using a name that was never imported. "
            "Check that EVERY class, function, and constant you reference "
            "(e.g. RobustScaler, ColumnTransformer, train_test_split, "
            "balanced_accuracy_score) has a corresponding import statement at the top.\n"
        )
    parts.append(
        "\nFix the error and return a corrected, **complete, self-contained** Python script. "
        "Make sure every class and function you use is explicitly imported at the top."
    )
    return "".join(parts)



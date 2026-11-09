# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

from __future__ import annotations

import os
import re

import requests

from src.contracts import ErrorCategory

_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OPENROUTER_DEFAULT_URL = "https://openrouter.ai/api/v1"

# Ollama used to be the only backend. Ollama's cloud models (``*-cloud``) were
# retired, so the stack now talks to OpenRouter's OpenAI-compatible API by
# default. These aliases let callers keep passing the old Ollama model names
# (e.g. the README/UI default ``gpt-oss:120b-cloud``) and still hit the right
# OpenRouter model id. Anything not in the map is sent to OpenRouter verbatim,
# so ``openai/gpt-oss-120b`` and any other OpenRouter slug also work directly.
_OPENROUTER_MODEL_ALIASES = {
    "gpt-oss:120b-cloud": "openai/gpt-oss-120b",
    "gpt-oss:120b": "openai/gpt-oss-120b",
    "gpt-oss:20b-cloud": "openai/gpt-oss-20b",
    "gpt-oss:20b": "openai/gpt-oss-20b",
}


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _openrouter_api_key() -> str:
    """Return the OpenRouter key, accepting either spelling of the env var."""
    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return ""


def _provider() -> str:
    """Which backend to call: ``openrouter`` (default) or ``ollama``.

    Honors an explicit ``LLM_PROVIDER`` env var; otherwise auto-selects
    OpenRouter when an API key is present and falls back to a local Ollama.
    """
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "openrouter" if _openrouter_api_key() else "ollama"


def _ollama_generate_url() -> str:
    base = os.environ.get("OLLAMA_URL", _OLLAMA_DEFAULT_URL)
    return base.rstrip("/") + "/api/generate"


def _openrouter_chat_url() -> str:
    base = (
        os.environ.get("OPENROUTER_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or _OPENROUTER_DEFAULT_URL
    )
    return base.rstrip("/") + "/chat/completions"


def _call_ollama(
    prompt: str, backend: str, seed: int, num_predict: int | None
) -> tuple[str, int | None, int | None]:
    """POST to a local/remote Ollama server. Returns (text, prompt_tok, completion_tok)."""
    options = {"seed": seed, "temperature": 0.7}
    if num_predict is not None:
        options["num_predict"] = num_predict
    body = {"model": backend, "prompt": prompt, "stream": False, "options": options}
    url = _ollama_generate_url()
    try:
        response = requests.post(url, json=body, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to reach Ollama at {url}. Is `ollama serve` running? Original: {exc}"
        ) from exc

    payload = response.json()
    text = _strip_fences(payload.get("response", ""))
    return text, payload.get("prompt_eval_count"), payload.get("eval_count")


def _call_openrouter(
    prompt: str, backend: str, seed: int, num_predict: int | None
) -> tuple[str, int | None, int | None]:
    """POST to OpenRouter's OpenAI-compatible chat API.

    Returns ``(text, prompt_tokens, completion_tokens)``. Network failures, auth
    errors (bad/missing key), and empty responses are surfaced as
    ``ConnectionError`` so the worker records them as ``infrastructure``
    failures instead of crashing the job.
    """
    key = _openrouter_api_key()
    if not key:
        raise ConnectionError(
            "OPENROUTER_API_KEY is not set. Add it to automl-infra/.env "
            "(OPENROUTER_API_KEY=sk-or-...) so the worker can reach OpenRouter, "
            "or set LLM_PROVIDER=ollama to use a local model instead."
        )

    model = _OPENROUTER_MODEL_ALIASES.get(backend, backend)
    url = _openrouter_chat_url()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # Optional attribution headers OpenRouter uses for its rankings page.
    referer = os.environ.get("OPENROUTER_HTTP_REFERER") or os.environ.get("OPENROUTER_SITE_URL")
    title = os.environ.get("OPENROUTER_X_TITLE") or os.environ.get("OPENROUTER_APP_NAME")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "seed": seed,
    }
    if num_predict is not None:
        # Ollama's num_predict -> OpenAI's max_tokens (completion cap).
        body["max_tokens"] = num_predict

    try:
        response = requests.post(url, json=body, headers=headers, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                detail = f" (HTTP {resp.status_code}: {resp.text[:500]})"
            except Exception:  # noqa: BLE001
                detail = ""
        raise ConnectionError(
            f"Failed to reach OpenRouter at {url} for model '{model}'.{detail} "
            f"Original: {exc}"
        ) from exc

    payload = response.json()
    choices = payload.get("choices")
    if not choices:
        err = payload.get("error") or payload
        raise ConnectionError(
            f"OpenRouter returned no choices for model '{model}': {err}"
        )

    message = choices[0].get("message") or {}
    text = _strip_fences(message.get("content") or "")
    usage = payload.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


def _generate(
    prompt: str, backend: str, seed: int, num_predict: int | None = None
) -> tuple[str, int | None, int | None]:
    """Route one generation request to the configured provider."""
    provider = _provider()
    if provider == "openrouter":
        return _call_openrouter(prompt, backend, seed, num_predict)
    if provider == "ollama":
        return _call_ollama(prompt, backend, seed, num_predict)
    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Set it to 'openrouter' or 'ollama'."
    )


def call_llm(prompt: str, backend: str, seed: int) -> str:
    """Generate code via the configured provider; return the text, fences stripped."""
    text, _p, _c = _generate(prompt, backend, seed)
    return text


def call_llm_with_usage(
    prompt: str, backend: str, seed: int, num_predict: int | None = None
) -> tuple[str, int | None, int | None]:
    """Same as :func:`call_llm` but also returns token telemetry.

    Returns ``(code_text, prompt_tokens, completion_tokens)``. Token counts come
    from the provider's usage fields (OpenRouter's ``usage.prompt_tokens`` /
    ``usage.completion_tokens``, or Ollama's ``prompt_eval_count`` /
    ``eval_count``); when a provider omits them the value is ``None`` and callers
    should persist ``NULL``.

    If ``num_predict`` is set, the request caps completion tokens (OpenRouter's
    ``max_tokens`` / Ollama's ``num_predict``). Used to guard against runaway
    generation on reasoning models for specific (model, condition) combos.
    """
    return _generate(prompt, backend, seed, num_predict)


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



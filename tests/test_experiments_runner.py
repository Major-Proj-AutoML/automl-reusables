# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

"""Unit tests for the OpenRouter LLM path and the experimental-import guard.

No network, no pytest-only features: runnable as ``pytest tests/...`` or as
``python tests/test_experiments_runner.py`` (used to smoke-test inside the
worker container, which has no pytest installed).
"""

from __future__ import annotations

import contextlib
import json
import os

from src.execution import runner as exec_runner
from src.experiments import runner


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._p = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._p

    def raise_for_status(self) -> None:
        return None


@contextlib.contextmanager
def _patched(post=None, env: dict | None = None):
    """Temporarily swap ``requests.post`` and selected env vars, then restore."""
    old_post = runner.requests.post
    old_env = {k: os.environ.get(k) for k in (env or {})}
    try:
        if post is not None:
            runner.requests.post = post
        for k, v in (env or {}).items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        runner.requests.post = old_post
        for k, v in old_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_ensure_experimental_imports_after_future() -> None:
    # The regression: enable-import must land AFTER `from __future__`, and the
    # result must still be valid Python.
    code = (
        "from __future__ import annotations\n"
        "from sklearn.impute import IterativeImputer\n"
        "print('x')\n"
    )
    out = exec_runner._ensure_experimental_imports(code)
    compile(out, "<generated>", "exec")  # raises SyntaxError if ordering is wrong
    assert "enable_iterative_imputer" in out
    assert out.index("from __future__") < out.index("enable_iterative_imputer")

    # No __future__ line -> still prepended and still valid.
    out2 = exec_runner._ensure_experimental_imports(
        "from sklearn.impute import IterativeImputer\n"
    )
    compile(out2, "<generated>", "exec")
    assert out2.startswith("from sklearn.experimental import enable_iterative_imputer")


def test_openrouter_parse_and_alias() -> None:
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured.update(url=url, body=json, headers=headers)
        return _FakeResp(
            {
                "choices": [{"message": {"content": "```python\nprint(1)\n```"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        )

    with _patched(
        post=fake_post,
        env={
            "OPENROUTER_API_KEY": "sk-test",
            "OPEN_ROUTER_API_KEY": None,
            "OPENROUTER_URL": None,
            "OPENROUTER_HTTP_REFERER": None,
            "OPENROUTER_SITE_URL": None,
            "OPENROUTER_X_TITLE": None,
            "OPENROUTER_APP_NAME": None,
        },
    ):
        text, p_tok, c_tok = runner._call_openrouter("hi", "gpt-oss:120b-cloud", 42, None)

    assert text == "print(1)"  # fences stripped
    assert (p_tok, c_tok) == (5, 7)
    assert captured["body"]["model"] == "openai/gpt-oss-120b"  # alias applied
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["url"].endswith("/chat/completions")


def test_openrouter_missing_key_raises() -> None:
    with _patched(env={"OPENROUTER_API_KEY": None, "OPEN_ROUTER_API_KEY": None}):
        try:
            runner._call_openrouter("hi", "m", 42, None)
            assert False, "expected ConnectionError"
        except ConnectionError:
            pass


def test_openrouter_no_choices_raises() -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResp({"error": {"message": "bad model"}})

    with _patched(post=fake_post, env={"OPENROUTER_API_KEY": "sk-test"}):
        try:
            runner._call_openrouter("hi", "m", 42, None)
            assert False, "expected ConnectionError"
        except ConnectionError:
            pass


def test_provider_selection() -> None:
    with _patched(env={"LLM_PROVIDER": "ollama"}):
        assert runner._provider() == "ollama"
    with _patched(
        env={"LLM_PROVIDER": None, "OPENROUTER_API_KEY": "k", "OPEN_ROUTER_API_KEY": None}
    ):
        assert runner._provider() == "openrouter"
    with _patched(
        env={"LLM_PROVIDER": None, "OPENROUTER_API_KEY": None, "OPEN_ROUTER_API_KEY": None}
    ):
        assert runner._provider() == "ollama"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
    print("OK")

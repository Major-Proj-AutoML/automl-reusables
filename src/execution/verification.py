"""Mechanical verification of an LLM's ReasoningTrace.

For each decision the LLM claims to have made, this module checks two things:

1. ``observed_value`` matches what the meta-features actually say (within a
   small tolerance). This catches the LLM fabricating or misremembering a
   number to justify a decision.
2. ``action`` (a class or function name) actually appears in the generated
   Python code — as an import target OR as a called name. This catches the
   LLM claiming a preprocessor it did not use.

Both checks are pure Python: AST parsing + dict lookup. No LLM involved.
"""

from __future__ import annotations

import ast
import math
from typing import Any

from src.contracts import (
    Decision,
    DecisionVerdict,
    MetaFeatures,
    ReasoningTrace,
    VerificationReport,
)

_VALUE_TOLERANCE = 1e-3


def _dotted_get(obj: Any, path: str) -> Any | None:
    """Follow a dotted path through nested Pydantic models / dicts.

    Returns None if any segment is missing. Never raises.
    """
    cur: Any = obj
    for segment in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(segment)
        else:
            cur = getattr(cur, segment, None)
    return cur


def _value_matches(claimed: float | None, observed: Any) -> bool | None:
    """Whether the observed value at the meta_feature path matches the LLM's claim.

    - Returns None if there is no path or no observation to compare against
      (i.e. the decision is not tied to a specific numeric meta-feature).
    - Returns True/False otherwise, using an absolute tolerance for scalar
      floats and a small relative tolerance for values near zero.
    """
    if claimed is None or observed is None:
        return None
    try:
        obs_float = float(observed)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(obs_float) or not math.isfinite(float(claimed)):
        return False
    return math.isclose(float(claimed), obs_float, abs_tol=_VALUE_TOLERANCE)


def _names_in_code(code: str) -> set[str]:
    """Every class/function/attribute name referenced in the source.

    Includes: imported names, called attributes, and bare identifiers used
    anywhere. A ``TargetEncoder`` in ``from category_encoders import TargetEncoder``
    is captured, and so is a bare ``LGBMClassifier(...)`` further down.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                # For ``import foo.bar`` we keep the last component so the
                # LLM's shorthand action name still resolves.
                short = (alias.asname or alias.name).split(".")[-1]
                names.add(short)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def verify_decision(
    decision: Decision, code_names: set[str], meta: MetaFeatures | None
) -> DecisionVerdict:
    reasons: list[str] = []
    action_present = decision.action in code_names
    if not action_present:
        reasons.append(
            f"action '{decision.action}' was cited but does not appear in the generated code."
        )

    value_matches: bool | None = None
    if decision.meta_feature is not None and meta is not None:
        observed = _dotted_get(meta, decision.meta_feature)
        if observed is None and decision.observed_value is not None:
            # The LLM cited a path that does not exist in the meta-features —
            # a claim we cannot support is worse than no claim at all.
            value_matches = False
            reasons.append(
                f"could not resolve meta_feature path '{decision.meta_feature}'."
            )
        else:
            value_matches = _value_matches(decision.observed_value, observed)
            if value_matches is False:
                reasons.append(
                    f"claimed observed_value={decision.observed_value} for "
                    f"'{decision.meta_feature}' but real value is {observed}."
                )
    elif decision.meta_feature is not None and meta is None:
        reasons.append(
            "meta_feature was cited but no MetaFeatures were provided to verify against."
        )

    return DecisionVerdict(
        step=decision.step,
        action=decision.action,
        value_matches=value_matches,
        action_present=action_present,
        reasons=reasons,
    )


def _decision_is_faithful(v: DecisionVerdict) -> bool:
    """A decision is faithful iff its action is present AND its value is
    not contradicted. A ``None`` value_matches (not checkable) is treated
    as faithful — we cannot refute what we cannot measure."""
    return v.action_present and v.value_matches is not False


def verify_reasoning(
    trace: ReasoningTrace,
    code: str,
    meta: MetaFeatures | None,
) -> VerificationReport:
    """Verify a full ReasoningTrace against the generated code and meta-features."""
    code_names = _names_in_code(code)
    verdicts = [verify_decision(d, code_names, meta) for d in trace.decisions]
    n_faithful = sum(1 for v in verdicts if _decision_is_faithful(v))
    n = len(verdicts)
    faithful = n > 0 and n_faithful == n
    notes = None
    if n == 0:
        notes = "Trace contained zero decisions."
    elif not faithful:
        notes = f"{n - n_faithful} of {n} decisions failed verification."
    return VerificationReport(
        n_decisions=n,
        n_faithful=n_faithful,
        faithful=faithful,
        verdicts=verdicts,
        overall_notes=notes,
    )

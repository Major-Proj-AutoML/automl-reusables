# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright 2026 The VESIT AutoML Project contributors

import logging
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

_log = logging.getLogger(__name__)

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

class SimpleStats(BaseModel):
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_ratio_overall: float
    missing_ratio_per_column: dict[str, float]
    class_balance_ratio: Optional[float]  # null for regression
    categorical_cardinality: dict[str, int]

class DistributionalStats(BaseModel):
    skewness_per_numeric: dict[str, float]
    kurtosis_per_numeric: dict[str, float]
    outlier_ratio_per_numeric: dict[str, float]  # IQR-based

class InformationStats(BaseModel):
    mutual_info_to_target: dict[str, float]
    mean_abs_correlation: float
    max_pairwise_correlation: float
    target_entropy: Optional[float]  # null for regression

class LandmarkScores(BaseModel):
    decision_stump_score: float
    # Legacy field name; holds GaussianNB score on classification and Ridge on
    # regression — the "naive_bayes" label misleads on regression. Retained for
    # backward compatibility with cached meta_features JSONB. New readers should
    # prefer `linear_score` (see `linear_score_or_legacy`).
    naive_bayes_score: float
    one_nn_score: float
    metric_used: Literal["balanced_accuracy", "neg_rmse"]
    # Task-agnostic alias for `naive_bayes_score`. Nullable so legacy JSONBs
    # (which lack this key) still hydrate.
    linear_score: Optional[float] = None

    @property
    def linear_score_or_legacy(self) -> float:
        if self.linear_score is not None:
            return self.linear_score
        _log.debug("reading legacy field naive_bayes_score as linear_score")
        return self.naive_bayes_score

class MetaFeatures(BaseModel):
    """The single source of truth for what we extract from a dataset."""
    dataset_id: str
    task_type: TaskType
    simple: SimpleStats
    distributional: DistributionalStats
    information: InformationStats
    landmarks: LandmarkScores

class RunResult(BaseModel):
    """One pipeline execution result."""
    dataset_id: str
    condition: Literal["B0", "B1", "B2"]
    llm_backend: str
    seed: int
    iteration: int
    success: bool
    error_category: Optional[str]  # None if success
    error_message: Optional[str]
    test_score: Optional[float]
    runtime_seconds: float
    generated_code_path: str

class ErrorCategory(str, Enum):
    """Taxonomy of LLM code-generation failures."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    API_HALLUCINATION = "api_hallucination"
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_ERROR = "type_error"
    DEPRECATED_API = "deprecated_api"
    METRIC_MISMATCH = "metric_mismatch"
    MISSING_NAME = "missing_name"
    SUSPICIOUS_LEAKAGE = "suspicious_leakage"
    REASONING_UNFAITHFUL = "reasoning_unfaithful"
    TIMEOUT = "timeout"
    RUNTIME_OTHER = "runtime_other"
    RESOURCE_LIMIT = "resource_limit"

class GeneratedPipeline(BaseModel):
    """What the LLM produces — the code to execute."""
    code: str
    condition: Literal["B0", "B1", "B2"]
    llm_backend: str
    dataset_id: str
    seed: int
    iteration: int


class Decision(BaseModel):
    """One structured claim by the LLM about a preprocessing/modeling choice.

    Every field is machine-verifiable:
      - meta_feature is a dotted path into the MetaFeatures object.
      - observed_value must match the value at that path (within tolerance).
      - action is a code symbol (class or function name) expected to appear
        in the generated Python source.
      - applied_to lists the column names the action targets.
    """
    step: str = Field(..., description="Human-readable step name, e.g. 'scaling', 'imputation'.")
    rule_id: Optional[int] = Field(None, description="Which B2 rule the LLM cites, if any.")
    meta_feature: Optional[str] = Field(
        None,
        description="Dotted path into MetaFeatures, e.g. 'distributional.outlier_ratio_per_numeric.Fare'.",
    )
    observed_value: Optional[float] = Field(None, description="Value the LLM claims it observed.")
    # Threshold is descriptive-only (we don't verify against it) so accept any
    # scalar the LLM emits — e.g. a numeric bound (0.05) or a symbolic hint
    # ("n_categorical > n_numeric"). Parsing must not fail on this field.
    threshold: Optional[float | str] = Field(None, description="Threshold or symbolic condition from the rule.")
    action: str = Field(..., description="Class/function name expected in the generated code.")
    applied_to: list[str] = Field(default_factory=list, description="Columns the action targets.")


class ReasoningTrace(BaseModel):
    """Structured reasoning trace emitted by the LLM alongside its code."""
    decisions: list[Decision] = Field(default_factory=list)
    notes: Optional[str] = None


class DecisionVerdict(BaseModel):
    """Per-decision verification outcome."""
    step: str
    action: str
    value_matches: Optional[bool] = Field(
        None, description="True/False if a meta_feature+observed_value was supplied; None if not checkable."
    )
    action_present: bool = Field(..., description="Whether the action symbol appears in the generated code.")
    reasons: list[str] = Field(default_factory=list, description="Human-readable notes on any failures.")


class VerificationReport(BaseModel):
    """Result of mechanically verifying a ReasoningTrace against code + meta-features."""
    n_decisions: int
    n_faithful: int
    faithful: bool
    verdicts: list[DecisionVerdict]
    overall_notes: Optional[str] = None

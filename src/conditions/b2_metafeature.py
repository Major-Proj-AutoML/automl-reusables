import pandas as pd

from src.contracts import MetaFeatures
from src.conditions.base import PromptCondition

_DECISION_RULES = """\
PREPROCESSING RULES:
1. If missing_ratio_overall > 0.05: prefer IterativeImputer over SimpleImputer(strategy='mean').
2. If missing_ratio_overall > 0.30: consider dropping columns with >50% missing before imputation.
3. If any categorical_cardinality > 20: use TargetEncoder instead of OneHotEncoder for those columns.
4. If n_categorical > n_numeric: consider gradient boosting with native categorical support (e.g., LGBMClassifier(categorical_features=...)).

FEATURE ENGINEERING RULES:
5. If max_pairwise_correlation > 0.95: drop one of the near-duplicate feature pair.
6. If any abs(skewness) > 2.0: apply PowerTransformer(method='yeo-johnson') to those features.
7. If any outlier_ratio > 0.05: use RobustScaler instead of StandardScaler for those features.
8. If n_cols / n_rows > 0.1: apply feature selection (SelectKBest with mutual_info) before modeling.
9. If mean_abs_correlation > 0.5 and n_cols > 50: consider PCA for dimensionality reduction.

CLASS IMBALANCE RULES:
10. If class_balance_ratio < 0.3: apply SMOTE or set class_weight='balanced' in the estimator.
11. If class_balance_ratio < 0.1: combine SMOTE with Tomek links; do NOT use plain accuracy as the metric.

MODEL SELECTION HINTS (from landmarking; scores are in `landmarks.metric_used` units — balanced_accuracy for classification, negative RMSE for regression):
12. If decision_stump_score > 0.75 (classification only): the problem may be simple — try LogisticRegression or shallow DecisionTreeClassifier first.
13. If one_nn_score - linear_score > 0.1: data has local structure — prefer tree ensembles (RandomForest, XGBoost) or KNN. (For legacy cached meta-features, `naive_bayes_score` holds the same value as `linear_score`.)
14. If linear_score - one_nn_score > 0.1: features are relatively independent — a linear model (LogisticRegression on classification, Ridge on regression) is viable.

IMPORTANT: For each pipeline component you choose, cite which meta-feature or rule informed your decision.\
"""


class B2MetaFeatureGuided(PromptCondition):
    @property
    def condition_name(self) -> str:
        return "B2"

    def build_prompt(
        self,
        task_description: str,
        meta: MetaFeatures,
        df_head: pd.DataFrame,
        target_col: str,
    ) -> str:
        dtype_lines = "\n".join(f"  - {col}: {dtype}" for col, dtype in df_head.dtypes.items())
        schema_section = (
            "## Dataset Schema\n\n"
            f"**Rows:** {meta.simple.n_rows}  |  **Columns:** {meta.simple.n_cols}\n\n"
            f"**Task type:** {meta.task_type.value}\n\n"
            f"**Target column:** {target_col}\n\n"
            f"**Column dtypes:**\n{dtype_lines}\n\n"
            f"**First 3 rows:**\n{df_head.head(3).to_markdown(index=False)}\n"
        )
        meta_section = (
            "## Dataset Meta-Feature Profile\n\n"
            f"```json\n{meta.model_dump_json(indent=2)}\n```\n"
        )
        rules_section = (
            "## Decision Rules (apply where conditions match)\n\n"
            f"{_DECISION_RULES}\n"
        )
        reasoning_section = (
            "## Structured Reasoning (mandatory — machine-verified)\n\n"
            "After the `SCORE: <number>` line, your script MUST also print exactly one line of the form:\n\n"
            "  REASONING: <json>\n\n"
            "where <json> is a single-line JSON object with this shape:\n\n"
            "```json\n"
            "{\n"
            '  "decisions": [\n'
            "    {\n"
            '      "step": "scaling",\n'
            '      "rule_id": 7,\n'
            '      "meta_feature": "distributional.outlier_ratio_per_numeric.Fare",\n'
            '      "observed_value": 0.128,\n'
            '      "threshold": 0.05,\n'
            '      "action": "RobustScaler",\n'
            '      "applied_to": ["Fare"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Rules for producing the trace:\n"
            "- One decision per meaningful choice you made (imputation, scaling, encoding, model, feature-selection, imbalance handling).\n"
            "- `meta_feature` is a dotted path into the meta-feature JSON above (e.g. `simple.class_balance_ratio`, `information.max_pairwise_correlation`). Omit if the decision is not tied to a specific meta-feature.\n"
            "- `observed_value` must be the exact number you read from that path (copy it verbatim, no rounding). If the meta_feature points to a scalar, use it. If it points to a dict of per-column values, pick the value for the relevant column.\n"
            "- `action` must be the class or function name that actually appears in your imports (e.g. `RobustScaler`, `IterativeImputer`, `TargetEncoder`, `LGBMClassifier`).\n"
            "- The trace is verified mechanically against your generated code and the meta-feature values. Fabricating numbers or citing symbols you did not actually import will cause the run to be flagged as unfaithful and discarded.\n"
        )
        return (
            "You are an AutoML assistant. Generate a complete scikit-learn pipeline for the following task:\n\n"
            f"{task_description}\n\n"
            "Return a **single, self-contained** Python script that loads the data, preprocesses it, trains a "
            "model, and evaluates on a held-out test set. Use balanced_accuracy_score for classification or "
            "root_mean_squared_error for regression.\n\n"
            "IMPORTANT: The script must include ALL import statements at the top. Every class and function you "
            "use (e.g. RobustScaler, ColumnTransformer, train_test_split, etc.) must be explicitly imported. "
            "The script will run in an isolated environment with no pre-imported names.\n\n"
            f"{schema_section}\n"
            f"{meta_section}\n"
            f"{rules_section}\n"
            f"{reasoning_section}"
        )

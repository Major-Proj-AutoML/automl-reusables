from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wilcoxon


def load_results(path: str = "results/experiment_results.jsonl") -> pd.DataFrame:
    """Read a JSONL of RunResult records into a DataFrame."""
    p = Path(path)
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame.from_records(records)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, llm_backend): mean/median score on successes, success rate, mean runtime."""
    grouped = df.groupby(["condition", "llm_backend"], dropna=False, sort=True)
    rows = []
    for (condition, llm), group in grouped:
        succ = group[group["success"] == True]
        rows.append(
            {
                "condition": condition,
                "llm_backend": llm,
                "mean_score": succ["test_score"].mean() if len(succ) else float("nan"),
                "median_score": succ["test_score"].median() if len(succ) else float("nan"),
                "success_rate": float(group["success"].sum()) / len(group),
                "mean_runtime": group["runtime_seconds"].mean(),
                "n_runs": len(group),
            }
        )
    return pd.DataFrame(rows)


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count failures by (condition, error_category). Excludes successes."""
    failed = df[df["success"] == False]
    counts = (
        failed.groupby(["condition", "error_category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return counts


def wilcoxon_test(df: pd.DataFrame, condition_a: str, condition_b: str) -> dict[str, Any]:
    """Paired Wilcoxon over per-dataset mean success scores for two conditions."""
    succ = df[df["success"] == True]
    per_dataset = (
        succ.groupby(["dataset_id", "condition"])["test_score"].mean().reset_index()
    )
    pivot = per_dataset.pivot(index="dataset_id", columns="condition", values="test_score")
    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return {
            "statistic": None,
            "p_value": None,
            "n_datasets": 0,
            "condition_a_mean": None,
            "condition_b_mean": None,
        }
    paired = pivot[[condition_a, condition_b]].dropna()
    if len(paired) == 0:
        return {
            "statistic": None,
            "p_value": None,
            "n_datasets": 0,
            "condition_a_mean": None,
            "condition_b_mean": None,
        }
    a = paired[condition_a]
    b = paired[condition_b]
    stat, p = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_datasets": int(len(paired)),
        "condition_a_mean": float(a.mean()),
        "condition_b_mean": float(b.mean()),
    }


def iteration_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """RQ3: Mean iterations to first success by condition (and optionally llm_backend).

    Requires the ``iterations_used`` column written by run_sweep.py.
    Falls back to counting iteration numbers from RunResult-style records.
    """
    if "iterations_used" in df.columns:
        iter_col = "iterations_used"
    elif "iteration" in df.columns:
        iter_col = "iteration"
    else:
        return pd.DataFrame()

    succ = df[df["success"] == True]
    if succ.empty:
        return pd.DataFrame()

    group_cols = ["condition"]
    if "llm_backend" in df.columns:
        group_cols.append("llm_backend")

    result = (
        succ.groupby(group_cols, dropna=False)[iter_col]
        .agg(mean_iterations="mean", median_iterations="median", count="size")
        .reset_index()
    )
    return result


def model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """RQ4: Per-model performance breakdown — mean score, success rate, mean iterations."""
    if "llm_backend" not in df.columns:
        return pd.DataFrame()

    rows = []
    for llm, group in df.groupby("llm_backend", dropna=False):
        succ = group[group["success"] == True]
        iter_col = "iterations_used" if "iterations_used" in df.columns else None
        row: dict[str, Any] = {
            "llm_backend": llm,
            "n_runs": len(group),
            "success_rate": float(group["success"].sum()) / len(group),
            "mean_score": succ["test_score"].mean() if len(succ) else float("nan"),
            "median_score": succ["test_score"].median() if len(succ) else float("nan"),
        }
        if iter_col and iter_col in group.columns:
            row["mean_iterations"] = succ[iter_col].mean() if len(succ) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


_SIZE_BINS = {"small": (0, 1000), "medium": (1000, 10_000), "large": (10_000, float("inf"))}


def size_stratified_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """RQ5: Performance by dataset size bucket (small / medium / large).

    Requires a ``dataset_name`` column to look up sizes from SELECTED_CLASSIFICATION,
    or a ``n_rows`` column if pre-joined.
    """
    from src.experiments.datasets import SELECTED_CLASSIFICATION, _DATA_ROOT

    if "n_rows" not in df.columns:
        size_map: dict[str, int] = {}
        for ds_id, name in SELECTED_CLASSIFICATION.items():
            meta_path = _DATA_ROOT / name / "meta.json"
            train_path = _DATA_ROOT / name / "train.csv"
            if train_path.exists():
                n = sum(1 for _ in train_path.open(encoding="utf-8")) - 1
                size_map[name] = n
            elif "dataset_name" in df.columns:
                matching = df[df["dataset_name"] == name]
                if not matching.empty:
                    size_map[name] = 0

        if "dataset_name" in df.columns:
            df = df.copy()
            df["n_rows"] = df["dataset_name"].map(size_map)
        else:
            return pd.DataFrame()

    def _bucket(n: float) -> str:
        for label, (lo, hi) in _SIZE_BINS.items():
            if lo <= n < hi:
                return label
        return "large"

    df = df.copy()
    df["size_bucket"] = df["n_rows"].apply(_bucket)

    rows = []
    for (bucket, condition), group in df.groupby(
        ["size_bucket", "condition"], dropna=False, sort=True
    ):
        succ = group[group["success"] == True]
        rows.append(
            {
                "size_bucket": bucket,
                "condition": condition,
                "n_runs": len(group),
                "success_rate": float(group["success"].sum()) / len(group),
                "mean_score": succ["test_score"].mean() if len(succ) else float("nan"),
            }
        )
    return pd.DataFrame(rows)

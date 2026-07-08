import pandas as pd


def is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < 20:
        return True
    return not pd.api.types.is_numeric_dtype(s.dtype)


def classify_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        if is_categorical(df[col]):
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def numeric_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        col for col in df.columns
        if col != target_col and not is_categorical(df[col])
    ]

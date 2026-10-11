from src.column_utils import _get_dtype, _check_if_string
from typing import Any
import pandas as pd

def _check_if_valid_for_t_test(params: dict[Any, Any], welch_or_student) -> None:
    for name, value in params.items():
        if welch_or_student:
            if isinstance(value, list):
                if "N" in name:
                    for item in value:
                        if item <= 0:
                            raise ValueError(f"Invalid value for: {name}. {item} is smaller than or equal to 0")
                continue
        if value is None:
            continue
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            if ("N" in name or "std" in name) and value <= 0:
                raise ValueError(f"Invalid value for: {name}. {value} is smaller than 0")
            continue
        else:
            raise ValueError(f"Invalid data type {type(value)} for {name}. Argument has to either be a number, bool or a list, depending on the argument")

def _calculate_mean(df: pd.DataFrame, cols: str | list) -> float | dict:
    if isinstance(cols, str):
        if _get_dtype(df[cols].dtype) != "numeric": 
            raise ValueError(f"{cols} has invalid data type: {df[cols].dtype}. To compare means, all columns must contain numeric values")
        mean = df[cols].mean()
        return mean
    
    if isinstance(cols, list):
        _check_if_string(cols)
        means: dict[str, float] = {}
        for col in cols:
            if _get_dtype(df[col].dtype) != "numeric":
                raise ValueError(f"{col} has invalid data type: {df[col].dtype}. To compare means, all columns must contain numeric values")
            means.update({f"mean_{col}": round(df[col].mean(), 2)})
        
        return means
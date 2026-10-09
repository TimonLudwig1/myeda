import pandas as pd 
from typing import Any

def _is_valid_dtype(dtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    )

def _get_dtype(dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif dtype is list: 
            return "list"
    elif pd.api.types.is_object_dtype(dtype):
        return "categorical"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    else: 
        raise ValueError(f"Column contains datatype not suitable for plotting or calculating. Datatype: {dtype}")

def _check_if_string(item: Any) -> None:
    if isinstance(item, list):
        if not item:
            raise ValueError(f"{item} is empty.")
        for value in item:
            _check_if_string(value)
    elif not isinstance(item, str):
        raise ValueError(f"{item} is not a string. Pass a list of column names as strings")
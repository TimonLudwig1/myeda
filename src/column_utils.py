import pandas as pd 
from typing import Any

def is_valid_dtype(dtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    )

def get_dtype(dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_object_dtype(dtype):
        return "categorical"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    else: 
        raise ValueError(f"Column contains datatype not suitable for plotting. Datatype: {dtype}")

def check_if_string(item: Any) -> None:
    if isinstance(item, list):
        if not item:
            raise ValueError(f"{item} is empty.")
        for value in item:
            check_if_string(value)
    elif not isinstance(item, str):
        raise ValueError(f"{item} is not a string. Pass a list of column names as strings")
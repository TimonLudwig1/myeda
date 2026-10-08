import numpy as np
from math import sqrt
import pandas as pd 
from column_utils import _get_dtype, _check_if_string
from typing import Any
from scipy import stats

# config variables and constants

VALID_COMBINATIONS_T_TEST = [
    {"mean1": "numeric", "std1": "numeric", "N1": "numeric", "mean2": "numeric", "std2": "numeric", "N2": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "numeric", "mean2": "numeric", "std2": "None", "N2": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "list", "mean2": "numeric", "std2": "None", "N2": "None"},
    {"mean1": "list", "std1": "numeric", "N1": "numeric", "mean2": "None", "std2": "numeric", "N2": "numeric"},
    {"mean1": "list", "std1": "list", "N1": "numeric", "mean2": "None", "std2": "None", "N2": "numeric"},
    {"mean1": "list", "std1": "numeric", "N1": "list", "mean2": "None", "std2": "numeric", "N2": "None"},
    {"mean1": "list", "std1": "list", "N1": "list", "mean2": "None", "std2": "None", "N2": "None"},
]

# helpers

def _check_if_valid_for_t_test(params: dict[Any, Any]) -> None:
    for name, value in params.items():
        if not isinstance(value, (float, list)):
            raise ValueError(f"Invalid data type {type(value)} for {name}. Argument has to either be a number or a list")

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

def one_sample_t_test():
    # mean of column against given value - you can pass either just the column and it calculates the mean itself or you can pass a mean
    pass

def students_t_test(
    mean1: float | dict | list, 
    std1: float | list, 
    N1: float | list,
    mean2: float | None = None, 
    std2: float | None = None,
    N2: float | None = None,
) -> float:
    
    # t-test of diff of two means with similar variances - two sample
    # t = (mean1 - mean2) / sqr(std1^2/N1 + std2^2/N2)
    # use helper function 
    params = {
        "mean1": mean1,
        "std1": std1,
        "N1": N1,
        "mean2": mean2,
        "std2": std2,
        "N2": N2,
    }
    param_types = {
        "mean1": _get_dtype(type(mean1)),
        "std1": _get_dtype(type(std1)),
        "N1": _get_dtype(type(N1)),
        "mean2": _get_dtype(type(mean2)) if mean2 is not None else "None",
        "std2": _get_dtype(type(std2)) if std2 is not None else "None",
        "N2": _get_dtype(type(N2)) if N2 is not None else "None",
    }

    _check_if_valid_for_t_test(params)


    if param_types in VALID_COMBINATIONS_T_TEST:
        # standardize input into a dict, then get values out of dict for calculation
        if param_types["mean1"] == "list":
            mean1, mean2 = mean1  # type: ignore

        if param_types["std1"] == "list":
            std1, std2 = std1  # type: ignore

        if param_types["N1"] == "list": 
            N1, N2 = N1  # type: ignore

        t_statistic = (mean1 - mean2) / (sqrt((std1**2 / N1) + (std2**2 / N2))) # type: ignore
        return t_statistic
    
    else:
        raise ValueError("Invalid combination of datatypes. mean1, std1 and N1 are required arguments. If you do not pass a list provide the other value as a seperate argument")


def welch_t_test():
    #t-test from difference of two means with different variances
    pass

def paired_t_test():
    # t-test on paired observations 
    pass

# main functions

def compare_means(df: pd.DataFrame, cols: list[str]):
    if type(cols) != list:
        raise ValueError(f"{cols} has to be a list containing at least two column names as strings")
    _check_if_string(cols)
    if len(cols) < 2:
        raise ValueError(f"{cols} has to contain at least two column names")

    if len(cols) == 2:
        # do a t-test on the two means 
        
        pass

    else:
        # ANOVA
        pass

def check_significant_difference():
    """Checks if the difference of a sample mean is significantly different to a given value"""
    pass

def compare_proportions():
    pass

def check_relationship():
    pass

def compare_variances():
    pass

def relationship_analysis(df: pd.DataFrame, col: str | list):
    pass

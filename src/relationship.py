import numpy as np
from math import sqrt
import pandas as pd 
from column_utils import _get_dtype, _check_if_string
from typing import Any
import scipy.stats

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

REQUIRED_KEYS_WELCH_TEST = {
    "mean1", "std1", "N1",
    "mean2", "std2", "N2",
    "critical_level"
}

# helpers

def _check_if_valid_for_t_test(params: dict[Any, Any]) -> None:
    for name, value in params.items():
        if value is None:
            continue
        elif isinstance(value, list):
            for item in value:
                if item <= 0:
                    raise ValueError(f"Invalid value for: {item}. {item} is smaller than 0")
        elif isinstance(value, float):
            if value <= 0:
                raise ValueError(f"Invalid value for: {value}. {value} is smaller than 0")
        elif isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Invalid value for: {value}. {value} is smaller than 0")
        else:
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

def significance_check(t_statistic: float, t_crit: float, critical_level: float):
    if abs(t_statistic) > abs(t_crit):
        stat_significance = f"statistically significant at the {critical_level * 100}% level"
        is_significant = True
    else:
        stat_significance = f"not statistically significant at the {critical_level * 100}% level"
        is_significant = False

    return stat_significance, is_significant


def welch_t_test(
    mean1: float | dict | list, 
    std1: float | list = 0, 
    N1: float | list = 0,
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    mean2: float | None = None, 
    std2: float | None = None,
    N2: float | None = None,
) -> dict:

    # t-test for difference of two means with different variances
    # add bool for left, right tailed or two tailed

    # if just a single dict is passed 
    if isinstance(mean1, dict):
        params = mean1.copy()

        if params.keys() != REQUIRED_KEYS_WELCH_TEST:
            raise ValueError("Invalid parameter keys. Check for missing keys and exact spelling")

        mean1 = params["mean1"]
        mean2 = params["mean2"]

        std1 = params["std1"]
        std2 = params["std2"]

        N1 = params["N1"]
        N2 = params["N2"]

        critical_level = params["critical_level"]

        param_types = {
            "mean1": _get_dtype(type(mean1)),
            "std1": _get_dtype(type(std1)),
            "N1": _get_dtype(type(N1)),
            "mean2": _get_dtype(type(mean2)) if mean2 is not None else "None",
            "std2": _get_dtype(type(std2)) if std2 is not None else "None",
            "N2": _get_dtype(type(N2)) if N2 is not None else "None",
    }

    else:
        params = {
            "mean1": mean1,
            "std1": std1,
            "N1": N1,
            "mean2": mean2,
            "std2": std2,
            "N2": N2,
            "critical_level": critical_level,
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

        params.update({
            "mean1": mean1,
            "mean2": mean2,
            "std1": std1,
            "std2": std2,
            "N1": N1,
            "N2": N2,
        })

        t_statistic = (mean1 - mean2) / (sqrt((std1**2 / N1) + (std2**2 / N2))) # type: ignore

        dof = (((std1**2/N1) + (std2**2/N2))**2) / (((std1**2 / N1)**2 / (N1 - 1)) + ((std2**2 / N2)**2 / (N2 - 1)))  # type: ignore

        if sum([two_tailed, left_tailed, right_tailed]) != 1:
            raise ValueError("Exactly one test type must be selected")
        
        if two_tailed: 
            t_crit = float(scipy.stats.t.ppf((1 - (critical_level/2)), dof))
            p_value = float(2 * scipy.stats.t.sf(abs(t_statistic), dof))
            stat_significance, is_significant =  significance_check(t_statistic, t_crit, critical_level)


        elif left_tailed: # mean1 significantly smaller than mean2 
            t_crit = float(scipy.stats.t.ppf((critical_level), dof))
            p_value = float(scipy.stats.t.cdf(t_statistic, dof))
            stat_significance, is_significant = significance_check(t_statistic, t_crit, critical_level)

        elif right_tailed: # mean1 significantly larger than mean2 
            t_crit = float(scipy.stats.t.ppf((1 - critical_level), dof))
            p_value = float(scipy.stats.t.sf(t_statistic, dof))
            stat_significance, is_significant = significance_check(t_statistic, t_crit, critical_level)
        
        else:
            raise ValueError("at least one test type has to be chosen")

        results = {
            "significance": stat_significance,
            "dof": dof,
            "p_value": p_value,
            "is_significant": is_significant,
        }
        results.update(params)

        return results
    
    else:
        raise ValueError("Invalid combination of datatypes. mean1, std1 and N1 are required arguments. If you do not pass a list provide the other value as a seperate argument")

def one_sample_t_test():
    # mean of column against given value - you can pass either just the column and it calculates the mean itself or you can pass a mean
    # t_statistic = (mean1 - mean2) / (sqrt((std1**2 / N1) + (std2**2 / N2))) 
    pass


def students_t_test():
    # t-test of diff of two means with similar variances - two sample
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

test1 = welch_t_test(10, 2, 100, mean2=12, std2=3, N2=120, left_tailed=True)

test2 = welch_t_test([10, 12], [2, 3], [100, 120], left_tailed=True)

test3 = welch_t_test({
    "mean1": 10, "mean2": 12,
    "std1": 2, "std2": 3,
    "N1": 100, "N2": 120,
    "critical_level": 0.05
}, left_tailed=True)

print(test1)
print(test2)
print(test3)
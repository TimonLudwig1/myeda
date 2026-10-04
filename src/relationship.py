import numpy as np
from math import sqrt
import pandas as pd 
from auto_plot import get_dtype, check_if_string
from typing import Any
from scipy import stats

# helpers

def check_if_valid_for_t_test(items: list[Any]) -> None:
    for item in items:
        if not isinstance(item, (float, list)):
            raise ValueError(f"Invalid data type {type(item)} for {item}. Argument has to either be a number or a list")

def calculate_mean(df: pd.DataFrame, cols: str | list) -> float | dict:
    if isinstance(cols, str):
        if get_dtype(df[cols].dtype) != "numeric": 
            raise ValueError(f"{cols} has invalid data type: {df[cols].dtype}. To compare means, all columns must contain numeric values")
        mean = df[cols].mean()
        return mean
    
    if isinstance(cols, list):
        check_if_string(cols)
        means: dict[str, float] = {}
        for col in cols:
            if get_dtype(df[col].dtype) != "numeric":
                raise ValueError(f"{col} has invalid data type: {df[col].dtype}. To compare means, all columns must contain numeric values")
            means.update({f"mean_{col}": round(df[col].mean(), 2)})
        
        return means

def one_sample_t_test():
    # mean of column against given value
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
    if mean2:
        if std2 is None or N2 is None:
            raise ValueError(f"{std2} and {N2} are missing")
        if not isinstance(mean1, float):
            raise ValueError(f"Invalid data type: {type(mean1)}. {mean1} has to either be an int or float")
        if not isinstance(std1, float):
            raise ValueError(f"Invalid data type: {type(std1)}. {std1} has to either be an int or float")
        if not isinstance(N1, float):
            raise ValueError(f"Invalid data type: {type(N1)}. {N1} has to either be an int or float")
        # maybe check if mean2 and std2 are only int or float as well 
        t_statistic = (mean1 - mean2) / (sqrt((std1**2 / N1) + (std2**2 / N2)))
        return t_statistic
    else:
        if not isinstance(mean1, list):
            raise ValueError(f"Invalid data type {type(mean1)}. When passing just a number as the first sample mean, provide the second sample mean as well. Or pass a list of the two sample means. Other data types are not supported")
        if isinstance(std1, float): 
            # std2 has to be float - otherwise pass std as a list 
            if not isinstance(std2, float):
                raise ValueError("When passing standard deviation 1 as a number, pass standard deviation as a number as well. Or pass a list of both standard deviations for std1")
        if isinstance(N1, float):
            raise ValueError("When passing a list of sample means and standard deviations, provide the sample sizes as a list as well. Or pass each argument as single values")

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
    check_if_string(cols)
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

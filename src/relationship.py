import math
import numpy as np
from math import sqrt
import pandas as pd 
from src.column_utils import _get_dtype, _check_if_string
from typing import Any
import scipy.stats
import pandas as pd

# split this up into seperate files soon!

# config variables and constants

VALID_COMBINATIONS_WELCH_T_TEST = [
    {"mean1": "numeric", "std1": "numeric", "N1": "numeric", "mean2": "numeric", "std2": "numeric", "N2": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "numeric", "mean2": "numeric", "std2": "None", "N2": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "list", "mean2": "numeric", "std2": "None", "N2": "None"},
    {"mean1": "list", "std1": "numeric", "N1": "numeric", "mean2": "None", "std2": "numeric", "N2": "numeric"},
    {"mean1": "list", "std1": "list", "N1": "numeric", "mean2": "None", "std2": "None", "N2": "numeric"},
    {"mean1": "list", "std1": "numeric", "N1": "list", "mean2": "None", "std2": "numeric", "N2": "None"},
    {"mean1": "list", "std1": "list", "N1": "list", "mean2": "None", "std2": "None", "N2": "None"},
]

VALID_COMBINATIONS_ONE_SAMPLE_T_TEST = [
    {"mean": "numeric", "std": "numeric", "N": "numeric", "hypothesized_mean": "numeric", "critical_level": "numeric"},
]

REQUIRED_KEYS_WELCH_TEST = {
    "mean1", "std1", "N1",
    "mean2", "std2", "N2",
    "critical_level"
}
REQUIRED_KEYS_ONE_SAMPLE_TEST = {
    "mean", "std", "n",
    "hypothesized_mean", "critical_level"
}

# helpers

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

def significance_check(p_value: float, critical_level: float):
    if p_value < critical_level:
        stat_significance = f"statistically significant at the {critical_level * 100}% level"
        is_significant = True
    else:
        stat_significance = f"not statistically significant at the {critical_level * 100}% level"
        is_significant = False

    return stat_significance, is_significant

def format_ci(lower: float, upper: float) -> str:
    left = "(" if math.isinf(lower) else "["
    right = ")" if math.isinf(upper) else "]"
    return f"{left}{lower} , {upper}{right}"

def _tailed_test(two_tailed: bool, left_tailed: bool, right_tailed: bool,t_statistic, dof, critical_level: float, mean: float, standard_error: float):
    if two_tailed: 
        t_crit = float(scipy.stats.t.ppf((1 - (critical_level/2)), dof))
        p_value = float(2 * scipy.stats.t.sf(abs(t_statistic), dof))
        stat_significance, is_significant =  significance_check(p_value, critical_level)
        test_type = "two_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="two")   

    elif left_tailed: # mean1 significantly smaller than mean2 
        t_crit = float(scipy.stats.t.ppf((critical_level), dof))
        p_value = float(scipy.stats.t.cdf(t_statistic, dof))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "left_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="left")   

    elif right_tailed: # mean1 significantly larger than mean2 
        t_crit = float(scipy.stats.t.ppf((1 - critical_level), dof))
        p_value = float(scipy.stats.t.sf(t_statistic, dof))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "right_tailed"
        lower_bound, upper_bound = calculate_confidence_interval(mean=mean, t_crit=t_crit, standard_error=standard_error, type="right")   

    else:
        raise ValueError("at least one test type has to be chosen. If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")
    
    ci = format_ci(lower_bound, upper_bound)

    return t_crit, p_value, stat_significance, is_significant, test_type, lower_bound, upper_bound, ci

def calculate_confidence_interval(mean: float, t_crit: float, std: float | None = None, n: float | None = None, standard_error: float | None = None, type: str | None = "two") -> tuple[float, float]:

    if standard_error is None:
        if std is None or n is None: 
            raise ValueError("Missing required arguments")
        standard_error = (std / sqrt(n))

    margin = abs(t_crit) * standard_error

    if type == "two":
        return round(mean - margin, 4), round(mean + margin, 4)
    elif type == "right": 
        return round(mean - margin, 4), math.inf
    elif type == "left":
        return -math.inf, round(mean + margin, 4)
    raise ValueError(f"Unknown type: {type}")
    

def _independant_t_test(
    mean1: float | dict | list, 
    std1: float | list = 0, 
    N1: float | list = 0,
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    equal_var: bool = False,
    hypothetical_difference: float = 0,
    mean2: float | None = None, 
    std2: float | None = None,
    N2: float | None = None,
) -> dict:

    # t-test helper 

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

    _check_if_valid_for_t_test(params, welch_or_student=True)

    if sum([two_tailed, left_tailed, right_tailed]) != 1:
        raise ValueError("Exactly one test type must be selected.  If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")

    if param_types in VALID_COMBINATIONS_WELCH_T_TEST:
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
        if equal_var: 
            pooled_variance = ((((N1 - 1)*(std1**2)) + ((N2 - 1)*std2**2)) / (N1 + N2 - 2)) # type: ignore
            standard_error = sqrt(pooled_variance * ((1/N1) + (1/N2)))  # type: ignore
            dof = (N1 + N2 - 2) # type: ignore
            
        else:
            standard_error = sqrt((std1**2 / N1) + (std2**2 / N2)) # type: ignore
            dof = (((std1**2/N1) + (std2**2/N2))**2) / (((std1**2 / N1)**2 / (N1 - 1)) + ((std2**2 / N2)**2 / (N2 - 1)))  # type: ignore

        diff_in_means = ((mean1 - mean2) - hypothetical_difference) # type: ignore
        t_statistic = diff_in_means / standard_error  
        
        t_crit, p_value, stat_significance, is_significant,test_type, lower_bound, upper_bound, ci = _tailed_test(two_tailed, left_tailed, right_tailed, t_statistic, dof, critical_level, diff_in_means, standard_error)

        results = {
            "significance": stat_significance,
            "test_type": test_type,
            "dof": dof,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_interval": ci,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "t_crit": t_crit,
        }
        results.update(params)

        return results
    
    else:
        raise ValueError("Invalid combination of datatypes. mean1, std1 and N1 are required arguments. If you do not pass a list provide the other value as a seperate argument")

def one_sample_t_test(
    mean: float | str | dict | list[str | float], 
    hypothesized_mean: float | None = 0,
    std: float | None = None,
    n: float | None = None,
    critical_level: float = 0.05,
    paired: bool = False,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    df: pd.DataFrame | None = None,
    ):

    if df is not None: 
        if paired: 
            if isinstance(mean, list):
                _check_if_string(mean)
                if len(mean) > 2:
                    raise ValueError(f"{mean} contains too many arguments. For a paired test pass only two columns")
                mean1 = _calculate_mean(df=df, cols=mean[0])    # type: ignore
                mean2 = _calculate_mean(df=df, cols=mean[1])    # type: ignore
                df["differences"] = df[mean[0]] - df[mean[1]] 
                mean = mean1 - mean2    # type: ignore
                std = df["differences"].std()
            else: 
                raise ValueError("when working with a DataFrame for a paired test, pass the columns for the paired values as a list of column names as str.")
            
        elif not isinstance(mean, str):
            raise ValueError(f"if you're working with a DataFrame, pass a string of the column you want to use the mean of")
        else:
            mean = _calculate_mean(df, cols=mean)  

    if isinstance(mean, dict):
        params = mean.copy()

        if params.keys() != REQUIRED_KEYS_ONE_SAMPLE_TEST:
            raise ValueError("Invalid parameter keys. Check for missing keys and exact spelling")

        mean = params["mean"]
        std = params["std"]
        n = params["std"]
        hypothesized_mean = params["hypothesized_mean"]
        critical_level = params["critical_level"]

        param_types = {
            "mean": _get_dtype(type(mean)),
            "std": _get_dtype(type(std)),
            "N":_get_dtype(type(n)),
            "hypothesized_mean": _get_dtype(type(hypothesized_mean)), 
            "critical_level": _get_dtype(type(critical_level)),            
        }

    else:
        if paired:
            # could write a small helper for numeric and len check
            if isinstance(mean, list):
                for _ in mean:
                    if not isinstance(_, (int, float)):
                        raise ValueError(f"Invalid data type {type(_)} for {_}. Value for both means have to be numeric")
                if len(mean) > 2:
                    raise ValueError(f"{std} contains too many arguments. For a paired test pass only two values")
                mean1 = mean[0]    
                mean2 = mean[1]  
                mean = mean1 - mean2    # type: ignore

        params = {
            "mean": mean,
            "std": std,
            "N":n,
            "hypothesized_mean": hypothesized_mean, 
            "critical_level": critical_level,
        }

        param_types = {
            "mean": _get_dtype(type(mean)),
            "std": _get_dtype(type(std)),
            "N":_get_dtype(type(n)),
            "hypothesized_mean": _get_dtype(type(hypothesized_mean)), 
            "critical_level": _get_dtype(type(critical_level)),          
        }

    _check_if_valid_for_t_test(params, welch_or_student=False)

    for name, value in params.items():
        if value is None:
            raise ValueError(f"Missing required argument: '{name}'")
        
    if sum([two_tailed, left_tailed, right_tailed]) != 1:
        raise ValueError("Exactly one test type must be selected")

    if param_types in VALID_COMBINATIONS_ONE_SAMPLE_T_TEST:

        standard_error = std / sqrt(n)  # type: ignore
        dof = n - 1 # type: ignore
        t_statistic = (mean - hypothesized_mean) / standard_error   # type: ignore

        t_crit, p_value, stat_significance, is_significant,test_type, lower_bound, upper_bound, ci = _tailed_test(two_tailed, left_tailed, right_tailed, t_statistic, dof, critical_level, mean, standard_error)    # type: ignore

        results = {
            "significance": stat_significance,
            "test_type": test_type,
            "dof": dof,
            "p_value": p_value,
            "is_significant": is_significant,
            "confidence_interval": ci,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "t_crit": t_crit,
        }
        if paired:
            params = {
                "diff_in_means": mean,
                "std_of_diff": std,
                "N":n,
                "hypothesized_mean": hypothesized_mean, 
                "critical_level": critical_level,
            }
        results.update(params)

        return results

    else:
        raise ValueError("Invalid combination of datatypes")


def welch_t_test(
    mean1: float | dict | list, 
    std1: float | list = 0, 
    N1: float | list = 0,
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    hypothetical_difference: float = 0,
    mean2: float | None = None, 
    std2: float | None = None,
    N2: float | None = None,
) -> dict:
    # maybe change this to: def welch_t_test(*args, **kwargs) -> dict:
    #   return _independant_t_test(*args, **kwargs, equal_var=False) instead of passing all args again
    return _independant_t_test(
        mean1=mean1, 
        std1=std1, 
        N1=N1, 
        critical_level=critical_level, 
        two_tailed=two_tailed, 
        left_tailed=left_tailed, 
        right_tailed=right_tailed, 
        equal_var=False, 
        hypothetical_difference=hypothetical_difference, 
        mean2=mean2,
        std2=std2,
        N2 = N2,
    )

def students_t_test(
    mean1: float | dict | list, 
    std1: float | list = 0, 
    N1: float | list = 0,
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    hypothetical_difference: float = 0,
    mean2: float | None = None, 
    std2: float | None = None,
    N2: float | None = None,
) -> dict:
    return _independant_t_test(
        mean1=mean1, 
        std1=std1, 
        N1=N1, 
        critical_level=critical_level, 
        two_tailed=two_tailed, 
        left_tailed=left_tailed, 
        right_tailed=right_tailed, 
        equal_var=True, 
        hypothetical_difference=hypothetical_difference, 
        mean2=mean2,
        std2=std2,
        N2 = N2,
    )


def paired_t_test(
    mean: float | str | dict | list[str | float], 
    hypothesized_mean: float | None = 0,
    std: float | None = None,
    n: float | None = None,
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False,
    df: pd.DataFrame | None = None,
) -> dict:
    return one_sample_t_test(
        mean=mean,
        hypothesized_mean=hypothesized_mean,
        std=std,
        n=n,
        critical_level=critical_level,
        paired=True,
        two_tailed=two_tailed,
        left_tailed=left_tailed,
        right_tailed=right_tailed,
        df=df,
    ) 

def  anova_test():
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

def compare_proportions():
    pass

def check_relationship():
    pass

def compare_variances():
    pass

def relationship_analysis(df: pd.DataFrame, col: str | list):
    pass


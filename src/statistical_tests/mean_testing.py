from math import sqrt
import pandas as pd 
from src.column_utils import _get_dtype, _check_if_string
from typing import Any
import pandas as pd
from input_validation import _calculate_mean, _check_if_string, _check_if_valid_for_t_test
from stats_utils import _tailed_test

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




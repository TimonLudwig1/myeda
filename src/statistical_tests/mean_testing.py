from math import sqrt
import pandas as pd 
from src.column_utils import _get_dtype, _check_if_string
from typing import Any
import pandas as pd
from src.statistical_tests.input_validation import _calculate_mean, _check_if_valid_for_test
from src.statistical_tests.stats_utils import _tailed_test, f_test

# config variables and constants

VALID_COMBINATIONS_WELCH_T_TEST = [
    {"mean1": "numeric", "std1": "numeric", "N1": "numeric", "mean2": "numeric", "std2": "numeric", "N2": "numeric", "critical_level": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "numeric", "mean2": "numeric", "std2": "None", "N2": "numeric", "critical_level": "numeric"},
    {"mean1": "numeric", "std1": "list", "N1": "list", "mean2": "numeric", "std2": "None", "N2": "None", "critical_level": "numeric"},
    {"mean1": "list", "std1": "numeric", "N1": "numeric", "mean2": "None", "std2": "numeric", "N2": "numeric", "critical_level": "numeric"},
    {"mean1": "list", "std1": "list", "N1": "numeric", "mean2": "None", "std2": "None", "N2": "numeric", "critical_level": "numeric"},
    {"mean1": "list", "std1": "numeric", "N1": "list", "mean2": "None", "std2": "numeric", "N2": "None", "critical_level": "numeric"},
    {"mean1": "list", "std1": "list", "N1": "list", "mean2": "None", "std2": "None", "N2": "None", "critical_level": "numeric"},
]

VALID_COMBINATIONS_ONE_SAMPLE_T_TEST = [
    {"mean": "numeric", "std": "numeric", "N": "numeric", "hypothesized_mean": "numeric", "critical_level": "numeric"},
]

VALID_COMBINATIONS_ANOVA_TEST = [
    {"num_groups": "numeric", "N_per_group": "list", "means": "list", "critical_level": "numeric", "within_group_variance": "numeric", "between_group_variance": "numeric"},
]
REQUIRED_KEYS_WELCH_TEST = {
    "mean1", "std1", "N1",
    "mean2", "std2", "N2",
    "critical_level"
}
REQUIRED_KEYS_ONE_SAMPLE_TEST = {
    "mean", "std", "N",
    "hypothesized_mean", "critical_level"
}  
REQUIRED_KEYS_ANOVA = {
    "num_groups", "N_per_group", 
    "means", "critical_level", 
    "within_group_variance", "between_group_variance"
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

    param_types = {}

    for k, v in params.items():
        param_types[k] = _get_dtype(type(v)) if v is not None else "None"

    _check_if_valid_for_test(params, welch_or_student=True)

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
            "t_statistic": t_statistic,
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
                mean_str = _check_if_string(mean)
                if len(mean_str) > 2:
                    raise ValueError(f"{mean} contains too many arguments. For a paired test pass only two columns")
                mean1 = _calculate_mean(df=df, cols=mean_str[0])    
                mean2 = _calculate_mean(df=df, cols=mean_str[1])    
                df["differences"] = df[mean_str[0]] - df[mean_str[1]] 
                mean = mean1 - mean2    #type: ignore - fix this in _calculate_mean       
                std = df["differences"].std()
            else: 
                raise ValueError("when working with a DataFrame for a paired test, pass the columns for the paired values as a list of column names as str.")
            
        elif not isinstance(mean, str):
            raise ValueError(f"if you're working with a DataFrame, pass a string of the column you want to use the mean of")
        else:
            mean = _calculate_mean(df, cols=mean)  

        params = {
                    "mean": mean,
                    "std": std,
                    "N":n,
                    "hypothesized_mean": hypothesized_mean, 
                    "critical_level": critical_level,
                }
        param_types = {}

        for k, v in params.items():
            param_types[k] = _get_dtype(type(v)) if v is not None else "None"

    elif isinstance(mean, dict):
        params = mean.copy()

        if params.keys() != REQUIRED_KEYS_ONE_SAMPLE_TEST:
            raise ValueError("Invalid parameter keys. Check for missing keys and exact spelling")

        mean = params["mean"]
        std = params["std"]
        n = params["N"]
        hypothesized_mean = params["hypothesized_mean"]
        critical_level = params["critical_level"]

        param_types = {}

        for k, v in params.items():
            param_types[k] = _get_dtype(type(v)) if v is not None else "None"

    else:
        if paired:
            if isinstance(mean, list):
                if len(mean) > 2:
                    raise ValueError(f"{std} contains too many arguments. For a paired test pass only two values")
                mean1, mean2 = mean  

                if not isinstance(mean1, (int, float)) or not isinstance(mean2, (int, float)):
                    raise ValueError("Values for both means have to be numeric")
                
                mean = mean1 - mean2    

        params = {
            "mean": mean,
            "std": std,
            "N":n,
            "hypothesized_mean": hypothesized_mean, 
            "critical_level": critical_level,
        }

        param_types = {}
        
        for k, v in params.items():
            param_types[k] = _get_dtype(type(v)) if v is not None else "None"

    _check_if_valid_for_test(params, welch_or_student=False)

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

def  anova_test(
    num_groups: float | dict, 
    N_per_group: list | None,
    means: list | None, 
    within_group_variance: float | None,
    between_group_variance: float | None, 
    critical_level: float = 0.05,
    two_tailed: bool = False,
    left_tailed: bool = False,
    right_tailed: bool = False, 
    df: pd.DataFrame | None = None
):

    if df is not None:
        # pass a list of col names you want to test as str
        if not isinstance(means, list):
            raise ValueError("{means} must be a list")
        means_values = []
        N_per_group = []
        num_groups = len(means)
        within_sq_dev = 0

        for col in means:
            group = df[col]

            group_mean = _calculate_mean(df, col)
            group_n = df[col].count()
            means_values.append(group_mean)
            N_per_group.append(group_n)

            # within group sum of squares
            within_sq_dev += ((group - group_mean) ** 2).sum()

        N = sum(N_per_group)
        # weighted grand mean
        grand_mean = sum(
            n * mean for n, mean in zip(N_per_group, means_values)
        ) / N

        # between groups sum of squares
        sq_dev_groups = sum(
            n * (mean - grand_mean)**2 for n, mean in zip(N_per_group, means_values)
        )

        between_group_variance = sq_dev_groups / (num_groups - 1)
        within_group_variance = within_sq_dev / (N - num_groups)

        params = {
                    "num_groups": num_groups,
                    "N_per_group": N_per_group,
                    "means": means_values,
                    "critical_level": critical_level,
                    "within_group_variance": within_group_variance,
                    "between_group_variance": between_group_variance,
                }
        
        param_types = {}
        
        for k, v in params.items():
            param_types[k] = _get_dtype(type(v)) if v is not None else "None"


    elif isinstance(num_groups, dict):
        params = num_groups.copy()

        if params.keys() != REQUIRED_KEYS_ANOVA:
            raise ValueError("Invalid parameter keys. Check for missing keys and exact spelling")
        
        num_groups = params["num_groups"]
        N_per_group = params["N_per_group"]
        means = params["means"]
        critical_level = params["critical_level"]
        within_group_variance = params["within_group_variance"]
        between_group_variance = params["between_group_variance"]

    else:
        params = {
            "num_groups": num_groups,
            "N_per_group": N_per_group,
            "means": means,
            "critical_level": critical_level,
            "within_group_variance": within_group_variance,
            "between_group_variance": between_group_variance,
        }
        
    param_types = {}

    for k, v in params.items():
        param_types[k] = _get_dtype(type(v)) if v is not None else "None"

    _check_if_valid_for_test(params, welch_or_student=True)
    
    if sum([two_tailed, left_tailed, right_tailed]) != 1:
        raise ValueError("Exactly one test type must be selected.  If you're using a dict, test type has to be passed as a seperate argument outside the dictionary") 
    
    if param_types in VALID_COMBINATIONS_ANOVA_TEST:
        if num_groups != len(N_per_group) or num_groups != len(means):    # type: ignore   
            raise ValueError("number of groups does not match number of mean entries or number of sample size entries")
        N = sum(N_per_group)  # type: ignore  
        f_statistic, f_crit, p_value, stat_significance, is_significant, test_type  = f_test(k=num_groups, N=N, within_group_variance=within_group_variance, between_group_variance=between_group_variance, critical_level= critical_level, two_tailed=two_tailed, right_tailed=right_tailed, left_tailed=left_tailed)  # type: ignore

        results = {
            "significance": stat_significance, 
            "test_type": test_type,
            "p_value": p_value,
            "is_significant": is_significant,
            "f_statistic": f_statistic,
            "f_crit": f_crit,
        }
        results.update(params)

        return results
    else: 
        raise ValueError(f"Invalid combination of datatypes: {param_types}")
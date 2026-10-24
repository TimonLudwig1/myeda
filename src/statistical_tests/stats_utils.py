import scipy.stats
import warnings
import numpy as np
import math
from math import sqrt
import pandas as pd
from src.column_utils import _check_if_string, _get_dtype
from dataclasses import dataclass

@dataclass
class Chi2Result:
    n: int
    V: float
    distinct_combinations: int
    stat_significance: str
    dof: int
    p_value: float
    is_significant: bool
    X2: float
    chi2_crit: float

@dataclass
class TailedTestResult:
    t_crit: float
    p_value: float
    stat_significance: str
    is_significant: bool
    test_type: str
    lower_bound: float
    upper_bound: float
    ci: str

@dataclass
class TailedProportionTestResult:
    z_crit: float
    p_value: float
    stat_significance: str
    is_significant: bool
    test_type: str

@dataclass 
class FTestResult:
    f_statistic: float
    f_crit: float
    p_value: float
    stat_significance: str
    is_significant: bool
    test_type: str

def significance_check(p_value: float, critical_level: float) -> tuple[str, bool]:
    if p_value < critical_level:
        stat_significance = f"statistically significant at the {critical_level * 100}% level"
        is_significant = True
    else:
        stat_significance = f"not statistically significant at the {critical_level * 100}% level"
        is_significant = False

    return stat_significance, is_significant

def calculate_confidence_interval(
        mean: float, 
        t_crit: float, 
        std: float | None = None, 
        n: float | None = None, 
        standard_error: float | None = None, 
        type: str | None = "two"
) -> tuple[float, float]:

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

def format_ci(lower: float, upper: float) -> str:
    left = "(" if math.isinf(lower) else "["
    right = ")" if math.isinf(upper) else "]"
    return f"{left}{lower} , {upper}{right}"
  

def _tailed_test(
        two_tailed: bool, 
        left_tailed: bool, 
        right_tailed: bool,
        t_statistic: float, 
        dof: float, 
        critical_level: float, 
        mean: float, 
        standard_error: float
) -> TailedTestResult:
    
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

    return TailedTestResult(
        t_crit=t_crit,
        p_value=p_value,
        stat_significance=stat_significance,
        is_significant=is_significant,
        test_type=test_type,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        ci=ci
    ) 

def f_test(
    k: float, 
    N: float, 
    within_group_variance: float, 
    between_group_variance: float, 
    critical_level: float = 0.05, 
    two_tailed: bool = False, 
    left_tailed: bool = False, 
    right_tailed: bool = False
) -> FTestResult:
    
    f_statistic = between_group_variance / within_group_variance
    dof_between = k - 1
    dof_within = N - k

    if two_tailed:
        f_crit = float(scipy.stats.f.ppf(1 - (critical_level/2), dof_between, dof_within))
        p_value = float(2* scipy.stats.f.sf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "two_tailed"
    elif left_tailed:
        f_crit = float(scipy.stats.f.ppf((critical_level), dof_between, dof_within))
        p_value = float(scipy.stats.f.cdf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "left_tailed"
    elif right_tailed:
        f_crit = float(scipy.stats.f.ppf((1 - critical_level), dof_between, dof_within))
        p_value = float(scipy.stats.f.sf(f_statistic, dof_between, dof_within))
        stat_significance, is_significant = significance_check(p_value, critical_level)
        test_type = "right_tailed"
    else:
        raise ValueError("at least one test type has to be chosen. If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")

    return FTestResult(
        f_statistic=f_statistic,
        f_crit=f_crit,
        p_value=p_value,
        stat_significance=stat_significance,
        is_significant=is_significant,
        test_type=test_type
    ) 

def _tailed_proportion_test(
        z_statistic: float, 
        alpha: float, 
        two_tailed: bool = False, 
        left_tailed: bool = False, 
        right_tailed: bool = False
) -> TailedProportionTestResult:
    
    if two_tailed: 
        z_crit = float(scipy.stats.norm.ppf((1 - (alpha/2))))
        p_value = float(2 * scipy.stats.norm.sf(abs(z_statistic)))
        stat_significance, is_significant =  significance_check(p_value, alpha)
        test_type = "two_tailed"
       
    elif left_tailed: 
        z_crit = float(scipy.stats.norm.ppf((alpha)))
        p_value = float(scipy.stats.norm.cdf(z_statistic))
        stat_significance, is_significant = significance_check(p_value, alpha)
        test_type = "left_tailed"
  
    elif right_tailed: 
        z_crit = float(scipy.stats.norm.ppf((1 - alpha)))
        p_value = float(scipy.stats.norm.sf(z_statistic))
        stat_significance, is_significant = significance_check(p_value, alpha)
        test_type = "right_tailed" 

    else:
        raise ValueError("at least one test type has to be chosen. If you're using a dict, test type has to be passed as a seperate argument outside the dictionary")

    return TailedProportionTestResult(
        z_crit=z_crit,
        p_value=p_value,
        stat_significance=stat_significance,
        is_significant=is_significant,
        test_type=test_type
    )

def _input_handling_chi2(
        table: pd.DataFrame | dict[str, list], 
        cols: list[str] | None = None, 
        is_crosstab: bool = False
) -> pd.DataFrame:
    
    if cols is not None:
        cols_str = _check_if_string(cols)

        if isinstance(table, dict):

            for k, v in table.items():
                if not isinstance(k, str):
                    raise ValueError(f"Invalid data type found for key. Found: {type(k)}, expected: str")
                elif not isinstance(v, list):
                    raise ValueError(f"Invalid data type found for value. Found: {type(v)}, expected: list")
                
            df = pd.DataFrame(table)
            table = pd.crosstab(df[cols_str[0]], df[cols_str[1]])    

        elif isinstance(table, pd.DataFrame):
            if is_crosstab:
                table = table
            else:
                if table.shape[1] < 2:
                    raise ValueError("Invalid DataFrame shape. Need at least two columns to build a crosstab")
                table = pd.crosstab(table[cols_str[0]], table[cols_str[1]])
        else: 
            raise ValueError("Invalid input datatype")
    else:
        if isinstance(table, dict):

            for k, v in table.items():
                if not isinstance(k, str):
                    raise ValueError(f"Invalid data type found for key. Found: {type(k)}, expected: str")
                elif not isinstance(v, list):
                    raise ValueError(f"Invalid data type found for value. Found: {type(v)}, expected: list")
                            
            df = pd.DataFrame(table)
            table = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1])

        elif isinstance(table, pd.DataFrame):
            if is_crosstab:
                table = table
            else:
                if table.shape[1] < 2:
                    raise ValueError("Invalid DataFrame shape. Need at least two columns to build a crosstab")
                
                table = pd.crosstab(table.iloc[:, 0], table.iloc[:, 1])
        else: 
            raise ValueError("Invalid input datatype")

    if (table < 0).any().any():
        raise ValueError("Invalid sample data. Values cannot be negative")

    return table

def _calculate_chi2(
    table: pd.DataFrame,
    alpha: float = 0.05,
    Yates: bool = True,
) -> Chi2Result:

    totals_cols = table.sum(axis=0) # underneath
    totals_rows = table.sum(axis=1) # right

    if (totals_cols == 0).any() or (totals_rows == 0).any():
        raise ValueError("Invalid sample data. Row or column totals cannot be equal to zero")
    for i in table.columns:
        if _get_dtype(table[i].dtype) != "numeric":
            raise ValueError(f"Non numeric entries found in column: {i}")
    num_rows = len(table)
    num_cols = len(table.columns)
    distinct_combinations = num_rows*num_cols

    observed = table.to_numpy()

    n = int(observed.sum()) # .sum() on a xd array just sums up every entry

    expected = np.outer(totals_rows, totals_cols) / n

    if (expected == 0).any():   # any on a np.array without specifying axis (or axis=None) checks entire array
        raise ValueError("Empty row or column: expected count of 0.")

    if num_cols == 2 and num_rows == 2 and Yates:
        X2 = float(((abs(expected - observed) - 0.5 )**2 / expected).sum())
    else:
        X2 = float(((expected - observed)**2 / expected).sum())

    if expected.min() < 1 or (expected < 5).mean() > 0.2:
        warnings.warn("Expected counts are small; the chi-square approximation may be unreliable.")

    dof = (num_rows - 1) * (num_cols - 1)
    chi2_crit = float(scipy.stats.chi2.ppf(1 - alpha, dof))  # inverse of cdf, gives quantile for pct value
    p_value = float(scipy.stats.chi2.sf(X2, dof))  # .sf : probability of right tail for given X2 and dof
    V = float(np.sqrt(X2 / (n * min(num_rows - 1, num_cols - 1))))
    stat_significance, is_significant = significance_check(p_value, alpha)

    return Chi2Result(
        n=n,
        V=V,
        distinct_combinations=distinct_combinations,
        stat_significance=stat_significance,
        dof=dof,
        p_value=p_value,
        is_significant=is_significant,
        X2=X2,
        chi2_crit=chi2_crit
    )
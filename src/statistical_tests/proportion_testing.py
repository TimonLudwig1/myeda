# is proportion different 2 or >2 groups 
import warnings
from src.column_utils import _get_dtype, _check_if_string
from typing import Any
from src.statistical_tests.stats_utils import _tailed_proportion_test, significance_check, _input_handling_chi2, _calculate_chi2
from math import sqrt
import numpy as np
import numpy.typing as npt
import pandas as pd

# helper
def _normalize_labels(labels: str | list[str]) -> set[str]:
    if isinstance(labels, str):
        labels = [labels]
    return {l.strip().lower() for l in labels}

def _convert_to_bool_array(success: str | list[str], fail: str, data: list[str]) -> tuple[npt.NDArray[np.bool_], int]:

    _check_if_string([success, fail])
    data_str = _check_if_string(data)

    success_set = _normalize_labels(success)
    fail_set = _normalize_labels(fail)

    if fail_set.intersection(success_set):
        raise ValueError("success category cannot be same as fail category")
    
    data= [d.strip().lower() for d in data_str] 
    n = len(data)

    invalid = set(data) - success_set.union(fail_set)
    if invalid:
        raise ValueError(f"Entries other than the defined categories found: {invalid}")
    
    success_array = np.array([d in success_set for d in data], dtype=bool)
    return success_array, n

def _proportion_array_handler(
    data: list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series,
    success: str | list[str] | None = None,
    fail: str | None = None,
) -> tuple[float, int]:

    if isinstance(data, pd.Series):
        if data.isna().any():
            raise ValueError("Series contains NaN values")
        if _get_dtype(data.dtype) == "boolean":
            n = len(data)
            success_array = data.to_numpy()
        elif set(data) <= {0, 1}:
            n = len(data)
            success_array = data.to_numpy().astype(bool)
        elif success is not None and fail is not None:
            if _get_dtype(data.dtype) != "categorical":
                raise ValueError("Non-object type Series found")
            else:
                data = data.to_list()
                data_str = _check_if_string(data)
                success_array, n = _convert_to_bool_array(success=success, fail=fail, data=data_str)
        else:
            raise ValueError("Non bool Series was passed without definitions for success/fail categories")
        
    elif isinstance(data, list):
        if not data:
            raise ValueError("List is empty")
        
        if set(data) <= {0, 1}:
            n = len(data)
            success_array = np.array(data)
        
        elif success is not None and fail is not None:
            data_str = _check_if_string(data)
            success_array, n = _convert_to_bool_array(success=success, fail=fail, data=data_str)
        else:
            raise ValueError("Non bool list was passed without definitions for success/fail categories")

    elif isinstance(data, np.ndarray):
        if np.isin(data, [0, 1, True, False]).all():
            success_array = data
            n = len(data)
        else:
            raise ValueError("Non-boolean np.array was passed")
    else:
        raise TypeError(f"Unsupported data type {type(data)} passed")
    p_hat = success_array.sum() / n

    return float(p_hat), int(n)

def one_samp_z_test_proportions(
    p0: float,      # hypothesized under H0
    p_hat: float | None = None,   # observed from sample
    n: int | None = None,
    alpha: float = 0.05,
    two_tailed: bool = False,
    right_tailed: bool = False,
    left_tailed: bool = False,
    min_count: int = 10,
    data: list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series | None = None,
    success: str | None = None,
    fail: str | None = None,
) -> dict[str, object]:
    
    if sum([two_tailed, right_tailed, left_tailed]) != 1:
        raise ValueError("Exactly one of two_tailed, left_tailed, right_tailed must be True.")

    if data is not None:
        p_hat, n = _proportion_array_handler(data=data, success=success, fail=fail)

    elif p_hat is None or n is None:
        raise ValueError("Neither data nor p_hat and n are given")
    
    if n * p0 < min_count or n * (1 - p0) < min_count:
        raise ValueError(
            "Success-failure condition not met: n*p0 and n*(1-p0) must both "
            f"be >= {min_count} (got n*p0 = {n * p0:.1f}, "
            f"n*(1-p0) = {n * (1 - p0):.1f})."
        )
    standard_error = sqrt(( (p0*(1-p0)) / n))
    z_score = (p_hat - p0) / standard_error

    result = _tailed_proportion_test(z_statistic=z_score, alpha=alpha, two_tailed=two_tailed, right_tailed=right_tailed, left_tailed=left_tailed)

    results = {
        "sample_proportion": p_hat,
        "hypothesized_population_proportion": p0,
        "sample_size": n,
        "z_score": z_score,
        "standard_error": standard_error,
        "z_crit": result.z_crit,
        "p_value": result.p_value,
        "statistical_significance": result.stat_significance,
        "is_significant": result.is_significant,
        "test_type": result.test_type
    }
    return results

# rn groups have to be in long format: cols needs to be [group, value]
def two_samp_z_test_proportions(
    p1_hat: float | list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series | None = None,
    p2_hat: float | list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series | None = None,
    n1: int | list[int] | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    two_tailed: bool = False,
    right_tailed: bool = False,
    left_tailed: bool = False,
    min_count: int = 10, 
    data: pd.DataFrame | None = None,
    cols: list[str] | None = None,
    success: str | None = None,
    fail: str | None = None,
) -> dict[str, object]:
    
    if sum([two_tailed, right_tailed, left_tailed]) != 1:
            raise ValueError("Exactly one of two_tailed, left_tailed, right_tailed must be True.")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            if data.isna().to_numpy().any():
                raise ValueError("NaN-Values found in DataFrame.")
            if cols is None:
                raise ValueError("When working with a DataFrame, pass a list of column names for cols=")
            else:
                if len(cols) < 2 or len(cols) > 2:
                    raise ValueError("Invalid number of column names. Specify exactly two column names")
                cols = _check_if_string(cols)
                # filter rows for the two groups with loc 
                groups = data[cols[0]].unique()
                group_one = data.loc[data[cols[0]] == groups[0], cols[1]]
                group_two = data.loc[data[cols[0]] == groups[1], cols[1]]
                p1_hat, n1 = _proportion_array_handler(data=group_one, success=success, fail=fail)
                p2_hat, n2 = _proportion_array_handler(data=group_two, success=success, fail=fail)
        else:
            raise ValueError(f"Invalid data type for 'data'. Found {type(data)}. Needed pd.DataFrame")
    else:
        if isinstance(p1_hat, float) and isinstance(p2_hat, float):
            if isinstance(n1, list):
                if n2 is not None:
                    warnings.warn("n2 was passed while n1 was passed as a list. The value for n2 will be ignored")
                if len(n1) > 2 or len(n1) < 2:
                    raise ValueError(f"Invalid number of entries found in list of sample_sizes. Number of entries: {len(n1)}")
                for i in n1:
                    if not isinstance(i, int):
                        raise ValueError(f"Non-numeric entry found in {n1}")
                n2 = n1[1]
                n1 = n1[0]

            elif isinstance(n1, int):
                if n2 is None:
                    raise ValueError("n2 is missing. n1 was passed as an int, so n2 must be passed as an int as well")
                elif not isinstance(n2, int):
                    raise ValueError(f"Invalid data type for n2. Got {type(n2)}, expected: int")
            else:
                raise ValueError(f"Invalid data type for n1. Found {type(n1)}, expected: int | list")

        elif isinstance((p1_hat), (list, np.ndarray, pd.Series)) and isinstance(p2_hat, (list, np.ndarray, pd.Series)):
            if n1 is not None or n2 is not None:
                warnings.warn("n1 and n2 will be calculated using p1_hat and p2_hat")
            p1_hat, n1 = _proportion_array_handler(data=p1_hat, success=success, fail=fail)
            p2_hat, n2 = _proportion_array_handler(data=p2_hat, success=success, fail=fail)
        else:
            raise ValueError("Invalid combination of data types")
                
    if n1 * p1_hat < min_count or n1 * (1 - p1_hat) < min_count:
        raise ValueError(
            "Success-failure condition not met: n1*p1_hat and n1*(1-p1_hat) must both "
            f"be >= {min_count} (got n1*p1_hat = {n1 * p1_hat:.1f}, "
            f"n1*(1-p1_hat) = {n1 * (1 - p1_hat):.1f})."
        )
    if n2 * p2_hat < min_count or n2 * (1 - p2_hat) < min_count:
        raise ValueError(
            "Success-failure condition not met: n2*p2_hat and n2*(1-p2_hat) must both "
            f"be >= {min_count} (got n2*p2_hat = {n2 * p2_hat:.1f}, "
            f"n2*(1-p2_hat) = {n2 * (1 - p2_hat):.1f})."
        )

    p_c_hat = (n1*p1_hat + n2*p2_hat) / (n1 + n2)

    standard_error = sqrt(p_c_hat * (1 - p_c_hat) * (1/n1 + 1/n2))
    z_statistic = float((p1_hat - p2_hat) / standard_error)

    result = _tailed_proportion_test(z_statistic=z_statistic, alpha=alpha, two_tailed=two_tailed, 
                                                                                                                                        left_tailed=left_tailed, right_tailed=right_tailed)

    results = {
        "proportion_group_1": p1_hat,
        "proportion_group_2": p2_hat,
        "sample_size_1": n1,
        "sample_size_2": n2,
        "significance_level": alpha,
        "z_score": z_statistic,
        "standard_error": standard_error,
        "z_crit": result.z_crit,
        "p_value": result.p_value,
        "statistical_significance": result.stat_significance,
        "is_significant": result.is_significant,
        "test": "two_sample_z_test_for_proportions",
        "test_type": result.test_type
    }

    return results

def chi2_test_homogeneity(
        data: pd.DataFrame | dict[str, list],
        cols: list[str] | None = None, 
        alpha: float = 0.05,
        Yates: bool = True,
        is_crosstab: bool = False
) -> dict[str, object]:

    data = _input_handling_chi2(table=data, cols=cols, is_crosstab=is_crosstab)

    result = _calculate_chi2(table=data, alpha=alpha, Yates=Yates) 

    test_type = "Chi-Square Test of Homogeneity"

    results = {
        "n": result.n,
        "Cramér's V": result.V,
        "unique_category_combinations": result.distinct_combinations,
        "significance": result.stat_significance,
        "test": test_type,
        "dof": result.dof,
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "X²": result.X2,
        "chi2_crit": result.chi2_crit,
        }

    return results

# main method: 2 groups: z test, >2 groups chi2 

def compare_proportions(
    data: pd.DataFrame | None,
    cols: list[str] | None,
    p1_hat: list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series | None = None,
    p2_hat: float | list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series | None = None,
    n1: int | list[int] | None = None,
    n2: int | None = None,
    alpha: float = 0.05,
    two_tailed: bool = False,
    right_tailed: bool = False,
    left_tailed: bool = False,
    min_count: int = 10, 
    success: str | None = None,
    fail: str | None = None,
    Yates: bool = True,
    is_crosstab: bool = False
) -> dict[str, object]:

    if is_crosstab:     # hast to be chi2 then
        if isinstance(data, pd.DataFrame):
            return chi2_test_homogeneity(
                data=data, cols=cols, alpha=alpha,Yates=Yates,is_crosstab=is_crosstab
                )
        else: 
            raise ValueError("is_instance set to true, even tho ""data"f" is not a pd.DataFrame. Got {type(data)}, expected: pd.Dataframe")
        
    elif data is not None and cols is not None:
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(f"Invalid data taype for `data`. Got {type(data)}, expected: pd.DataFrame")
        # data is a df, for chi 2 one col is group and one col is category, for z test one col per sample group
        if len(data[cols[0]].unique()) == 2:
            return two_samp_z_test_proportions(
                p1_hat=p1_hat, p2_hat=p2_hat, n1=n1, n2=n2, alpha=alpha, 
                two_tailed=two_tailed, right_tailed=right_tailed, left_tailed=left_tailed,
                min_count=min_count, data=data, cols=cols, success=success, fail=fail
            )
        elif len(data[cols[0]].unique()) <= 2:
            raise ValueError(f"Invalid number of cols passed. Expected 2: [group, values], got: {len(data[cols[0]].unique())}")
        else:
            return chi2_test_homogeneity(
                            data=data, cols=cols, alpha=alpha,Yates=Yates,is_crosstab=is_crosstab
                            )

    elif data is None and p1_hat is not None:
        return two_samp_z_test_proportions(
                        p1_hat=p1_hat, p2_hat=p2_hat, n1=n1, n2=n2, alpha=alpha, 
                        two_tailed=two_tailed, right_tailed=right_tailed, left_tailed=left_tailed,
                        min_count=min_count, data=data, cols=cols, success=success, fail=fail
                    )
    elif data is None and p1_hat is None:
        raise ValueError("Invalid combination of arguments. data and p1_hat missing")

    else:
        # unreachable
        raise ValueError("Something went wrong")
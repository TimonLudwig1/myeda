# is proportion different 2 or >2 groups 
from column_utils import _get_dtype, _check_if_string
from typing import Any
from src.statistical_tests.stats_utils import _tailed_proportion_test
from math import sqrt
import numpy as np
import numpy.typing as npt
import pandas as pd

# helper

def _convert_to_bool_array(success: str, fail: str, data: list[str]) -> tuple[npt.NDArray[np.bool_], int]:
    """
    Convert a list of category strings to a boolean success array.

    All inputs are stripped and lowercased before an exact comparison, so
    "Yes " matches success="yes".

    Args:
        success: Category counted as a success.
        fail: Category counted as a failure. Must differ from `success`
            after normalization.
        data: Observations as strings.

    Returns:
        Tuple (success_array, n): boolean numpy array (True = success)
        and the number of observations.

    Raises:
        ValueError: If success and fail are identical, or if `data`
            contains entries other than the two categories (the offending
            entries are listed in the message).
    """

    _check_if_string([success, fail])
    success = success.strip().lower()
    fail = fail.strip().lower()

    if success == fail:
        raise ValueError("success category cannot be same as fail category")
    
    data = [d.strip().lower() for d in data]
    n = len(data)

    invalid = set(data) - {success, fail}
    if invalid:
        raise ValueError(f"Entries other than the defined categories found: {invalid}")
    
    success_array = np.array([success == d for d in data])
    return success_array, n

def _proportion_array_handler(
    data: list[str] | npt.NDArray[np.bool_] | pd.Series,
    success: str | None = None,
    fail: str | None = None,
) -> tuple[float, int]:
    """
    Reduce raw data to the sample proportion and sample size.

    Accepts a pandas Series, list or numpy array. Boolean and 0/1 data are
    counted directly. Other data (strings/categories) are converted via
    `_convert_to_bool_array` and therefore require `success` and `fail`;
    for boolean or 0/1 data these two arguments are ignored.

    Args:
        data: Observations. Must not be empty or contain NaN.
        success: Category counted as a success (non-boolean data only).
        fail: Category counted as a failure (non-boolean data only).

    Returns:
        Tuple (p_hat, n) with the proportion of successes and the number
        of observations.

    Raises:
        ValueError: If the data are empty, contain NaN or values outside
            the defined categories, or if non-boolean data are passed
            without `success`/`fail`.
        TypeError: If `data` is not a Series, list or numpy array.
    """

    if isinstance(data, pd.Series):
        if data.isna().any():
            raise ValueError("Series contains NaN values")
        if _get_dtype(data.dtype) == "boolean":
            n = len(data)
            success_array = data.to_numpy()

        elif success is not None and fail is not None:
            if _get_dtype(data.dtype) != "categorical":
                raise ValueError("Non-object type Series found")
            else:
                data = data.to_list()
                _check_if_string(data)
                success_array, n = _convert_to_bool_array(success=success, fail=fail, data=data)
        else:
            raise ValueError("Non bool Series was passed without definitions for success/fail categories")
        
    elif isinstance(data, list):
        if not data:
            raise ValueError("List is empty")
        
        if set(data) <= {0, 1}:
            n = len(data)
            success_array = np.array(data)
        
        elif success is not None and fail is not None:
            _check_if_string(data)
            success_array, n = _convert_to_bool_array(success=success, fail=fail, data=data)
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

    return p_hat, n
    

def one_samp_z_test_proportions(
    p0: float,      # hypothesized under H0
    p_hat: float | None = None,   # observed from sample
    n: int | None = None,
    alpha: float = 0.05,
    two_tailed: bool = False,
    right_tailed: bool = False,
    left_tailed: bool = False,
    min_count: int = 10,
    data: list[str] | npt.NDArray[np.bool_] | pd.Series | None = None,
    success: str | None = None,
    fail: str | None = None,
) -> dict[str, Any]:
    """
    One-sample z-test for a proportion.

    Tests whether an observed sample proportion differs from the
    hypothesized population proportion p0. The input can be either raw data
    (`data`, plus `success`/`fail` if the data are not already boolean) or
    the summary values `p_hat` and `n`. If `data` is given, `p_hat` and `n`
    are ignored.

    The success-failure condition (n*p0 >= min_count and
    n*(1-p0) >= min_count) is checked with p0, since the test assumes H0
    is true.

    Args:
        p0: Hypothesized population proportion under H0.
        p_hat: Observed sample proportion. Required together with `n`
            if `data` is not given.
        n: Sample size. Required together with `p_hat` if `data` is not given.
        alpha: Significance level. Default 0.05.
        two_tailed: Test H1: p != p0.
        right_tailed: Test H1: p > p0.
        left_tailed: Test H1: p < p0.
        min_count: Minimum for n*p0 and n*(1-p0) in the success-failure
            condition. Default 10.
        data: Raw observations as a pandas Series, list or numpy array.
            Boolean or 0/1 values are used as is. Other values require
            `success` and `fail`.
        success: Category counted as a success (case- and
            whitespace-insensitive). Only used for non-boolean data.
        fail: Category counted as a failure. Only used for non-boolean data.

    Returns:
        Dict with the keys sample_proportion, hypothesized_population_proportion,
        sample_size, z_score, z_crit, statistical_significance,
        is_significant and test_type. For two-tailed tests, z_crit is the
        positive critical value (the rejection region is beyond +/- z_crit).

    Raises:
        ValueError: If not exactly one of the three tail flags is True,
            if neither `data` nor both `p_hat` and `n` are given, if the
            success-failure condition is not met, or if the data are invalid.
    """
    
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

    z_score = (p_hat - p0) / sqrt(( (p0*(1-p0)) / n))

    z_crit, p_value, stat_significance, is_significant, test_type = _tailed_proportion_test(z_statistic=z_score, alpha=alpha, two_tailed=two_tailed, right_tailed=right_tailed, left_tailed=left_tailed)

    results = {
        "sample_proportion": p_hat,
        "hypothesized_population_proportion": p0,
        "sample_size": n,
        "z_score": z_score,
        "z_crit": z_crit,
        "statistical_significance": stat_significance,
        "is_significant": is_significant,
        "test_type": test_type
    }
    return results

def two_samp_z_test_proportions():
    pass

# main method

def compare_proportions():
    pass
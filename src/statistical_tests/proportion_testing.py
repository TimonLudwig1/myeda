# is proportion different 2 or >2 groups 
import warnings
from src.column_utils import _get_dtype, _check_if_string
from typing import Any
from src.statistical_tests.stats_utils import _tailed_proportion_test, significance_check
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
    if set(data) <= {0, 1}:
        n = len(data)
        success_array = np.array(data)
        return success_array, n
    else:
        [success, fail] = _check_if_string([success, fail])
        data_str = _check_if_string(data)

        success = success.strip().lower()
        fail = fail.strip().lower()

        if success == fail:
            raise ValueError("success category cannot be same as fail category")
        
        data= [d.strip().lower() for d in data_str] 
        n = len(data)

        invalid = set(data) - {success, fail}
        if invalid:
            raise ValueError(f"Entries other than the defined categories found: {invalid}")
        
        success_array = np.array([success == d for d in data])
        return success_array, n

def _proportion_array_handler(
    data: list[str] | list[bool] | npt.NDArray[np.bool_] | pd.Series,
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

    return p_hat, n
    
# add option for data to be dict
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
        sample_size, z_score,, standard_error, z_crit, p_value, statistical_significance,
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
    standard_error = sqrt(( (p0*(1-p0)) / n))
    z_score = (p_hat - p0) / standard_error

    z_crit, p_value, stat_significance, is_significant, test_type = _tailed_proportion_test(z_statistic=z_score, alpha=alpha, two_tailed=two_tailed, 
                                                                                                                                        right_tailed=right_tailed, left_tailed=left_tailed)

    results = {
        "sample_proportion": p_hat,
        "hypothesized_population_proportion": p0,
        "sample_size": n,
        "z_score": z_score,
        "standard_error": standard_error,
        "z_crit": z_crit,
        "p_value": p_value,
        "statistical_significance": stat_significance,
        "is_significant": is_significant,
        "test_type": test_type
    }
    return results

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
):
    """
    Two-sample z-test for the difference between two independent proportions.

    Tests H0: p1 = p2 and uses the pooled proportion for the standard error.
    The input can be given as sample proportions plus sample sizes, as raw
    observations of each group, or as two columns of a DataFrame.

    Parameters
    ----------
    p1_hat, p2_hat : float or list or numpy.ndarray or pandas.Series, optional
        Group 1 and group 2. Both must be of the same kind:

        - Floats: the sample proportions (between 0 and 1). ``n1`` and ``n2``
          are required in this case.
        - Raw observations: booleans, or labels such as strings that are
          matched via ``success`` / ``fail``. The proportions and sample sizes
          are calculated from the data, so ``n1`` and ``n2`` are ignored
          (a warning is raised if they are passed anyway).

        Ignored if ``data`` is passed.
    n1 : int or list of int, optional
        Sample size of group 1. Can also be a list ``[n1, n2]``; ``n2`` is then
        ignored (with a warning). Only used if proportions are passed as floats.
    n2 : int, optional
        Sample size of group 2. Only used if proportions are passed as floats.
    alpha : float, default 0.05
        Significance level.
    two_tailed : bool, default False
        Test H1: p1 != p2.
    right_tailed : bool, default False
        Test H1: p1 > p2.
    left_tailed : bool, default False
        Test H1: p1 < p2.

        Exactly one of ``two_tailed``, ``right_tailed`` and ``left_tailed``
        must be True.
    min_count : int, default 10
        Minimum number of expected successes and failures per group
        (success-failure condition). Both ``n * p_hat`` and ``n * (1 - p_hat)``
        must be at least ``min_count`` for each group.
    data : pandas.DataFrame, optional
        DataFrame holding the raw observations. Each of the two columns in
        ``cols`` is treated as one independent group.
    cols : list of str, optional
        Names of exactly two columns of ``data`` (group 1, group 2).
        Required if ``data`` is passed.
    success, fail : str, optional
        Labels that mark a success or a failure in the raw observations,
        e.g. ``success="passed"``, ``fail="failed"``.

    Returns
    -------
    dict
        Dictionary with the keys ``proportion_group_1``, ``proportion_group_2``,
        ``sample_size_1``, ``sample_size_2``, ``significance_level``,
        ``z_score``, ``standard_error``, ``z_crit``, ``p_value``,
        ``statistical_significance`` (readable summary), ``is_significant``
        (bool) and ``test_type``.

    Raises
    ------
    ValueError
        If not exactly one test direction is selected, if the input types or
        combination are invalid, if ``cols`` does not contain exactly two
        column names, if a sample size is missing, or if the success-failure
        condition is not met for one of the groups.

    Warns
    -----
    UserWarning
        If ``n1`` / ``n2`` are passed although they are ignored or calculated
        from the data.

    Notes
    -----
    The two groups must be independent. For paired data (the same subjects
    measured twice), use McNemar's test instead.

    The test statistic is::

        z = (p1_hat - p2_hat) / sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    with ``p_pooled = (n1 * p1_hat + n2 * p2_hat) / (n1 + n2)``.
    """
    
    if sum([two_tailed, right_tailed, left_tailed]) != 1:
            raise ValueError("Exactly one of two_tailed, left_tailed, right_tailed must be True.")
    if data is not None:
        if isinstance(data, pd.DataFrame):
            if cols is None:
                raise ValueError("When working with a DataFrame, pass a list of column names for cols=")
            else:
                if len(cols) < 2 or len(cols) > 2:
                    raise ValueError("Invalid number of column names. Specify exactly two column names")
                cols = _check_if_string(cols)
                p1_hat, n1 = _proportion_array_handler(data=data[cols[0]], success=success, fail=fail)
                p2_hat, n2 = _proportion_array_handler(data=data[cols[1]], success=success, fail=fail)
        else:
            raise ValueError(f"Invalid data type for 'data'. Found {type(data)}. Needed pd.DataFrame")
    else:
        if isinstance(p1_hat, float) and isinstance(p2_hat, float):
            if n1 is not None:
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
            else:
                raise ValueError("n1 is None. If p1_hat and p2_hat are passed as a float, n1 cannot be None")

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

    z_crit, p_value, stat_significance, is_significant, test_type = _tailed_proportion_test(z_statistic=z_statistic, alpha=alpha, two_tailed=two_tailed, 
                                                                                                                                        left_tailed=left_tailed, right_tailed=right_tailed)

    results = {
        "proportion_group_1": p1_hat,
        "proportion_group_2": p2_hat,
        "sample_size_1": n1,
        "sample_size_2": n2,
        "significance_level": alpha,
        "z_score": z_statistic,
        "standard_error": standard_error,
        "z_crit": z_crit,
        "p_value": p_value,
        "statistical_significance": stat_significance,
        "is_significant": is_significant,
        "test_type": test_type
    }

    return results

def chi_sq_test_homogeneity():
    pass



# main method

def compare_proportions():
    pass
# association or correlation? pearson, chi-squared, spearman
import warnings
from column_utils import _get_dtype
import numpy as np 
from numpy.typing import NDArray
from src.column_utils import _check_if_string
import pandas as pd 
from src.statistical_tests.input_validation import _calculate_mean
from src.statistical_tests.stats_utils import significance_check
from scipy.stats import rankdata, chi2

# helpers
def _array_conversion(cols, b):
    if not isinstance(cols, (list, np.ndarray)) or not isinstance(b, (list, np.ndarray)):
        raise ValueError("Invalid arg data types")
    if len(cols) != len(b):
        raise ValueError("arrays need to have the same amount of entries")
    if isinstance(cols, list):
        x_arr = np.array(cols)
    else:
        x_arr = cols

    if isinstance(b, list):
        y_arr = np.array(b)
    else:
        y_arr = b

    if not pd.api.types.is_numeric_dtype(x_arr) or not pd.api.types.is_numeric_dtype(y_arr):
        raise ValueError("array contains non-numeric values")

    return x_arr, y_arr
    
def pearson_correlation(
        df: pd.DataFrame | None = None, 
        cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
        b: list[float] | NDArray[np.float64] | None = None, 
) -> float:
    if df is not None and cols is not None:
        _check_if_string(cols)
        if len(cols) != 2:
            raise ValueError(f"exactly two columns required, got {len(cols)}")
        
        x, y = cols

        data = df[[x, y]].dropna()

        x_arr = data[x].to_numpy(dtype=float)
        y_arr = data[y].to_numpy(dtype=float)

        dev_x = x_arr - _calculate_mean(df, x)  # type: ignore
        dev_y = y_arr - _calculate_mean(df, y)  # type: ignore
    
    elif cols is not None and b is not None:
        x_arr, y_arr = _array_conversion(cols, b)

        dev_x = x_arr - x_arr.mean()
        dev_y = y_arr - y_arr.mean()
    else:
        raise ValueError("Invalid combination of args")
    
    sum_of_products = (dev_x*dev_y).sum()

    ss_x = (dev_x**2).sum()
    ss_y = (dev_y**2).sum()

    return sum_of_products / np.sqrt(ss_x*ss_y)

# add option to just pass a list of vals or a np.array and no df
def spearman_correlation(
        df: pd.DataFrame | None = None, 
        cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
        b: list[float] | NDArray[np.float64] | None = None, 
) -> float:
    if df is not None and cols is not None:
        _check_if_string(cols)
        if len(cols) != 2:
            raise ValueError(f"exactly two columns required, got {len(cols)}")

        x, y = cols

        data = df[[x, y]].dropna()

        rank_x = df[x].rank().to_numpy()
        rank_y = df[y].rank().to_numpy()

    elif cols is not None and b is not None:
        x_arr, y_arr = _array_conversion(cols, b)

        rank_x = rankdata(x_arr)
        rank_y = rankdata(y_arr)
    else:
        raise ValueError("Invalid combination of args")
    
    dev_x = rank_x - rank_x.mean()
    dev_y = rank_y - rank_y.mean()

    return (dev_x * dev_y).sum() / np.sqrt((dev_x**2).sum() * (dev_y**2).sum())

def chi_sq_independence(table: pd.DataFrame, alpha: float = 0.05, Yates: bool = True) -> dict:
    """
    pass Yates = Flase if you do not want the Yate's continuity correction. Default: True
    """
    # start just with a ready crosstab - later pass arrays/lists and It'll make the crosstab automatically
    # add input checks next
    if (table < 0).any().any():
        raise ValueError("Invalid sample data. Values cannot be negative")
    totals_cols = table.sum(axis=0) # underneath
    totals_rows = table.sum(axis=1) # right

    if (totals_cols == 0).any() or (totals_rows == 0).any():
        raise ValueError("Invalid smaple data. Row or column totals cannot be equal to zero")
    for i in table.columns:
        if _get_dtype(table[i] != "numeric"):
            raise ValueError(f"Non numeric entries found in column: {i}")
    num_rows = len(table)
    num_cols = len(table.columns)
    num_data_points = num_rows*num_cols
    grand_total = table.sum().sum()

    observed = table.to_numpy()

    n = observed.sum() # .sum() on a xd array just sums up every entry

    expected = np.outer(totals_cols, totals_rows) / grand_total

    if (expected == 0).any():   # any on a np.array without specifying axis (or axis=None) checks entire array
        raise ValueError("Empty row or column: expected count of 0.")

    if Yates:
        X2 = ((expected - observed)**2 / expected).sum()        
    else:
        X2 = ((abs(expected - observed) - 0.5 )**2 / expected).sum()

    if expected.min() < 1 or (expected < 5).mean() > 0.2:
        warnings.warn("Expected counts are small; the chi-square approximation may be unreliable.")

    dof = (num_rows - 1) * (num_cols - 1)
    chi2_crit = chi2.ppf(1 - alpha, dof)  # inverse of cdf, gives quantile for pct value
    p_value = float(chi2.sf(X2, dof))  # .sf : probability of right tail for given X2 and dof
    V = np.sqrt(X2 / (n * min(num_rows - 1, num_cols - 1)))
    stat_significance, is_significant = significance_check(p_value, alpha)

    results = {
        "n": n,
        "Cramér's V": V,
        "sample_data_points": num_data_points,
        "significance": stat_significance,
        "test_type": "Chi-Square Test of Independence",
        "dof": dof,
        "p_value": p_value,
        "is_significant": is_significant,
        "X²": X2,
        "chi2_crit": chi2_crit,
        }

    return results


    

# main method
def check_relationship():
    # auto check if cat or num, and maybe auto check if spearman or perason with F statistic + derivative
    pass


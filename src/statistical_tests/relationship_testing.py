# association or correlation? pearson, chi-squared, spearman
import numpy as np 
from numpy.typing import NDArray
from src.column_utils import _check_if_string
import pandas as pd 
from src.statistical_tests.input_validation import _calculate_mean
from scipy.stats import rankdata

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

def chi_sq_independence():
    pass

# main method
def check_relationship():
    # auto check if cat or num, and maybe auto check if spearman or perason with F statistic + derivative
    pass


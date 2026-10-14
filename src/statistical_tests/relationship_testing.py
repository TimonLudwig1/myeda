# association or correlation? pearson, chi-squared, spearman
import numpy as np 
from src.column_utils import _check_if_string
import pandas as pd 
from src.statistical_tests.input_validation import _calculate_mean

# helpers
...

# add option to just pass a list of vals or a np.array and no df
def pearson_correlation(df: pd.DataFrame, cols: list[str]) -> float:
    _check_if_string(cols)
    if len(cols) != 2:
        raise ValueError(f"exactly two columns required, got {len(cols)}")
    
    x, y = cols

    data = df[[x, y]].dropna()

    x_arr = data[x].to_numpy(dtype=float)
    y_arr = data[y].to_numpy(dtype=float)

    dev_x = x_arr - _calculate_mean(df, x)
    dev_y = y_arr - _calculate_mean(df, y)

    sum_of_products = (dev_x*dev_y).sum()

    ss_x = (dev_x**2).sum()
    ss_y = (dev_y**2).sum()

    return sum_of_products / np.sqrt(ss_x*ss_y)

# add option to just pass a list of vals or a np.array and no df
def spearman_correlation(df: pd.DataFrame, cols: list[str]):
    _check_if_string(cols)
    if len(cols) != 2:
        raise ValueError(f"exactly two columns required, got {len(cols)}")

    x, y = cols

    data = df[[x, y]].dropna()

    rank_x = df[x].rank().to_numpy()
    rank_y = df[y].rank().to_numpy()
    
    dev_x = rank_x - rank_x.mean()
    dev_y = rank_y - rank_y.mean()

    return (dev_x * dev_y).sum() / np.sqrt((dev_x**2).sum() * (dev_y**2).sum())

def chi_sq_independence():
    pass

# main method
def check_relationship():
    # auto check if cat or num, and maybe auto check if spearman or perason with F statistic + derivative
    pass


# association or correlation? pearson, chi-squared, spearman
import warnings
from src.column_utils import _get_dtype
import numpy as np 
from numpy.typing import NDArray
from src.column_utils import _check_if_string
import pandas as pd 
from src.statistical_tests.input_validation import _calculate_mean
from src.statistical_tests.stats_utils import significance_check, _input_handling_chi2, _calculate_chi2
from scipy.stats import rankdata, chi2


#handler and helper
def _handle_relationship_input(
    data: pd.DataFrame | dict[str, list] | None, 
    cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
    b: list[float] | NDArray[np.float64] | None = None, 
): 

    if data is not None:
        if isinstance(data, pd.DataFrame):
            df_raw = data
        elif isinstance(data, dict):
            df_raw = pd.DataFrame(data)

        if cols is not None:
            if isinstance(cols, list):
                cols_str = _check_if_string(cols)
                df = df_raw[cols_str]
                return df
            else:
                raise ValueError("cols is not a list of strings. When working with a DataFrame, pass a list of strings")
            
        else:
            df = df_raw.iloc[:, [0, 2]] # take first two cols if cols is not specified but df/dict passed
            return df
        
    elif data is None and cols is not None and b is not None:
        df = pd.DataFrame(
            data=[cols, b]
        )
        return df
    
    else:
        raise ValueError("Invalid combination of args")

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

def heuristic_corr_choice(
    df: pd.DataFrame, 
    cols: list[str] | list[float] | NDArray[np.float64] | None, 
    b: list[float] | NDArray[np.float64] | None,
    linear_tolerance: float = 0.08,
    min_monotonicity: float = 0.5,
):

    pearson_r = pearson_correlation(df=df, cols=cols, b=b)
    spearman_r = spearman_correlation(df=df, cols=cols, b=b)

    abs_p = abs(pearson_r)
    abs_s = abs(spearman_r)

    # Heuristic step 1: check if a meaningful monotonic relationship exists
    if abs_s < min_monotonicity:
        corr_type = "spearman"
        warnings.warn(f"`is_linear` was not specified; {corr_type} was selected automatically based on a heuristic. Relationship is weak/non-monotonic (Spearman |r_s| = {abs_s:.2f}). This choice may be unreliable.")
        return spearman_r
    
    # step 2: check linearity via magnitude divergence between spearman and pearson
    delta = abs_s - abs_p

    if delta <= linear_tolerance:
        corr_type = "pearson"
        warnings.warn(f"`is_linear` was not specified; {corr_type} was selected automatically based on a heuristic. This choice may be unreliable.")
        return pearson_r
    else:
        corr_type = "spearman"
        warnings.warn(f"`is_linear` was not specified; {corr_type} was selected automatically based on a heuristic. Relationship is monotonic but non-linear (|r_s| - |r| = {delta:.3f} > {linear_tolerance}. This choice may be unreliable.")
        return spearman_r

# add dict input option + when no cols selected always just use first + second   
def pearson_correlation(
        df: pd.DataFrame | None = None, 
        cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
        b: list[float] | NDArray[np.float64] | None = None, 
) -> float:
    if df is not None and cols is not None:
        cols_str = _check_if_string(cols)
        if len(cols_str) != 2:
            raise ValueError(f"exactly two columns required, got {len(cols_str)}")
        
        x, y = cols_str

        data = df[[x, y]].dropna()

        x_arr = data[x].to_numpy(dtype=float)
        y_arr = data[y].to_numpy(dtype=float)

        dev_x = x_arr - _calculate_mean(df, x)  
        dev_y = y_arr - _calculate_mean(df, y)  
    
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

# add dict input option + when no cols selected always just use first + second
def spearman_correlation(
        df: pd.DataFrame | None = None, 
        cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
        b: list[float] | NDArray[np.float64] | None = None, 
) -> float:
    if df is not None and cols is not None:
        cols_str = _check_if_string(cols)
        if len(cols_str) != 2:
            raise ValueError(f"exactly two columns required, got {len(cols_str)}")

        x, y = cols_str

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

def chi2_test_independence(
        table: pd.DataFrame | dict[str, list], 
        cols: list[str] | None = None, 
        alpha: float = 0.05, 
        Yates: bool = True, 
        is_crosstab: bool = False
) -> dict:
    """
    Performes a Chi-squared test of independence, based on the data you provide.

    Parameters:
    - table: pd.Dataframe | dict | pre made pd.crosstab - the input table containing the observations for the test. Pass is_crosstab = True if you are working with a premade crosstab
    - cols: list[str] optional - The names of the two columns you want to create the crosstab from. If None, the first and second column will be used.
    - alpha: the significance level for the test. Default: = 0.05.
    - Yates: True | False if you want to use the Yate's continuity correction. Default: True
    - is_crosstab: True | False if your DataFrame is already a crosstable. Default: False
    """
    # still more input data (type) checks missing 
    table = _input_handling_chi2(table=table, cols=cols, is_crosstab=is_crosstab)
    
    result = _calculate_chi2(table=table, alpha=alpha, Yates=Yates) 

    test_type = "Chi-Square Test of Independence"

    results = {
        "n": result.n,
        "Cramér's V": result.V,
        "unique_category_combinations": result.distinct_combinations,
        "significance": result.stat_significance,
        "test_type": test_type,
        "dof": result.dof,
        "p_value": result.p_value,
        "is_significant": result.is_significant,
        "X²": result.X2,
        "chi2_crit": result.chi2_crit,
        }

    return results

# main method
def check_relationship(
    df: pd.DataFrame | dict[str, list] | None, 
    cols: list[str] | list[float] | NDArray[np.float64] | None = None, 
    b: list[float] | NDArray[np.float64] | None = None, 
    is_linear: bool | None = None,
    alpha: float = 0.05, 
    Yates: bool = True, 
    is_crosstab: bool = False,
    linear_tolerance: float = 0.08,
    min_monotonicity: float = 0.5,
)-> dict | float:
    
    data = _handle_relationship_input(data=df, cols=cols, b=b)  

    param_types = {
        _get_dtype(data.iloc[:, 0].dtype),
        _get_dtype(data.iloc[:, 1].dtype)
    }

    if param_types == {"numeric", "numeric"}:
        if is_linear is not None:
            # pearson or spearman
            if is_linear:
                # pearson
                return pearson_correlation(df=data, cols=cols, b=b)
            else: 
                # spearman
                return spearman_correlation(df=data, cols=cols, b=b)
        else:
            # heuristic choice
            return heuristic_corr_choice(df=data, cols=cols, b=b, linear_tolerance=linear_tolerance, min_monotonicity=min_monotonicity)

    elif param_types == {"categorical", "categorical"}:
        # chi2
        return chi2_test_independence(table=data, alpha=alpha, Yates=Yates, is_crosstab=is_crosstab)    
    else: 
        raise ValueError("Invalid combination of input data types for variables")

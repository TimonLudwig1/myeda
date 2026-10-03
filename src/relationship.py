import pandas as pd 
from auto_plot import get_dtype
from typing import Any

def compare_means(df: pd.DataFrame, cols: list[str]):
    if type(cols) != list:
        raise ValueError(f"{cols} has to be a list containing at least two column names as strings")
    
    #types: list[str] = [] - not sure if I even need this for mean comparison, since it always has to be numeric anyways

    means = {}

    for col in cols:
        dtype = get_dtype(df[col].dtype)
        if dtype != "numeric":
            raise ValueError(f"{col} has invalid data type: {dtype}. To compare means, all columns must contain numeric values")
        #types.append(dtype)

        means.update({f"mean_{col}": round(df[col].mean(), 2)})

    print(means)
    if len(cols) == 2:
        pass

    else:
        pass

def compare_proportions():
    pass

def check_relationship():
    pass

def compare_variances():
    pass

def relationship_analysis(df: pd.DataFrame, col: str | list):
    pass

compare_means(pd.read_csv(r"/Users/timonludwig/Documents/GitHub/myeda/data/eda_testdata.csv"), ["age", "income"])
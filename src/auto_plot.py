import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Callable
from typing import Any

def is_valid_dtype(dtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    )

def get_dtype(dtype) -> str:
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    elif pd.api.types.is_object_dtype(dtype):
        return "categorical"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    else: 
        raise ValueError(f"Column contains datatype not suitable for plotting. Datatype: {dtype}")

  
# plotting functions 

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, other_col: str | list[str] | None = None):
    fig, ax = plt.subplots()

    if other_col: 
        if type(other_col) is list:
            # treat the entries of the other_cols list as unordered and check the df cols for each entry
            pass
        else: 
            #draw one scatter per category 
            for category, group in df.groupby(other_col):
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    label=category
                )
            ax.set(title=f"{x_col} against {y_col}", xlabel=x_col, ylabel=y_col)
            ax.legend()
            plt.show()
            return fig, ax
    else:
        ax.scatter(df[x_col], df[y_col])

        ax.set_title(f"{x_col} against {y_col}")
        ax.set_xlabel(f"{x_col}")
        ax.set_ylabel(f"{y_col}")
        ax.legend(title=other_col)
        plt.tight_layout()
        plt.show()
        return fig, ax

def plot_stacked_bar(df: pd.DataFrame, x_col: str, y_col: str, other_col: str, agg: str):
    fig, ax = plt.subplots()
    pivot = (
        df.groupby([x_col, other_col])[y_col]
        .agg(agg)
        .unstack(other_col)
        .fillna(0)
    )
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_label(x_col)
    ax.legend(title=f"Sum of {y_col}")
    ax.legend(title=", ".join(other_col), bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return fig, ax 

def plot_grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, agg: str, other_col: str | None = None):
    if other_col:
        fig, ax = plt.subplots()
        df.groupby([x_col, other_col])[y_col].agg(agg).unstack().plot(kind='bar', ax=ax)
        ax.set_title(f"{x_col} against {y_col} and {other_col}")
        ax.set_xlabel(f"{x_col}")
        ax.set_ylabel(f"{y_col}")
        ax.legend(title=other_col)
        plt.tight_layout()
        plt.show()
        return fig, ax
    else:
        fig, ax = plt.subplots()
        counts = df.groupby([x_col, y_col]).size()
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"count of each unique combination of {x_col} and {y_col}")
        ax.set_xlabel("combinations")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        plt.show()
        return fig, ax

def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, other_col: str, agg: str):
    fig, ax = plt.subplots()
    pivot = (
        df.groupby([x_col, other_col])[y_col]
        .agg(agg)
        .unstack()
        .fillna(0)
    )
    sns.heatmap(pivot, cmap="magma")
    ax.set_title(f"{x_col} against {y_col} and {other_col}")
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.legend(title=other_col)
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_bar(df: pd.DataFrame, x_col: str, y_col: str | None):
    fig, ax = plt.subplots()
    ax.bar(df[x_col], df[y_col])
    ax.set(title=f"{y_col} for {x_col}", xlabel=x_col, ylabel=y_col)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_boxplot(df:pd.DataFrame, x_col: str, y_col:str):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set(title=f"{y_col} by {x_col}", xlabel=x_col, ylabel=y_col)
    plt.tight_layout()
    plt.show()
    return fig, ax

def plot_histogram(df:pd.DataFrame, x_col:str):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(df[x_col])
    plt.title(f"distribution of {x_col}")

    plt.subplot(1,2,2)
    df[x_col].plot.density(bw_method='scott', color='blue', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.show()
    return fig 

# handler functions - handlers that just call the plot functions are setup, in case the plotting logic gets more complicated 

def handle_single_num(df: pd.DataFrame, col: str, **_):
    return plot_histogram(df, col)

def handle_single_cat_or_bool(df: pd.DataFrame, col: str, **_):
    counts = df.groupby(col).size()                 
    plot_df = counts.to_frame("count").reset_index()  

    return plot_bar(plot_df, col, "count")

def handle_num_num(df: pd.DataFrame, x_col: str, y_col: str, **_):
    return plot_scatter(df, x_col, y_col)

def handle_num_cat(df: pd.DataFrame, x_col: str, y_col: str, distribution: bool, agg: str):
    if distribution:
        return plot_boxplot(df, x_col, y_col)
    else:
        groups = df.groupby(x_col)
        plot_df = groups[y_col].agg(agg).reset_index()

        return plot_bar(plot_df, x_col, y_col)

def handle_num_datetime(df: pd.DataFrame, x_col: str, y_col: str, agg: str, **_):
    groups = df.groupby(pd.Grouper(key=x_col, freq="D"))
    plot_df = groups[y_col].agg(agg).reset_index()

    return plot_scatter(plot_df, x_col, y_col)

def handle_cat_cat(df: pd.DataFrame, x_col: str, y_col: str, agg: str = "count", **_):
    return plot_grouped_bar(df, x_col, y_col, agg)

def handle_num_num_cat(df: pd.DataFrame, x_col: str, y_col: str, other_col: str, **_):
    return plot_scatter(df, x_col, y_col, other_col)

def handle_num_cat_cat(df: pd.DataFrame, x_col: str, y_col: str, other_col: str, agg: str, additive: bool, **_):
    if not pd.api.types.is_numeric_dtype(df[y_col].dtype):
        raise ValueError("please use a numeric column for the y_column")
    
    if len(df[x_col].unique()) <= 6 and len(df[other_col].unique()) <= 5:
        if additive:
            return plot_stacked_bar(df, x_col, y_col, other_col, agg)
        else: 
            return plot_grouped_bar(df, x_col, y_col, agg, other_col)
    else:
        return plot_heatmap(df, x_col, y_col, other_col, agg)

TYPE_ORDER = {
    "numeric": 0,
    "categorical": 1,
    "datetime": 2
}
PLOT_MAP: dict[tuple, Callable[..., Any]] = {
    ("numeric",): handle_single_num,
    ("categorical",): handle_single_cat_or_bool, 
    ("numeric", "numeric"): handle_num_num, 
    ("numeric", "categorical"): handle_num_cat, 
    ("numeric", "datetime"): handle_num_datetime,
    ("categorical", "categorical"): handle_cat_cat,
    ("numeric", "numeric", "categorical"): handle_num_num_cat,
    ("numeric", "categorical", "categorical"): handle_num_cat_cat,
}

def auto_plot(df: pd.DataFrame, x_col: str, y_col: str | None = None, other_cols: str | list[str] | None = None, agg: str = "sum", additive: bool=False, distribution: bool = False, **kwargs):
    """
    Automatically generates a plot based on the provided DataFrame and specified columns.

    Parameters:
    - df: pd.DataFrame - The input DataFrame containing the data to plot.
    - x_col: str - The name of the column to use for the x-axis.
    - y_col: str - The name of the column to use for the y-axis. Optional if you want to plot only one variable.
    - other_cols: str | list - The name(s) of additional columns to consider for plotting. Can be a single column name or a list of column names.
    - **kwargs: Additional keyword arguments to pass to the plotting function.

    Returns:
    - matplotlib fig, ax containing the generated plot.
    """

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    if not is_valid_dtype(df[x_col].dtype): 
        raise ValueError(f"Column '{x_col}' must be either numeric, categorical (object), or datetime.")
    
    if y_col is not None:
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame.")
        
        if not is_valid_dtype(df[y_col].dtype):
            raise ValueError(f"Column '{y_col}' must be either numeric categorical (object), or datetime.")

        if df[y_col].equals(df[x_col]):
            raise ValueError(f"Column '{y_col}' cannot be the same as column '{x_col}'.")

        if other_cols is not None:
            if isinstance(other_cols, str):
                types = [get_dtype(df[other_cols]), get_dtype(df[y_col]), get_dtype(df[x_col])]               
                key = tuple(sorted( # sorted method loops through types
                    types,
                    key=lambda dtype: TYPE_ORDER[dtype] #lambda  <parameter> : <expression> (return)
                ))
                handler = PLOT_MAP[key]
                handler(
                    df = df,
                    x_col = x_col,
                    y_col = y_col, 
                    other_col = other_cols,
                    agg = agg,
                    additive = additive
                )

            elif isinstance(other_cols, list):
                d_types_other_cols = [df[col].dtype for col in other_cols]
                # check: length of list: I can't be bothered rn to do more than 2 additional variables (so 4 total) - so for now if len(other_cols) > 2 error:
                if len(d_types_other_cols) > 2:
                    raise ValueError("Currently, only up to 2 additional columns can be considered for plotting. Please provide 2 or fewer additional columns.")
                # check to make sure all columns are either numeric, categorical, or datetime
                for dtype_other in d_types_other_cols:
                    if not is_valid_dtype(dtype_other):
                        raise ValueError(f"Column '{dtype_other}' must be either numeric categorical (object), or datetime.")      
                # need to build handlers for this case    
                pass
            else:
                raise ValueError("other_cols must be a string or a list of strings.")
     
        # here: x and y but no other cols - so plots for 2 variables.
        else:
            types = [get_dtype(x_col), get_dtype(y_col)]
            key = tuple(sorted(
                types,
                key=lambda dtype: TYPE_ORDER[dtype]
            ))
            handler = PLOT_MAP[key]
            handler(
                df = df,
                x_col = x_col,
                y_col=y_col,
                distribution=distribution,
                agg = agg
            )
    # plots for one variable
    else:
        key = (get_dtype(df[x_col].dtype),)
        handler = PLOT_MAP[key]
        handler(
            df = df, 
            col = x_col
        ) 

    if other_cols is not None:
         raise ValueError("To plot more than one variable, fill y_col first. other_cols is for additional variables to consider for plotting, not for plotting more than one variable.")




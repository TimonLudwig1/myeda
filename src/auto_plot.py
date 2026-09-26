import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def is_valid_dtype(dtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    )

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

def plot_grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, other_col: str, agg: str):
    fig, ax = plt.subplots()
    df.groupby([x_col, other_col])[y_col].agg(agg).unstack().plot(kind='bar', ax=ax)
    ax.set_title(f"{x_col} against {y_col} and {other_col}")
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.legend(title=other_col)
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
    - A matplotlib Axes object containing the generated plot.
    """
    # everything that is just plain messy code will eventually be rewritten in functions. This rn is just to test stuff

    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")
    d_type_x = df[x_col].dtype

    if not is_valid_dtype(d_type_x): 
        raise ValueError(f"Column '{x_col}' must be either numeric categorical (object), or datetime.")
    
    if y_col is not None:
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame.")
        d_type_y = df[y_col].dtype

        if not is_valid_dtype(d_type_y):
            raise ValueError(f"Column '{y_col}' must be either numeric categorical (object), or datetime.")

        if df[y_col].equals(df[x_col]):
            raise ValueError(f"Column '{y_col}' cannot be the same as column '{x_col}'.")
 
        if other_cols is not None:
            if isinstance(other_cols, str):
                d_type_other_cols = df[other_cols].dtype
                # d_type_other_cols is not a list, so a single column - 3 variables
                # two numeric, one cat
                if pd.api.types.is_numeric_dtype(d_type_x) and pd.api.types.is_numeric_dtype(d_type_y) and pd.api.types.is_object_dtype(d_type_other_cols):
                    return plot_scatter(df, x_col, y_col, other_cols)

                # one numeric two cat
                elif (pd.api.types.is_object_dtype(d_type_x) or pd.api.types.is_datetime64_any_dtype(d_type_x)) and pd.api.types.is_numeric_dtype(d_type_y) and pd.api.types.is_object_dtype(d_type_other_cols):
                    # check number of unique values in the cat cols
                    if len(df[x_col].unique()) <= 6 and len(df[other_cols].unique()) <= 5:
                        # check if we use stacked bar - groups can also be datetime! Important: datetime has to be groups, make sure to force dates as x
                        if additive:
                            return plot_stacked_bar(df, x_col, y_col, other_cols, agg)
                        else: 
                            return plot_grouped_bar(df, x_col, y_col, other_cols, agg)   
                    else:                
                        # if too many cols - heatmap
                        return plot_heatmap(df, x_col, y_col, other_cols, agg)
                    
            elif isinstance(other_cols, list):
                d_types_other_cols = [df[col].dtype for col in other_cols]
                # check: length of list: I can't be bothered rn to do more than 2 additional variables (so 4 total) - so for now if len(other_cols) > 2 error:
                if len(d_types_other_cols) > 2:
                    raise ValueError("Currently, only up to 2 additional columns can be considered for plotting. Please provide 2 or fewer additional columns.")
                # check to make sure all columns are either numeric, categorical, or datetime
                for dtype_other in d_types_other_cols:
                    if not is_valid_dtype(dtype_other):
                        raise ValueError(f"Column '{dtype_other}' must be either numeric categorical (object), or datetime.")         
                return plot_scatter(df, x_col, y_col, other_cols)
            else:
                raise ValueError("other_cols must be a string or a list of strings.")
            
            # now check data types for plot decision logic:
            # first: 3 variables total: x, y, and one other variable.      
        # here: x and y but no other cols - so plots for 2 variables.
        else:
            #num + num (or date)
            if (pd.api.types.is_numeric_dtype(d_type_x) or pd.api.types.is_datetime64_any_dtype(d_type_x)) and pd.api.types.is_numeric_dtype(d_type_y):
                #scatter - if x is datetime, group by date 
                if pd.api.types.is_datetime64_any_dtype(d_type_x):
                    groups = df.groupby(pd.Grouper(key=x_col, freq="D"))
                    plot_df = groups[y_col].agg(agg).reset_index()
                else: 
                    plot_df = df

                return plot_scatter(plot_df, x_col, y_col)
            
            elif pd.api.types.is_object_dtype(d_type_x) and pd.api.types.is_numeric_dtype(d_type_y):
                #num and cat - here for now just bar plot, when this becomes a function, pass an arg that decides if you compare groups or distributions (distributions=False as default)
                if distribution:
                    return plot_boxplot(df, x_col, y_col)
                else:
                    groups = df.groupby(x_col)
                    plot_df = groups[y_col].agg(agg).reset_index()
                    return plot_bar(plot_df, x_col, y_col)
            else:
                # fix this when writing the function. Do some kind of frequency count and plot heatmap/bars or smth
                raise ValueError("If you want to plot two variables against each other, use a frequency method first and pass 3 arguments to plot")

    # here: x but no y - so plots for 1 variable. Categorical has to always be counted for bar plot/frequency count. You can't just plot categories. That would kind of be just a list. 
    else:
        if pd.api.types.is_numeric_dtype(d_type_x):
            return plot_histogram(df, x_col)
        else:
        # cat col
            groups = df.groupby(x_col)
            count_df = groups.agg(
                x_col_count=(x_col, 'count')
            ).reset_index()

            return plot_bar(count_df, x_col, f"{x_col}_count")
           

    if other_cols is not None:
         raise ValueError("To plot more than one variable, fill y_col first. other_cols is for additional variables to consider for plotting, not for plotting more than one variable.")

# things to fix:
# plots for when we pass a list of other_cols 

    # more than 3 variables 
    # │     
    # ├── time + multiple numerical series - 3+ total (time series or x and y num with different lines)
    # │   └── Multi-line plot
    # │
    # ├── Hierarchical categories + numerical size - 3+ total ?? 
    # │   └── Treemap
    # │
    # ├── Flow between categories - 3+ total
    # │   └── Sankey diagram
    # │
    # └── Set membership / overlap - 3+ total
    #     └── Venn diagram

# examples for 4 or more variables:
# Scatter plot: x = income, y = spending, color = customer segment, point size = account value.
# Grouped bar chart split into panels: x = product, bar height = revenue, color = sales channel, one panel per region. 
# Heatmaps split into panels: rows = city, columns = product, color = revenue, one panel per year.
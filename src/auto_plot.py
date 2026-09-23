from lets_plot import ylab
from matplotlib.pylab import xlabel, ylabel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def is_valid_dtype(dtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    )

def auto_plot(df: pd.DataFrame, x_col: str, y_col: str | None = None, other_cols: str | list | None = None, additive: bool=False, **kwargs):
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
            elif isinstance(other_cols, list):
                d_type_other_cols = [df[col].dtype for col in other_cols]
                # check: length of list: I can't be bothered rn to do more than 2 additional variables (so 4 total) - so for now if len(other_cols) > 2 error:
                if len(d_type_other_cols) > 2:
                    raise ValueError("Currently, only up to 2 additional columns can be considered for plotting. Please provide 2 or fewer additional columns.")
                # check to make sure all columns are either numeric, categorical, or datetime
                for dtype_other in d_type_other_cols:
                    if not is_valid_dtype(dtype_other):
                        raise ValueError(f"Column '{dtype_other}' must be either numeric categorical (object), or datetime.")
            else:
                raise ValueError("other_cols must be a string or a list of strings.")

            #here: x and y and other cols - so plots for 3 or more variables.
            
            # now check data types for plot decision logic:
            # first: 3 variables total: x, y, and one other variable.
            if type(d_type_other_cols) != list:
                # d_type_other_cols is not a list, so a single column - 3 variables
                # two numeric, one cat first - num for x, y, cat for color
                if pd.api.types.is_numeric_dtype(d_type_x) and pd.api.types.is_numeric_dtype(d_type_y) and pd.api.types.is_object_dtype(d_type_other_cols):
                    fig, ax = plt.subplots()
                    ax.scatter(df[x_col], df[y_col])

                    ax.set_title(f"{x_col} against {y_col}")
                    ax.set_xlabel(f"{x_col}")
                    ax.set_ylabel(f"{y_col}")
                    ax.legend()
                    plt.show()
                    return plt

                # one numeric two cat (rn, I'm checking only y as numeric, but x and y as categories and other as numeric for scatter with different size + opacity is also possible)
                elif (pd.api.types.is_object_dtype(d_type_x) or pd.api.types.is_datetime64_any_dtype(d_type_x)) and pd.api.types.is_numeric_dtype(d_type_y) and pd.api.types.is_object_dtype(d_type_other_cols):
                    # check number of unique values in the cat cols
                    if len(df[x_col].unique()) <= 6 and len(df[other_cols].unique()) <= 5:
                        # check if we use stacked bar - groups can also be datetime! Important: datetime has to be groups, make sure to force dates as x
                        if additive: 
                            fig, ax = plt.subplots()
                            #pivot to reshape long to wide
                            pivot = (
                                df.groupby([x_col, other_cols])[y_col]
                                .sum()
                                .unstack(other_cols)
                                .fillna(0)
                            )
                            pivot.plot(kind="bar", stacked=True, ax=ax)
                            ax.set_label(x_col)
                            ax.legend(title=f"Sum of {y_col}")
                            ax.legend(title=", ".join(other_cols), bbox_to_anchor=(1.02, 1), loc="upper left")
                            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                            plt.tight_layout()
                            plt.show()
                            return plt
                
                            # grouped bar plot
                        fig = plt.figure
                        fig, ax = plt.subplots()
                        df.groupby([x_col, other_cols])[y_col].mean().unstack().plot(kind='bar', ax=ax)
                        ax.set_title(f"{x_col} against {y_col} and {other_cols}")
                        ax.set_xlabel(f"{x_col}")
                        ax.set_ylabel(f"{y_col}")
                        ax.legend(title=other_cols)
                        plt.show()
                        return plt
                    
                    # if too many cols - heatmap
                    pivot = (
                        df.groupby([x_col, other_cols])[y_col]
                            .sum()
                            .unstack(other_cols)
                            .fillna(0)
                    )
                    sns.heatmap(pivot, cmap="magma")
                    plt.show()
                    return plt



        # here: x and y but no other cols - so plots for 2 variables.
        else:
            #num + num (or date)
            if (pd.api.types.is_numeric_dtype(d_type_x) or pd.api.types.is_datetime64_any_dtype(d_type_x)) and pd.api.types.is_numeric_dtype(d_type_y):
                #scatter - if x is datetime, group by date 
                if pd.api.types.is_datetime64_any_dtype(d_type_x):
                    groups = df.groupby(x_col)
                    grouped_plot_df = groups.agg(
                        sum=(y_col, 'sum')
                    )
                    fig, ax = plt.subplots()
                    ax.scatter(grouped_plot_df.index, grouped_plot_df['sum'])
                    ax.set(title=f"{x_col} against {y_col}", xlabel=x_col, ylabel=y_col)
                    ax.legend()
                    ax.grid(axis="y", alpha=0.3)
                    fig.tight_layout()
                    plt.show()
                    return plt

                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col])
                ax.set(title=f"{x_col} against {y_col}", xlabel=x_col, ylabel=y_col)
                ax.legend()
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                plt.show()
                return plt
            
            elif pd.api.types.is_object_dtype(d_type_x) and pd.api.types.is_numeric_dtype(d_type_y):
                #num and cat - here for now just bar plot, when this becomes a function, pass an arg that decides if you compare groups or distributions (distributions=False as default)
                fig, ax = plt.subplots()
                ax.bar(df[x_col], df[y_col])
                ax.set(title=f"{y_col} for {x_col}")
                ax.legend()
                fig.tight_layout()
                plt.show()
                return plt
            else:
                # fix this when writing the function. Do some kind of frequency count and plot heatmap/bars or smth
                raise ValueError("If you want to plot two variables against each other, use a frequency method first and pass 3 arguments to plot")

    # here: x but no y - so plots for 1 variable. Categorical has to always be counted for bar plot/frequency count. You can't just plot categories. That would kind of be just a list. 
    else:
        if pd.api.types.is_numeric_dtype(d_type_x):
            plt.subplot(1, 2, 1)
            plt.hist(df[x_col])
            plt.title(f"distribution of {x_col}")

            plt.subplot(1,2,2)
            df[x_col].plot.density(bw_method='scott', color='blue', linestyle='-', linewidth=2)
            plt.tight_layout()
            plt.show()
            return plt 
        
        # cat col - sum()
        groups = df.groupby(x_col)
        count_df = groups.agg(
            x_col_count=(x_col, 'count')
        )
        fig, ax = plt.subplots()
        ax.bar(count_df.index, count_df["x_col_count"])
        ax.set(title=f"number of entries for each unique value of {x_col}", xlabel=f"categories for {x_col}", ylabel="count")
        plt.tight_layout()
        plt.show()
        return plt

    if other_cols is not None:
         raise ValueError("To plot more than one variable, fill y_col first. other_cols is for additional variables to consider for plotting, not for plotting more than one variable.")

df = pd.read_csv(r"C:\Users\timon\Documents\GitHub\myeda\data\sales_by_region.csv")
df["date"] = pd.to_datetime(df["date"])
auto_plot(df,"sales_channel")


    # Determine the plot type based on the data types of the columns

    # Decision logic for plot type:

    #     How many variables / dimensions?

    # 1 variable
    # │
    # ├── Numerical
    # │   ├── Ordered observations? → Line plot
    # │   └── Distribution? → Histogram / Boxplot
    # │
    # └── Categorical
    #     └── Frequencies/counts → Bar plot


    # 2 variables
    # │
    # ├── Numerical + Numerical
    # │   ├── Ordered relationship? → Line plot / Connected scatter
    # │   └── Unordered → Scatter plot
    # │
    # ├── Categorical + Numerical
    # │   ├── Compare groups → Bar plot
    # │   └── Compare distributions → Box plot / Violin plot
    # │
    # └── Categorical + Categorical - doesn't make sense without any kind of frequency operation
    #     ├── Counts/proportions → Grouped/Stacked bar plot
    #     └── Association matrix → Heatmap


    # 3+ variables / dimensions
    # │
    # ├── Two numerical + category
    # │   └── Scatter plot with color/shape - 3 total - y has to be numerical, the remaining have to be cat + num, I'll make sure it doesn't matter if x or other_col is either. 
    # │
    # ├── Two categorical + numerical value
    # │   └── Heatmap / Grouped or stacked bar - 3 total - the numerical value is the intersect of the two categorical cols

    # IF (cat1 <= 6 and cat2 <= 5)
    #   Then grouped bar 
    # stacked bar semantic - so pass a additive = true/false as *arg to the function, default = false, so always grouped bar! 
    # Else:
    #   Heatmap 

    # ELSE 
#     Use Stacked Bar Chart (Best for showing absolute totals)
    # │     
    # ├── Ordered/time + multiple numerical series - 3+ total
    # │   └── Multi-line plot
    # │
    # ├── Hierarchical categories + numerical size - 3+ total
    # │   └── Treemap
    # │
    # ├── Flow between categories - 3+ total
    # │   └── Sankey diagram
    # │
    # └── Set membership / overlap - 3+ total
    #     └── Venn diagram




    # more than one variable as x against one variable as y: 
        # no: Is it ordered?:   
            #yes: numerical: line plot
            #yes: categorical: bar plot/box plot 
            #no: numerical: box plot 
            #no: categorical: histogram

        # yes: Are they similar?:
            #no: Are they ordered?:
                #no: numerical: scatter plot
                #no: categorical: bar plot
                #yes: numerical: connected scatter plot
                #yes: categorical: bar plot - pretty unlikely use case.

            #yes: Do they have hirarchy?: 
                #yes: numerical: venn diagram. Thechnically both cases are categorical. Check this if it's bugging
                #yes: categorical: Sankey diagram. 

                #no: Are they ordered?:
                    #yes: numerical: line plot
                    #yes: categorical: stacked bar plot
                    #no: numerical: heatmap
                    #no categorical: treemap

    # code chekc structure: first: if d_type_x and d_type_y are even there: no? then only a singel variable!



        
    
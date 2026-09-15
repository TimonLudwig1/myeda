import pandas as pd
import matplotlib.pyplot as plt

def auto_plot(df: pd.DataFrame, x_col: str, y_col: str, other_cols: str | list, **kwargs):
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

    if x_col:
        if x_col not in df.columns:
            raise ValueError(f"Column '{x_col}' not found in DataFrame.")
        d_type_x = df[x_col].dtype

        if d_type_x != 'object' and not pd.api.types.is_numeric_dtype(d_type_x) and not  pd.api.types.is_datetime64_any_dtype(d_type_x):
            raise ValueError(f"Column '{x_col}' must be either numeric categorical (object), or datetime.")
    else:
        raise ValueError("x_col must be provided.")
    
    if y_col:
        if y_col not in df.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame.")
        d_type_y = df[y_col].dtype

        if d_type_y != 'object' and not pd.api.types.is_numeric_dtype(d_type_y) and not  pd.api.types.is_datetime64_any_dtype(d_type_y):
                    raise ValueError(f"Column '{y_col}' must be either numeric categorical (object), or datetime.")

        if df[y_col].equals(df[x_col]):
            raise ValueError(f"Column '{y_col}' cannot be the same as column '{x_col}'.")
 
        if other_cols:
            if isinstance(other_cols, str):
                d_type_other_cols = df[other_cols].dtype
            elif isinstance(other_cols, list):
                d_type_other_cols = [df[col].dtype for col in other_cols]
                # insert check to make sure all columns are either numeric, categorical, or datetime - some other time
            else:
                raise ValueError("other_cols must be a string or a list of strings.")

            #here: x and y and other cols - so plots for 3 or more variables.
            # check: length of list: I can't be bothered rn to do more than 2 additional variables (so 4 total) - so for now if len(other_cols) > 2 error:
            if len(other_cols) > 2:
                raise ValueError("Currently, only up to 2 additional columns can be considered for plotting. Please provide 2 or fewer additional columns.")

            # now check data types for plot decision logic:
            pass 

        # here: x and y but no other cols - so plots for 2 variables.
        else:
            pass

    # here: x but no y - so plots for 1 variable.
    else:
        pass 
    if other_cols:
         raise ValueError("To plot more than one variable, fill y_col first. other_cols is for additional variables to consider for plotting, not for plotting more than one variable.")
    
    

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
    # └── Categorical + Categorical
    #     ├── Counts/proportions → Grouped/Stacked bar plot
    #     └── Association matrix → Heatmap


    # 3+ variables / dimensions
    # │
    # ├── Two numerical + category
    # │   └── Scatter plot with color/shape
    # │
    # ├── Two categorical + numerical value
    # │   └── Heatmap / Grouped or stacked bar
    # │
    # ├── Ordered/time + multiple numerical series
    # │   └── Multi-line plot
    # │
    # ├── Hierarchical categories + numerical size
    # │   └── Treemap
    # │
    # ├── Flow between categories
    # │   └── Sankey diagram
    # │
    # └── Set membership / overlap
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

    if d_type_x and d_type_y:
        # both columns are present, determine the plot type based on their data types
        if pd.api.types.is_numeric_dtype(d_type_x) and pd.api.types.is_numeric_dtype

        
    
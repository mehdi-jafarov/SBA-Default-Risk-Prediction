"""
table_builder.py

A utility module for generating summary tables and descriptive statistics
using pandas, particularly useful for classification outcome analysis.

Includes:
- Calculating outcome rates by group
- Identifying top and bottom groups by outcome
- Generating quartile summaries
"""

import pandas as pd

class TableBuilder:
    """
    Utility class for generating summary tables, grouping statistics,
    and quartile breakdowns from a pandas DataFrame.
    """

    @staticmethod
    def print_top_bottom(data, group_by_col, rate_label, n=5):
        """
        Print the top and bottom N groups by a specified rate column.

        Args:
            data (pd.DataFrame): Table containing the rate column.
            group_by_col (str): Name of the group column.
            rate_label (str): Label to identify the rate column (e.g., 'Default').
            n (int): Number of top and bottom groups to print.
        """
        # Automatically detect the rate column containing the label
        rate_col = [col for col in data.columns if rate_label in col][0]

        # Sort and get top and bottom N
        top_n = data.sort_values(by=rate_col, ascending=False).head(n)
        bottom_n = data.sort_values(by=rate_col, ascending=True).head(n)

        print(f"\nTop {n} {rate_label} Rates by {group_by_col}:\n")
        print(top_n)

        print(f"\nBottom {n} {rate_label} Rates by {group_by_col}:\n")
        print(bottom_n)

    @staticmethod
    def group_by_rate(
        df,
        group_by_col,
        target_col,
        labels,    
        map_data=None,
    ):
        """
        Calculate the percentage of specified target labels within each group.

        Args:
            df (pd.DataFrame): Input DataFrame with group and target columns.
            group_by_col (str): Column to group by (e.g., 'State').
            target_col (str): Column with target classification (e.g., 'LoanStatus').
            labels (list): List of labels to include in output columns, in order.
            map_data (dict, optional): Mapping for group descriptions.

        Returns:
            pd.DataFrame: A table with calculated rate percentages and optional metadata.
        """
        df = df.copy()

        # Count occurrences of each target label in each group
        counts = df.groupby(group_by_col)[target_col].value_counts().unstack(fill_value=0)
        denominator = counts.sum(axis=1)

        # Build initial rate table
        rates_table = counts[labels].div(denominator, axis=0) * 100
        rates_table = rates_table.reset_index()

        # Optionally add mapped group descriptions
        if map_data:
            rates_table['Description'] = rates_table[group_by_col].map(map_data)
            cols = [group_by_col, 'Description'] + labels
            rates_table = rates_table[cols]
        else:
            cols = [group_by_col] + labels
            rates_table = rates_table[cols]

        # Round all rate columns
        for col in labels:
            rates_table[col] = rates_table[col].round(2)

        rates_table = rates_table.sort_values(by=labels[0], ascending=True).reset_index(drop=True)
            
        return rates_table


    @staticmethod
    def get_quartiles(series):
        """
        Return quartiles and min/max from a numeric series.

        Args:
            series (pd.Series): The numeric data series.

        Returns:
            dict: A dictionary of quartiles and min/max values.
        """
        return {
            '100% maximum': series.max(),
            '75% quartile': series.quantile(0.75),
            '50% median': series.median(),
            '25% quartile': series.quantile(0.25),
            'Minimum': series.min()
        }

    @staticmethod
    def quartiles_by_outcomes(df, column, target_col):
        """
        Generate a table of quartile statistics for a numeric column,
        grouped by target outcomes (e.g., 'Default' vs 'Paid').

        Args:
            df (pd.DataFrame): The dataset containing both the column and target_col.
            column (str): Numeric column for which to compute quartiles.
            target_col (str): Categorical column to group by.

        Returns:
            pd.DataFrame: A summary table of quartiles per outcome category.
        """
        df = df.copy()
        labels = sorted(df[target_col].dropna().unique())
        quartile_labels = ['100% maximum', '75% quartile', '50% median', '25% quartile', 'Minimum']
        table = pd.DataFrame({'Quartiles': quartile_labels})

        # For each target category (e.g., 'Default', 'Paid')
        for label in labels:
            label_series = df[df[target_col] == label][column]
            label_quart = TableBuilder.get_quartiles(label_series)
            # Format numbers as currency-style (e.g., $1,000)
            table[label] = [f'${v:,.0f}' for v in label_quart.values()]

        return table

    @staticmethod
    def save_tables(tables_dict, extension='xlsx'):
        """
        Save multiple pandas DataFrames to files with specified extension.

        Args:
            tables_dict (dict): Dictionary where keys are filename strings (without extension),
                            and values are pandas DataFrames to save.
            extension (str): File extension/format to save tables in. Supported:
                         'csv', 'xls', 'xlsx', 'json'.
                         Defaults to 'csv'.

        Raises:
            ValueError: If an unsupported file extension is provided.
        """
        for name, table in tables_dict.items():
            if extension == 'csv':
                table.to_csv(f'{name}.{extension}', index=False)
            elif extension in ['xls', 'xlsx']:
                table.to_excel(f'{name}.{extension}', index=False)
            elif extension == 'json':
                table.to_json(f'{name}.{extension}')
            else:
                raise ValueError(f"Unsupported file extension: {extension}")

    @staticmethod
    def summary_to_dfs(summary):
        """
        Convert a statsmodels Summary object to a list of Pandas DataFrames.

        Parameters:
        - summary (statsmodels.iolib.summary.Summary): The summary object.

        Returns:
        - list[pd.DataFrame]: List of DataFrames, one for each table in the summary.
        """
        dfs = []
        for table in summary.tables:
            df = pd.DataFrame(table.data)
            df.columns = df.iloc[0]  # First row as header
            df = df.drop(index=0).reset_index(drop=True)
            dfs.append(df)
        return dfs


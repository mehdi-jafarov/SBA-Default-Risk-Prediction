"""
preprocessor.py

A utility module for common data preprocessing operations using pandas.

Includes:
- Date formatting
- Whitespace removal in string columns
- Cleaning and converting currency-like numeric columns
"""

import pandas as pd

class Preprocessor:
    """
    A collection of static methods for preprocessing pandas DataFrames.

    Methods are non-destructive (they return a modified copy of the input DataFrame).
    """

    @staticmethod
    def format_date(df, columns, date_format='%d-%b-%y'):
        """
        Convert specified columns to datetime objects using a specific format.

        Args:
            df (pd.DataFrame): The DataFrame containing the date columns.
            columns (list): List of column names to convert.
            date_format (str): The expected format of the input dates (e.g., '%d-%b-%y').

        Returns:
            pd.DataFrame: A copy of the DataFrame with converted datetime columns.
        """
        df = df.copy()
        for col in columns:
            # Use pd.to_datetime with error handling
            df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
        return df

    @staticmethod
    def unspace(df, columns):
        """
        Remove all spaces and strip whitespace from string columns.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): List of column names (string type) to clean.

        Returns:
            pd.DataFrame: A copy of the DataFrame with cleaned string columns.
        """
        df = df.copy()
        for col in columns:
            # Remove all spaces and strip leading/trailing whitespace
            df[col] = df[col].str.replace(' ', '', regex=False).str.strip()
        return df

    @staticmethod
    def unsign(df, columns, astype=float):
        """
        Remove currency formatting (e.g., '$', ',') and convert to numeric type.

        Args:
            df (pd.DataFrame): The DataFrame containing the columns.
            columns (list): List of column names to clean.
            astype (type or str): Desired output type (e.g., float, int, 'Int64').

        Returns:
            pd.DataFrame: A copy of the DataFrame with numeric columns.
        """
        df = df.copy()
        for col in columns:
            # Remove currency symbols and commas, then convert type
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(astype)
        return df

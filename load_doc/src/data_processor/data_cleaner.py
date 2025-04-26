"""
Data Cleaner Module

This module provides functions for cleaning and preprocessing data:
- Handling missing values
- Detecting and handling outliers
- Fixing data type issues
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                         columns: Optional[List[str]] = None, 
                         fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.

    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('drop', 'fill', 'interpolate')
        columns: List of columns to apply the strategy to. If None, applies to all columns.
        fill_values: Dict of column:value pairs for filling missing values (used with 'fill' strategy)

    Returns:
        DataFrame with missing values handled
    """
    if columns is None:
        columns = df.columns
    
    result_df = df.copy()
    
    if strategy == 'drop':
        if len(columns) == len(df.columns):
            result_df = result_df.dropna()
        else:
            result_df = result_df.dropna(subset=columns)
    
    elif strategy == 'fill':
        if fill_values is not None:
            for col, value in fill_values.items():
                if col in columns:
                    result_df[col] = result_df[col].fillna(value)
        else:
            # Use sensible defaults if no fill values provided
            for col in columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                elif pd.api.types.is_string_dtype(result_df[col]):
                    result_df[col] = result_df[col].fillna('')
                else:
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else None)
    
    elif strategy == 'interpolate':
        for col in columns:
            if pd.api.types.is_numeric_dtype(result_df[col]):
                result_df[col] = result_df[col].interpolate(method='linear')
    
    return result_df


def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers. If None, checks all numeric columns.
        method: Method for outlier detection ('iqr', 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, typically 3 for z-score)

    Returns:
        DataFrame with a boolean mask where True indicates an outlier
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            z_scores = abs((df[col] - mean) / std)
            outlier_mask[col] = z_scores > threshold
    
    return outlier_mask


def fix_data_types(df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Fix data types of columns in a DataFrame.

    Args:
        df: Input DataFrame
        type_dict: Dictionary mapping column names to their target data types
                   (e.g., {'age': 'int', 'price': 'float', 'date': 'datetime'})

    Returns:
        DataFrame with corrected data types
    """
    result_df = df.copy()
    
    for col, type_name in type_dict.items():
        if col in result_df.columns:
            try:
                if type_name == 'int':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce').astype('Int64')  # nullable int
                elif type_name == 'float':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                elif type_name == 'datetime':
                    result_df[col] = pd.to_datetime(result_df[col], errors='coerce')
                elif type_name == 'string' or type_name == 'str':
                    result_df[col] = result_df[col].astype(str)
                elif type_name == 'category':
                    result_df[col] = result_df[col].astype('category')
                elif type_name == 'bool' or type_name == 'boolean':
                    result_df[col] = result_df[col].astype(bool)
            except Exception as e:
                print(f"Error converting column {col} to {type_name}: {str(e)}")
    
    return result_df 
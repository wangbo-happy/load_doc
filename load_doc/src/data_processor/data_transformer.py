"""
Data Transformer Module

This module provides functions for transforming data:
- Standardization and normalization
- Feature engineering
- Dimensionality reduction
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA


def normalize(df: pd.DataFrame, columns: Optional[List[str]] = None, 
              method: str = 'z-score') -> pd.DataFrame:
    """
    Normalize numeric data in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of columns to normalize. If None, normalizes all numeric columns.
        method: Normalization method ('z-score', 'min-max')

    Returns:
        DataFrame with normalized values
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    result_df = df.copy()
    
    if method == 'z-score':
        scaler = StandardScaler()
        result_df[columns] = scaler.fit_transform(result_df[columns])
    
    elif method == 'min-max':
        scaler = MinMaxScaler()
        result_df[columns] = scaler.fit_transform(result_df[columns])
    
    return result_df


def encode_categorical(df: pd.DataFrame, columns: List[str], 
                      method: str = 'one-hot', drop_first: bool = False) -> pd.DataFrame:
    """
    Encode categorical variables in a DataFrame.

    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        method: Encoding method ('one-hot', 'label', 'ordinal')
        drop_first: Whether to drop the first category in one-hot encoding

    Returns:
        DataFrame with encoded categorical variables
    """
    result_df = df.copy()
    
    if method == 'one-hot':
        for col in columns:
            if drop_first:
                one_hot = pd.get_dummies(result_df[col], prefix=col, drop_first=True)
            else:
                one_hot = pd.get_dummies(result_df[col], prefix=col)
            
            result_df = pd.concat([result_df, one_hot], axis=1)
            result_df.drop(col, axis=1, inplace=True)
    
    elif method == 'label':
        for col in columns:
            result_df[f"{col}_encoded"] = result_df[col].astype('category').cat.codes
    
    elif method == 'ordinal':
        # Requires a mapping dict to be provided
        raise NotImplementedError("Ordinal encoding requires a mapping dictionary")
    
    return result_df


def create_time_features(df: pd.DataFrame, date_column: str, 
                         features: List[str] = ['year', 'month', 'day', 'dayofweek']) -> pd.DataFrame:
    """
    Create time-based features from a date column.

    Args:
        df: Input DataFrame
        date_column: Name of the column containing datetime values
        features: List of time features to create
                 (options: 'year', 'month', 'day', 'hour', 'minute', 
                  'dayofweek', 'quarter', 'weekofyear', 'dayofyear')

    Returns:
        DataFrame with additional time-based features
    """
    result_df = df.copy()
    
    # Ensure the column is in datetime format
    result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
    
    # Create the requested features
    feature_mapping = {
        'year': lambda x: x.dt.year,
        'month': lambda x: x.dt.month,
        'day': lambda x: x.dt.day,
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'dayofweek': lambda x: x.dt.dayofweek,
        'quarter': lambda x: x.dt.quarter,
        'weekofyear': lambda x: x.dt.isocalendar().week,
        'dayofyear': lambda x: x.dt.dayofyear
    }
    
    for feature in features:
        if feature in feature_mapping:
            result_df[f"{date_column}_{feature}"] = feature_mapping[feature](result_df[date_column])
    
    return result_df


def reduce_dimensions(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                     n_components: int = 2, method: str = 'pca') -> pd.DataFrame:
    """
    Reduce dimensionality of data using PCA or other methods.

    Args:
        df: Input DataFrame
        columns: List of columns to use for dimensionality reduction. If None, uses all numeric columns.
        n_components: Number of components to reduce to
        method: Dimensionality reduction method ('pca', 'tsne', etc.)

    Returns:
        DataFrame with reduced dimensions
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    result_df = df.copy()
    
    if method == 'pca':
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(result_df[columns])
        
        # Create new dataframe with components
        component_cols = [f"PC{i+1}" for i in range(n_components)]
        component_df = pd.DataFrame(components, columns=component_cols, index=result_df.index)
        
        # Add PCA components to the original dataframe
        for col in component_cols:
            result_df[col] = component_df[col]
    
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
    return result_df 
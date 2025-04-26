"""
Descriptive Statistics Module

This module provides functions for calculating descriptive statistics:
- Basic statistics (mean, median, mode, etc.)
- Distribution analysis
- Data summarization
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
import scipy.stats as stats


def get_basic_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate basic descriptive statistics for numerical columns.

    Args:
        df: Input DataFrame
        columns: List of columns to analyze. If None, analyzes all numeric columns.

    Returns:
        DataFrame with basic statistics for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Calculate basic statistics
    stats_df = df[columns].describe().T
    
    # Add additional statistics
    stats_df['skewness'] = df[columns].skew()
    stats_df['kurtosis'] = df[columns].kurtosis()
    stats_df['median'] = df[columns].median()
    stats_df['mode'] = df[columns].mode().iloc[0] if not df[columns].mode().empty else np.nan
    stats_df['missing'] = df[columns].isna().sum()
    stats_df['missing_pct'] = (df[columns].isna().sum() / len(df)) * 100
    
    return stats_df


def get_categorical_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Calculate descriptive statistics for categorical columns.

    Args:
        df: Input DataFrame
        columns: List of categorical columns to analyze. If None, analyzes all non-numeric columns.

    Returns:
        Dictionary mapping column names to DataFrames with their statistics
    """
    if columns is None:
        columns = df.select_dtypes(exclude=np.number).columns.tolist()
    
    result = {}
    
    for col in columns:
        # Get value counts and percentages
        value_counts = df[col].value_counts()
        value_pcts = df[col].value_counts(normalize=True) * 100
        
        # Combine into a single DataFrame
        col_stats = pd.DataFrame({
            'count': value_counts,
            'percentage': value_pcts,
            'missing': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100
        })
        
        # Add statistics
        col_stats.loc['total', 'count'] = df[col].count()
        col_stats.loc['unique', 'count'] = df[col].nunique()
        col_stats.loc['unique_pct', 'percentage'] = (df[col].nunique() / df[col].count()) * 100
        
        result[col] = col_stats
    
    return result


def check_normality(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   method: str = 'shapiro', alpha: float = 0.05) -> pd.DataFrame:
    """
    Check if numeric columns follow a normal distribution.

    Args:
        df: Input DataFrame
        columns: List of columns to check. If None, checks all numeric columns.
        method: Statistical test to use ('shapiro', 'ks', 'anderson')
        alpha: Significance level for the test

    Returns:
        DataFrame with test results for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    results = []
    
    for col in columns:
        data = df[col].dropna()
        
        if method == 'shapiro':
            # Shapiro-Wilk test (better for small samples)
            stat, p_value = stats.shapiro(data)
            test_name = 'Shapiro-Wilk'
        
        elif method == 'ks':
            # Kolmogorov-Smirnov test
            stat, p_value = stats.kstest(data, 'norm')
            test_name = 'Kolmogorov-Smirnov'
        
        elif method == 'anderson':
            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            stat = result.statistic
            p_value = 0  # Anderson-Darling doesn't return a p-value directly
            test_name = 'Anderson-Darling'
        
        else:
            raise ValueError(f"Unknown normality test method: {method}")
        
        is_normal = p_value > alpha if method != 'anderson' else stat < result.critical_values[2]
        
        results.append({
            'column': col,
            'test': test_name,
            'statistic': stat,
            'p_value': p_value if method != 'anderson' else None,
            'is_normal': is_normal
        })
    
    return pd.DataFrame(results)


def get_quantiles(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                 q: List[float] = [0.25, 0.5, 0.75]) -> pd.DataFrame:
    """
    Calculate quantiles for numeric columns.

    Args:
        df: Input DataFrame
        columns: List of columns to analyze. If None, analyzes all numeric columns.
        q: List of quantiles to calculate

    Returns:
        DataFrame with quantiles for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    return df[columns].quantile(q) 
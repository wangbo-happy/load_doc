"""
Correlation Analysis Module

This module provides functions for analyzing relationships between variables:
- Pearson correlation
- Spearman correlation
- Kendall correlation
- Point biserial correlation
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
import scipy.stats as stats


def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for selected columns.

    Args:
        df: Input DataFrame
        columns: List of columns to include in correlation analysis. If None, includes all numeric columns.
        method: Correlation method ('pearson', 'spearman', or 'kendall')

    Returns:
        DataFrame with correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    return df[columns].corr(method=method)


def calculate_correlation_significance(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                                    method: str = 'pearson', alpha: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix with significance tests.

    Args:
        df: Input DataFrame
        columns: List of columns to include in correlation analysis. If None, includes all numeric columns.
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        alpha: Significance level

    Returns:
        Tuple of (correlation matrix, p-value matrix)
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    n = df[columns].shape[0]
    corr_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    p_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
                p_matrix.loc[col1, col2] = 0.0
            else:
                if method == 'pearson':
                    corr, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                elif method == 'spearman':
                    corr, p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                elif method == 'kendall':
                    corr, p = stats.kendalltau(df[col1].dropna(), df[col2].dropna())
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                corr_matrix.loc[col1, col2] = corr
                p_matrix.loc[col1, col2] = p
    
    return corr_matrix, p_matrix


def get_top_correlations(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                       method: str = 'pearson', n: int = 10) -> pd.DataFrame:
    """
    Get the top n correlations between variables.

    Args:
        df: Input DataFrame
        columns: List of columns to include in correlation analysis. If None, includes all numeric columns.
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        n: Number of top correlations to return

    Returns:
        DataFrame with top correlations
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    corr_matrix = df[columns].corr(method=method)
    
    # Convert to long format and get absolute correlation
    corr_long = corr_matrix.unstack().reset_index()
    corr_long.columns = ['Variable1', 'Variable2', 'Correlation']
    
    # Remove self-correlations and duplicates
    corr_long = corr_long[corr_long['Variable1'] != corr_long['Variable2']]
    corr_long['Pair'] = corr_long.apply(lambda row: tuple(sorted([row['Variable1'], row['Variable2']])), axis=1)
    corr_long = corr_long.drop_duplicates(subset=['Pair'])
    corr_long['Absolute_Correlation'] = corr_long['Correlation'].abs()
    
    # Get top n correlations
    top_corr = corr_long.sort_values('Absolute_Correlation', ascending=False).head(n)
    top_corr = top_corr.drop(['Pair', 'Absolute_Correlation'], axis=1)
    
    return top_corr


def point_biserial_correlation(df: pd.DataFrame, continuous_var: str, binary_var: str) -> Dict:
    """
    Calculate point-biserial correlation between a continuous and a binary variable.

    Args:
        df: Input DataFrame
        continuous_var: Name of the continuous variable
        binary_var: Name of the binary variable (should be 0/1 or boolean)

    Returns:
        Dictionary with correlation results
    """
    # Ensure binary variable is 0/1
    binary_data = df[binary_var].astype(int)
    continuous_data = df[continuous_var]
    
    # Calculate point-biserial correlation
    correlation, p_value = stats.pointbiserialr(binary_data, continuous_data)
    
    return {
        'continuous_variable': continuous_var,
        'binary_variable': binary_var,
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def calculate_vif(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.

    Args:
        df: Input DataFrame
        columns: List of columns to check for multicollinearity. If None, includes all numeric columns.

    Returns:
        DataFrame with VIF values
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    
    X = df[columns]
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Variable"] = columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data.sort_values("VIF", ascending=False) 
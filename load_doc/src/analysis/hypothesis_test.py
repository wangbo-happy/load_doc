"""
Hypothesis Testing Module

This module provides functions for statistical hypothesis testing:
- T-tests
- ANOVA
- Chi-square tests
- Non-parametric tests
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
import scipy.stats as stats


def t_test_one_sample(data: Union[List, np.ndarray, pd.Series], 
                     popmean: float = 0, alpha: float = 0.05) -> Dict:
    """
    Perform a one-sample t-test.

    Args:
        data: Sample data
        popmean: Expected population mean to test against
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Ensure data is a numpy array
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    # Perform t-test
    statistic, p_value = stats.ttest_1samp(data, popmean)
    
    # Determine test result
    reject_null = p_value < alpha
    
    return {
        'test': 'One-sample t-test',
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': 'two-sided',
        'n': len(data),
        'sample_mean': np.mean(data),
        'popmean': popmean
    }


def t_test_two_sample(data1: Union[List, np.ndarray, pd.Series], 
                     data2: Union[List, np.ndarray, pd.Series],
                     equal_var: bool = True, alpha: float = 0.05) -> Dict:
    """
    Perform a two-sample t-test.

    Args:
        data1: First sample data
        data2: Second sample data
        equal_var: Whether to assume equal variances
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Ensure data are numpy arrays
    if isinstance(data1, pd.Series):
        data1 = data1.dropna().values
    if isinstance(data2, pd.Series):
        data2 = data2.dropna().values
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
    
    # Determine test result
    reject_null = p_value < alpha
    
    return {
        'test': 'Two-sample t-test',
        'equal_var': equal_var,
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': 'two-sided',
        'n1': len(data1),
        'n2': len(data2),
        'mean1': np.mean(data1),
        'mean2': np.mean(data2),
        'mean_diff': np.mean(data1) - np.mean(data2),
        'std1': np.std(data1, ddof=1),
        'std2': np.std(data2, ddof=1)
    }


def paired_t_test(data1: Union[List, np.ndarray, pd.Series], 
                 data2: Union[List, np.ndarray, pd.Series],
                 alpha: float = 0.05) -> Dict:
    """
    Perform a paired-sample t-test.

    Args:
        data1: First sample data
        data2: Second sample data
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Ensure data are numpy arrays
    if isinstance(data1, pd.Series):
        data1 = data1.dropna().values
    if isinstance(data2, pd.Series):
        data2 = data2.dropna().values
    
    # Ensure equal lengths
    assert len(data1) == len(data2), "Data samples must have the same length for paired t-test"
    
    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(data1, data2)
    
    # Determine test result
    reject_null = p_value < alpha
    
    return {
        'test': 'Paired t-test',
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': 'two-sided',
        'n': len(data1),
        'mean1': np.mean(data1),
        'mean2': np.mean(data2),
        'mean_diff': np.mean(data1) - np.mean(data2),
        'std_diff': np.std(data1 - data2, ddof=1)
    }


def one_way_anova(groups: List[Union[List, np.ndarray, pd.Series]], alpha: float = 0.05) -> Dict:
    """
    Perform a one-way ANOVA test.

    Args:
        groups: List of sample data for each group
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Convert all groups to numpy arrays
    groups = [g.dropna().values if isinstance(g, pd.Series) else g for g in groups]
    
    # Perform ANOVA
    statistic, p_value = stats.f_oneway(*groups)
    
    # Determine test result
    reject_null = p_value < alpha
    
    # Calculate group statistics
    group_stats = []
    for i, group in enumerate(groups):
        group_stats.append({
            'group': i + 1,
            'n': len(group),
            'mean': np.mean(group),
            'std': np.std(group, ddof=1)
        })
    
    return {
        'test': 'One-way ANOVA',
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'num_groups': len(groups),
        'group_stats': pd.DataFrame(group_stats)
    }


def chi_square_test(observed: Union[np.ndarray, pd.DataFrame], expected: Optional[np.ndarray] = None,
                   alpha: float = 0.05) -> Dict:
    """
    Perform a chi-square test.

    Args:
        observed: Observed frequency table
        expected: Expected frequency table. If None, assumes equal distribution.
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(observed, pd.DataFrame):
        observed = observed.values
    
    # Calculate chi-square test
    if expected is not None:
        statistic, p_value, dof, expected = stats.chi2_contingency(observed, expected)
    else:
        statistic, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Determine test result
    reject_null = p_value < alpha
    
    return {
        'test': 'Chi-square test',
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'degrees_of_freedom': dof,
        'observed': observed,
        'expected': expected
    }


def mann_whitney_test(data1: Union[List, np.ndarray, pd.Series], 
                     data2: Union[List, np.ndarray, pd.Series],
                     alpha: float = 0.05) -> Dict:
    """
    Perform a Mann-Whitney U test (non-parametric alternative to t-test).

    Args:
        data1: First sample data
        data2: Second sample data
        alpha: Significance level

    Returns:
        Dictionary with test results
    """
    # Ensure data are numpy arrays
    if isinstance(data1, pd.Series):
        data1 = data1.dropna().values
    if isinstance(data2, pd.Series):
        data2 = data2.dropna().values
    
    # Perform Mann-Whitney U test
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    # Determine test result
    reject_null = p_value < alpha
    
    return {
        'test': 'Mann-Whitney U test',
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': reject_null,
        'alpha': alpha,
        'alternative': 'two-sided',
        'n1': len(data1),
        'n2': len(data2),
        'median1': np.median(data1),
        'median2': np.median(data2)
    } 
"""
Tests for the analysis module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.descriptive_stats import get_basic_stats, get_categorical_stats, check_normality, get_quantiles
from src.analysis.hypothesis_test import t_test_one_sample, t_test_two_sample, paired_t_test, chi_square_test
from src.analysis.correlation_analysis import calculate_correlation_matrix, get_top_correlations


class TestDescriptiveStats(unittest.TestCase):
    """Test the descriptive_stats module."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducibility
        
        # Create a sample DataFrame with numerical and categorical data
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': np.random.normal(0, 1, 10),  # Normal distribution
            'C': np.random.uniform(0, 10, 10),  # Uniform distribution
            'D': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2', 'cat1', 'cat3', 'cat2', 'cat1', 'cat2']
        })

    def test_get_basic_stats(self):
        """Test getting basic statistics."""
        stats = get_basic_stats(self.data, columns=['A', 'B', 'C'])
        
        # Check that we have the expected statistics
        expected_stats_cols = ['count', '25%', '50%', '75%', 'mean', 'std', 'min', 'max', 'skewness', 'kurtosis', 'median', 'mode', 'missing', 'missing_pct']
        self.assertTrue(all(col in stats.columns for col in expected_stats_cols))
        
        # Check specific values
        self.assertEqual(stats.loc['A', 'count'], 10.0)
        self.assertEqual(stats.loc['A', 'mean'], 5.5)
        self.assertEqual(stats.loc['A', 'median'], 5.5)
        self.assertEqual(stats.loc['A', 'min'], 1.0)
        self.assertEqual(stats.loc['A', 'max'], 10.0)

    def test_get_categorical_stats(self):
        """Test getting categorical statistics."""
        stats = get_categorical_stats(self.data, columns=['D'])
        
        # Check that we have a result for our categorical column
        self.assertIn('D', stats)
        
        # Check that the counts add up
        cat_counts = stats['D']['count'].loc[['cat1', 'cat2', 'cat3']]
        self.assertEqual(cat_counts.sum(), 10)
        
        # Check specific counts
        self.assertEqual(stats['D']['count']['cat1'], 4)
        self.assertEqual(stats['D']['count']['cat2'], 4)
        self.assertEqual(stats['D']['count']['cat3'], 2)

    def test_check_normality(self):
        """Test normality check."""
        # Generate a normally distributed sample
        normal_data = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'uniform': np.random.uniform(0, 1, 100)
        })
        
        result = check_normality(normal_data, method='shapiro')
        
        # The normal column should have a higher p-value than the uniform column
        normal_result = result[result['column'] == 'normal'].iloc[0]
        uniform_result = result[result['column'] == 'uniform'].iloc[0]
        
        self.assertGreater(normal_result['p_value'], 0.01)  # Should not reject normality
        
        # Test with different method
        result_ks = check_normality(normal_data, method='ks')
        self.assertEqual(len(result_ks), 2)  # Two columns tested

    def test_get_quantiles(self):
        """Test quantile calculation."""
        quantiles = get_quantiles(self.data, columns=['A', 'B', 'C'], q=[0.25, 0.5, 0.75])
        
        # Check that we have the expected quantiles
        self.assertEqual(quantiles.index.tolist(), [0.25, 0.5, 0.75])
        self.assertEqual(quantiles.columns.tolist(), ['A', 'B', 'C'])
        
        # Check specific values for column A
        self.assertEqual(quantiles.loc[0.25, 'A'], 3.25)
        self.assertEqual(quantiles.loc[0.5, 'A'], 5.5)
        self.assertEqual(quantiles.loc[0.75, 'A'], 7.75)


class TestHypothesisTest(unittest.TestCase):
    """Test the hypothesis_test module."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducibility
        
        # Create sample data for hypothesis tests
        self.sample1 = np.random.normal(0, 1, 30)
        self.sample2 = np.random.normal(0.5, 1, 30)
        
        # Create categorical data for chi-square test
        self.observed = np.array([[10, 20, 30], [20, 30, 10]])

    def test_t_test_one_sample(self):
        """Test one-sample t-test."""
        result = t_test_one_sample(self.sample1, popmean=0)
        
        # Check that all expected keys are in the result
        expected_keys = ['test', 'statistic', 'p_value', 'reject_null', 'alpha', 'alternative', 'n', 'sample_mean', 'popmean']
        self.assertTrue(all(key in result for key in expected_keys))
        
        # Since the sample is from a normal distribution with mean 0, we should not reject the null hypothesis
        self.assertFalse(result['reject_null'])
        
        # Test with a different mean (should reject)
        result2 = t_test_one_sample(self.sample2, popmean=0)
        self.assertTrue(result2['reject_null'])

    def test_t_test_two_sample(self):
        """Test two-sample t-test."""
        result = t_test_two_sample(self.sample1, self.sample2)
        
        # Check that all expected keys are in the result
        expected_keys = ['test', 'equal_var', 'statistic', 'p_value', 'reject_null', 'alpha', 'alternative', 'n1', 'n2', 'mean1', 'mean2', 'mean_diff', 'std1', 'std2']
        self.assertTrue(all(key in result for key in expected_keys))
        
        # Since the samples are from different distributions, we should reject the null hypothesis
        self.assertTrue(result['reject_null'])
        
        # Test with equal variances
        result2 = t_test_two_sample(self.sample1, self.sample1)
        self.assertFalse(result2['reject_null'])

    def test_paired_t_test(self):
        """Test paired t-test."""
        # Create paired samples
        sample1 = np.random.normal(0, 1, 30)
        sample2 = sample1 + 0.5  # Add a constant offset
        
        result = paired_t_test(sample1, sample2)
        
        # Check that all expected keys are in the result
        expected_keys = ['test', 'statistic', 'p_value', 'reject_null', 'alpha', 'alternative', 'n', 'mean1', 'mean2', 'mean_diff', 'std_diff']
        self.assertTrue(all(key in result for key in expected_keys))
        
        # Since sample2 is consistently higher, we should reject the null hypothesis
        self.assertTrue(result['reject_null'])
        
        # The mean difference should be close to the offset we added
        self.assertAlmostEqual(result['mean_diff'], -0.5, places=1)

    def test_chi_square_test(self):
        """Test chi-square test."""
        result = chi_square_test(self.observed)
        
        # Check that all expected keys are in the result
        expected_keys = ['test', 'statistic', 'p_value', 'reject_null', 'alpha', 'degrees_of_freedom', 'observed', 'expected']
        self.assertTrue(all(key in result for key in expected_keys))
        
        # Check specific values
        self.assertEqual(result['degrees_of_freedom'], 2)
        self.assertTrue(np.array_equal(result['observed'], self.observed))


class TestCorrelationAnalysis(unittest.TestCase):
    """Test the correlation_analysis module."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)  # For reproducibility
        
        # Create variables with known correlations
        x = np.random.normal(0, 1, 100)
        y = x * 0.8 + np.random.normal(0, 0.5, 100)  # Positive correlation
        z = -x * 0.6 + np.random.normal(0, 0.7, 100)  # Negative correlation
        w = np.random.normal(0, 1, 100)  # Uncorrelated
        
        self.data = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'w': w
        })

    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        corr_matrix = calculate_correlation_matrix(self.data)
        
        # Check that we have a correlation matrix of the right shape
        self.assertEqual(corr_matrix.shape, (4, 4))
        
        # Check that diagonal is all 1
        self.assertTrue(all(corr_matrix.values[i, i] == 1.0 for i in range(4)))
        
        # Check that the matrix is symmetric
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))
        
        # Check that x,y correlation is positive and strong
        self.assertGreater(corr_matrix.loc['x', 'y'], 0.7)
        
        # Check that x,z correlation is negative and strong
        self.assertLess(corr_matrix.loc['x', 'z'], -0.5)
        
        # Check that x,w correlation is close to 0
        self.assertAlmostEqual(corr_matrix.loc['x', 'w'], 0, delta=0.3)

    def test_get_top_correlations(self):
        """Test getting top correlations."""
        top_corr = get_top_correlations(self.data, n=3)
        
        # Check that we have the expected number of results
        self.assertEqual(len(top_corr), 3)
        
        # Check that the correlations are sorted by absolute value
        abs_corrs = [abs(corr) for corr in top_corr['Correlation']]
        self.assertTrue(abs_corrs[0] >= abs_corrs[1] >= abs_corrs[2])
        
        # Check that the strongest correlation (x,y) is first
        strongest = top_corr.iloc[0]
        self.assertTrue(
            (strongest['Variable1'] == 'x' and strongest['Variable2'] == 'y') or
            (strongest['Variable1'] == 'y' and strongest['Variable2'] == 'x')
        )


if __name__ == '__main__':
    unittest.main() 
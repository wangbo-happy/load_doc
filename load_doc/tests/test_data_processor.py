"""
Tests for the data_processor module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import sqlite3

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor.data_loader import load_csv, load_excel, load_database
from src.data_processor.data_cleaner import handle_missing_values, detect_outliers, fix_data_types
from src.data_processor.data_transformer import normalize, encode_categorical, create_time_features


class TestDataLoader(unittest.TestCase):
    """Test the data_loader module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Create temporary files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.temp_dir.name, 'test.csv')
        self.excel_path = os.path.join(self.temp_dir.name, 'test.xlsx')
        self.db_path = os.path.join(self.temp_dir.name, 'test.db')
        
        # Save sample data to files
        self.data.to_csv(self.csv_path, index=False)
        self.data.to_excel(self.excel_path, index=False)
        
        # Create a test database
        conn = sqlite3.connect(self.db_path)
        self.data.to_sql('main_table', conn, index=False, if_exists='replace')
        conn.close()

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_load_csv(self):
        """Test loading data from a CSV file."""
        loaded_data = load_csv(self.csv_path)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_load_excel(self):
        """Test loading data from an Excel file."""
        loaded_data = load_excel(self.excel_path)
        pd.testing.assert_frame_equal(loaded_data, self.data)

    def test_load_database(self):
        """Test loading data from a database."""
        loaded_data = load_database(self.db_path, query="SELECT * FROM main_table")
        pd.testing.assert_frame_equal(loaded_data, self.data)


class TestDataCleaner(unittest.TestCase):
    """Test the data_cleaner module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame with missing values and outliers
        self.data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 100],  # outlier: 100
            'B': [10, np.nan, 30, 40, 50],
            'C': ['a', 'b', 'c', np.nan, 'e']
        })

    def test_handle_missing_values_drop(self):
        """Test dropping missing values."""
        result = handle_missing_values(self.data, strategy='drop')
        self.assertEqual(len(result), 2)  # 2 rows without missing values

    def test_handle_missing_values_fill(self):
        """Test filling missing values."""
        result = handle_missing_values(self.data, strategy='fill', 
                                     fill_values={'A': 0, 'B': 0, 'C': 'missing'})
        self.assertEqual(result['A'].isna().sum(), 0)
        self.assertEqual(result['B'].isna().sum(), 0)
        self.assertEqual(result['C'].isna().sum(), 0)

    def test_detect_outliers(self):
        """Test outlier detection."""
        outliers = detect_outliers(self.data, method='iqr')
        self.assertTrue(outliers['A'].iloc[4])  # The value 100 should be detected as an outlier

    def test_fix_data_types(self):
        """Test fixing data types."""
        data = pd.DataFrame({
            'A': ['1', '2', '3'],
            'B': [1.1, 2.2, 3.3],
            'C': ['2020-01-01', '2020-01-02', '2020-01-03']
        })
        
        type_dict = {'A': 'int', 'B': 'float', 'C': 'datetime'}
        result = fix_data_types(data, type_dict)
        
        self.assertEqual(result['A'].dtype, 'Int64')
        self.assertTrue(pd.api.types.is_float_dtype(result['B']))
        self.assertTrue(pd.api.types.is_datetime64_dtype(result['C']))


class TestDataTransformer(unittest.TestCase):
    """Test the data_transformer module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['cat1', 'cat2', 'cat1', 'cat3', 'cat2'],
            'date': pd.date_range(start='2020-01-01', periods=5, freq='D')
        })

    def test_normalize_z_score(self):
        """Test z-score normalization."""
        result = normalize(self.data, columns=['A', 'B'], method='z-score')
        
        # Check that the mean is approximately 0 and std is approximately 1
        self.assertAlmostEqual(result['A'].mean(), 0, places=10)
        self.assertAlmostEqual(result['B'].mean(), 0, places=10)
        self.assertAlmostEqual(result['A'].std(), 1, places=10)
        self.assertAlmostEqual(result['B'].std(), 1, places=10)

    def test_normalize_min_max(self):
        """Test min-max normalization."""
        result = normalize(self.data, columns=['A', 'B'], method='min-max')
        
        # Check that the min is 0 and max is 1
        self.assertAlmostEqual(result['A'].min(), 0, places=10)
        self.assertAlmostEqual(result['B'].min(), 0, places=10)
        self.assertAlmostEqual(result['A'].max(), 1, places=10)
        self.assertAlmostEqual(result['B'].max(), 1, places=10)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        result = encode_categorical(self.data, columns=['C'], method='one-hot')
        
        # Check that we have the expected columns
        expected_cols = ['A', 'B', 'date', 'C_cat1', 'C_cat2', 'C_cat3']
        self.assertListEqual(sorted(result.columns.tolist()), sorted(expected_cols))
        
        # Check that the encoding is correct
        self.assertEqual(result['C_cat1'].sum(), 2)  # 2 instances of 'cat1'
        self.assertEqual(result['C_cat2'].sum(), 2)  # 2 instances of 'cat2'
        self.assertEqual(result['C_cat3'].sum(), 1)  # 1 instance of 'cat3'

    def test_create_time_features(self):
        """Test creating time features."""
        result = create_time_features(self.data, 'date', features=['year', 'month', 'day'])
        
        # Check that we have the expected columns
        expected_cols = ['A', 'B', 'C', 'date', 'date_year', 'date_month', 'date_day']
        self.assertListEqual(sorted(result.columns.tolist()), sorted(expected_cols))
        
        # Check that the time features are correct
        self.assertEqual(result['date_year'].iloc[0], 2020)
        self.assertEqual(result['date_month'].iloc[0], 1)
        self.assertEqual(result['date_day'].iloc[0], 1)


if __name__ == '__main__':
    unittest.main() 
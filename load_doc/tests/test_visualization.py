"""
Tests for the visualization module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization.plot_factory import PlotFactory


class TestPlotFactory(unittest.TestCase):
    """Test the PlotFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        # For reproducibility
        np.random.seed(42)
        
        # Create a sample DataFrame
        self.data = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(2, 1.5, 100),
            'C': np.random.choice(['cat1', 'cat2', 'cat3'], 100),
            'D': pd.date_range(start='2020-01-01', periods=100, freq='D')
        })
        
        # Create a plot factory
        self.plot_factory = PlotFactory(figsize=(8, 6), dpi=72)
        
        # Disable showing plots
        plt.close('all')
        plt.ioff()

    def tearDown(self):
        """Clean up test fixtures."""
        plt.close('all')

    def test_init(self):
        """Test initialization of PlotFactory."""
        self.assertEqual(self.plot_factory.figsize, (8, 6))
        self.assertEqual(self.plot_factory.dpi, 72)
        
        # Test with different parameters
        pf = PlotFactory(style='seaborn', palette='Set1', figsize=(10, 8), dpi=100)
        self.assertEqual(pf.palette, 'Set1')
        self.assertEqual(pf.figsize, (10, 8))
        self.assertEqual(pf.dpi, 100)

    def test_set_style(self):
        """Test setting the plot style."""
        self.plot_factory.set_style('ggplot')
        self.plot_factory.set_style('default')
        
        # Test with invalid style (should use default)
        self.plot_factory.set_style('nonexistent_style')

    def test_histogram(self):
        """Test creating a histogram."""
        fig = self.plot_factory.histogram(self.data, column='A', bins=20, kde=True)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'A')
        self.assertEqual(ax.get_ylabel(), 'Frequency')
        self.assertEqual(ax.get_title(), 'Histogram of A')

    def test_density_plot(self):
        """Test creating a density plot."""
        fig = self.plot_factory.density_plot(self.data, columns=['A', 'B'])
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'Value')
        self.assertEqual(ax.get_ylabel(), 'Density')
        
        # Check that we have a legend with two items
        self.assertEqual(len(ax.get_legend().get_texts()), 2)

    def test_scatter_plot(self):
        """Test creating a scatter plot."""
        fig = self.plot_factory.scatter_plot(self.data, x='A', y='B', hue='C')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'A')
        self.assertEqual(ax.get_ylabel(), 'B')
        
        # Test with regression line
        fig = self.plot_factory.scatter_plot(self.data, x='A', y='B', show_reg_line=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_correlation_matrix(self):
        """Test creating a correlation matrix."""
        fig = self.plot_factory.correlation_matrix(self.data, columns=['A', 'B'])
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), 'Pearson Correlation Matrix')
        
        # Test with different method
        fig = self.plot_factory.correlation_matrix(self.data, columns=['A', 'B'], method='spearman')
        self.assertEqual(fig.axes[0].get_title(), 'Spearman Correlation Matrix')

    def test_bar_plot(self):
        """Test creating a bar plot."""
        # Create a value count first
        self.data['E'] = np.random.choice(['group1', 'group2', 'group3'], 100)
        self.data['F'] = np.random.uniform(0, 10, 100)
        
        # Test count plot
        fig = self.plot_factory.bar_plot(self.data, x='E')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'E')
        self.assertEqual(ax.get_ylabel(), 'Count')
        
        # Test value plot
        fig = self.plot_factory.bar_plot(self.data, x='E', y='F', bar_labels=True)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test horizontal orientation
        fig = self.plot_factory.bar_plot(self.data, x='E', y='F', orient='h')
        self.assertIsInstance(fig, plt.Figure)

    def test_pie_chart(self):
        """Test creating a pie chart."""
        fig = self.plot_factory.pie_chart(self.data, column='C')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), 'Pie Chart: C')
        
        # Test with explode
        fig = self.plot_factory.pie_chart(self.data, column='C', explode=[0.1, 0, 0])
        self.assertIsInstance(fig, plt.Figure)

    def test_line_plot(self):
        """Test creating a line plot."""
        # Make a time series
        ts_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
            'value1': np.random.normal(0, 1, 100).cumsum(),
            'value2': np.random.normal(0, 1, 100).cumsum(),
            'group': np.random.choice(['A', 'B'], 100)
        })
        
        fig = self.plot_factory.line_plot(ts_data, x='date', y='value1', markers=True)
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'date')
        self.assertEqual(ax.get_ylabel(), 'value1')
        
        # Test with multiple y columns
        fig = self.plot_factory.line_plot(ts_data, x='date', y=['value1', 'value2'])
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with trend line
        fig = self.plot_factory.line_plot(ts_data, x='date', y='value1', add_trend=True)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with hue
        fig = self.plot_factory.line_plot(ts_data, x='date', y='value1', hue='group')
        self.assertIsInstance(fig, plt.Figure)

    def test_box_plot(self):
        """Test creating a box plot."""
        fig = self.plot_factory.box_plot(self.data, x='C', y='A')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'C')
        self.assertEqual(ax.get_ylabel(), 'A')
        self.assertEqual(ax.get_title(), 'Box Plot')
        
        # Test with hue
        fig = self.plot_factory.box_plot(self.data, x='C', y='A', hue='C')
        self.assertIsInstance(fig, plt.Figure)

    def test_violin_plot(self):
        """Test creating a violin plot."""
        fig = self.plot_factory.violin_plot(self.data, x='C', y='A')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), 'C')
        self.assertEqual(ax.get_ylabel(), 'A')
        self.assertEqual(ax.get_title(), 'Violin Plot')
        
        # Test with split
        fig = self.plot_factory.violin_plot(self.data, x='C', y='A', hue='C', split=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_heatmap(self):
        """Test creating a heatmap."""
        # Create a correlation matrix for the heatmap
        corr_matrix = self.data[['A', 'B']].corr()
        
        fig = self.plot_factory.heatmap(corr_matrix, title='Test Heatmap')
        
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(fig.axes), 1)
        
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), 'Test Heatmap')

    def test_create_plot(self):
        """Test the create_plot interface."""
        # Test creating a plot via the general interface
        fig = self.plot_factory.create_plot('histogram', self.data, column='A')
        self.assertIsInstance(fig, plt.Figure)
        
        fig = self.plot_factory.create_plot('scatter', self.data, x='A', y='B')
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with invalid plot type
        with self.assertRaises(ValueError):
            self.plot_factory.create_plot('invalid_type', self.data)


if __name__ == '__main__':
    unittest.main() 
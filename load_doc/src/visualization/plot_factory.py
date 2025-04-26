"""
Plot Factory Module

This module provides a unified interface for creating various types of plots:
- Histograms and density plots
- Scatter plots and correlation matrices
- Bar plots and pie charts
- Line plots for time series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Union, Tuple, Callable
import os


class PlotFactory:
    """A factory class for creating various types of plots."""
    
    def __init__(self, style: str = 'default', palette: str = 'viridis', 
                figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize the PlotFactory with style settings.
        
        Args:
            style: Plot style ('default', 'ggplot', 'seaborn', etc.)
            palette: Color palette for plots
            figsize: Default figure size (width, height) in inches
            dpi: Dots per inch for figure resolution
        """
        self.set_style(style)
        self.palette = palette
        self.figsize = figsize
        self.dpi = dpi
    
    def set_style(self, style: str):
        """
        Set the plot style.
        
        Args:
            style: Plot style to use
        """
        if style == 'default':
            plt.style.use('default')
        elif style in plt.style.available:
            plt.style.use(style)
        else:
            print(f"Style '{style}' not found. Using default style.")
            plt.style.use('default')
    
    def create_plot(self, plot_type: str, df: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Create a plot based on the specified type.
        
        Args:
            plot_type: Type of plot to create
            df: DataFrame containing the data to plot
            **kwargs: Additional arguments for the specific plot type
        
        Returns:
            Matplotlib Figure object
        """
        plot_functions = {
            'histogram': self.histogram,
            'density': self.density_plot,
            'scatter': self.scatter_plot,
            'correlation': self.correlation_matrix,
            'bar': self.bar_plot,
            'pie': self.pie_chart,
            'line': self.line_plot,
            'box': self.box_plot,
            'violin': self.violin_plot,
            'heatmap': self.heatmap,
            'pair': self.pair_plot
        }
        
        if plot_type in plot_functions:
            return plot_functions[plot_type](df, **kwargs)
        else:
            raise ValueError(f"Plot type '{plot_type}' not supported")
    
    def histogram(self, df: pd.DataFrame, column: str, bins: int = 30, 
                kde: bool = True, title: Optional[str] = None, 
                xlabel: Optional[str] = None, ylabel: str = 'Frequency',
                show_stats: bool = True) -> plt.Figure:
        """
        Create a histogram for a numeric column.
        
        Args:
            df: DataFrame containing the data
            column: Column to plot
            bins: Number of bins for the histogram
            kde: Whether to overlay a kernel density estimate
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_stats: Whether to show basic statistics on the plot
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.histplot(df[column].dropna(), bins=bins, kde=kde, ax=ax)
        
        # Set labels
        ax.set_xlabel(xlabel or column)
        ax.set_ylabel(ylabel)
        ax.set_title(title or f'Histogram of {column}')
        
        # Add statistics if requested
        if show_stats:
            stats_text = (
                f"Mean: {df[column].mean():.2f}\n"
                f"Median: {df[column].median():.2f}\n"
                f"Std Dev: {df[column].std():.2f}\n"
                f"Min: {df[column].min():.2f}\n"
                f"Max: {df[column].max():.2f}"
            )
            plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        ha='right', va='top')
        
        plt.tight_layout()
        return fig
    
    def density_plot(self, df: pd.DataFrame, columns: List[str], 
                   title: Optional[str] = None, 
                   xlabel: str = 'Value', ylabel: str = 'Density',
                   legend_title: Optional[str] = None) -> plt.Figure:
        """
        Create a density plot for multiple numeric columns.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_title: Title for the legend
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        for column in columns:
            sns.kdeplot(df[column].dropna(), ax=ax, label=column)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or 'Density Plot')
        ax.legend(title=legend_title)
        
        plt.tight_layout()
        return fig
    
    def scatter_plot(self, df: pd.DataFrame, x: str, y: str, 
                    hue: Optional[str] = None, size: Optional[str] = None,
                    title: Optional[str] = None, 
                    xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    alpha: float = 0.7, show_reg_line: bool = False) -> plt.Figure:
        """
        Create a scatter plot for two numeric columns.
        
        Args:
            df: DataFrame containing the data
            x: Column for the x-axis
            y: Column for the y-axis
            hue: Column to use for color encoding
            size: Column to use for size encoding
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            alpha: Opacity of the points
            show_reg_line: Whether to show a regression line
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        scatter_kwargs = {
            'data': df,
            'x': x,
            'y': y,
            'hue': hue,
            'size': size,
            'alpha': alpha,
            'ax': ax
        }
        
        # Remove None values to avoid warnings
        scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}
        
        if show_reg_line:
            sns.regplot(x=x, y=y, data=df, scatter=False, ax=ax, line_kws={'color': 'red'})
        
        sns.scatterplot(**scatter_kwargs)
        
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title or f'Scatter Plot: {y} vs {x}')
        
        # Add correlation coefficient
        if show_reg_line:
            corr = df[[x, y]].corr().iloc[0, 1]
            plt.annotate(f"Correlation: {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        ha='left', va='top')
        
        plt.tight_layout()
        return fig
    
    def correlation_matrix(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'pearson', annot: bool = True,
                         cmap: str = 'coolwarm', 
                         title: Optional[str] = None) -> plt.Figure:
        """
        Create a correlation matrix heatmap.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to include in the correlation matrix
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate the heatmap with correlation values
            cmap: Colormap for the heatmap
            title: Plot title
        
        Returns:
            Matplotlib Figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        corr_matrix = df[columns].corr(method=method)
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, 
                   vmin=-1, vmax=1, center=0, square=True, ax=ax,
                   fmt='.2f', linewidths=0.5)
        
        ax.set_title(title or f'{method.capitalize()} Correlation Matrix')
        
        plt.tight_layout()
        return fig
    
    def bar_plot(self, df: pd.DataFrame, x: str, y: Optional[str] = None, 
               hue: Optional[str] = None, orient: str = 'v',
               title: Optional[str] = None, 
               xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               bar_labels: bool = False) -> plt.Figure:
        """
        Create a bar plot.
        
        Args:
            df: DataFrame containing the data
            x: Column for the categories
            y: Column for the values (if None, uses count of records)
            hue: Column to use for color encoding
            orient: Orientation ('v' for vertical, 'h' for horizontal)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bar_labels: Whether to add value labels to the bars
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if y is None:
            # Count plot
            if orient == 'h':
                sns.countplot(y=x, hue=hue, data=df, ax=ax)
            else:
                sns.countplot(x=x, hue=hue, data=df, ax=ax)
        else:
            # Value plot
            if orient == 'h':
                sns.barplot(y=x, x=y, hue=hue, data=df, ax=ax)
            else:
                sns.barplot(x=x, y=y, hue=hue, data=df, ax=ax)
        
        # Set labels
        if orient == 'h':
            ax.set_xlabel(ylabel or (y if y else 'Count'))
            ax.set_ylabel(xlabel or x)
        else:
            ax.set_xlabel(xlabel or x)
            ax.set_ylabel(ylabel or (y if y else 'Count'))
        
        ax.set_title(title or 'Bar Plot')
        
        # Add bar labels if requested
        if bar_labels:
            for p in ax.patches:
                if orient == 'h':
                    ax.annotate(f'{p.get_width():.1f}', 
                              (p.get_width(), p.get_y() + p.get_height()/2.),
                              ha='left', va='center')
                else:
                    ax.annotate(f'{p.get_height():.1f}', 
                              (p.get_x() + p.get_width()/2., p.get_height()),
                              ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def pie_chart(self, df: pd.DataFrame, column: str, 
                title: Optional[str] = None, 
                autopct: str = '%1.1f%%', legend: bool = True,
                explode: Optional[List[float]] = None) -> plt.Figure:
        """
        Create a pie chart for a categorical column.
        
        Args:
            df: DataFrame containing the data
            column: Column to use for the pie segments
            title: Plot title
            autopct: Format string for the percentage labels
            legend: Whether to show a legend
            explode: List of values to explode pie slices (0.1 is a good start)
        
        Returns:
            Matplotlib Figure object
        """
        counts = df[column].value_counts()
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if explode is None:
            explode = [0] * len(counts)
        elif len(explode) < len(counts):
            explode = explode + [0] * (len(counts) - len(explode))
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            explode=explode, 
            labels=None if legend else counts.index, 
            autopct=autopct,
            shadow=False, 
            startangle=90
        )
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
        
        ax.set_title(title or f'Pie Chart: {column}')
        
        if legend:
            ax.legend(wedges, counts.index, title=column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        return fig
    
    def line_plot(self, df: pd.DataFrame, x: str, y: Union[str, List[str]], 
                hue: Optional[str] = None, 
                title: Optional[str] = None, 
                xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                markers: bool = False, add_trend: bool = False) -> plt.Figure:
        """
        Create a line plot, useful for time series data.
        
        Args:
            df: DataFrame containing the data
            x: Column for the x-axis (usually time)
            y: Column or list of columns for the y-axis
            hue: Column to use for color encoding
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            markers: Whether to add markers to the lines
            add_trend: Whether to add a trend line
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if isinstance(y, str):
            y = [y]
        
        for column in y:
            if hue is not None:
                for category in df[hue].unique():
                    subset = df[df[hue] == category]
                    ax.plot(subset[x], subset[column], 
                          marker='o' if markers else None,
                          label=f'{column} - {category}')
            else:
                ax.plot(df[x], df[column], 
                      marker='o' if markers else None,
                      label=column if len(y) > 1 else None)
                
                if add_trend:
                    z = np.polyfit(range(len(df[x])), df[column], 1)
                    p = np.poly1d(z)
                    ax.plot(df[x], p(range(len(df[x]))), 
                          linestyle='--', color='red', 
                          label='Trend' if len(y) == 1 else f'{column} trend')
        
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or ('Value' if len(y) > 1 else y[0]))
        ax.set_title(title or 'Line Plot')
        
        if hue is not None or len(y) > 1 or add_trend:
            ax.legend()
        
        plt.tight_layout()
        return fig
    
    def box_plot(self, df: pd.DataFrame, x: Optional[str] = None, y: str = None,
               hue: Optional[str] = None,
               title: Optional[str] = None, 
               xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> plt.Figure:
        """
        Create a box plot for numeric columns.
        
        Args:
            df: DataFrame containing the data
            x: Categorical column for grouping on x-axis
            y: Numeric column to plot
            hue: Additional categorical column for nested grouping
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.boxplot(x=x, y=y, hue=hue, data=df, ax=ax)
        
        ax.set_xlabel(xlabel or x or '')
        ax.set_ylabel(ylabel or y or '')
        ax.set_title(title or 'Box Plot')
        
        plt.tight_layout()
        return fig
    
    def violin_plot(self, df: pd.DataFrame, x: Optional[str] = None, y: str = None,
                  hue: Optional[str] = None, split: bool = False,
                  title: Optional[str] = None, 
                  xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> plt.Figure:
        """
        Create a violin plot for numeric columns.
        
        Args:
            df: DataFrame containing the data
            x: Categorical column for grouping on x-axis
            y: Numeric column to plot
            hue: Additional categorical column for nested grouping
            split: Whether to split the violins when hue is provided
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.violinplot(x=x, y=y, hue=hue, data=df, split=split, ax=ax)
        
        ax.set_xlabel(xlabel or x or '')
        ax.set_ylabel(ylabel or y or '')
        ax.set_title(title or 'Violin Plot')
        
        plt.tight_layout()
        return fig
    
    def heatmap(self, data: pd.DataFrame, 
              title: Optional[str] = None, 
              cmap: str = 'viridis', annot: bool = True,
              fmt: str = '.2g') -> plt.Figure:
        """
        Create a heatmap.
        
        Args:
            data: DataFrame or array to display as a heatmap
            title: Plot title
            cmap: Colormap for the heatmap
            annot: Whether to annotate the heatmap with values
            fmt: Format string for the annotations
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        sns.heatmap(data, annot=annot, fmt=fmt, cmap=cmap, ax=ax)
        
        ax.set_title(title or 'Heatmap')
        
        plt.tight_layout()
        return fig
    
    def pair_plot(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                hue: Optional[str] = None, 
                title: Optional[str] = None) -> plt.Figure:
        """
        Create a pair plot (matrix of scatter plots and histograms).
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to include in the pair plot
            hue: Categorical column for color encoding
            title: Plot title
        
        Returns:
            Matplotlib Figure object
        """
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        g = sns.pairplot(df[columns + ([hue] if hue else [])], 
                        hue=hue, 
                        palette=self.palette,
                        height=3)
        
        if title:
            g.fig.suptitle(title, y=1.02)
        
        plt.tight_layout()
        return g.fig
    
    def save_plot(self, fig: plt.Figure, filename: str, 
                folder: str = "plots", 
                formats: List[str] = ['png', 'pdf'],
                dpi: Optional[int] = None) -> None:
        """
        Save the plot to disk in multiple formats.
        
        Args:
            fig: Matplotlib Figure object to save
            filename: Base filename (without extension)
            folder: Folder to save the plot in
            formats: List of formats to save (e.g., ['png', 'pdf', 'svg'])
            dpi: Dots per inch (resolution) for raster formats
        """
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # Save in each format
        for fmt in formats:
            filepath = os.path.join(folder, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, dpi=dpi or self.dpi, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        # Close the figure to free up memory
        plt.close(fig) 
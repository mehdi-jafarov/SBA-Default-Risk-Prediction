"""
plotter.py

A utility module providing reusable visualization functions using
matplotlib, seaborn, and plotly.

Includes:
- Horizontal binary percentage bar plots
- Boxplots by category
- Choropleth maps (e.g., US state-level)
- Standard bar plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score
import os

# Default color palette: Red and Blue
red_blue = ['#E41A1C', '#377EB8']


class Plotter:
    """
    A collection of static methods for common data visualizations.

    Each method returns a matplotlib or plotly figure object, allowing flexible use
    in notebooks, GUIs, or saving to files.
    """

    @staticmethod
    def plot_binary_split_hbar(
        data,
        label_col,
        positive_col,
        negative_col,
        positive_name=None,
        negative_name=None,
        colors=('#E41A1C', '#377EB8'),
        title='Binary Outcome by Group',
        xlabel='Percentage (%)',
        figsize=(6, 2.8)
    ):
        """
        Plots a horizontal stacked bar chart showing binary outcome percentages.

        Args:
            data (pd.DataFrame): Data containing the label and two percentage columns.
            label_col (str): Column name for group labels.
            positive_col (str): Column name for the positive outcome percentages.
            negative_col (str): Column name for the negative outcome percentages.
            positive_name (str, optional): Label for positive legend. Defaults to column name.
            negative_name (str, optional): Label for negative legend. Defaults to column name.
            colors (tuple): Colors for the bars (positive, negative).
            title (str): Plot title.
            xlabel (str): Label for the X-axis.
            figsize (tuple): Figure size in inches (width, height).

        Returns:
            matplotlib.figure.Figure: The generated bar chart figure.
        """
        data = data.sort_values(by=positive_col, ascending=True)

        labels = data[label_col]
        positives = data[positive_col]
        negatives = data[negative_col]

        positive_name = positive_name or positive_col
        negative_name = negative_name or negative_col

        fig, ax = plt.subplots(figsize=figsize)
        bar_height = 0.5

        # Draw positive and negative bars stacked horizontally
        ax.barh(labels, positives, color=colors[0], label=positive_name,
                edgecolor='none', height=bar_height)
        ax.barh(labels, negatives, left=positives, color=colors[1],
                label=negative_name, edgecolor='none', height=bar_height)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=12, weight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add percentage text inside the bars
        for i, (d, p) in enumerate(zip(positives, negatives)):
            ax.text(d * 0.5, i, f'{d:.1f}%', va='center', ha='center',
                    color='white', fontsize=9, weight='bold')
            ax.text(d + p * 0.5, i, f'{p:.1f}%', va='center', ha='center',
                    color='white', fontsize=9, weight='bold')

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_boxplot_by_category(
        data,
        x_col,
        y_col,
        x_labels=None,
        palette=red_blue,
        title='Boxplot by Category',
        ylabel=None,
        xlabel=None,
        size=(3, 4.5),
        showfliers=True,
        width=0.2,
        alpha=1,
        ylim=None
    ):
        """
        Plots a boxplot for a numerical column grouped by a categorical column.

        Args:
            data (pd.DataFrame): Dataset containing both categorical and numerical columns.
            x_col (str): Categorical column on the x-axis.
            y_col (str): Numerical column for boxplot.
            x_labels (list, optional): Custom labels for the x-axis.
            palette (list): Color palette for boxes.
            title (str): Plot title.
            ylabel (str): Label for the y-axis.
            xlabel (str): Label for the x-axis.
            size (tuple): Figure size in inches (width, height).
            showfliers (bool): Whether to show outliers.
            width (float): Width of each box.
            alpha (float): Transparency of boxes.
            ylim (tuple, optional): Y-axis limits.

        Returns:
            matplotlib.figure.Figure: The generated boxplot figure.
        """
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=size)
        sns.boxplot(
            x=x_col,
            y=y_col,
            hue=x_col,                
            data=data,
            palette=palette,
            legend=False,            
            showmeans=True,
            showfliers=showfliers,
            width=width,
            meanprops={"marker": "+", "markerfacecolor": "black", "markeredgecolor": "black"},
            boxprops=dict(alpha=alpha),
            ax=ax
        )

        if x_labels:
            ax.set_xticks(ticks=range(len(x_labels)))
            ax.set_xticklabels(labels=x_labels)

        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_choropleth(
        title,
        data,
        locations,
        color_col,
        scope='usa',
        locationmode='USA-states',
        color_scale='RdBu_r'
    ):
        """
        Plots a choropleth map using Plotly.

        Args:
            title (str): Title of the map.
            data (pd.DataFrame): DataFrame containing values to visualize.
            locations (str): Column containing region codes (e.g., 'CA', 'NY').
            color_col (str): Column used to color the map.
            scope (str): Map scope (e.g., 'usa', 'world').
            locationmode (str): Interpretation of locations ('ISO-3', 'USA-states', etc.).
            color_scale (str): Color scale name from Plotly.

        Returns:
            plotly.graph_objs.Figure: Choropleth figure.
        """
        fig = px.choropleth(
            data_frame=data,
            title=title,
            locations=locations,
            color=color_col,
            scope=scope,
            locationmode=locationmode,
            color_continuous_scale=color_scale
        )
        return fig

    @staticmethod
    def plot_barplot(
        title,
        data,
        x,
        y,
        size=(3, 4.5),
        width=0.4,
        palette=red_blue
    ):
        """
        Creates a simple vertical bar plot.

        Args:
            title (str): Plot title.
            data (pd.DataFrame): Data source.
            x (str): Column to use for x-axis categories.
            y (str): Column for bar heights.
            size (tuple): Size of the figure (width, height).
            width (float): Width of bars.
            palette (list): Colors to use for the bars.

        Returns:
            matplotlib.figure.Figure: Barplot figure.
        """
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=size)

        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=x,          
            palette=palette,
            legend=False,  
            ax=ax,
            width=width
        )

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel(x, fontsize=10)
        ax.set_ylabel(y, fontsize=10)
        ax.tick_params(axis='both', labelsize=9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return fig

    @staticmethod
    def plot_threshold_perf(x, y, x_label='', y_label='', size=(6, 4), title=''):
        """
        Plot a performance metric against varying threshold values and return the figure object.

        Parameters:
        - x (list or array-like): Values on the x-axis (e.g., threshold cutoffs).
        - y (list or array-like): Corresponding performance metric values on the y-axis.
        - x_label (str, optional): Label for the x-axis. Default is an empty string.
        - y_label (str, optional): Label for the y-axis. Default is an empty string.
        - size (tuple, optional): Figure size as (width, height). Default is (6, 4).
        - title (str, optional): Title of the plot. Default is an empty string.

        Returns:
        - matplotlib.figure.Figure: The figure object for further customization or saving.

        Notes:
        - The function creates a line plot with markers, enables a grid, 
          and uses a dashed line style with a blue color palette.
        - This function does not automatically call `plt.show()`, so the caller 
          can decide when or how to display the plot.
        """
        fig, ax = plt.subplots(figsize=size)
        ax.plot(x, y, color='#377EB8', marker='o', linestyle='--')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True)
        return fig



    @staticmethod
    def plot_roc_auc(y_test, y_pred_prob):
        """
        Plots the ROC Curve and returns the matplotlib figure.

        Parameters:
        -----------
        y_test : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        
        y_pred_prob : array-like of shape (n_samples,)
            Predicted probabilities or scores for the positive class.

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the ROC plot.
        """
        # Compute false positive rate, true positive rate, and thresholds
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        # Compute Area Under the Curve (AUC) score
        auc_score = roc_auc_score(y_test, y_pred_prob)

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(6, 4))
        # Plot ROC curve
        ax.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}', color='#377EB8')
        # Plot diagonal line for random classifier reference
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        # Label axes
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        # Add legend and grid
        ax.legend()
        ax.grid(True)

        return fig
    
    @staticmethod
    def save_figures(figures_dict, dpi=300):
        """
    Save multiple matplotlib figures with specified filenames.

    Args:
        fig_dict (dict): Keys are filename strings (without extension),
                         values are matplotlib figure objects.
        dpi (int): Resolution of saved images.
    """
        for name, figure in figures_dict.items():
            figure.savefig(f'{name}.png', dpi=dpi)

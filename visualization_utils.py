import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationUtils:
    """
    Provides plotting utilities for convergence trajectories, regret bounds, and comparative analysis.
    """

    def __init__(self, config=None):
        """
        Initializes the VisualizationUtils class with optional configuration.

        Parameters:
            config (dict): Configuration settings for the visualization utilities.
        """
        self.config = config or {}

    def plot_regret_trajectories(self, trajectories, labels, colors=None, xlabel='Iterations', ylabel='Regret', title='Regret Trajectories'):
        """
        Plots the regret trajectories for multiple runs or algorithms.

        Parameters:
            trajectories (list of list of float): List of regret trajectories, each represented as a list of floats.
            labels (list of str): List of labels for each trajectory.
            colors (list of str, optional): List of colors to use for the trajectories. Defaults to None.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Iterations'.
            ylabel (str, optional): Label for 'y-axis. Defaults to 'Regret'.
            title (str, optional): Title of the plot. Defaults to 'Regret Trajectories'.
        """
        plt.figure(figsize=(10, 6))
        if colors is None:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(trajectories)))

        for trajectory, label, color in zip(trajectories, labels, colors):
            plt.plot(trajectory, label=label, color=color)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_directness_gap(self, directness_values, labels, xlabel='Iteration', ylabel='Directness Gap', title='Directness Gap Comparison'):
        """
        Visualizes the directness gap for different algorithms or settings.

        Parameters:
            directness_values (list of list of float): List of directness gap values for each algorithm/setting.
            labels (list of str): List of labels for each set of values.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Iteration'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Directness Gap'.
            title (str, optional): Title of the plot. Defaults to 'Directness Gap Comparison'.
        """
        plt.figure(figsize=(10, 6))
        for values, label in zip(directness_values, labels):
            plt.plot(values, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def create_parameter_heatmap(self, data, x_label, y_label, title, cmap='viridis'):
        """
        Creates a heatmap to visualize parameter values or performance metrics.

        Parameters:
            data (ndarray): 2D array containing the data to be visualized.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            title (str): Title of the heatmap.
            cmap (str, optional): Colormap to use. Defaults to 'viridis'.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, cmap=cmap)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def generate_summary_plots(self, performance_data, labels, xlabel='Iterations', ylabel='Performance Metric', title='Performance Comparison'):
        """
        Generates summary plots for comparing performance metrics of different algorithms.

        Parameters:
            performance_data (list of list of float): List of performance data for each algorithm.
            labels (list of str): List of labels for each set of data.
            xlabel (str, optional): Label for the x-axis. Defaults to 'Iterations'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Performance Metric'.
            title (str, optional): Title of the plot. Defaults to 'Performance Comparison'.
        """
        plt.figure(figsize=(10, 6))
        for data, label in zip(performance_data, labels):
            plt.plot(data, label=label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == '__main__':
    utils = VisualizationUtils()

    # Sample regret trajectories
    trajectories = [
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.6, 0.5, 0.4, 0.3, 0.2],
        [0.7, 0.6, 0.5, 0.4, 0.3]
    ]
    labels = ['Algorithm 1', 'Algorithm 2', 'Algorithm 3']
    utils.plot_regret_trajectories(trajectories, labels)

    # Sample directness gap values
    directness_gap_values = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ]
    labels = ['Setting A', 'Setting B']
    utils.visualize_directness_gap(directness_gap_values, labels)

    # Sample parameter heatmap data
    param_data = np.random.rand(5, 5)
    x_labels = ['Param 1', 'Param 2', 'Param 3', 'Param 4', 'Param 5']
    y_labels = ['Value 1', 'Value 2', 'Value 3', 'Value 4', 'Value 5']
    utils.create_parameter_heatmap(param_data, x_label='Parameters', y_label='Values', title='Parameter Heatmap')

    # Sample performance comparison data
    performance_data = [
        [10, 20, 30, 40, 50],
        [8, 15, 22, 30, 40]
    ]
    labels = ['Algorithm X', 'Algorithm Y']
    utils.generate_summary_plots(performance_data, labels)
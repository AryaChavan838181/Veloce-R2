import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def show_plot_before_after(dataset, column, scaling_function, plot_type='box'):
    """
    Displays plots for a column before and after applying a scaling function.

    Parameters:
    - dataset: The DataFrame containing the data.
    - column: The column to be visualized.
    - scaling_function: The function to apply for scaling (e.g., np.log1p).
    - plot_type: The type of plot to display ('box', 'line', etc.).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if plot_type == 'box':
        sns.boxplot(y=dataset[column], ax=axes[0])
    elif plot_type == 'line':
        axes[0].plot(dataset.index, dataset[column], color='blue')
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    axes[0].set_title(f'{column} (Original)')
    axes[0].set_ylabel(column)

    dataset[column] = scaling_function(dataset[column])

    if plot_type == 'box':
        sns.boxplot(y=dataset[column], ax=axes[1])
    elif plot_type == 'line':
        axes[1].plot(dataset.index, dataset[column], color='green')
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    axes[1].set_title(f'{column} (Transformed)')
    axes[1].set_ylabel(column)

    plt.tight_layout()
    plt.show()
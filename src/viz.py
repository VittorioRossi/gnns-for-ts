import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def viz_ts(time_series, time_points=None, x_label='Time', y_label='Value', title='Time Series Plot'):
    """
    Visualizes a generic time series.

    This function takes in a time series data and optional labels and title.
    It creates a visualization of the time series using a line plot.

    Parameters:
    time_series (list): A list of values representing the time series.
    time_points (list, optional): A list of time points corresponding to the time series.
    x_label (str, optional): Label for the x-axis. Default is 'Time'.
    y_label (str, optional): Label for the y-axis. Default is 'Value'.
    title (str, optional): Title of the plot. Default is 'Time Series Plot'.

    Returns:
    None
    """

    if time_points is None:
        time_points = list(range(len(time_series)))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=time_points, y=time_series)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

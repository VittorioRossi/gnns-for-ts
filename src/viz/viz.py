import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as anim

def setup_plot(time_series, time_points=None, x_label='Time', y_label='Value', title='Time Series Plot'):
    """
    Prepares and returns the plot's figure and axes with basic configuration.
    """
    if time_points is None:
        time_points = list(range(len(time_series)))
    
    fig, ax = plt.subplots()
    ax.set_xlim(min(time_points), max(time_points))
    ax.set_ylim(min(time_series), max(time_series))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return fig, ax

def viz_ts(time_series, time_points=None, x_label='Time', y_label='Value', title='Time Series Plot', *args, **kwargs):
    """
    Plots a static time series graph.
    """
    fig, ax = setup_plot(time_series, time_points, x_label, y_label, title)
    sns.lineplot(x=time_points or list(range(len(time_series))), y=time_series, ax=ax, *args, **kwargs)
    return fig, ax


def viz_animated_ts(time_series, time_points=None, x_label='Time', y_label='Value', title='Animated Time Series Plot', interval=50, smoothness_factor=5):
    """
    Create an animated time series plot.

    Parameters:
    - time_series (array-like): The time series data to be plotted.
    - time_points (array-like, optional): The corresponding time points for the time series data. If not provided, the time points will be generated based on the length of the time series and the smoothness factor.
    - x_label (str, optional): The label for the x-axis. Default is 'Time'.
    - y_label (str, optional): The label for the y-axis. Default is 'Value'.
    - title (str, optional): The title of the plot. Default is 'Animated Time Series Plot'.
    - interval (int, optional): The interval between frames in milliseconds. Default is 50.
    - smoothness_factor (int, optional): The factor used to determine the number of time points. Default is 5.

    Returns:
    - a (matplotlib.animation.FuncAnimation): The animated plot.

    """
    if time_points is None:
        time_points = np.linspace(0, len(time_series)-1, len(time_series)*smoothness_factor)
        time_series = np.interp(time_points, np.arange(len(time_series)), time_series)
    else:
        original_length = len(time_points)
        time_points = np.linspace(time_points[0], time_points[-1], original_length * smoothness_factor)
        time_series = np.interp(time_points, np.linspace(0, original_length-1, original_length), time_series)
    
    fig, ax = setup_plot(time_series, time_points, x_label, y_label, title)
    
    line, = ax.plot([], [], lw=2)
    
    def init():
        line.set_data([], [])
        return (line,)
    
    def animate(i):
        x = time_points[:i+1]
        y = time_series[:i+1]
        line.set_data(x, y)
        return (line,)
    
    a = anim.FuncAnimation(fig, animate, init_func=init, frames=len(time_points), blit=True, interval=interval)
    
    return a


def plot_predictions(true_x, true_y, pred_y, window_size, target_size):
    """
    Plots the true time series against the predicted one.

    Args:
        true_x (numpy.ndarray or torch.Tensor): The true values for the input window.
        true_y (numpy.ndarray or torch.Tensor): The true values for the target window.
        pred_y (numpy.ndarray or torch.Tensor): The predicted values for the target window.
        window_size (int): The size of the input window.
        target_size (int): The size of the target window.
    """
    if isinstance(true_x, torch.Tensor):
        true_x = true_x.numpy()
    if isinstance(true_y, torch.Tensor):
        true_y = true_y.numpy()
    if isinstance(pred_y, torch.Tensor):
        pred_y = pred_y.numpy()

    plt.figure(figsize=(12, 6))
    
    true_series = np.concatenate((true_x, true_y))
    pred_series = np.concatenate((true_x, pred_y))
    plt.plot(range(window_size + target_size), true_series, label='True values', linestyle='-', color='blue')
    plt.plot(range(window_size+target_size), pred_series, label='Predicted future values', linestyle='-', color='red')

    plt.title('Hourly Traffic Volume: True Values vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Traffic Volume')
    plt.legend()
    plt.show()
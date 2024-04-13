from src.utils import aggregate_time_series

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

import numpy as np
        
from typing import Callable

class TimeSeriesDataset(Dataset):
    """
    A PyTorch dataset class for time series data with Graph Convolutional Neural Networks (GCNNs).

    Args:
        time_series_data (numpy.ndarray): The input time series data.
        window_size (int): The window size for the time series data.
        target_size (int): The number of time steps to predict.
        new_granularity (int, optional): The new granularity for the data. You can pass the number of timesteps to aggregate. Defaults to None.
        aggregation_func (Callable[[np.ndarray], np.ndarray], optional): The aggregation function to use. Defaults to np.sum.

    Attributes:
        time_series_data (numpy.ndarray): The input time series data.
        window_size (int): The window size for the time series data.
        target_size (int): The number of time steps to predict.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(self, 
                 time_series_data: np.ndarray, 
                 window_size: int,
                 target_size: int,
                 new_granularity: int = None,
                 aggregation_func: Callable[[np.ndarray], np.ndarray] = np.sum,
                 standardize:bool = True):
        """
        Initializes a TimeSeriesGCNNDataset object.

        Args:
            time_series_data (numpy.ndarray): The input time series data.
            window_size (int): The window size for the time series data.
            target_size (int): The number of time steps to predict.
            new_granularity (int, optional): The new granularity for the data. You can pass the number of timesteps to aggregate. Defaults to None.
            aggregation_func (Callable[[np.ndarray], np.ndarray], optional): The aggregation function to use. Defaults to np.sum.
        """

        self.window_size = window_size
        self.target_size = target_size
        self.time_series_data = time_series_data # not aggregated data

        self.num_nodes = time_series_data.shape[1]

        self.standardize = standardize

        if new_granularity is not None:
            if new_granularity > time_series_data.shape[0] or new_granularity < 1:
                raise ValueError("Invalid new_granularity value. It should be between 1 and the number of timesteps in the data.")

            self.time_series_data = self._aggregate_time_series(time_series_data, new_granularity, aggregation_func)

        if self.standardize:
            self.time_series_data = self._standardize_data(self.time_series_data)

    def _aggregate_time_series(self, data, new_granularity, aggregation_func):
        
        return aggregate_time_series(data, new_granularity, aggregation_func)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.time_series_data.shape[0] - self.window_size - self.target_size - 1

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input data and the target.
        """

        _node_features_shape = (self.num_nodes, self.window_size)
        _target_shape = (self.num_nodes, self.target_size)

        node_features = self.time_series_data[idx:idx+self.window_size].T
        target = self.time_series_data[idx+self.window_size:idx+self.window_size+self.target_size].T
        
        if (node_features.shape != _node_features_shape) or (target.shape != _target_shape):
            raise IndexError(f"Index out of bound.")


        assert node_features.shape == (self.num_nodes, self.window_size), f"Invalid shape for node_features. Expected {(self.num_nodes, self.window_size)}, but got {node_features.shape}."
        assert target.shape == (self.num_nodes, self.target_size), f"Invalid shape for target. Expected {(self.num_nodes, self.target_size)}, but got {target.shape}."

        return {"input": torch.tensor(node_features, dtype=torch.float), 
                "target": torch.tensor(target, dtype=torch.float)}
    
    def _standardize_data(self, data):
        """
        Standardizes the input data.

        Args:
            data (numpy.ndarray): The input data to standardize.

        Returns:
            numpy.ndarray: The standardized data.
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        return (data - self.mean) / self.std

    def _inverse_standardize_data(self, data):
        """
        Inverse standardizes the input data.

        Args:
            data (numpy.ndarray): The input data to inverse standardize.

        Returns:
            numpy.ndarray: The inverse standardized data.
        """
        if self.standardize:
            return data * self.std[:, np.newaxis] + self.mean[:, np.newaxis]
        return data

class TimeSeriesGCNNDataset(TimeSeriesDataset):
    def __init__(self, 
                 time_series_data: np.ndarray,
                 edge_index: np.ndarray,
                 edge_attr:np.ndarray,
                 window_size: int, 
                 target_size: int, 
                 **kwargs):
        super().__init__(time_series_data, window_size, target_size,**kwargs)
        
        self.edge_index = self._get_edge_index(edge_index)
        self.edge_attr = self._get_edge_attr(edge_attr)

    def __getitem__(self, idx):
        node_features, target = super().__getitem__(idx).values()
        
        data = Data(x=node_features,
                    y=target,
                    edge_index=self.edge_index,
                    edge_attr=self.edge_attr)
        
        return data
    
    def _get_edge_index(self, edge_index):
        return torch.tensor(edge_index, dtype=torch.long).t()

    def _get_edge_attr(self, edge_attr):
        return torch.tensor(edge_attr, dtype=torch.float)
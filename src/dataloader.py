from numpy.core.fromnumeric import sum as sum
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
                 aggregation_func: Callable[[np.ndarray], np.ndarray] = np.sum):
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

        if new_granularity is not None:
            if new_granularity > time_series_data.shape[0] or new_granularity < 1:
                raise ValueError("Invalid new_granularity value. It should be between 1 and the number of timesteps in the data.")

            self.time_series_data = self._aggregate_time_series(time_series_data, new_granularity, aggregation_func)

    def _aggregate_time_series(self, data, new_granularity, aggregation_func):
        """
        Aggregates the time series data based on the new granularity.

        Args:
            data (numpy.ndarray): The input time series data.
            new_granularity (int): The new granularity for the data.
            aggregation_func (Callable[[np.ndarray], np.ndarray]): The aggregation function to use.

        Returns:
            numpy.ndarray: The aggregated time series data.
        """
        return aggregation_func(data.reshape(-1, new_granularity, *data.shape[1:]), axis=1)

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


class TimeSeriesGCNNDataset(TimeSeriesDataset):
    def __init__(self, 
                 time_series_data: np.ndarray,
                 edge_index: np.ndarray,
                 edge_attr:np.ndarray,
                 window_size: int, 
                 target_size: int, 
                 new_granularity: int = None, 
                 aggregation_func: Callable[[np.ndarray], np.ndarray] = np.sum):
        super().__init__(time_series_data, window_size, target_size, new_granularity, aggregation_func)
        
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



class TrafficDataset(Dataset):
    """
    A PyTorch dataset class for traffic data.



    Args:
        data_seq (numpy.ndarray): The input traffic data sequence.
        num_of_weeks (int): The number of weeks to consider in the dataset.
        num_of_days (int): The number of days to consider in the dataset.
        num_of_hours (int): The number of hours to consider in the dataset.
        num_for_predict (int): The number of time steps to predict.
        points_per_hour (int, optional): The number of data points per hour. Defaults to 12.

    Attributes:
        data_seq (numpy.ndarray)
        num_of_weeks (int)
        num_of_days (int) 
        num_of_hours (int) 
        num_for_predict (int)
        points_per_hour (int) 
        max_look_back (int): The maximum lookback period based on the input parameters.
        total_samples (int): The total number of samples in the dataset.

    Methods:
        __getitem__(self, idx): Retrieves a sample from the dataset.
        __len__(self): Returns the total number of samples in the dataset.
        get_sample_indices(self, label_start_idx): Retrieves the indices for a sample.
        search_data(self, num_of_batches, label_start_idx, units): Searches for data indices.
    """

    def __init__(self, data_seq, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=12):
        self.data_seq = data_seq
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict
        self.points_per_hour = points_per_hour

        # Calculate the maximum lookback period to determine the start of the dataset
        self.max_look_back = max(self.num_of_weeks * 7 * 24, self.num_of_days * 24, self.num_of_hours) * self.points_per_hour
        self.total_samples = len(data_seq) - self.max_look_back - self.num_for_predict

    def __getitem__(self, idx):
        # Adjust idx to account for the lookback period
        adjusted_idx = idx + self.max_look_back 
        sample = self.get_sample_indices(adjusted_idx)
        
        if sample is None:  # In case of any indexing error, return zeros or handle appropriately
            zeros = torch.zeros((self.data_seq.shape[1], self.num_for_predict, self.data_seq.shape[2]))
            return zeros, zeros, zeros, zeros
        
        week_sample, day_sample, hour_sample, target = sample
        
        # Convert numpy arrays to torch tensors
        week_sample = torch.from_numpy(week_sample).float()
        day_sample = torch.from_numpy(day_sample).float()
        hour_sample = torch.from_numpy(hour_sample).float()
        target = torch.from_numpy(target).float()

        return week_sample, day_sample, hour_sample, target

    def __len__(self):
        return self.total_samples

    def get_sample_indices(self, label_start_idx):
        week_indices = self.search_data(self.num_of_weeks, label_start_idx, 7 * 24)
        day_indices = self.search_data(self.num_of_days, label_start_idx, 24)
        hour_indices = self.search_data(self.num_of_hours, label_start_idx, 1)

        if not week_indices or not day_indices or not hour_indices:
            return None

        week_sample = np.concatenate([self.data_seq[i: j] for i, j in week_indices], axis=0)
        day_sample = np.concatenate([self.data_seq[i: j] for i, j in day_indices], axis=0)
        hour_sample = np.concatenate([self.data_seq[i: j] for i, j in hour_indices], axis=0)
        target = self.data_seq[label_start_idx: label_start_idx + self.num_for_predict]

        return week_sample, day_sample, hour_sample, target

    def search_data(self, num_of_batches, label_start_idx, units):
        if self.points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")
        
        # If the current index is more self.num_for_predict away from the end of the data, return None
        if label_start_idx + self.num_for_predict > self.data_seq.shape[0]:
            return None
        
        x_idx = []

        for i in range(1, num_of_batches + 1):
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + self.points_per_hour * units
            if start_idx >= 0:
                x_idx.append((start_idx, end_idx))
            else:
                return None
            
        
        if len(x_idx) != num_of_batches:
            return None
        
        return x_idx[::-1]  # Reversed to ensure chronological order


        

def load_data(graph_signal_matrix_filename, batch_size=64, num_of_weeks=1, num_of_days=1, num_of_hours=1, num_for_predict=1):
    """
    Load data from a graph signal matrix file and create a data loader.

    Parameters:
    graph_signal_matrix_filename (str): The filename of the graph signal matrix file.
    batch_size (int, optional): The batch size for the data loader. Default is 64.
    num_of_weeks (int, optional): The number of weeks to consider in the data. Default is 1.
    num_of_days (int, optional): The number of days to consider in the data. Default is 1.
    num_of_hours (int, optional): The number of hours to consider in the data. Default is 1.
    num_for_predict (int, optional): The number of time steps to predict. Default is 1.

    Returns:
    DataLoader: The data loader containing the loaded data.

    """
    data_seq = np.load(graph_signal_matrix_filename)['data']

    dataset = TrafficDataset(data_seq, num_of_weeks, num_of_days, num_of_hours, num_for_predict)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrafficDataset(Dataset):
    """
    A PyTorch dataset class for traffic data.

    Args:
        data_seq (numpy.ndarray): The input traffic data sequence.
        num_of_weeks (int): The number of weeks to include in each sample.
        num_of_days (int): The number of days to include in each sample.
        num_of_hours (int): The number of hours to include in each sample.
        num_for_predict (int): The number of time steps to predict.
        points_per_hour (int, optional): The number of data points per hour. Defaults to 12.

    Attributes:
        samples (list): A list of preprocessed samples.

    Methods:
        preprocess(): Preprocesses the data sequence and returns a list of samples.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample at the given index.
        get_sample_indices(label_start_idx): Returns the sample indices for the given label start index.
        search_data(num_of_batches, label_start_idx, units): Calculates indices for sampling data based on class attributes.
    """

    def __init__(self, data_seq, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour=12):
        self.data_seq = data_seq
        self.num_of_weeks = num_of_weeks
        self.num_of_days = num_of_days
        self.num_of_hours = num_of_hours
        self.num_for_predict = num_for_predict
        self.points_per_hour = points_per_hour
        
        self.samples = self.preprocess()

    def preprocess(self):
        """
        Preprocesses the data sequence and returns a list of samples.

        Returns:
            list: A list of preprocessed samples.
        """
        all_samples = []
        for idx in range(self.data_seq.shape[0]):
            sample = self.get_sample_indices(idx)
            if not sample:
                continue

            week_sample, day_sample, hour_sample, target = sample
            all_samples.append((
                np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
                np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
            ))
        return all_samples
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple: A tuple containing the week sample, day sample, hour sample, and target.
        """
        week_sample, day_sample, hour_sample, target = self.samples[idx]
        return torch.tensor(week_sample, dtype=torch.float), \
               torch.tensor(day_sample, dtype=torch.float), \
               torch.tensor(hour_sample, dtype=torch.float), \
               torch.tensor(target, dtype=torch.float)

    def get_sample_indices(self, label_start_idx):
        """
        Get the sample indices for the given label start index.

        Args:
            label_start_idx (int): The index of the label start.

        Returns:
            tuple: A tuple containing the week sample indices, day sample indices, hour sample indices, and target indices.
                - week_sample (numpy.ndarray): The concatenated week sample indices.
                - day_sample (numpy.ndarray): The concatenated day sample indices.
                - hour_sample (numpy.ndarray): The concatenated hour sample indices.
                - target (numpy.ndarray): The target indices.
        """
        week_indices = self.search_data(self.num_of_weeks, label_start_idx, 7 * 24)
        if not week_indices:
            return None

        day_indices = self.search_data(self.num_of_days, label_start_idx, 24)
        if not day_indices:
            return None

        hour_indices = self.search_data(self.num_of_hours, label_start_idx, 1)
        if not hour_indices:
            return None

        week_sample = np.concatenate([self.data_seq[i: j] for i, j in week_indices], axis=0)
        day_sample = np.concatenate([self.data_seq[i: j] for i, j in day_indices], axis=0)
        hour_sample = np.concatenate([self.data_seq[i: j] for i, j in hour_indices], axis=0)
        target = self.data_seq[label_start_idx: label_start_idx + self.num_for_predict]

        return week_sample, day_sample, hour_sample, target

    def search_data(self, num_of_batches, label_start_idx, units):
        """
        Calculates indices for sampling data based on class attributes.

        Args:
            num_of_batches (int): The number of batches (weeks, days, or hours) to be used for training.
            label_start_idx (int): The first index of the predicting target.
            units (int): Represents time units in hours (e.g., 7*24 for weeks, 24 for days, 1 for hours).

        Returns:
            list: A list of tuples where each tuple contains start and end indices for sampling.
        """
        if self.points_per_hour < 0:
            raise ValueError("points_per_hour should be greater than 0!")

        if label_start_idx + self.num_for_predict > self.data_seq.shape[0]:
            return None

        x_idx = []
        for i in range(1, num_of_batches + 1):
            start_idx = label_start_idx - self.points_per_hour * units * i
            end_idx = start_idx + self.points_per_hour * units  # Adjusted to ensure correct sample length
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



if __name__ == "__main__":

    print("Loading the data...")
    # Load the data
    data_loader = load_data('data/PEMS04/pems04.npz')

    # Print the first lines of the dataset
    for idx, (week_sample, day_sample, hour_sample, target) in enumerate(data_loader):
        print(f"Sample {idx+1}:")
        print("Week sample:", week_sample)
        print("Day sample:", day_sample)
        print("Hour sample:", hour_sample)
        print("Target:", target)
        print()
        if idx == 1:
            break
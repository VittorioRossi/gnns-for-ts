import numpy as np
import torch

from typing import Any

def create_train_test_split(data, 
                            train_ratio = 0.7,
                            test_ratio = 0.2,
                            window_size=10, 
                            target_size=5) -> tuple[list, list, list]:
    """
    Splits the input data into training, test, and validation sets.

    Args:
        data (ndarray): The input data of shape (num_timesteps, num_nodes, num_features).
        train_ratio (float, optional): The ratio of data to be used for training. Defaults to 0.7.
        test_ratio (float, optional): The ratio of data to be used for testing. Defaults to 0.2.
        window_size (int, optional): The size of the sliding window. Defaults to 10.
        target_size (int, optional): The size of the target window. Defaults to 5.

    Returns:
        tuple: A tuple containing the training data, test data, and validation data.
            - train_data (ndarray): The training data of shape (num_train_samples, num_nodes, num_features).
            - test_data (ndarray): The test data of shape (num_test_samples, num_nodes, num_features).
            - val_data (ndarray): The validation data of shape (num_val_samples, num_nodes, num_features).

    Raises:
        ValueError: If there is not enough data for the given window size and target size.
    """
    num_timesteps = data.shape[0]

    # Calculate the number of samples for training and testing
    num_train_samples = int(num_timesteps * train_ratio)

    # Calculate the number of samples required for test data
    num_test_samples = int(num_timesteps * test_ratio)

    # Calculate the number of samples required for validation data
    num_val_samples = num_timesteps - num_train_samples - num_test_samples

    # Check if the test data has enough samples
    if num_test_samples < 1:
        raise ValueError("Not enough data for the given window size and target size")

    # Split the data into training, test, and validation sets
    train_data = data[:num_train_samples]
    test_data = data[num_train_samples:num_train_samples+num_test_samples]
    val_data = data[num_train_samples+num_test_samples:]

    return train_data, test_data, val_data


def compute_test_loss(model, loss_fn, test_data_loader, device):
    """
    Computes the test loss for a given model and test data.

    Args:
        model (nn.Module): The PyTorch model.
        loss_fn (nn.Module): The loss function.
        test_data_loader (DataLoader): The test data loader.
        device (str): The device to run the computation on.

    Returns:
        float: The test loss.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        loss = 0
        for batch in test_data_loader:
            batch = batch.to(device)  # Move batch to device
            outputs = model(batch.x, batch.edge_index, batch.edge_weight)  # Forward pass
            loss += loss_fn(outputs, batch.y).item()  # Compute the loss

        loss /= len(test_data_loader)  # Average the loss over all batches
    
    return loss


def aggregate_time_series(data, new_granularity, aggregation_func):
    """
    Aggregates the time series data based on the new granularity.

    Args:
        data (numpy.ndarray): The input time series data.
        new_granularity (int): The new granularity for the data.
        aggregation_func (Callable[[np.ndarray], np.ndarray]): The aggregation function to use.

    Returns:
        numpy.ndarray: The aggregated time series data.
    """
    if data.shape[0] % new_granularity != 0:
        # Pad the sequence with zeros
        pad_length = new_granularity - (data.shape[0] % new_granularity)
        data = np.pad(data, [(0, pad_length), (0, 0)], mode='constant')
    
    return aggregation_func(data.reshape(-1, new_granularity, *data.shape[1:]), axis=1)



class EarlyStopper:
    """
    https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

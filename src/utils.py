import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

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

class Evaluator:
    def __init__(self, metrics:dict[str, object], 
                     save_history:bool = True, 
                     save_best:bool=False, 
                     save_best_metric:str = "") -> None:
            """
            Initialize the Utils class.

            Parameters:
            - metrics (dict[str, function]): A dictionary containing the metrics name and the function to compute the metric.
            - save_history (bool): Flag indicating whether to save the history of metrics. Default is True.
            - save_best (bool): Flag indicating whether to save the best metric. Default is False.
            - save_best_metric (str | None): The name of the metric to save as the best. Default is None.
            """
            self.metrics = metrics

            if save_history:
                self.history = {metric: [] for metric in metrics.keys()}

            if save_best:
                if save_best_metric is None:
                    raise ValueError("save_best_metric must be provided if save_best is True")
                elif save_best_metric not in metrics.keys():
                    raise ValueError(f"{save_best_metric} is not in the metrics list")

                self.save_best = True
                self.best_metric = float('inf')
                self.save_best_metric = save_best_metric
    
    def evaluate(self, y_true, y_pred, save_history = False) -> dict[str, float]:
        metrics = {metric: 0 for metric in self.metrics.keys()}
        for metric, func in self.metrics.items():
            metrics[metric] = func(y_true, y_pred)

            if hasattr(self, 'history') and save_history:
                self.history[metric].append(metrics[metric])

        return metrics

    def evaluate_dataloader(self, data_loader, model, device) -> dict[str, float]:
        """
        Evaluates the metrics of a model on a dataset.

        Args:
            data_loader (DataLoader): The data loader.
            model (nn.Module): The model.
            device (str): The device to run the computation on.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)  # Move batch to device
                outputs = model(batch.x, batch.edge_index, batch.edge_weight)  # Forward pass

                y_true = batch.y.cpu().numpy()
                y_pred = outputs.cpu().numpy()
                metrics = self.evaluate(y_true, y_pred, save_history=True)
        
        if hasattr(self, 'save_best'):
            self._update_best(metrics, model)

        return metrics
    
    def _update_best(self, metrics, model):
            """
            Updates the best model based on the given metrics.

            Args:
                metrics (dict): A dictionary containing the evaluation metrics.
                model: The model whose state_dict will be saved as the best model.

            Returns:
                None
            """
            if self.save_best:
                if metrics[self.save_best_metric] < self.best_metric:
                    self.best_metric = metrics[self.save_best_metric]
                    self.best_metrics = metrics
                    self.best_model = model.state_dict()


def train_model(model, train_loader, val_loader, criterion, device, epochs=100, batch_size=128, learning_rate=1e-4, patience=2, min_delta=0.0005):
    """
    Trains the given model using the provided datasets and training parameters.
    
    Parameters:
        model (torch.nn.Module): The model to train.
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for testing/validation.
        criterion (callable): The loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
        epochs (int): The maximum number of epochs to train.
        batch_size (int): The size of each batch during training.
        learning_rate (float): The learning rate for the optimizer.
        patience (int): The patience for early stopping.
        min_delta (float): The minimum delta change to qualify as an improvement for early stopping.
    """
    
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()

    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')

        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            test_loss = compute_test_loss(model, criterion, val_loader, device)
            pbar.set_postfix({'Train loss': total_loss / len(pbar), 
                            'Test loss': test_loss})

        if early_stopper.early_stop(test_loss):
            print("Early stopping")
            break

    return model
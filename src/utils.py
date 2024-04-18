import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from tqdm.auto import tqdm

from typing import Any
from copy import deepcopy

from src.data import TimeSeriesGCNNDataset
from torch_geometric.data import DataLoader

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

    def evaluate_dataloader(self, data_loader, model, device, **kwargs) -> dict[str, float]:
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
        metrics = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)  # Move batch to device
                outputs = model(batch.x, batch.edge_index, batch.edge_weight)  # Forward pass

                y_true = batch.y.cpu().numpy()
                y_pred = outputs.cpu().numpy()
                crt_mtr = self.evaluate(y_true, y_pred, save_history=True)
                metrics.append(crt_mtr)
        
        if hasattr(self, 'save_best'):
            self._update_best(metrics, model)

        return {metric: np.mean([m[metric] for m in metrics]) for metric in metrics[0].keys()}
    

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


def create_loader(raw_data:np.ndarray, 
                  adj_matrix:np.ndarray,
                  cost_matrix:np.ndarray,
                  train_split = 0.7, 
                  test_split = 0.2,
                  window_size = 12,
                  target_size = 1,
                  granularity = 12,
                  batch_size = 64,
                  standardize = True,
                  **kwargs) -> tuple:
                  
    train_data, test_data, val_data = create_train_test_split(raw_data, 
                                                              train_ratio=train_split, 
                                                              test_ratio=test_split)

    train_dataset = TimeSeriesGCNNDataset(train_data,
                                          adj_matrix,
                                          cost_matrix,
                                          window_size=window_size,
                                          target_size=target_size,
                                          new_granularity=granularity, 
                                          standardize=standardize)

    test_dataset = TimeSeriesGCNNDataset(test_data,
                                         adj_matrix,
                                         cost_matrix,
                                         window_size=window_size,
                                         target_size=target_size,
                                         new_granularity=granularity, 
                                         standardize=standardize)                                         

    val_dataset = TimeSeriesGCNNDataset(val_data,
                                         adj_matrix,
                                         cost_matrix,
                                         window_size=window_size,
                                         target_size=target_size,
                                         new_granularity=granularity, 
                                         standardize=standardize)    

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                device, 
                epochs=100, 
                learning_rate=1e-4, 
                patience=2, 
                min_delta=0.0005,
                reset_weights=True,
                **kwargs
                ):
    """
    Trains the given model using the provided datasets and training parameters.
    
    Parameters:
        model (torch.nn.Module): The model to train.
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for testing/validation.
        criterion (callable): The loss function.
        device (torch.device): The device to run the model on (CPU or GPU).
        epochs (int): The maximum number of epochs to train.
        learning_rate (float): The learning rate for the optimizer.
        patience (int): The patience for early stopping.
        min_delta (float): The minimum delta change to qualify as an improvement for early stopping.
        reset_weights (bool): Flag indicating whether to reset the model weights before training. Default is True.
    """
    
    if reset_weights:
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

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



def load_raw(path_to_raw_data, path_to_raw_dist):
    raw = np.load(path_to_raw_data)["data"]
    raw_adj = pd.read_csv(path_to_raw_dist)[["from", "to"]].to_numpy()
    raw_dist = pd.read_csv(path_to_raw_dist)[["cost"]].to_numpy()
    return raw[:, :, 0], raw_adj, raw_dist
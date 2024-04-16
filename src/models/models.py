import torch
from torch.nn import Linear, ReLU, Dropout, ModuleList, Sequential
from torch_geometric.nn import GCNConv

class BaselineGCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) model.

    Args:
        num_of_features (int): Number of input features.
        num_feature_out (int): Number of output features.

    Attributes:
        conv1 (GCNConv): First graph convolutional layer.
        conv2 (GCNConv): Second graph convolutional layer.
        conv3 (GCNConv): Third graph convolutional layer.
        lin (Linear): Linear layer for output.

    """

    def __init__(self, num_of_features, num_feature_out, hidden_channels=512):
        super().__init__()
        self.graph_convolution = GCNConv(num_of_features, hidden_channels)
        self.linear = Linear(hidden_channels, num_feature_out)
        self.activation = ReLU(inplace=False)


    def forward(self, x, edge_index, edge_weight):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Input features.
            edge_index (LongTensor): Graph edge indices.

        Returns:
            out (Tensor): Output tensor.
            h (Tensor): Hidden tensor.

        """
        h = self.graph_convolution(x, edge_index, edge_weight)
        h = self.activation(h)
        
        h = self.linear(h)
        out = self.activation(h)

        return out


class MultiLayerGCN(torch.nn.Module):
    """
    A dynamic and modular Graph Convolutional Network (GCN) model that supports various types of GCN layers.

    Args:
        num_features (int): Number of input features.
        num_features_out (int): Number of output features.
        layer_configs (list): Configuration for each layer, including type and parameters.
        linear_layers (list of int): Number of features for each linear layer.
        dropout (float): Dropout rate.
    """
    def __init__(self, num_features, num_features_out, layer_configs, linear_layers, dropout=0.2):
        super(MultiLayerGCN, self).__init__()
        self.layers = ModuleList()

        # Building the GCN layers dynamically based on configuration
        in_channels = num_features
        for config in layer_configs:
            layer_type = config['type']
            layer_params = {k: v for k, v in config.items() if k != 'type'}
            layer = layer_type(in_channels, **layer_params)
            self.layers.append(layer)
            in_channels = layer_params.get('out_channels', in_channels)  # Update in_channels for next layer

        # Building the linear layers
        linear_layers = [in_channels] + linear_layers
        linear_modules = []
        for in_features, out_features in zip(linear_layers[:-1], linear_layers[1:]):
            linear_modules.append(Linear(in_features, out_features))
            linear_modules.append(ReLU(inplace=True))
            linear_modules.append(Dropout(dropout))

        linear_modules.append(Linear(linear_layers[-2], num_features_out))
        self.linear_layers = Sequential(*linear_modules)

    def forward(self, x, edge_index, edge_weight):
        # Process through GCN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)

        # Process through linear layers
        x = self.linear_layers(x)
        return x

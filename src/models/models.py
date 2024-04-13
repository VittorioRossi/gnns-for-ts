import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, GraphNorm


class NNAR(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNAR, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)  # Input to hidden layer
        self.relu = ReLU()                          # Activation function
        self.fc2 = Linear(hidden_size, output_size) # Hidden to output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Activation function applied to hidden layer
        x = self.fc2(x)             # Output layer
        return x



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

    def __init__(self, 
                 num_of_features, 
                 num_feature_out, 
                 embedding_inner_layer = 512,
                 embedding_size: int = 128,
                 hidden_channels: int = 1024,
                 dropout=0.2,
                 graph_normalization=False):
        
        super().__init__()
        self.graph_convolution_1 = GCNConv(num_of_features, embedding_inner_layer)
        self.graph_convolution_2 = GCNConv(embedding_inner_layer, embedding_size)
        self.graph_normalization = GraphNorm(embedding_inner_layer) if graph_normalization else lambda x: x
        
        self.linear_1 = Linear(embedding_size, hidden_channels)
        self.linear_2 = Linear(hidden_channels, num_feature_out)

        self.dropout = Dropout(p=dropout) if dropout > 0 else lambda x: x
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
        h = self.graph_convolution_1(x, edge_index, edge_weight)
        h = self.activation(h)
        
        h = self.graph_normalization(h)
        h = self.graph_convolution_2(h, edge_index, edge_weight)
        h = self.activation(h)

        h = self.dropout(h)

        h = self.linear_1(h)
        h = self.activation(h)

        h = self.dropout(h)

        h = self.linear_2(h)
        out = self.activation(h)

        return out


class EncoderDecoder:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, edge_weight):
        x = self.encoder(x, edge_index, edge_weight)
        x = self.decoder(x, edge_index, edge_weight)
        return x
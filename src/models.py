import torch
from torch.nn import Linear, ReLU
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
        self.conv1 = GCNConv(num_of_features, hidden_channels)
        self.lin = Linear(hidden_channels, num_feature_out)
        self.activ = ReLU(inplace=False)


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
        h = self.conv1(x, edge_index, edge_weight)
        h = self.activ(h)
        
        h = self.lin(h)
        out = self.activ(h)

        return out

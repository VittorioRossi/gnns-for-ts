import torch
from torch_geometric.nn import GCNConv
from torch.nn import ReLU, LSTM, Linear, Module

class GCNLayer(Module):
    def __init__(self, in_channels, out_channels, normalization_layer=None, activation_function=ReLU):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.normalization = normalization_layer(out_channels) if normalization_layer else torch.nn.Identity()
        self.activation = activation_function() if activation_function else torch.nn.Identity()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.normalization(x)
        x = self.activation(x)
        return x
    

import torch
from torch.nn import Module, Linear, ReLU, LSTM
from torch_geometric.nn import GCNConv
from torch.nn.functional import relu

class GCNLSTMCell(Module):
    def __init__(self, in_channels, out_channels, normalization_layer=None, activation_function=ReLU, lstm_features=None):
        super(GCNLSTMCell, self).__init__()
        # In the initialization, make sure that the LSTM features are defined
        self.conv = GCNConv(in_channels, out_channels)  
        self.normalization = normalization_layer(out_channels) if normalization_layer else torch.nn.Identity()
        self.lstm = LSTM(out_channels, lstm_features or out_channels, batch_first=True)
        self.fc = Linear(lstm_features or out_channels, out_channels)
        self.activation = activation_function() if activation_function else torch.nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (batch_size, num_nodes, sequence_length) 
        # where sequence_length is effectively the number of features per node
        batch_size, num_nodes, sequence_length = x.shape
    
        # Apply GCN and normalization
        x = self.conv(x, edge_index, edge_weight)  # Apply GCN
        x = self.normalization(x)
        
        # Reshape back to (batch_size, num_nodes, sequence_length, out_channels)
        x = x.view(batch_size, sequence_length, num_nodes).permute(0, 2, 1)  # Reshape for LSTM processing

        # Process temporal features for each node
        x_final = torch.zeros(batch_size, num_nodes, self.fc.out_features, device=x.device)
        for node in range(num_nodes):
            node_features = x[:, node]  # shape (batch_size, sequence_length, out_channels)
            node_features, _ = self.lstm(node_features)  # Apply LSTM
            node_features = self.fc(node_features[:, -1])  # Take the last timestep
            x_final[:, node] = self.activation(node_features)

        return x_final

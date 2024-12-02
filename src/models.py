import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from torch_geometric.utils import add_self_loops, degree

ACTIVATIONS = {
    'relu': F.relu,
    'softplus': F.softplus,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'elu': F.elu,
    'leaky_relu': F.leaky_relu,
    'swish': lambda x: x * F.sigmoid(x),  # Swish activation function.
    'gelu': F.gelu,  # Gaussian Error Linear Unit.
    'mish': lambda x: x * F.tanh(F.softplus(x)),  # Mish activation function.
    'selu': F.selu,  # Scaled Exponential Linear Unit.
    'prelu': nn.PReLU(),  # Parametric ReLU.
    'celu': F.celu,  # Continuously Differentiable Exponential Linear Unit.
}


class GCN(nn.Module):
    def __init__(self, nfeat, hidden_layers, nclass, dropout, normalization=True, activation='relu'):
        super(GCN, self).__init__()

        # Check if provided activation is valid
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(ACTIVATIONS.keys())}.")

        self.layers = torch.nn.ModuleList()
        layer_sizes = [nfeat] + hidden_layers

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GCNConv(in_dim, out_dim))

        # Define fully connected layers
        self.fc1 = nn.Linear(layer_sizes[-1], 128)
        self.fc2 = nn.Linear(128, nclass)
       
        self.dropout = dropout
        self.normalization = normalization
        self.activation = ACTIVATIONS[activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Normalize the adjacency matrix if needed
        if self.normalization:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index
            deg = degree(row, dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            normalized_edge_weight = deg_inv_sqrt[row] * data.edge_attr * deg_inv_sqrt[col]
        else:
            normalized_edge_weight = data.edge_attr

        # Iterate through GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, normalized_edge_weight)
            # Do not apply activation & dropout in the last layer
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pass through fully connected layers
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class GCN_with_attention(nn.Module):
    def __init__(self, nfeat, hidden_layers, nclass, dropout, normalization=True, activation='relu'):
        super(GCN_with_attention, self).__init__()

        # Check if provided activation is valid
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(ACTIVATIONS.keys())}.")

        self.layers = torch.nn.ModuleList()
        layer_sizes = [nfeat] + hidden_layers

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(GCNConv(in_dim, out_dim))

        # Define fully connected layers
        self.fc1 = nn.Linear(layer_sizes[-1], 128)
        self.self_attention = nn.MultiheadAttention(128, num_heads=1)
        self.fc2 = nn.Linear(128, nclass)
       
        self.dropout = dropout
        self.normalization = normalization
        self.activation = ACTIVATIONS[activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Normalize the adjacency matrix if needed
        if self.normalization:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            row, col = edge_index
            deg = degree(row, dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            normalized_edge_weight = deg_inv_sqrt[row] * data.edge_attr * deg_inv_sqrt[col]
        else:
            normalized_edge_weight = data.edge_attr

        # Iterate through GCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, normalized_edge_weight)
            # Do not apply activation & dropout in the last layer
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pass through fully connected layers
        x = self.activation(self.fc1(x))
        # Apply self-attention to individual nodes' embeddings
        attention, _ = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attention = attention.squeeze(0)  # Remove the batch dimension
        x = self.fc2(attention)
        return x



class GraphSAGE(nn.Module):
    def __init__(self, nfeat, hidden_layers, nclass, dropout, aggregator='mean', activation='relu'):
        super(GraphSAGE, self).__init__()

        # Check if provided activation is valid
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(ACTIVATIONS.keys())}.")

        self.layers = torch.nn.ModuleList()

        # The first layer is unique as it uses 'nfeat' as the input dimension.
        self.layers.append(SAGEConv(nfeat, hidden_layers[0], aggregator=aggregator))

        # For subsequent layers
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator=aggregator))

        self.fc1 = nn.Linear(hidden_layers[-1], 128)
        self.fc2 = nn.Linear(128, nclass)

        self.dropout = dropout
        self.activation = ACTIVATIONS[activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Iterate through GraphSAGE layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # Do not apply activation & dropout in the last layer
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pass through fully connected layers
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


class GraphSAGE_with_attention(nn.Module):
    def __init__(self, nfeat, hidden_layers, nclass, dropout, aggregator='mean', activation='relu'):
        super(GraphSAGE_with_attention, self).__init__()

        # Check if provided activation is valid
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(ACTIVATIONS.keys())}.")

        self.layers = torch.nn.ModuleList()

        # The first layer is unique as it uses 'nfeat' as the input dimension.
        self.layers.append(SAGEConv(nfeat, hidden_layers[0], aggregator=aggregator))

        # For subsequent layers
        for in_dim, out_dim in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator=aggregator))

        self.fc1 = nn.Linear(hidden_layers[-1], 128)
        self.self_attention = nn.MultiheadAttention(128, num_heads=1)
        self.fc2 = nn.Linear(128, nclass)

        self.dropout = dropout
        self.activation = ACTIVATIONS[activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Iterate through GraphSAGE layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            # Do not apply activation & dropout in the last layer
            if i != len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Pass through fully connected layers
        x = self.activation(self.fc1(x))
        # Apply self-attention to individual nodes' embeddings
        attention, _ = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        attention = attention.squeeze(0)  # Remove the batch dimension
        x = self.fc2(attention)
        return x

        #return F.log_softmax(x, dim=1)

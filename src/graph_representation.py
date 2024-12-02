import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import snf
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

def create_graph(X, y, distance_metric, K, mu, threshold):
    num_samples = X.shape[0]
    # Compute patient-wise similarity using SNF affinity matrix function
    affinity_matrix_single_modality = snf.make_affinity([X.astype(float)], metric= distance_metric, K=K, mu=mu)

    # Create the PyTorch Geometric Data object
    graph = Data(x=torch.tensor(X, dtype=torch.float),
                 edge_index=None,  # We'll create edge indices later
                 edge_attr=None,  # No edge attributes initially
                 y=torch.tensor(y, dtype=torch.long))

    # Create edges based on the patient-wise similarity
    edges = []
    edge_weights = []

    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            weight = affinity_matrix_single_modality[0][i, j]  # Use SNF affinity matrix as edge weight
            #weight = similarity_matrix[i,j]
            if weight > threshold:
                edges.append((i, j))
                edge_weights.append(weight)

    # Extract node indices for source and target nodes from the edges
    src, tgt = zip(*edges)
    src = torch.tensor(src, dtype=torch.long)
    tgt = torch.tensor(tgt, dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    # Update the graph with edge information
    graph.edge_index = torch.stack([src, tgt], dim=0)
    graph.edge_attr = edge_weights

    return graph

# visualize indiviual-wise similarity matrix
def visualize_similarity_matrix(graph):
    # Calculate the node similarity matrix using edge weights
    adjacency_matrix = to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr, max_num_nodes=graph.num_nodes)[0]
    node_similarity_matrix = torch.matmul(adjacency_matrix, adjacency_matrix.t())
    # Plot the node similarity matrix as a heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(node_similarity_matrix, cmap='coolwarm', origin='upper', vmin=0, vmax=1)
    plt.title('Node Similarity Matrix with Edge Weights')
    plt.show()

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

# Define a wrapper function for your model
class GCNWrapper(nn.Module):
    def __init__(self, model):
        super(GCNWrapper, self).__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        data = Data(x=x, edge_index=edge_index)
        return self.model(data)

def graph_node_features_explanation(model, model_path, test_graph, top_features, node_index=None):
    model = model # load the model
    #model_path = 'gcn_model.pkl'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Wrap your original model
    new_model = GCNWrapper(model)

    # Initialize the Explainer with the averaged model
    explainer = Explainer(
        model=new_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    results = []  # To store the results for each node

    if node_index is not None:
        # Generate explanation for the specified node index
        explanation = explainer(test_graph.x.cpu(), test_graph.edge_index.cpu(), index=node_index)
        explanation.visualize_feature_importance(feat_labels=top_features, top_k=30)
       
    else:
        # Loop through nodes and generate explanations for all nodes
        num_nodes = test_graph.num_nodes
        #for node_index in tqdm(range(num_nodes)):
        for i, node_index in enumerate(range(num_nodes)):
            explanation = explainer(test_graph.x.cpu(), test_graph.edge_index.cpu(), index=node_index)
            feature_names = top_features
            feature_scores = explanation.get('node_mask').sum(dim=0).tolist()

            # Create a dictionary for each node with feature names and scores
            node_data = {
                'NodeIndex': node_index,
                'Target': test_graph.y[node_index].item(),
            }
            node_data.update(zip(feature_names, feature_scores))

            results.append(node_data)

    # Create the DataFrame from the results list
    result_df = pd.DataFrame(results)

    return result_df

"""
import matplotlib.pyplot as plt

#result_df= results_df[results_df['Target'] == 1]
result_df = results_df.drop(columns=['Target'])

# Ensure 'NodeIndex' is in the DataFrame and set it as the index
if 'NodeIndex' in result_df.columns:
    result_df.set_index('NodeIndex', inplace=True)
# Calculate the mean score for each variable across all NodeIndex
mean_scores = result_df.mean()

# Sort the variables based on their mean scores in descending order
top_variables = mean_scores.sort_values(ascending=False).index[:50]


# Plot the top 10 variables in decreasing order with 'coolwarm' colormap
plt.figure(figsize=(10, 6))
plt.bar(top_variables, mean_scores[top_variables], color=plt.cm.coolwarm(mean_scores[top_variables]))
plt.xlabel('Variable')
plt.ylabel('Mean Score')
plt.title('Top Variables with the Highest Mean Scores')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

"""

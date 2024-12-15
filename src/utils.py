import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.impute import SimpleImputer
import time
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN, RandomOverSampler, BorderlineSMOTE, SMOTE, SMOTEN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import seaborn as sns
from torch_geometric.utils import to_dense_adj
from graph_representation import create_graph

def create_genotype_dataframe(data_path, file_id, raw=None, labels_data_path):
   
    #Create a dataset by merging multiple data frames based on 'SampleID' column.
    #Args:
    #    data_path (str): Path to the data files.
    #    imputed (bool): Flag indicating whether to process imputed data.
    #    raw (bool): Flag indicating whether to process raw data.
    #    file_id (str): Identifier for the specific files to be processed.
    #    labels_data (str): Path to the labels data file.
    #Returns:
    #    df (DataFrame): Merged data frame containing the combined dataset.
    
    if not raw:
        imputed_data_files = glob.glob(os.path.join(data_path, 'processed_imputed', file_id, '*.csv'))
        print('\033[91m[INFO]\033[0m Number of imputed files uploaded: {}'.format(len(imputed_data_files))) #red color info
 
        # Create an empty list to store the data frames
        dfs = []

        # Read each imputed data file into a data frame
        for df_path in tqdm(imputed_data_files):
            df = pd.read_csv(df_path)
            df.rename(columns={"Unnamed: 0": "SampleID"}, inplace=True)
            dfs.append(df)
        print('\033[92m[INFO]\033[0m Combining all the sub files into single dataframe...')
        # Merge the imputed data frames based on the 'SampleID' column
        combined_df = dfs[0]  # Initialize the combined dataframe with the first dataframe
        for df in dfs[1:]:
            combined_df = pd.merge(combined_df, df, on='SampleID', how='outer')

    if raw:
        raw_data_files = glob.glob(os.path.join(data_path, 'processed_raw', file_id, '*.csv'))
        print('\033[91m[INFO]\033[0m Number of raw files uploaded: {}'.format(len(raw_data_files)))

        # Create an empty list to store the data frames
        dfs = []

        # Read each raw data file into a data frame
        for df_path in tqdm(raw_data_files):
            df = pd.read_csv(df_path)
            df.rename(columns={"Unnamed: 0": "SampleID"}, inplace=True)
            dfs.append(df)

        print('\033[92m[INFO]\033[0m Combining all the sub files into single dataframe...')
        # Concatenate the raw data frames into a single data frame
        combined_df = pd.concat(dfs, sort=False)
        # Impute missing values using SimpleImputer
        imputer = SimpleImputer(strategy='median')  # You can change the strategy as needed
        combined_df.iloc[:, 1:] = imputer.fit_transform(combined_df.iloc[:, 1:])  # Assuming first column is 'SampleID'

        combined_df.drop_duplicates(inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
    
    
    print('\033[91m[INFO]\033[0m Loading labels dataframe....')
    labels_df = pd.read_csv(labels_data_path)
    labels_df = labels_df.rename(columns={'PatientID': 'SampleID'})

    # Merge the combined data frame with the labels data frame based on 'SampleID'
    df = combined_df.merge(labels_df, on='SampleID')
    print('\033[92m[INFO]\033[0m Genotype dataframe created successfully....')

    return combined_df



def variants_filter(gt_df, labels_path, gwas_threshold=5e-8, n_significant_variants=100):
    # Load the labels dataframe
    labels_df = pd.read_csv(labels_path)
    labels_df = labels_df.rename(columns={'PatientID': 'SampleID'})
    # Merge the genotype variants dataframe with labels data
    print('\033[92m[INFO]\033[0m Combining genotype dataframe with the labels data...')
    df = gt_df.merge(labels_df, on='SampleID')

    X = df.drop(columns=['SampleID', 'CaseControl']).values
    y = df['CaseControl'].values
    feature_names = df.drop(columns=['SampleID', 'CaseControl']).columns.tolist()

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    print('\033[91m[INFO]\033[0m Computing the chi-square to filter GWAS significant variants...')
    start_time = time.time()  # Record start time
    p_values = []
    for feature in X.T:
        contingency_table = pd.crosstab(feature, y)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        p_values.append(p)
    
    end_time = time.time()  # Record end time
    chi_square_computation_time = end_time - start_time
    print(f'\033[92m[INFO]\033[0m Chi-square computation done in {chi_square_computation_time:.2f} seconds.')

    negative_log10_p_values = [-np.log10(p) for p in p_values]
    # Filter variants based on the gwas_threshold
    significant_variants = [feature_names[i] for i, p_value in enumerate(negative_log10_p_values) if p_value <= -np.log10(gwas_threshold)]
    # Sort the filtered variants based on negative_log10_p_values in increasing order
    sorted_variants = sorted(significant_variants, key=lambda x: negative_log10_p_values[feature_names.index(x)])
    # Take the top n_significant_variants from the sorted list
    top_n_significant_variants = sorted_variants[:n_significant_variants]
    return top_n_significant_variants

def class_balancing(X, y, type, method, random_state=42):
    """
    Apply class imbalance handling techniques to balance the dataset.

    Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target variable.
        type (str): The class balance type ('over_sampling' or 'combined').
        method (str): The imbalance handling method to be applied.
        random_state (int, RandomState instance or None, optional): Controls the random seed for reproducibility.
                                                                   Defaults to 42.

    Returns:
        X_sampled (array-like): The balanced feature matrix.
        y_sampled (array-like): The balanced target variable.
    """

    if type == 'over_sampling':
        if method == 'ADASYN':
            oversampler = ADASYN(random_state=random_state)
        elif method == 'RandomOverSampler':
            oversampler = RandomOverSampler(random_state=random_state)
        elif method == 'BorderlineSMOTE':
            oversampler = BorderlineSMOTE(random_state=random_state)
        elif method == 'SMOTE':
            oversampler = SMOTE(random_state=random_state)
        elif method == 'SMOTEN':
            oversampler = SMOTEN(random_state=random_state)
        elif method == 'KMeansSMOTE':
            oversampler = KMeansSMOTE(random_state=random_state)
        elif method == 'SVMSMOTE':
            oversampler = SVMSMOTE(random_state=random_state)
        else:
            raise ValueError('Invalid over-sampling method specified.')

        X_sampled, y_sampled = oversampler.fit_resample(X, y)

    elif type == 'combined':
        if method == 'SMOTEENN':
            combiner = SMOTEENN(random_state=random_state)
        elif method == 'SMOTETomek':
            combiner = SMOTETomek(random_state=random_state)
        else:
            raise ValueError('Invalid combination method specified.')

        X_sampled, y_sampled = combiner.fit_resample(X, y)

    else:
        raise ValueError('Invalid type specified.')

    return X_sampled, y_sampled



from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

X = raw_df.drop(columns=['SampleID', 'CaseControl']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scaling the X values
y = raw_df['CaseControl'].values
feature_names = raw_df.columns[2:].tolist()

# Function to normalize feature importance
def normalize_importance(importance):
    return importance / np.sum(importance)

xgb_model = xgb.XGBClassifier().fit(X,y)
xgb_importance = normalize_importance(xgb_model.feature_importances_)

# Display feature importance for each model and aggregated evidence
feature_importances = pd.DataFrame({
    'Feature': feature_names,
    'XGBoost': xgb_importance,
})

# Selecting top features based on aggregated evidence
num_top_features = 200
top_features = list(feature_importances.nlargest(num_top_features, 'XGBoost')['Feature'])


def plot_roc_curves(model, model_path, test_graph, n_iterations, save_fig_path=None):
    # Lists to store ROC AUC values for each bootstrapped sample
    roc_auc_values = []

    # Lists to store ROC curves for each bootstrapped sample
    roc_curves = []

    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_out = model(test_graph)
    test_pred = test_out.max(1)[1]
    test_pred_probs = F.softmax(test_out, dim=1)[:, 1].detach().numpy()

    test_true = test_graph.y.cpu().numpy()

    for _ in range(n_iterations):
        # Generate a random sample with replacement (bootstrapping)
        sample_idx = np.random.choice(len(test_true), len(test_true), replace=True)
        sampled_y_true = test_true[sample_idx]
        sampled_y_pred_probs = test_pred_probs[sample_idx]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(sampled_y_true, sampled_y_pred_probs)

        # Calculate the area under the ROC curve (AUC)
        roc_auc = auc(fpr, tpr)

        # Append ROC AUC value to the list
        roc_auc_values.append(roc_auc)

        # Append ROC curve to the list
        roc_curves.append((fpr, tpr))

    # Calculate the mean and standard deviation of ROC AUC values from bootstrapping
    mean_roc_auc = np.mean(roc_auc_values)
    std_dev_roc_auc = np.std(roc_auc_values)

    label = f'Mean ROC AUC = {mean_roc_auc:.4f} ± {std_dev_roc_auc:.4f}'

    # Plot the ROC curve with confidence intervals
    plt.figure(figsize=(8, 6))
    for fpr, tpr in roc_curves:
        plt.plot(fpr, tpr, color='lightblue', lw=1, alpha=0.2)

    # Calculate and plot the mean ROC curve
    mean_fpr, mean_tpr, _ = roc_curve(test_true, test_pred_probs)
    plt.plot(mean_fpr, mean_tpr, color='darkblue', lw=2, label=label)

    # Plot confidence intervals as shaded regions
    for i in range(1, 4):  # Change the range as needed
        plt.fill_between(mean_fpr, mean_tpr - i * std_dev_roc_auc, mean_tpr + i * std_dev_roc_auc, color='lightblue', alpha=0.2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Bootstrapped Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if save_fig_path:
        plt.savefig(save_fig_path)

    plt.show()

# Usage example:
# plot_roc_curves(model, 'gcn_model.pkl', test_graph, n_iterations=100, save_fig_path='roc_curves.png')

def plot_mean_roc_curves(models, model_paths, test_graph, labels, n_iterations=10, save_fig_path=None):
    # Initialize colors for plotting
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Initialize a figure
    plt.figure(figsize=(8, 6))

    # Initialize a dictionary to store mean ROC AUC values for each model
    mean_roc_auc_values = {}
    std_roc_auc_values = {}

    # Loop through each model and plot its mean ROC curve
    for i, (model, model_path, label) in enumerate(zip(models, model_paths, labels)):
        roc_curves = []
        roc_auc_values = []
        # Load the model
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_out = model(test_graph)
        test_pred_probs = F.softmax(test_out, dim=1)[:, 1].detach().numpy()
        test_true = test_graph.y.cpu().numpy()

        for _ in range(n_iterations):
            # Generate a random sample with replacement (bootstrapping)
            sample_idx = np.random.choice(len(test_true), len(test_true), replace=True)
            sampled_y_true = test_true[sample_idx]
            sampled_y_pred_probs = test_pred_probs[sample_idx]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(sampled_y_true, sampled_y_pred_probs)

            # Calculate the area under the ROC curve (AUC)
            roc_auc = auc(fpr, tpr)

            # Append ROC AUC value to the list
            roc_auc_values.append(roc_auc)

            # Append ROC curve to the list
            roc_curves.append((fpr, tpr))

        # Calculate the mean and standard deviation of ROC AUC values from bootstrapping
        mean_roc_auc = np.mean(roc_auc_values)
        std_dev_roc_auc = np.std(roc_auc_values)

        # Store the mean ROC AUC value in the dictionary
        mean_roc_auc_values[label] = mean_roc_auc
        std_roc_auc_values[label]= std_dev_roc_auc

        # Create a label string with mean ROC AUC value
        label_with_mean_auc = f'{label} (Mean ROC AUC = {mean_roc_auc:.4f})'

        # Plot the ROC curve with confidence intervals (light gray)
        for fpr, tpr in roc_curves:
            plt.plot(fpr, tpr, color='darkgray', lw=1, alpha=0.2)

        # Calculate and plot the mean ROC curve (solid color)
        mean_fpr, mean_tpr, _ = roc_curve(test_true, test_pred_probs)
        plt.plot(mean_fpr, mean_tpr, lw=2, label=label_with_mean_auc, color=colors[i % len(colors)])

    # Plot the legend with mean ROC AUC values
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Bootstrapped Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Attach the mean ROC AUC values to the legend
    legend_text = [f'{label} ({mean_roc_auc_values[label]:.3f} ± {std_roc_auc_values[label]:.3f})' for label in labels]
    plt.legend(legend_text, loc='lower right')

    if save_fig_path:
        plt.savefig(save_fig_path)

    plt.show()




def patient_wise_similarity_matrix_visualization(X, y, metric, K, edge_weight, fig_save_path):
    # Create a graph (you should replace this with your actual 'create_graph' function)
    graph = create_graph(X, y, metric, K, edge_weight, 0.0)  # Adjust the parameters accordingly

    # Calculate the adjacency matrix
    adjacency_matrix = to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr, max_num_nodes=graph.num_nodes)[0]

    # Calculate the node similarity matrix
    node_similarity_matrix = torch.matmul(adjacency_matrix, adjacency_matrix.t())

    # Use 'y' values as cluster assignments
    cluster_assignments = y

    # Create a palette with red and green colors for the clusters
    cluster_palette = {0: 'red', 1: 'green'}  # Modify with your actual cluster labels and colors

    # Plot the clustermap with cluster assignments using fastcluster for clustering
    plt.figure(figsize=(8, 8))
    clustermap = sns.clustermap(node_similarity_matrix, cmap='coolwarm', row_cluster=True, col_cluster=True,
                                row_colors=[cluster_palette[i] for i in cluster_assignments],
                                col_colors=[cluster_palette[i] for i in cluster_assignments],
                                cbar_kws={'label': 'Distance'},
                                method='single')  # Use 'single' linkage method for fastcluster

    # Customize the font size and style for x and y labels
    for label in clustermap.ax_heatmap.get_xticklabels():
        label.set_fontsize(12)
        label.set_fontstyle('normal')
        label.set_weight('bold')
    for label in clustermap.ax_heatmap.get_yticklabels():
        label.set_fontsize(12)
        label.set_fontstyle('normal')
        label.set_weight('bold')

    # Customize the colorbar label using the cax parameter
    cbar = clustermap.ax_cbar.axes
    cbar.set_ylabel('Distance', fontsize=12, fontstyle='normal', fontweight='bold')

    # Customize the colorbar values (tick labels) to be bold
    for label in cbar.get_yticklabels():
        label.set_fontweight('bold')

    # Save the figure with maximum quality (high DPI) to the specified path
    #plt.savefig(fig_save_path, dpi=600, bbox_inches='tight')
    plt.show()

# Example usage:
# patient_wise_similarity_matrix_visualization(X, y, 'euclidean', 30, 0.5, 'path/to/save/clustermap_high_quality.png')


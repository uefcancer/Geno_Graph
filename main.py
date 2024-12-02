import os
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from utils import class_balancing
from sklearn.model_selection import train_test_split
from graph_representation import create_graph
from models import GCN_with_attention
from optuna_hyper_parameters import gcn_objective
import optuna
import torch
from learning import training_with_cv, evaluation_using_bootstrap

def main(cfg):
    # Load the genotype data
    print(f'\033[92m[INFO]\033[0m Loading the data...')
    data_path = os.path.join(cfg.data_path, cfg.type, f"{cfg.file_id}.pkl")
    df = pd.read_pickle(data_path)

    results_path = os.path.join(cfg.results_path, cfg.type, cfg.file_id)
    os.makedirs(results_path, exist_ok=True)

    # Load the features dataframe
    print(f'\033[92m[INFO]\033[0m Filtering the top K SNPs ...')
    features_path = os.path.join(r'results\top_features\biobank\genotype', cfg.type, cfg.file_id)
    file_path = os.path.join(features_path, 'top_2000_features.txt')
    top_features = [line.strip() for line in open(file_path, "r")][:cfg.top_k]
    #top_features = [line.strip() for line in open(r'results\top_features\biobank\genotype\raw\maf_0.01_hwe_0.0001_r2_0.5\top_100_features.txt', "r")][:cfg.top_k]


    # Filter the top K SNPs
    important_columns = ['SampleID', 'CaseControl'] + top_features
    df = df[important_columns]

    X = df.drop(columns=['SampleID', 'CaseControl']).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Scaling the X values

    y = df['CaseControl'].values

    print(f'\033[92m[INFO]\033[0m Handling imbalance data...')
    # Class balancing
    X, y = class_balancing(X, y, type=cfg.balance_type, method=cfg.balance_method, random_state=cfg.random_state)

    print(f'\033[92m[INFO]\033[0m Splitting the data into development and independent test set...')
    # Train test splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=cfg.random_state, stratify=y)

    print(f'\033[92m[INFO]\033[0m Creating the development and test graps...')
    # Creating training and test graphs
    train_graph = create_graph(X_train, y_train, cfg.distance_metric, cfg.K, cfg.mu, cfg.edge_threshold)
    test_graph = create_graph(X_test, y_test, cfg.distance_metric, cfg.K, cfg.mu, cfg.edge_threshold)

    print(f'\033[92m[INFO]\033[0m Finding best parameters using Optuna...')
    # Run Optuna to optimize hyperparameters
    train_idx, val_idx = train_test_split(torch.arange(len(train_graph.y)), test_size=0.3, stratify=train_graph.y)
    study = optuna.create_study(direction="maximize")  # Use maximize for accuracy optimization
    study.optimize(lambda trial: gcn_objective(trial, train_graph, train_idx, val_idx, cfg.model_type), n_trials=cfg.n_trails)  # 30 trials

    # Print the best hyperparameters found by Optuna
    print("Best hyperparameters: ", study.best_params)
    print("Best accuracy: ", study.best_value)

    #results_path = os.path.join(cfg.results_path, cfg.type, cfg.file_id)
    #os.makedirs(results_path, exist_ok=True)

    best_params = study.best_params
    #save_best_params = os.path.join(results_path, f"{cfg.feature_method}_{cfg.top_k}_gcn_with_attention_model_best_params.txt")
    save_best_params = os.path.join(results_path, f"{cfg.top_k}_gcn_att_model_best_params.txt")
    # Save the best hyperparameters to a text file
    with open(save_best_params, 'w') as f:
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    # Training GCN model with default parameters
    print(f'\033[92m[INFO]\033[0m Loading Graph convolution attention model with best parameters')
    model = GCN_with_attention(nfeat=train_graph.x.size(1),
                hidden_layers=[best_params['hidden_dim']],
                nclass=train_graph.y.max().item() + 1,
                dropout=best_params['dropout'],
                normalization=False,
                activation=best_params['activation'])

        
    #save_model = os.path.join(results_path, f"{cfg.feature_method}_{cfg.top_k}_gcn_with_attention_model.pkl")
    #save_train_results = os.path.join(results_path, f"{cfg.feature_method}_{cfg.top_k}_gcn_with_attention_train.csv")
    #save_test_results = os.path.join(results_path, f"{cfg.feature_method}_{cfg.top_k}_gcn_with_attention_test.csv")
    #save_final_results = os.path.join(results_path, f"{cfg.feature_method}_{cfg.top_k}_gcn_with_attention_final_results.txt")

    save_model = os.path.join(results_path, f"{cfg.top_k}_gcn_att_model.pkl")
    save_train_results = os.path.join(results_path, f"{cfg.top_k}_gcn_att_train.csv")
    save_test_results = os.path.join(results_path, f"{cfg.top_k}_gcn_att_test.csv")
    save_final_results = os.path.join(results_path, f"{cfg.top_k}_gcn_att_final_results.txt")


    print(f'\033[92m[INFO]\033[0m 5 Fold stratified cross validation training...')
    training_with_cv(train_graph=train_graph,
                                        model=model,
                                        optimizer_name=best_params['optimizer'],
                                        learning_rate=best_params['lr'],
                                        weight_decay=best_params['weight_decay'],
                                        criterion=best_params['loss_fn'],
                                        num_epochs=best_params['n_epochs'],
                                        n_splits=5,
                                        model_save_path=save_model,
                                        feature_selection_method=None,
                                        cv_results_save_path=save_train_results)

    print(f'\033[92m[INFO]\033[0m Performance evaluation on the independent test set...')
    evaluation_using_bootstrap(model=model,
                               model_path=save_model,
                               test_graph=test_graph,
                               n_iterations=cfg.n_iterations,
                               results_df_path=save_test_results,
                               test_results=save_final_results)

    print(f'\033[92m[INFO]\033[0m Done, all the results are saved sucessfully...')

if __name__ == '__main__':
    # Argument parser for GCN model configuration
    parser = argparse.ArgumentParser(description="Read arguments for the GCN with attention model.")

    # Dataset arguments
    parser.add_argument('--data_path', type=str, default="data/biobank/genotype/processed")
    parser.add_argument('--type', type=str, default='raw')
    parser.add_argument('--file_id', type=str, default='maf_0.01_hwe_1e-05_r2_0.8', help="File ID for dataset.")


    # Feature selection
    #parser.add_argument('--feature_selection_path', type=str, default="results/feature_selection/biobank/genotype")
    parser.add_argument('--top_k', type=int, default=200, choices=[100, 200, 500, 1000, 2000], help='Top K SNPs filtered using feature selection approaches')
    #parser.add_argument('--feature_method', type=str, default='chi_square', choices=['chi_square', 'annova', 'decision_tree', 'elastic_net', 'mutual_info', 'xgboost', 'random_forest', 'lasso', 'logistic_regression', 'ensemble_sum', 'ensemble_am', 'ensemble_gm', 'ensemble_hm'])

    # Utils
    parser.add_argument('--random_state', type=int, default=42)

    # Class balancing
    parser.add_argument('--class_balancing', default=True, help="Enable class balancing.")
    parser.add_argument('--balance_type', type=str, default='over_sampling', choices=['over_sampling'], help="Type of class balancing.")
    parser.add_argument('--balance_method', type=str, default='SMOTE', help="Method for class balancing.")

    # Graph representation
    parser.add_argument('--distance_metric', type=str, default='hamming')
    parser.add_argument('--K', type=int, default=30, help="Number of neighbors")
    parser.add_argument('--mu', type=float, default=0.5, help="Mu value")
    parser.add_argument('--edge_threshold', type=float, default=0.0, help="Edge weight threshold")

    # Hyper-parameters
    parser.add_argument('--model_type', type=str, default='gcn_with_attention')
    parser.add_argument('--n_trails', type=int, default=5)

    # Results saving paths
    parser.add_argument('--results_path', type=str, default='results/experiment_01/biobank/genotype')
    # Evaluation arguments
    parser.add_argument('--n_iterations', type=int, default=500, help="Number of iterations for evaluation")

    cfg = parser.parse_args()
    main(cfg)

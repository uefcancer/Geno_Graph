import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from metrics import compute_metrics
from models import GCN, GCN_with_attention, GraphSAGE_with_attention, GraphSAGE

ACTIVATIONS = {
    'relu': F.relu,
    'softplus': F.softplus,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'elu': F.elu,
    'leaky_relu': F.leaky_relu,
    'swish': lambda x: x * F.sigmoid(x),
    'gelu': F.gelu,
    'mish': lambda x: x * F.tanh(F.softplus(x)),
    'selu': F.selu,
    'prelu': nn.PReLU(),
    'celu': F.celu,
}

def gcn_objective(trial, train_graph, train_idx, val_idx, model_type):
    # Hyperparameters to be tuned:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "AdamW"])
    n_epochs = trial.suggest_int("n_epochs", 10, 200, step=10) 
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, log=True)
    activation_name = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))


    # Loss function selection
    loss_name = trial.suggest_categorical("loss_fn", ["nll_loss", "cross_entropy"])
    if loss_name == "nll_loss":
        loss_fn = F.nll_loss
        output_layer = nn.LogSoftmax(dim=1)  # Softmax for NLL Loss
    elif loss_name == "cross_entropy":
        loss_fn = F.cross_entropy
        output_layer = nn.Identity()  # Identity activation for Cross-Entropy Loss

    if model_type =='gcn':
        model = GCN(train_graph.x.size(1), [hidden_dim], train_graph.y.max().item() + 1, 
                dropout=dropout, 
                normalization=False, activation=activation_name)
    elif model_type =='gcn_with_attention':
        model = GCN_with_attention(train_graph.x.size(1), [hidden_dim], train_graph.y.max().item() + 1, 
                dropout=dropout, 
                normalization=False, activation=activation_name)


    # Add the output layer based on the selected loss function
    model.add_module("output", output_layer)

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Define early stopping parameters
    patience = 10
    min_delta = 0.001
    no_improve = 0
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph)
        train_loss = loss_fn(out[train_idx], train_graph.y[train_idx])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        # Evaluate validation predictions
        model.eval()
        val_out = model(train_graph)
        pred = val_out[val_idx].max(1)[1]
        pred_probs = F.softmax(val_out[val_idx], dim=1)[:, 1].detach().numpy()
        results = compute_metrics(train_graph.y[val_idx].cpu(), pred.cpu(), pred_probs)

        # Early stopping
        if results['auc_prc'] > best_accuracy + min_delta:
            best_accuracy = results['auc_prc']
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return results['auc_prc']

def multi_objective_gcn(trial, train_graph, train_idx, val_idx, model_type):
    # Hyperparameters to be tuned:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "AdamW"])
    n_epochs = trial.suggest_int("n_epochs", 10, 200, step=10) 
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, log=True)
    activation_name = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))

    # Loss function and model type
    loss_name = trial.suggest_categorical("loss_fn", ["nll_loss", "cross_entropy"])
    if loss_name == "nll_loss":
        loss_fn = F.nll_loss
        output_layer = nn.LogSoftmax(dim=1)
    else:
        loss_fn = F.cross_entropy
        output_layer = nn.Identity()

    # Model selection
    if model_type == 'gcn':
        model = GCN(train_graph.x.size(1), [hidden_dim], train_graph.y.max().item() + 1,
                    dropout=dropout, normalization=False, activation=activation_name)
    else:
        model = GCN_with_attention(train_graph.x.size(1), [hidden_dim], train_graph.y.max().item() + 1,
                                   dropout=dropout, normalization=False, activation=activation_name)

    model.add_module("output", output_layer)
    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop with early stopping
    best_f1 = 0.0
    best_auc_prc = 0.0
    patience = 10
    no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph)
        train_loss = loss_fn(out[train_idx], train_graph.y[train_idx])
        train_loss.backward()
        optimizer.step()

        model.eval()
        val_out = model(train_graph)
        pred = val_out[val_idx].max(1)[1]
        pred_probs = F.softmax(val_out[val_idx], dim=1)[:, 1].detach().numpy()
        results = compute_metrics(train_graph.y[val_idx].cpu(), pred.cpu(), pred_probs)

        # Update best metrics and check for improvement
        f1_score = results['f1_score']
        auc_prc = results['auc_prc']
        if f1_score > best_f1 or auc_prc > best_auc_prc:
            if f1_score > best_f1:
                best_f1 = f1_score
            if auc_prc > best_auc_prc:
                best_auc_prc = auc_prc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return best_f1, best_auc_prc

# Create a study object for multi-objective optimization
#study = optuna.create_study(directions=["maximize", "maximize"])
#study.optimize(lambda trial: multi_objective_gcn(trial, train_graph, train_idx, val_idx, model_type), n_trials=100)


def gsage_objective(trial, train_graph, train_idx, val_idx, model_type):
    # Hyperparameters to be tuned:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "AdamW"])
    n_epochs = trial.suggest_int("n_epochs", 10, 200, step=10) 
    dropout = trial.suggest_float("dropout", 0.0, 0.6)
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, log=True)
    activation_name = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))
    aggregator = trial.suggest_categorical("aggregator", ["mean", "sum", "max"])

    # Loss function selection
    loss_name = trial.suggest_categorical("loss_fn", ["nll_loss", "cross_entropy"])
    if loss_name == "nll_loss":
        loss_fn = F.nll_loss
        output_layer = nn.LogSoftmax(dim=1)  # Softmax for NLL Loss
    elif loss_name == "cross_entropy":
        loss_fn = F.cross_entropy
        output_layer = nn.Identity()  # Identity activation for Cross-Entropy Loss

    if model_type =='gsage':
        model =GraphSAGE(nfeat=train_graph.x.size(1),
                hidden_layers=[hidden_dim],
                nclass=train_graph.y.max().item() + 1,
                dropout=dropout,
                aggregator=aggregator,
                activation=activation_name)
        
    elif model_type =='gsage_with_attention':
        model =GraphSAGE_with_attention(nfeat=train_graph.x.size(1),
                hidden_layers=[hidden_dim],
                nclass=train_graph.y.max().item() + 1,
                dropout=dropout,
                aggregator=aggregator,
                activation=activation_name)


    # Add the output layer based on the selected loss function
    model.add_module("output", output_layer)

    optimizer_class = getattr(optim, optimizer_name)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)


    # Define early stopping parameters
    patience = 10
    min_delta = 0.001
    no_improve = 0
    best_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph)
        train_loss = loss_fn(out[train_idx], train_graph.y[train_idx])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()

        # Evaluate validation predictions
        model.eval()
        val_out = model(train_graph)
        pred = val_out[val_idx].max(1)[1]
        pred_probs = F.softmax(val_out[val_idx], dim=1)[:, 1].detach().numpy()
        results = compute_metrics(train_graph.y[val_idx].cpu(), pred.cpu(), pred_probs)

        # Early stopping
        if results['auc_prc'] > best_accuracy + min_delta:
            best_accuracy = results['auc_prc']
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return results['auc_prc']


"""
print(f'\033[92m[INFO]\033[0m Finding best parameters using Optuna...')
# Run Optuna to optimize hyperparameters
train_idx, val_idx = train_test_split(torch.arange(len(train_graph.y)), test_size=0.3, stratify=train_graph.y)
study = optuna.create_study(direction="maximize")  # Use maximize for accuracy optimization
study.optimize(lambda trial: gsage_objective(trial, train_graph, train_idx, val_idx, 'gsage_with_attention'), n_trials=50)  # 30 trials

# Print the best hyperparameters found by Optuna
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)
"""

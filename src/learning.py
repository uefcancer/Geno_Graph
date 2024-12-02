import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from metrics import compute_metrics, bootstrap_confidence_interval
import numpy as np
import pandas as pd
import os

def get_optimizer(optimizer_name, model, learning_rate, weight_decay):
    optimizers = {
        'Adam': optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'SGD': optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'RMSprop': optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9),
        'Adagrad': optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'Adadelta': optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'AdamW': optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Optimizer {optimizer_name} not supported. Choose from {list(optimizers.keys())}.")
    return optimizers[optimizer_name]

def training_with_cv(train_graph, model, optimizer_name, 
                     learning_rate, weight_decay, criterion='nll_loss', 
                     num_epochs=100, n_splits=5, model_save_path='model_paths/gcn',
                       cv_results_save_path='results.csv'):
    skf = StratifiedKFold(n_splits=n_splits)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(torch.arange(len(train_graph.y)), train_graph.y)):
        optimizer = get_optimizer(optimizer_name, model, learning_rate, weight_decay)
        best_auc = 0.0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_graph)
            loss_func = getattr(F, criterion if hasattr(F, criterion) else 'nll_loss')
            train_loss = loss_func(out[train_idx], train_graph.y[train_idx])
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_out = model(train_graph)
                    val_loss = loss_func(val_out[val_idx], train_graph.y[val_idx])
                    print(f"Fold {fold+1}, Epoch {epoch}, Train Loss: {train_loss.item()}, Val Loss: {val_loss.item()}")

        model.eval()
        with torch.no_grad():
            val_out = model(train_graph)
            pred = val_out[val_idx].max(1)[1]
            pred_probs = F.softmax(val_out[val_idx], dim=1)[:, 1]

        metrics = compute_metrics(train_graph.y[val_idx].cpu().numpy(), pred.cpu().numpy(), pred_probs.cpu().numpy())
        if metrics['auc_prc'] > best_auc:
            best_auc = metrics['auc_prc']
            torch.save(model.state_dict(), f"{model_save_path}_fold{fold+1}.pt")

        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(cv_results_save_path, index=False)
    return results_df

def training(train_graph, model, optimizer_name, learning_rate, weight_decay, criterion='nll_loss', num_epochs=100, model_save_path='model_paths/gcn_model.pkl'):
    train_idx, val_idx = train_test_split(np.arange(len(train_graph.y)), test_size=0.3, stratify=train_graph.y)
    optimizer = get_optimizer(optimizer_name, model, learning_rate, weight_decay)
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_graph)
        loss_func = getattr(F, criterion if hasattr(F, criterion) else 'nll_loss')
        train_loss = loss_func(out[train_idx], train_graph.y[train_idx])
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(train_graph)
                val_loss = loss_func(val_out[val_idx], train_graph.y[val_idx])
                print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Eval Loss: {val_loss.item()}")

    model.eval()
    with torch.no_grad():
        pred = val_out[val_idx].max(1)[1]
        pred_probs = F.softmax(val_out[val_idx], dim=1)[:, 1]
        metrics = compute_metrics(train_graph.y[val_idx].cpu().numpy(), pred.cpu().numpy(), pred_probs.cpu().numpy())

        if metrics['auc_prc'] > best_auc:
            best_auc = metrics['auc_prc']
            torch.save(model.state_dict(), model_save_path)

    return metrics

def evaluation_using_bootstrap(model, model_path, test_graph, n_iterations, results_df_path, test_results):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        test_out = model(test_graph)
        test_pred = test_out.max(1)[1]
        # Ensure the softmax operation is within the no_grad block and detached properly
        test_pred_probs = F.softmax(test_out, dim=1)[:, 1].detach().numpy()

    results_data = []

    for _ in range(n_iterations):
        sample_idx = np.random.choice(len(test_graph.y.cpu()), len(test_graph.y.cpu()), replace=True)
        sampled_y_true = test_graph.y.cpu().numpy()[sample_idx]
        sampled_y_pred = test_pred[sample_idx]
        sampled_y_pred_probs = test_pred_probs[sample_idx]

        metrics = compute_metrics(sampled_y_true, sampled_y_pred, sampled_y_pred_probs)
        results_data.append(metrics)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_df_path, index=False)

    test_metrics = compute_metrics(test_graph.y.cpu().numpy(), test_pred, test_pred_probs)
    metric_cis = {metric: bootstrap_confidence_interval(results_df[metric].values) for metric in test_metrics.keys()}

    with open(test_results, 'w') as file:
        for metric_name, metric_value in test_metrics.items():
            ci = metric_cis[metric_name]
            file.write(f"{metric_name.capitalize()}: {metric_value:.3f} (95% CI: {ci[0]:.3f}-{ci[1]:.3f})\n")

    return test_metrics, metric_cis

from sklearn.metrics import (
    precision_score, recall_score, accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)
import numpy as np

def compute_metrics(true_labels, pred_labels, pred_probs):
    # Calculate basic metrics
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    auc_prc = average_precision_score(true_labels, pred_probs)  # AUC-PRC
    auc_roc = roc_auc_score(true_labels, pred_probs)
    mcc = matthews_corrcoef(true_labels, pred_labels)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    
    # Calculate specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'auc_prc': auc_prc,  # Renamed from average_precision to auc_prc
        'auc_roc': auc_roc,
        'mcc': mcc,
        'kappa': kappa,
        'specificity': specificity
    }

# Example usage
# results = compute_metrics(true_labels, pred_labels, pred_probs)
# print(results)

def bootstrap_confidence_interval(metric_values, alpha=0.05):
    p_low = 100 * alpha / 2
    p_high = 100 * (1 - alpha / 2)
    lower = np.percentile(metric_values, p_low)
    upper = np.percentile(metric_values, p_high)
    return lower, upper

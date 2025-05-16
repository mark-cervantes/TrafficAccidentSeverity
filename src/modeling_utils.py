"""
modeling_utils.py

Utility functions for model evaluation and performance tracking.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from abc import ABC, abstractmethod


# def compute_classification_metrics(y_true, y_pred, y_prob=None):
#     """Compute common classification metrics."""
#     metrics = {
#         'Precision': precision_score(y_true, y_pred, zero_division=0),
#         'Recall': recall_score(y_true, y_pred, zero_division=0),
#         'F1': f1_score(y_true, y_pred, zero_division=0)
#     }
#     if y_prob is not None:
#         metrics['ROC_AUC'] = roc_auc_score(y_true, y_prob)
#     return metrics
"""
modeling_utils.py

Utility functions for model evaluation and performance tracking.
"""

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Compute common classification metrics with robust ROC AUC handling."""
    avg_param = 'weighted'
    metrics = compute_basic_metrics(y_true, y_pred, avg_param)

    if y_prob is not None:
        metrics['ROC_AUC'] = compute_roc_auc(y_true, y_prob, avg_param)
    return metrics


def compute_basic_metrics(y_true, y_pred, avg_param):
    """Compute basic classification metrics."""
    return {
        'Precision': precision_score(y_true, y_pred, average=avg_param, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=avg_param, zero_division=0),
        'F1': f1_score(y_true, y_pred, average=avg_param, zero_division=0)
    }


def compute_roc_auc(y_true, y_prob, avg_param):
    """Compute ROC AUC score with error handling."""
    unique_labels = pd.unique(y_true)
    is_multiclass = len(unique_labels) > 2
    roc_auc_value = float('nan')  # Default to NaN

    if is_multiclass:
        roc_auc_value = handle_multiclass_roc_auc(y_true, y_prob, unique_labels, avg_param)
    else:  # Binary classification
        try:
            roc_auc_value = roc_auc_score(y_true, y_prob)
        except ValueError as ve:
            print(f"Warning: ValueError computing binary ROC AUC: {ve}. Setting ROC_AUC to NaN.")
        except Exception as e:
            print(f"Warning: Unexpected error computing binary ROC AUC: {e}. Setting ROC_AUC to NaN.")
    return roc_auc_value


def handle_multiclass_roc_auc(y_true, y_prob, unique_labels, avg_param):
    """Handle ROC AUC computation for multiclass classification."""
    sorted_unique_labels = np.sort(unique_labels)
    if hasattr(y_prob, 'ndim') and y_prob.ndim == 2 and y_prob.shape[1] == len(sorted_unique_labels):
        try:
            return roc_auc_score(y_true, y_prob, multi_class='ovr', average=avg_param, labels=sorted_unique_labels)
        except ValueError as ve:
            print(f"Warning: ValueError computing multiclass ROC AUC (e.g., only one class in y_true, or labels mismatch): {ve}. Setting ROC_AUC to NaN.")
        except Exception as e:
            print(f"Warning: Unexpected error computing multiclass ROC AUC: {e}. Setting ROC_AUC to NaN.")
    else:
        print(f"Warning: ROC AUC for multiclass not computed. y_prob shape {getattr(y_prob, 'shape', 'N/A')} "
              f"is incompatible for {len(sorted_unique_labels)} classes (expected 2D array with {len(sorted_unique_labels)} columns). Setting ROC_AUC to NaN.")
    return float('nan')

class MetricStrategy(ABC):
    """Abstract base class for metric computation strategies."""
    @abstractmethod
    def compute(self, y_true, y_pred, y_prob=None):
        pass


class BasicMetricsStrategy(MetricStrategy):
    """Strategy for computing basic classification metrics."""
    def compute(self, y_true, y_pred, y_prob=None):
        avg_param = 'weighted'
        return compute_basic_metrics(y_true, y_pred, avg_param)


class RocAucStrategy(MetricStrategy):
    """Strategy for computing ROC AUC metrics."""
    def compute(self, y_true, y_pred, y_prob=None):
        avg_param = 'weighted'
        if y_prob is not None:
            return {'ROC_AUC': compute_roc_auc(y_true, y_prob, avg_param)}
        return {}


class CombinedMetricsStrategy(MetricStrategy):
    """Strategy for computing both basic and ROC AUC metrics."""
    def compute(self, y_true, y_pred, y_prob=None):
        avg_param = 'weighted'
        metrics = compute_basic_metrics(y_true, y_pred, avg_param)
        if y_prob is not None:
            metrics['ROC_AUC'] = compute_roc_auc(y_true, y_prob, avg_param)
        return metrics


def compute_classification_metrics(y_true, y_pred, y_prob=None, strategy=None):
    """Compute classification metrics using the specified strategy."""
    if strategy is None:
        strategy = CombinedMetricsStrategy()  # Default to combined metrics
    return strategy.compute(y_true, y_pred, y_prob)


def init_performance_excel(filepath: str):
    """Create an empty Excel sheet with appropriate columns for logging model performance."""
    columns = [
        'Model_Name', 'Timestamp', 'Hyperparameter_Set_Tried', 'CV_Score_for_Set',
        'Selected_Final_Hyperparameters', 'Training_Time_Seconds',
        'Train_Precision', 'Train_Recall', 'Train_F1', 'Train_ROC_AUC',
        'Test_Precision', 'Test_Recall', 'Test_F1', 'Test_ROC_AUC',
        'Class_Imbalance_Strategy', 'Notes'
    ]
    df = pd.DataFrame(columns=columns)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, index=False)


def append_performance_record(filepath: str, record: dict):
    """Append a single performance record (dict) to the Excel sheet."""
    df = pd.read_excel(filepath)
    # Add timestamp if not provided
    if 'Timestamp' not in record:
        record['Timestamp'] = datetime.utcnow().isoformat()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_excel(filepath, index=False)

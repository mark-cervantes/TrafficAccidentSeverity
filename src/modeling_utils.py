"""
modeling_utils.py

Utility functions for model evaluation and performance tracking.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """Compute common classification metrics."""
    metrics = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0)
    }
    if y_prob is not None:
        metrics['ROC_AUC'] = roc_auc_score(y_true, y_prob)
    return metrics


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

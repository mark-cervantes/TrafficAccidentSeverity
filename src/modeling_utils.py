"""
modeling_utils.py

Utility functions for model evaluation and performance tracking.
"""
import pandas as pd
import numpy as np
from pathlib import Path
# from datetime import datetime # Unused import
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from abc import ABC, abstractmethod
import joblib # Added for save/load model

# This is the primary function used by notebooks, leveraging the strategy pattern below.
def compute_classification_metrics(y_true, y_pred, y_prob=None, strategy=None):
    """Compute classification metrics using the specified strategy."""
    if strategy is None:
        strategy = CombinedMetricsStrategy()  # Default to combined metrics
    return strategy.compute(y_true, y_pred, y_prob)

# Helper functions and classes for the strategy-based metrics computation:
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


# AI-Programmer Guide:
# Use this function to log model performance metrics to an Excel file.
# - `filepath`: Path to the .xlsx file.
# - `record`: A dictionary where keys are column headers and values are the data for that row.
#             Include any model-specific parameters or metrics here.
#             If 'Experiment' key is omitted, it will be auto-incremented for the given sheet.
# - `sheet_name`: (Optional) Name of the sheet to append to (defaults to 'Performance').
#                 If the sheet doesn't exist, it's created with base columns:
#                 ['Experiment', 'Model', 'Runtime (s)', 'Train R²', 'Test R²', 'MSE', 'MAE'].
#                 The `record` can add more columns beyond these base ones.
# Example:
#   append_performance_record(
#       filepath='reports/model_performance.xlsx',
#       record={
#           'Model': 'MyAwesomeNet',
#           'Learning_Rate': 0.001,
#           'Epochs': 100,
#           'Train R²': 0.95,
#           'Test R²': 0.92,
#           'MSE': 15.3,
#           'MAE': 3.1,
#           'Custom_Metric': '0.88' # Any new column will be added
#       },
#       sheet_name='AwesomeNet_Experiments'
#   )
# This preserves other sheets and handles file/sheet creation automatically.
def append_performance_record(filepath: str, record: dict, sheet_name: str = 'Performance'):
    """
    Append a single performance record (dict) to a specified sheet in an Excel file.
    If the sheet or file doesn't exist, it's created with common columns.
    Allows for flexible addition of new columns from the record.
    """
    file_path_obj = Path(filepath)
    common_columns = ['Experiment', 'Model', 'Runtime (s)', 'Train R²', 'Test R²', 'MSE', 'MAE']
    
    all_sheets_data = {}
    
    # Ensure parent directory exists
    file_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if file_path_obj.exists():
        try:
            # Read all sheets from the existing Excel file
            all_sheets_data = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
        except FileNotFoundError:
            # This case should ideally be caught by file_path_obj.exists(), but as a fallback.
            print(f"Info: File {filepath} not found. A new file will be created.")
            all_sheets_data = {} # Start with no sheets
        except ValueError as ve: # Handles empty file or other read issues
             print(f"Warning: Could not properly read Excel file {filepath} (Error: {ve}). May overwrite or start fresh.")
             all_sheets_data = {}
        except Exception as e: # Catch other potential errors like bad zipfile for corrupted .xlsx
            print(f"Warning: Error reading Excel file {filepath} (Error: {e}). A new file or sheet might be created.")
            all_sheets_data = {}

    # Get the DataFrame for the target sheet, or create it if it doesn't exist
    if sheet_name in all_sheets_data:
        df_sheet = all_sheets_data[sheet_name]
        # Ensure df_sheet is a DataFrame (it could be None if sheet was empty and read_excel returned None)
        if df_sheet is None:
            df_sheet = pd.DataFrame(columns=common_columns)
        elif not isinstance(df_sheet, pd.DataFrame): # Should not happen with sheet_name=None
             df_sheet = pd.DataFrame(columns=common_columns)

    else:
        # Sheet doesn't exist, initialize with common columns
        df_sheet = pd.DataFrame(columns=common_columns)
        print(f"Info: Sheet '{sheet_name}' not found in {filepath}. Creating new sheet with common columns.")

    # Auto-assign 'Experiment' number if not provided in the record for this sheet
    if 'Experiment' not in record:
        current_max_experiment = 0
        if 'Experiment' in df_sheet.columns and not df_sheet.empty:
            # Attempt to convert 'Experiment' column to numeric, coercing errors to NaN, then drop NaNs and find max
            numeric_experiments = pd.to_numeric(df_sheet['Experiment'], errors='coerce').dropna()
            if not numeric_experiments.empty:
                current_max_experiment = numeric_experiments.max()
        record['Experiment'] = int(current_max_experiment) + 1

    # Convert the new record to a DataFrame
    # The columns of record_df will be derived from the keys in the 'record' dictionary
    record_df = pd.DataFrame([record])

    # Concatenate the existing sheet DataFrame with the new record DataFrame
    # This will add new columns if they exist in record_df but not in df_sheet
    # and fill NaNs appropriately.
    df_updated_sheet = pd.concat([df_sheet, record_df], ignore_index=True)

    # Update the dictionary of all sheets with the modified/new sheet
    all_sheets_data[sheet_name] = df_updated_sheet
    
    # Ensure 'Experiment' column is Int64Dtype (nullable integer) if it exists
    if 'Experiment' in all_sheets_data[sheet_name].columns:
        all_sheets_data[sheet_name]['Experiment'] = all_sheets_data[sheet_name]['Experiment'].astype(pd.Int64Dtype())


    # Write all sheets back to the Excel file
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for s_name, s_data in all_sheets_data.items():
                s_data.to_excel(writer, sheet_name=s_name, index=False)
    except Exception as e:
        print(f"Error: Could not write to Excel file {filepath}. Error: {e}")

def save_model(model, filepath: str, model_name: str = "Model"):
    """
    Save a trained model to a file using joblib.

    Args:
        model: The trained model object to save.
        filepath (str or Path): The path where the model will be saved.
        model_name (str): Name of the model, used for logging messages.
    """
    file_path_obj = Path(filepath)
    try:
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file_path_obj)
        print(f"{model_name} saved successfully to {file_path_obj}")
    except Exception as e:
        print(f"Error saving {model_name} to {file_path_obj}: {e}")

def load_model(filepath: str, model_name: str = "Model"):
    """
    Load a trained model from a file using joblib.

    Args:
        filepath (str or Path): The path from where the model will be loaded.
        model_name (str): Name of the model, used for logging messages.

    Returns:
        The loaded model object, or None if loading fails.
    """
    file_path_obj = Path(filepath)
    try:
        if not file_path_obj.exists():
            print(f"Error: Model file not found at {file_path_obj}")
            return None
        model = joblib.load(file_path_obj)
        print(f"{model_name} loaded successfully from {file_path_obj}")
        return model
    except Exception as e:
        print(f"Error loading {model_name} from {file_path_obj}: {e}")
        return None

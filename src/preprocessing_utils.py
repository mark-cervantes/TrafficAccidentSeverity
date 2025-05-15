"""
preprocessing_utils.py

Reusable preprocessing functions for the EDSA traffic accident forecasting project.
"""
import pandas as pd
import numpy as np
import os


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data into a DataFrame."""
    df = pd.read_csv(path)
    return df


def drop_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop casualty-related outcome columns to prevent data leakage."""
    outcome_cols = [
        'killed_driver', 'killed_passenger', 'killed_pedestrian',
        'injured_driver', 'injured_passenger', 'injured_pedestrian',
        'killed_uncategorized', 'injured_uncategorized',
        'killed_total', 'injured_total'
    ]
    return df.drop(columns=[c for c in outcome_cols if c in df.columns])


def drop_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier and redundant datetime columns."""
    drop_cols = ['INCIDENTDETAILS_ID', 'LOCATION_TEXT', 'ADDRESS', 'DATETIME_PST']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def parse_datetime(df: pd.DataFrame,
                   date_col: str = 'DATE_UTC',
                   time_col: str = 'TIME_UTC',
                   new_col: str = 'DATETIME_UTC') -> pd.DataFrame:
    """Combine date and time columns into a single datetime column."""
    df[new_col] = pd.to_datetime(
        df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce'
    )
    return df.drop(columns=[c for c in [date_col, time_col] if c in df.columns])


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day_of_week, day, month, year, is_weekend, and season from DATETIME_UTC."""
    df['hour'] = df['DATETIME_UTC'].dt.hour
    df['day_of_week'] = df['DATETIME_UTC'].dt.dayofweek
    df['day'] = df['DATETIME_UTC'].dt.day
    df['month'] = df['DATETIME_UTC'].dt.month
    df['year'] = df['DATETIME_UTC'].dt.year
    df['is_weekend'] = df['day_of_week'] >= 5

    def _get_season(m: int) -> str:
        if m in [12, 1, 2]: return 'Winter'
        if m in [3, 4, 5]: return 'Spring'
        if m in [6, 7, 8]: return 'Summer'
        return 'Fall'

    df['season'] = df['month'].apply(_get_season)
    return df


def impute_missing_categorical(df: pd.DataFrame,
                               cat_cols: list,
                               fill_value: str = 'Unknown') -> pd.DataFrame:
    """Impute missing values in categorical columns with specified fill_value."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    return df


def encode_categorical(df: pd.DataFrame,
                       cat_cols: list) -> pd.DataFrame:
    """One-Hot Encode specified categorical columns."""
    existing = [c for c in cat_cols if c in df.columns]
    return pd.get_dummies(df, columns=existing, prefix=existing, dummy_na=False)


def parse_desc_features(df: pd.DataFrame,
                        desc_col: str = 'DESC') -> pd.DataFrame:
    """Create basic text features from a description column and drop it."""
    if desc_col in df.columns:
        df['desc_word_count'] = df[desc_col].astype(str).str.split().apply(len)
        df['desc_contains_collision'] = (
            df[desc_col].str.contains('collision', case=False, na=False)
        ).astype(int)
        df = df.drop(columns=[desc_col])
    return df


def save_preprocessed(df: pd.DataFrame, output_path: str) -> None:
    """Ensure directory exists and save DataFrame as CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

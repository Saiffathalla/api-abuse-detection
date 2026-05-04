import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def encode_categoricals(df: pd.DataFrame, categorical_cols: list) -> tuple[pd.DataFrame, dict]:
    """Label-encode each categorical column; return df and encoder dict."""
    encoders = {}
    df = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def encode_target(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(series.astype(str))
    return y, le


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

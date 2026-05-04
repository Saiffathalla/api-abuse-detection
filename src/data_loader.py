import pandas as pd
import numpy as np


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"\nNumerical Summary:\n{df.describe()}")
    print(f"\nCategorical Counts:")
    for col in df.select_dtypes(include='object').columns:
        print(f"  {col}: {df[col].nunique()} unique — {df[col].value_counts().head(5).to_dict()}")

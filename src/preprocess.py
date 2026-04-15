import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def encode_labels(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def scale_features(df: pd.DataFrame, columns: list):
    """Return scaled DataFrame and the fitted scaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

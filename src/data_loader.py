import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_raw(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/raw/."""
    path = DATA_DIR / "raw" / filename
    return pd.read_csv(path)


def load_processed(filename: str) -> pd.DataFrame:
    """Load a CSV file from data/processed/."""
    path = DATA_DIR / "processed" / filename
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to data/processed/."""
    path = DATA_DIR / "processed" / filename
    df.to_csv(path, index=False)

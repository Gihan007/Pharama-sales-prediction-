import pandas as pd
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

def load_category_data(category, base_path=None):
    """Load CSV for a given drug category (C1-C8)."""
    data_dir = Path(base_path) if base_path else DATA_DIR
    file_path = data_dir / f"{category}.csv"
    df = pd.read_csv(file_path, parse_dates=["datum"], index_col="datum")
    return df

def load_all_categories(base_path=None):
    categories = [f"C{i}" for i in range(1, 9)]
    return {cat: load_category_data(cat, base_path) for cat in categories}

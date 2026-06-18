import numpy as np
import pandas as pd


def make_json_serializable(obj):
    """Convert NumPy/Pandas values into plain JSON-compatible Python values."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    if pd.isna(obj):
        return None
    return obj

import numpy as np
import pandas as pd

def convert_numpy_types(data: dict) -> dict:
    """Converts numpy types to native Python types before DB insert"""
    clean_data = {}
    for key, value in data.items():
        if isinstance(value, (np.int64, np.int32)):
            clean_data[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            clean_data[key] = float(value)
        else:
            clean_data[key] = value
    return clean_data


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        '1stFlrSF': 'FirstFlrSF',
        '2ndFlrSF': 'SecondFlrSF'
    }
    return df.rename(columns=rename_map)


def validate_input_features(df: pd.DataFrame, required_columns: list):
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    return df


def preprocess_input(data: dict, expected_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df = rename_columns(df)
    df = validate_input_features(df, expected_columns)
    df = df[expected_columns]
    return df

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.feature_engineering import run_feature_engineering


def pre_process(train_fe_path: str, test_fe_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs feature engineering pipeline on train and test data.
    Args:
        train_fe_path (str): Path to train fe
        test_fe_path (str):  Path to test fe
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of train and test data after feature engineering
    """
    train_fe_path = Path(train_fe_path)
    test_fe_path = Path(test_fe_path)

    if not train_fe_path.exists() or not test_fe_path.exists():
        print("Running feature engineering pipeline...")
        run_feature_engineering()
    else:
        print("FE files already exist. Skipping FE pipeline.")

    train_df_fe = pd.read_csv(train_fe_path)
    test_df_fe = pd.read_csv(test_fe_path)

    train_cols = set(train_df_fe.columns)
    test_cols = set(test_df_fe.columns)

    shared_features = [col for col in train_cols if col in test_cols]
    test_df_fe = test_df_fe[shared_features]

    if 'rating' not in train_df_fe.columns:
        raise ValueError("The train file is missing the 'rating' column.")
    train_df_fe = train_df_fe[shared_features]

    return train_df_fe, test_df_fe

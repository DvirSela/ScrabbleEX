from typing import List
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import SCRABBLE_LETTER_VALUES


def add_ohe_to_df(df: pd.DataFrame, cols_name: List[str], drop_original: bool = True) -> pd.DataFrame:
    """
    Adds one-hot encoded features for the specified list of columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        cols_name (List[str]): List of column names to one-hot encode.
        drop_original (bool): Whether to drop the original columns.

    Raises:
        ValueError: If any specified column is not found in the dataframe.

    Returns:
        pd.DataFrame: Dataframe with one-hot encoded features.
    """
    df = df.copy()
    missing_cols = [col for col in cols_name if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found in dataframe: {', '.join(missing_cols)}")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_features = ohe.fit_transform(df[cols_name])

    new_col_names = []
    for i, col in enumerate(cols_name):
        categories = ohe.categories_[i]
        new_col_names.extend([f'{col}_{cat}' for cat in categories])

    ohe_df = pd.DataFrame(
        ohe_features,
        columns=new_col_names,
        index=df.index
    )

    df = pd.concat([df, ohe_df], axis=1)
    if drop_original:
        df.drop(columns=cols_name, inplace=True)

    return df


def impute_missing_with_empty_string(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Imputes missing values in the specified column with an empty string.
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): List of column names to impute.
    Returns:
        pd.DataFrame: Dataframe with imputed values.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            imputer = SimpleImputer(strategy='constant', fill_value='')
            df[[col]] = imputer.fit_transform(df[[col]])
    return df


def merge_with_another_df(df: pd.DataFrame, to_merge_df: pd.DataFrame, on: str = None, how='left') -> pd.DataFrame:
    """
    Merges the input dataframe with another dataframe on the specified column.
    Args:
        df (pd.DataFrame): Input dataframe.
        to_merge_df (pd.DataFrame): Dataframe to merge with.
        on (str): Column name to merge on.
    Raises:
        ValueError: If 'on' parameter is not specified.
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    if on is None:
        raise ValueError("The 'on' parameter must be specified for merging.")
    df = df.copy()
    return df.merge(to_merge_df, on=on, how=how)


def calculate_rack_value(rack_str: str) -> int:
    """
    Calculates the total Scrabble point value of a player's rack.
    Args:
        rack_str (str): String representing the player's rack.
    Returns:
        int: Total point value of the rack.
    """
    return sum(SCRABBLE_LETTER_VALUES.get(letter.upper(), 0) for letter in rack_str)


def save_processed_file(df: pd.DataFrame, path: str):
    """
    Saves the processed dataframe to a CSV file, creating directories if they do not exist.
    Args:
        df (pd.DataFrame): Dataframe to save.
        path (str): Path to save the CSV file (including filename).
    Raises:
        OSError: If there is an error creating directories or saving the file.
    """
    try:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    except OSError as e:
        print(f"Error saving file to {path}: {e}")


def remove_casual_only_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows for players who have only played casual games.
    Assumes that the 'casual_only_player' column is already present in df.
    Args:
        df (pd.DataFrame): Input dataframe with 'casual_only_player' column.
    Raises:
        ValueError: If 'casual_only_player' column is not found in the dataframe.
    Returns:
        pd.DataFrame: Dataframe with casual-only players filtered out.
    """
    if 'casual_only_player' not in df.columns:
        raise ValueError(
            "The dataframe must contain a 'casual_only_player' column.")
    df = df.copy()
    initial_rows = df.shape[0]
    df = df[df['casual_only_player'] == 0].reset_index(drop=True)
    rows_removed = initial_rows - df.shape[0]
    print(f"Removed {rows_removed} rows belonging to casual-only players.")
    return df


def get_rack_left(rack_before: str, move: str) -> str:
    """
    Helper function to calculate the tiles left on the rack after a move.

    Assumes:
    - rack_before: All tiles on the rack (e.g., 'AEINRS?')
    - move: The move string (e.g., 'tA.kI?S')
    - Tiles from rack are UPPERCASE or '?'
    - Tiles from board are lowercase or '.'
    """
    try:
        rack_list = list(rack_before)
        tiles_played = [c for c in move if c.isupper() or c == '?']

        for tile in tiles_played:
            if tile in rack_list:
                rack_list.remove(tile)
            else:
                # This should'nt happen
                pass
        return "".join(rack_list)
    except Exception:
        return ""

from typing import List, Tuple
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from consts import DATA_DIR, BOTS


def add_ohe_to_df(df: pd.DataFrame, col_name: str, drop_original: bool = True) -> pd.DataFrame:
    """
    Adds one hot encoded features for the specified column.
    Args:
        df (pd.DataFrame): Input dataframe.
        col_name (str): Column name to one hot encode.
    Raises:
        ValueError: If the specified column is not found in the dataframe.
    Returns:
        pd.DataFrame: Dataframe with one hot encoded features.
    """
    df = df.copy()
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in dataframe.")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_features = ohe.fit_transform(df[[col_name]])
    ohe_df = pd.DataFrame(
        ohe_features,
        columns=[f'{col_name}_{cat}' for cat in ohe.categories_[0]],
        index=df.index
    )
    df = pd.concat([df, ohe_df], axis=1)
    if drop_original:
        df.drop(columns=[col_name], inplace=True)
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


def merge_with_games(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with games_df on 'game_id'.
    Args:
        df (pd.DataFrame): Input dataframe.
        games_df (pd.DataFrame): Games dataframe to merge with.
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    df = df.copy()
    return df.merge(games_df, on='game_id', how='left')


def add_casual_only_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary flag indicating whether a player has only played casual games.
    Args:
        df (pd.DataFrame): Input dataframe with 'nickname' and 'rating_mode' columns.
    Returns:
        pd.DataFrame: Dataframe with an additional 'casual_only_player' column.
    """
    df = df.copy()
    player_has_only_casual = (
        df.groupby('nickname')['rating_mode']
        .apply(lambda x: set(x) == {'CASUAL'})
    )
    df['casual_only_player'] = df['nickname'].map(
        player_has_only_casual).astype(int)
    return df


def add_board_letters_used(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a feature for each game and player indicating the total number of letters he used on the board (so not from his rack).
    These letters are calculated from the 'move' column in turns_df, and are indicated by lowercase letters and '.'s.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.
    Returns:
        pd.DataFrame: Dataframe with new column 'total_board_letters_used'.
    """
    df = df.copy()
    turns_df = turns_df.copy()
    turns_df['board_letters_used'] = turns_df['move'].apply(
        lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x)))
    player_board_letters = turns_df.groupby(['game_id', 'nickname'])[
        'board_letters_used'].sum().reset_index()
    player_board_letters.rename(
        columns={'board_letters_used': 'total_board_letters_used'}, inplace=True)
    df = df.merge(player_board_letters, on=['game_id', 'nickname'], how='left')
    return df


def add_points_per_game_stats(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features for player points per game (min, max, avg).
    Calculates statistics from the turns_df and merges them into the
    main dataframe based on 'game_id' and 'nickname'.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'points' columns.
    Returns:
        pd.DataFrame: Dataframe with new columns: 
                      'avg_points_per_game', 'min_points_per_game', 'max_points_per_game'.
    """
    df = df.copy()

    player_game_stats = turns_df.groupby(['game_id', 'nickname'])['points'].agg(
        avg_points_per_game='mean',
        min_points_per_game='min',
        max_points_per_game='max'
    ).reset_index()
    df = df.merge(player_game_stats, on=['game_id', 'nickname'], how='left')
    return df


def add_bot_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a one hot encoded feature for bot type.
    """
    df = df.copy()
    bot_rows = df[df['nickname'].isin(BOTS)]
    bot_map = bot_rows.set_index('game_id')['nickname']
    df.loc[:, 'opponent_bot'] = df['game_id'].map(bot_map)
    df = add_ohe_to_df(df, 'opponent_bot')
    return df


def add_time_related_features(df: pd.DataFrame, timestamp_col='created_at') -> pd.DataFrame:
    """
    Adds time-related features extracted from the timestamp_col column.
    Features added: 'hour_of_day', 'day_of_week', 'is_weekend', 'time_of_day'.
    Args:
        df (pd.DataFrame): Input dataframe with timestamp_col column.
        timestamp_col (str): Name of the timestamp column.
    Raises:
        ValueError: If the specified timestamp column is not found in the dataframe.
    Returns:
        pd.DataFrame: Dataframe with new time-related features.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in dataframe.")
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['day_of_week'] = df[timestamp_col].dt.day_name()
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['time_of_day'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=['early_morning', 'morning', 'afternoon', 'evening', 'night']
    )
    df = add_ohe_to_df(df, 'time_of_day')
    return df


def add_game_variant_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one hot encoded features for a new game_variant feature: f'{Lexicon} - {time_control_name}'
    Args:
        df (pd.DataFrame): Input dataframe with 'lexicon' and 'time_control_name' columns.
    Return:
        pd.DataFrame: Dataframe with new one hot encoded game_variant features.
    """
    df = df.copy()
    df['game_variant'] = df['lexicon'] + ' - ' + df['time_control_name']
    df = add_ohe_to_df(df, 'game_variant')
    return df


def build_feature_pipeline(games_df: pd.DataFrame, turns_df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Builds the feature engineering pipeline.
    Args:
        games_df (pd.DataFrame): Games dataframe.
        turns_df (pd.DataFrame): Turns dataframe.
    Returns:
        Tuple[Pipeline, pd.DataFrame]: Feature engineering pipeline and processed turns_df.
    """
    turns_df = impute_missing_with_empty_string(turns_df, ['rack', 'location'])

    pipeline = Pipeline([
        ('merge_games', FunctionTransformer(
            lambda df: merge_with_games(df, games_df))),
        ('add_casual_flag', FunctionTransformer(add_casual_only_flag)),
        ('add_avg_points', FunctionTransformer(
            lambda df: add_points_per_game_stats(df, turns_df))),
        ('add_board_letters', FunctionTransformer(
            lambda df: add_board_letters_used(df, turns_df))),
        ('add_bot_type', FunctionTransformer(add_bot_type)),
        ('add_time_features', FunctionTransformer(
            lambda df: add_time_related_features(df, 'created_at'))),
        ('add_game_variant', FunctionTransformer(add_game_variant_feature)),
    ])
    return pipeline, turns_df


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


def main():
    """
    Main function that runs the feature engineering pipeline on training and test datasets, and saves the results to CSV files.
    """
    games_df = pd.read_csv(DATA_DIR / 'raw' / 'games.csv')
    turns_df = pd.read_csv(DATA_DIR / 'raw' / 'turns.csv')
    train_df = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'raw' / 'test.csv')

    fe_pipeline, turns_df = build_feature_pipeline(games_df, turns_df)

    train_df_fe = fe_pipeline.fit_transform(train_df)
    test_df_fe = fe_pipeline.transform(test_df)
    print(f'{'-'*20} train_df_fe head {'-'*20}')
    print(train_df_fe.head())
    print(f'{'-'*20} test_df_fe head {'-'*20}')
    print(test_df_fe.head())

    save_processed_file(train_df_fe, DATA_DIR / 'processed' / 'train_fe.csv')
    save_processed_file(test_df_fe, DATA_DIR / 'processed' / 'test_fe.csv')


if __name__ == "__main__":
    main()

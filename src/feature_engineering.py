import time
from typing import List, Tuple
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from consts import DATA_DIR, BOTS


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
    This calculation is ONLY performed for turns where turn_type == 'Play'.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.
    Returns:
        pd.DataFrame: Dataframe with new column 'total_board_letters_used'.
    """
    df = df.copy()
    turns_df = turns_df.copy()

    mask = turns_df['turn_type'] == 'Play'
    turns_df['board_letters_used'] = 0
    turns_df.loc[mask, 'board_letters_used'] = turns_df.loc[mask, 'move'].apply(
        lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x))
    )
    player_board_letters = turns_df.groupby(['game_id', 'nickname'])[
        'board_letters_used'].sum().reset_index()
    player_board_letters.rename(
        columns={'board_letters_used': 'total_board_letters_used'}, inplace=True)
    df = df.merge(player_board_letters, on=['game_id', 'nickname'], how='left')

    df['total_board_letters_used'] = df['total_board_letters_used'].fillna(
        0).astype(int)
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
        max_points_per_game='max',
        median_points_per_game='median',
        std_points_per_game='std'
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
    df = add_ohe_to_df(df, ['opponent_bot'])
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
    df = add_ohe_to_df(df, ['time_of_day'])
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
    df = add_ohe_to_df(df, ['game_variant'])
    return df


def add_bingo_proxy_feature(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a feature for the number of bingos played by each player in each game.
    A bingo is defined as a move that uses all 7 letters from the player's rack, and is awarded a 50-point bonus.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.
    Returns:
        pd.DataFrame: Dataframe with new column 'number_of_bingos'.
    """
    df = df.copy()
    is_bingo_proxy = (turns_df['points'] > 50) & (
        turns_df['turn_type'] == 'Play')
    bingo_counts = (
        turns_df[is_bingo_proxy]
        .groupby(['game_id', 'nickname'], as_index=False)
        .size()
        .rename(columns={'size': 'bingo_count'})
    )
    df = df.merge(
        bingo_counts,
        on=['game_id', 'nickname'],
        how='left'
    )
    df['bingo_count'] = df['bingo_count'].fillna(0).astype(int)
    return df


def add_turn_type_counts(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features for counts of different turn types per player per game.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'turn_type' columns.
    Returns:
    df (pd.DataFrame): Dataframe with new columns for each turn type count.
    """
    df = df.copy()
    turn_type_counts = turns_df.pivot_table(
        index=['game_id', 'nickname'],
        columns='turn_type',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    turn_type_counts.columns.name = None
    turn_type_counts = turn_type_counts.add_prefix('turn_count_')
    turn_type_counts.rename(columns={
        'turn_count_game_id': 'game_id',
        'turn_count_nickname': 'nickname'
    }, inplace=True)
    df = df.merge(turn_type_counts, on=['game_id', 'nickname'], how='left')
    return df


def add_overall_player_stats(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds overall player statistics:
    - Total games played
    - min,max,std,avg points per all games played
    - min, max, avg 'max_overtime_minutes' per all games played
    Args:
        df (pd.DataFrame): Input dataframe with 'nickname' column.
        turns_df (pd.DataFrame): Turns dataframe with 'nickname' and 'points' columns.
    Returns:
        pd.DataFrame: Dataframe with new overall player statistics features.
    """
    df = df.copy()
    overall_stats = turns_df.groupby('nickname')['points'].agg(
        total_games_played='count',
        overall_avg_points='mean',
        overall_min_points='min',
        overall_max_points='max',
        overall_std_points='std'
    ).reset_index()
    overall_stats['total_games_played'] = turns_df.groupby(
        'nickname')['game_id'].nunique().values
    # 'max_overtime_minutes' is a column in df (from games_df)
    overtime_stats = df.groupby('nickname')['max_overtime_minutes'].agg(
        overall_avg_overtime='mean',
        overall_min_overtime='min',
        overall_max_overtime='max'
    ).reset_index()
    overall_stats = overall_stats.merge(
        overtime_stats, on='nickname', how='left')
    df = df.merge(overall_stats, on='nickname', how='left')
    return df


def calculate_past_window_features(df: pd.DataFrame, past_windows: List[int]) -> pd.DataFrame:
    """
    Calculates the player's history of games statistics over N past games.

    This function MUST be called after the following features have been added to df:
    - 'created_at' and 'winner' (from 'merge_with_games')
    - 'avg_points_per_game' (from 'add_points_per_game_stats')
    - 'bingo_count' (from 'add_bingo_proxy_feature')

    Args:
        df (pd.DataFrame): Input dataframe with per-game stats.
        past_windows (List[int]): List of past windows (e.g., [5, 10]).

    Returns:
        pd.DataFrame: Dataframe with new past-window features.
    """
    # This was written with the help of gemini-2.5
    df = df.copy()

    df['is_human'] = (~df['nickname'].isin(BOTS)).astype(int)
    human_win_condition = (df['is_human'] == 1) & (df['winner'] == 1)
    bot_win_condition = (df['is_human'] == 0) & (df['winner'] == 0)
    df['is_win'] = (human_win_condition | bot_win_condition).astype(int)
    df = df.drop(columns=['is_human'])

    df = df.sort_values(by=['nickname', 'created_at'])
    grouped = df.groupby('nickname', sort=False)

    df['past_avg_points'] = grouped['avg_points_per_game'].shift(1)
    df['past_bingo_count'] = grouped['bingo_count'].shift(1)
    df['past_is_win'] = grouped['is_win'].shift(1)

    for window in past_windows:
        # Create a new groupby object (on the sorted df)
        # to calculate rolling features on the 'past_' columns.
        grouped_past = df.groupby('nickname', sort=False)

        # Define the rolling window function. min_periods=1 ensures we get
        # stats even if the player has played fewer than 'window' games.
        def rolling_obj(col):
            return grouped_past[col].rolling(window=window, min_periods=1)

        # Points stats
        df[f'past_{window}_avg_points'] = rolling_obj(
            'past_avg_points').mean().reset_index(level=0, drop=True).fillna(0)
        df[f'past_{window}_min_points'] = rolling_obj(
            'past_avg_points').min().reset_index(level=0, drop=True).fillna(0)
        df[f'past_{window}_max_points'] = rolling_obj(
            'past_avg_points').max().reset_index(level=0, drop=True).fillna(0)

        # Count stats
        # .count() counts the non-NaN 'past_is_win' values in the window
        df[f'past_{window}_num_games'] = rolling_obj('past_is_win').count(
        ).reset_index(level=0, drop=True).fillna(0).astype(int)

        # .sum() adds up the bingos/wins in the window
        df[f'past_{window}_num_bingos'] = rolling_obj('past_bingo_count').sum(
        ).reset_index(level=0, drop=True).fillna(0).astype(int)
        df[f'past_{window}_num_wins'] = rolling_obj('past_is_win').sum(
        ).reset_index(level=0, drop=True).fillna(0).astype(int)

    df = df.drop(columns=['past_avg_points',
                 'past_bingo_count', 'past_is_win'])
    df = df.sort_index()

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
        ('add_number_of_bingos', FunctionTransformer(
            lambda df: add_bingo_proxy_feature(df, turns_df))),
        ('add_turn_type_counts', FunctionTransformer(
            lambda df: add_turn_type_counts(df, turns_df))),
        ('add_overall_player_stats', FunctionTransformer(
            lambda df: add_overall_player_stats(df, turns_df))),
        ('add_past_window_features', FunctionTransformer(
            lambda df: calculate_past_window_features(df, past_windows=[1, 5, 10, 20]))),
        ('add_rest_of_ohe', FunctionTransformer(
            lambda df: add_ohe_to_df(df, ['time_control_name', 'game_end_reason', 'lexicon', 'rating_mode', 'day_of_week']))),
        ('drop_unused_cols', FunctionTransformer(
            lambda df: df.drop(columns=['created_at', 'winner'])))
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
    start_time = time.time()
    games_df = pd.read_csv(DATA_DIR / 'raw' / 'games.csv')
    turns_df = pd.read_csv(DATA_DIR / 'raw' / 'turns.csv')
    train_df = pd.read_csv(DATA_DIR / 'raw' / 'train.csv')
    test_df = pd.read_csv(DATA_DIR / 'raw' / 'test.csv')

    fe_pipeline, turns_df = build_feature_pipeline(games_df, turns_df)

    train_df_fe = fe_pipeline.fit_transform(train_df)
    test_df_fe = fe_pipeline.transform(test_df)
    print(f'{'-'*20} train_df_shape: {train_df_fe.shape} {'-'*20}')
    print(f'{'-'*20} test_df_shape: {test_df_fe.shape} {'-'*20}')
    print(
        f"Feature engineering completed in {time.time() - start_time:.2f} seconds.")
    print(f'Saving processed files to {DATA_DIR / "processed"} directory.')
    save_processed_file(train_df_fe, DATA_DIR / 'processed' / 'train_fe.csv')
    save_processed_file(test_df_fe, DATA_DIR / 'processed' / 'test_fe.csv')


if __name__ == "__main__":
    main()

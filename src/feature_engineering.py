from typing import List, Tuple
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from consts import DATA_DIR, BOTS


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
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    bot_ohe = ohe.fit_transform(df[['opponent_bot']])
    bot_ohe_df = pd.DataFrame(
        bot_ohe,
        columns=[f'bot_type_{cat}' for cat in ohe.categories_[0]],
        index=df.index
    )
    df = pd.concat([df, bot_ohe_df], axis=1)
    df.drop(columns=['opponent_bot'], inplace=True)
    return df


def add_time_related_features(df: pd.DataFrame, timestamp_col='created_at') -> pd.DataFrame:
    """
    Adds time-related features extracted from the timestamp_col column.
    Features added: 'hour_of_day', 'day_of_week', 'is_weekend'.
    Args:
        df (pd.DataFrame): Input dataframe with timestamp_col column.
        timestamp_col (str): Name of the timestamp column.
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
    ])
    return pipeline, turns_df


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



if __name__ == "__main__":
    main()

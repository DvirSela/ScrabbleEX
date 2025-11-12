import time
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from textstat import textstat

from src.config import DATA_DIR, BOTS, PAST_WINDOWS_SIZES
from src.util import add_ohe_to_df, merge_with_another_df, impute_missing_with_empty_string, calculate_rack_value, save_processed_file


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
    Adds a feature for each game and player indicating the mean number of letters he used on the board (so not from his rack).
    These letters are calculated from the 'move' column in turns_df, and are indicated by lowercase letters and '.'s.
    This calculation is ONLY performed for turns where turn_type == 'Play'.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.
    Returns:
        pd.DataFrame: Dataframe with new column 'mean_board_letters_used'.
    """
    df = df.copy()
    turns_df = turns_df[['game_id', 'nickname', 'turn_type', 'move']].copy()

    mask = turns_df['turn_type'] == 'Play'
    turns_df['board_letters_used'] = 0
    turns_df.loc[mask, 'board_letters_used'] = turns_df.loc[mask, 'move'].apply(
        lambda x: str(x).count(".") + sum(int(c.islower()) for c in str(x))
    )
    player_board_letters = turns_df.groupby(['game_id', 'nickname'])[
        'board_letters_used'].mean().reset_index()
    player_board_letters.rename(
        columns={'board_letters_used': 'mean_board_letters_used'}, inplace=True)
    df = df.merge(player_board_letters, on=['game_id', 'nickname'], how='left')

    df['mean_board_letters_used'] = df['mean_board_letters_used'].fillna(
        0).astype(int)
    return df


def add_points_per_game_stats(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features for player points per game (min, max, avg), and non-bingo plays related stats.
    Calculates statistics from the turns_df and merges them into the
    main dataframe based on 'game_id' and 'nickname'.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'points' columns.
    Returns:
        pd.DataFrame: Dataframe with new columns: 
                      'avg_points_per_game', 'min_points_per_game', 'max_points_per_game',
                      'median_points_per_game', 'std_points_per_game', and
                      'average_non_bingo', 'median_non_bingo', 'std_non_bingo'.
    """
    df = df.copy()

    player_game_stats = turns_df.groupby(['game_id', 'nickname'])['points'].agg(
        avg_points_per_game='mean',
        min_points_per_game='min',
        max_points_per_game='max',
        median_points_per_game='median',
        std_points_per_game='std'
    ).reset_index()
    player_non_bingo_average = turns_df[(turns_df['turn_type'] == 'Play') & (turns_df['points'] < 50)]\
        .groupby(['game_id', 'nickname'])['points'].agg(
            average_non_bingo='mean',
            median_non_bingo='median',
            std_non_bingo='std'
    ).reset_index()
    df = df.merge(player_game_stats, on=['game_id', 'nickname'], how='left')
    df = df.merge(player_non_bingo_average, on=[
                  'game_id', 'nickname'], how='left')
    df['average_non_bingo'] = df['average_non_bingo'].fillna(0).astype(int)
    df['median_non_bingo'] = df['median_non_bingo'].fillna(0).astype(int)
    df['std_non_bingo'] = df['std_non_bingo'].fillna(0).astype(int)
    return df


def add_bot_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a one hot encoded feature for bot type.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
    Returns:
        pd.DataFrame: Dataframe with new one hot encoded bot type features and 'opponent_bot_strength' column.
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
    Features added: 'hour_of_day', 'day_of_week', 'is_weekend', 'time_of_day', 'month',
    and cyclical encodings for hour, day of week, and month.
    Args:
        df (pd.DataFrame): Input dataframe with timestamp_col column.
        timestamp_col (str): Name of the timestamp column.
    Returns:
        pd.DataFrame: Dataframe with new time-related features.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['day_of_week'] = df[timestamp_col].dt.day_name()
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['month'] = df[timestamp_col].dt.month
    df['time_of_day'] = pd.cut(
        df['hour_of_day'],
        bins=[-1, 5, 11, 17, 21, 24],
        labels=['early_morning', 'morning', 'afternoon', 'evening', 'night']
    )
    # Cyclical encodings, as suggested in here:
    # https://medium.com/@axelazara6/why-we-need-encoding-cyclical-features-79ecc3531232
    # formula is sin(2 * pi * x / max_x) and cos(2 * pi * x / max_x)
    df['_dow_num'] = df[timestamp_col].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['_dow_num'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['_dow_num'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df.drop(columns=['_dow_num'], inplace=True)
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
    Adds a feature for the ratio of turns that are bingos played by each player in each game.
    A bingo is defined as a move that uses all 7 letters from the player's rack, 
    and is awarded a 50-point bonus.

    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.

    Returns:
        pd.DataFrame: Dataframe with new column 'bingo_ratio'.
    """
    df = df.copy()

    total_turns = turns_df.groupby(
        ['game_id', 'nickname']).size().reset_index(name='total_turns')
    is_bingo_proxy = (turns_df['points'] > 50) & (
        turns_df['turn_type'] == 'Play')
    bingo_counts = (
        turns_df[is_bingo_proxy]
        .groupby(['game_id', 'nickname'], as_index=False)
        .size()
        .rename(columns={'size': 'bingo_count'})
    )
    df = df.merge(total_turns, on=['game_id', 'nickname'], how='left')

    df['total_turns'] = df['total_turns'].fillna(0)
    df = df.merge(
        bingo_counts,
        on=['game_id', 'nickname'],
        how='left'
    )
    df['bingo_count'] = df['bingo_count'].fillna(0).astype(int)
    df['bingo_ratio'] = np.where(
        df['total_turns'] > 0, df['bingo_count'] / df['total_turns'], 0)
    df = df.drop(columns=['total_turns', 'bingo_count'])

    return df


def add_turn_type_counts_ratios(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features for counts and ratios of different turn types per player per game.
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
    # now adding the ratios - dividing by total turns played by the player in that game
    turn_type_cols = [
        col for col in turn_type_counts.columns if col.startswith('turn_count_') and col not in ['game_id', 'nickname']
    ]
    turn_type_counts['total_turns'] = turn_type_counts[turn_type_cols].sum(
        axis=1)
    for col in turn_type_cols:
        ratio_col = col.replace('turn_count_', 'turn_ratio_')
        turn_type_counts[ratio_col] = turn_type_counts[col] / \
            turn_type_counts['total_turns']
    turn_type_counts.drop(columns=['total_turns'], inplace=True)
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
    - 'bingo_ratio' (from 'add_bingo_proxy_feature')

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
    df['past_bingo_ratio'] = grouped['bingo_ratio'].shift(1)
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

        df[f'past_{window}_avg_bingo_ratio'] = rolling_obj(
            'past_bingo_ratio').mean().reset_index(level=0, drop=True).fillna(0)
        df[f'past_{window}_num_wins'] = rolling_obj('past_is_win').sum(
        ).reset_index(level=0, drop=True).fillna(0).astype(int)

    df = df.drop(columns=['past_avg_points',
                 'past_bingo_ratio', 'past_is_win'])
    df = df.sort_index()

    return df


def went_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a feature indicating whether the player went first in the game.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
    Returns:
        pd.DataFrame: Dataframe with new column 'went_first'.
    """
    df = df.copy()
    df['went_first'] = (df['first'] == df['nickname']).astype(int)
    return df


def add_score_margin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds opponent_score, score_margin (player - opponent).
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'score' columns.
    Returns:
        pd.DataFrame: Dataframe with new columns 'opponent_score' and 'score_margin'.
    """
    df = df.copy()
    total_score = df.groupby('game_id')['score'].transform('sum')
    df['opponent_score'] = total_score - df['score']
    df['score_margin'] = df['score'] - df['opponent_score']
    return df


def add_pacing_features(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds pacing-related features:
    - turns_per_minute: Number of turns played per minute.
    - points_per_minute: Number of points scored per minute.
    - num_turns: Total number of turns played in the game.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id', 'nickname', and 'game_duration_seconds' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id' and 'nickname' columns.
    Returns:
        pd.DataFrame: Dataframe with new pace-related features.
    """
    df = df.copy()
    turns_per_player = (
        turns_df.groupby(['game_id', 'nickname'], as_index=False)
        .size()
        .rename(columns={'size': 'num_turns'})
    )
    df = df.merge(turns_per_player, on=['game_id', 'nickname'], how='left')
    df['num_turns'] = df['num_turns'].fillna(0).astype(int)

    minutes = (df['game_duration_seconds'] / 60).replace(0, np.nan)
    df['turns_per_minute'] = (df['num_turns'] / minutes).fillna(0.0)
    df['points_per_minute'] = (df['score'] / minutes).fillna(0.0)
    return df


def add_rack_value_features(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features related to the player's rack value and scoring efficiency.
    Features Added:
    - avg_rack_value: The player's average rack point value across all turns in the game.
    - rack_efficiency_avg_turn_points: Avg points per turn divided by avg_rack_value.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id', 'nickname',
                           and 'avg_points_per_game' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'rack' columns.

    Returns:
        pd.DataFrame: Dataframe with new rack value features.
    """
    df = df.copy()
    turns_df = turns_df[['game_id', 'nickname', 'rack']].copy()
    turns_df['vowel_count'] = turns_df['rack'].apply(
        lambda x: sum(1 for c in str(x).upper() if c in 'AEOUI'))

    turns_df['consonant_count'] = turns_df['rack'].apply(lambda x: sum(
        1 for c in str(x).upper() if c.isalpha() and c not in 'AEOUI'))
    turns_df['turn_rack_value'] = turns_df['rack'].apply(calculate_rack_value)

    player_game_rack_letter_stats = turns_df.groupby(['game_id', 'nickname']).agg(
        avg_vowel_count=('vowel_count', 'mean'),
        avg_consonant_count=('consonant_count', 'mean')
    ).reset_index()
    df = df.merge(player_game_rack_letter_stats, on=[
                  'game_id', 'nickname'], how='left')
    df['avg_vowel_count'] = df['avg_vowel_count'].fillna(0)
    df['avg_consonant_count'] = df['avg_consonant_count'].fillna(0)

    player_game_rack_stats = turns_df.groupby(['game_id', 'nickname'])['turn_rack_value'].agg(
        avg_rack_value='mean'
    ).reset_index()
    df = df.merge(player_game_rack_stats, on=[
                  'game_id', 'nickname'], how='left')
    df['avg_rack_value'] = df['avg_rack_value'].fillna(0)

    df['rack_efficiency_avg_turn_points'] = df['avg_points_per_game'] / \
        df['avg_rack_value']
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df['rack_efficiency_avg_turn_points'] = df['rack_efficiency_avg_turn_points'].fillna(
        0)

    return df


def add_lead_features(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds turn-by-turn score differential features.
    This tracks the lead/deficit for a player throughout the game.
    Features Added:
    - avg_lead: The player's average lead (or deficit) during the game.
    - std_lead: The volatility of the lead.
    - max_lead: The biggest lead the player ever had.
    - min_lead: The worst deficit (biggest negative lead) the player ever had.
    - final_lead: The player's lead at the end of their last turn.

    Args:
        df (pd.DataFrame): Input dataframe with 'game_id', 'nickname'.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'turn_number', 
                                 'nickname', and 'score' columns.

    Returns:
        pd.DataFrame: Dataframe with new lead-related features.
    """
    df = df.copy()
    turns_data = turns_df[['game_id',
                           'turn_number', 'nickname', 'score']].copy()
    turns_data = turns_data.sort_values(by=['game_id', 'turn_number'])

    turns_data['opponent_score_at_turn'] = turns_data.groupby(
        'game_id')['score'].shift(1).fillna(0)

    turns_data['turn_lead'] = turns_data['score'] - \
        turns_data['opponent_score_at_turn']

    lead_stats = turns_data.groupby(['game_id', 'nickname'])['turn_lead'].agg(
        avg_lead='mean',
        std_lead='std',
        max_lead='max',
        min_lead='min',
        final_lead='last'
    ).reset_index()

    df = df.merge(lead_stats, on=['game_id', 'nickname'], how='left')

    fill_cols = ['avg_lead', 'std_lead', 'max_lead', 'min_lead', 'final_lead']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def add_textstat_features(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds textstat features based on the 'move' column in turns_df.
    Features Added:
    - average_difficult_words: Average number of difficult words used by the player in the game.
    Args:
        df (pd.DataFrame): Input dataframe with 'game_id' and 'nickname' columns.
        turns_df (pd.DataFrame): Turns dataframe with 'game_id', 'nickname', and 'move' columns.
    Returns:
        pd.DataFrame: Dataframe with new textstat features.
    """
    df = df.copy()
    turns_df = turns_df[['game_id', 'nickname', 'turn_type', 'move']].copy()

    play_turns = turns_df[turns_df['turn_type'] == 'Play'].copy()
    play_turns['num_difficult_words'] = play_turns['move'].apply(
        lambda x: textstat.difficult_words(
            str(x).replace('.', '').replace('?', ' '))
    )
    difficult_words_stats = play_turns.groupby(['game_id', 'nickname'])['num_difficult_words'].agg(
        sum_difficult_words='sum'
    ).reset_index()
    df = df.merge(difficult_words_stats, on=[
                  'game_id', 'nickname'], how='left')
    df['sum_difficult_words'] = df['sum_difficult_words'].fillna(0)
    return df


def add_count_turn_types(df: pd.DataFrame, turns_df: pd.DataFrame) -> pd.DataFrame:
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
            lambda df: merge_with_another_df(df, games_df, on='game_id'))),
        ('add_casual_flag', FunctionTransformer(add_casual_only_flag)),
        ('add_went_first', FunctionTransformer(went_first)),
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
            lambda df: add_turn_type_counts_ratios(df, turns_df))),
        ('add_overall_player_stats', FunctionTransformer(
            lambda df: add_overall_player_stats(df, turns_df))),
        ('add_past_window_features', FunctionTransformer(
            lambda df: calculate_past_window_features(df, past_windows=PAST_WINDOWS_SIZES))),
        ('add_rest_of_ohe', FunctionTransformer(
            lambda df: add_ohe_to_df(df, ['time_control_name', 'game_end_reason', 'rating_mode', 'day_of_week']))),
        ('drop_unused_cols', FunctionTransformer(
            lambda df: df.drop(columns=['created_at', 'winner']))),
        ('add_score_margin', FunctionTransformer(add_score_margin_features)),
        ('add_pacing_features', FunctionTransformer(
            lambda df: add_pacing_features(df, turns_df))),
        ('add_rack_value_features', FunctionTransformer(
            lambda df: add_rack_value_features(df, turns_df))),
        ('add_lead_features', FunctionTransformer(
            lambda df: add_lead_features(df, turns_df))),
        ('add_textstat_features', FunctionTransformer(
            lambda df: add_textstat_features(df, turns_df))),
        # ('add_count_turn_types', FunctionTransformer(
        #     lambda df: add_count_turn_types(df, turns_df))),
    ])
    return pipeline, turns_df


def run_feature_engineering():
    """
    Main function to run the feature engineering pipeline.
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

    train_no_bots = train_df_fe[~train_df_fe['nickname'].isin(BOTS)]
    test_no_bots = test_df_fe[~test_df_fe['nickname'].isin(BOTS)]
    save_processed_file(train_no_bots, DATA_DIR /
                        'processed' / 'train_fe_no_bots.csv')
    save_processed_file(test_no_bots, DATA_DIR /
                        'processed' / 'test_fe_no_bots.csv')

import time
from pathlib import Path
from typing import Tuple, Dict, Any, List
from functools import partial
import pickle
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .config import DATA_DIR, SEED, N_TRIALS, SUBMISSION_DIR
from .feature_engineering import run_feature_engineering
from .util import remove_casual_only_players


optuna.logging.set_verbosity(optuna.logging.WARNING)


def pre_process(train_fe_path: Path, test_fe_path: Path, force_run_fe: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs feature engineering pipeline on train and test data.
    Args:
        train_fe_path (str): Path to train fe
        test_fe_path (str):  Path to test fe
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of train and test data after feature engineering
    """
    if force_run_fe or not train_fe_path.exists() or not test_fe_path.exists():
        print(
            f"Running feature engineering pipeline{' (forced)' if force_run_fe else ''}...")
        run_feature_engineering()
    else:
        print("FE files already exist. Skipping FE pipeline.")

    train_df_fe = pd.read_csv(train_fe_path)
    test_df_fe = pd.read_csv(test_fe_path)

    if 'rating' not in train_df_fe.columns:
        raise ValueError("The train file is missing the 'rating' column.")

    train_cols = set(train_df_fe.drop(columns=['rating']).columns)
    test_cols = set(test_df_fe.columns)

    shared_features = list(train_cols.intersection(test_cols))

    test_df_fe = test_df_fe[shared_features]
    train_df_fe = train_df_fe[shared_features + ['rating']]

    return train_df_fe, test_df_fe


def clean_data(train_df: pd.DataFrame, true_test_df: pd.DataFrame, cols_to_drop: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Drops unnecessary columns and prepares X, y, and test sets.
    Args:
        train_df (pd.DataFrame): The training dataframe
        test_df (pd.DataFrame): The test dataframe
        cols_to_drop (List[str]): List of columns to drop from both dataframes
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]: train_x, train_y, test_df_processed, test_game_ids
    """
    for col in cols_to_drop:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])

    test_game_ids = true_test_df['game_id']

    if 'game_id' in train_df.columns:
        train_df = train_df.drop(columns=['game_id'])

    train_y = train_df['rating']
    train_x = train_df.drop(columns=['rating'])
    return train_x, train_y, true_test_df, test_game_ids


def create_pipeline(model: Any, needs_scaling: bool) -> Pipeline:
    """
    Creates an sklearn Pipeline, adding a StandardScaler if specified.
    Args:
        model (Any): The model to be used in the pipeline
        needs_scaling (bool): Whether to include StandardScaler in the pipeline
    """
    if needs_scaling:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    return Pipeline([('model', model)])


def get_model_and_params(model_name: str, trial: optuna.trial.Trial) -> Any:
    """
    Suggests hyperparameters for a given model name using an Optuna trial.
    Args:
        model_name (str): Name of the model
        trial (optuna.trial.Trial): The Optuna trial object
    Returns:
        Any: An instance of the model with suggested hyperparameters
    """
    if model_name == 'Ridge':
        alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)
        return Ridge(alpha=alpha, random_state=SEED)

    if model_name == 'KNeighborsRegressor':
        n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 2)  # 1 for Manhattan, 2 for Euclidean
        return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p, n_jobs=-1)

    if model_name == 'XGBRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        return XGBRegressor(**params, random_state=SEED, n_jobs=-1)

    if model_name == 'LGBMRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        }
        return LGBMRegressor(**params, random_state=SEED, verbose=-1, log_level='silent', n_jobs=-1)

    if model_name == 'CatBoostRegressor':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
        return CatBoostRegressor(**params, random_state=SEED, verbose=0)
    if model_name == 'RandomForestRegressor':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 100),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
        }
        return RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(f"Model {model_name} not recognized for tuning.")


def objective(trial: optuna.trial.Trial, model_name: str, needs_scaling: bool, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    The Optuna objective function.
    Trains a model with suggested params and returns the validation RMSE.
    Args:
        trial (optuna.trial.Trial): The Optuna trial object
        model_name (str): Name of the model to tune
        needs_scaling (bool): Whether the model needs feature scaling
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
    Returns:
        float: The validation RMSE
    """
    model = get_model_and_params(model_name, trial)
    pipeline = create_pipeline(model, needs_scaling)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse


def plot_model_results(ax: plt.Axes, y_test: pd.Series, y_pred: np.ndarray, model_name: str, metrics: Dict[str, float]):
    """
    Plots prediction and true value histograms on a given matplotlib axis.
    Args:
        ax (plt.Axes): The matplotlib axis to plot on
        y_test (pd.Series): True target values
        y_pred (np.ndarray): Predicted target values
        model_name (str): Name of the model for the title
        metrics (Dict[str, float]): Evaluation metrics to display on the plot
    """
    ax.hist(y_test, bins=50, alpha=0.7, label='True Values (y_test)',
            color='blue', edgecolor='black')
    ax.hist(y_pred, bins=50, alpha=0.7, label='Predictions (y_pred)',
            color='orange', edgecolor='black')
    ax.set_title(f"{model_name} (Tuned) Predictions vs. True Values")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    ax.legend()
    text_str = (
        f"RMSE: {metrics['RMSE']:.2f}\n"
        f"MAE: {metrics['MAE']:.2f}\n"
    )
    ax.text(
        0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )


def generate_submission_file(pipeline: Pipeline, true_test_df: pd.DataFrame, test_game_ids: pd.Series, model_name: str, output_dir: Path):
    """
    Generates and saves a submission CSV file for a given model.
    Args:
        pipeline (Pipeline): The trained sklearn Pipeline
        test_df (pd.DataFrame): The test features dataframe
        test_game_ids (pd.Series): Series of game IDs for the test set
        model_name (str): Name of the model for the filename
        output_dir (Path): Directory to save the submission file
    """
    print(f"Generating submission file for {model_name}...")
    test_predictions = pipeline.predict(
        true_test_df.drop(columns=['first', 'game_id', 'nickname']))
    submission_df = pd.DataFrame(
        {'game_id': test_game_ids, 'rating': test_predictions})

    # This is temporary - will be moved to function
    submission_df = submission_df.merge(
        true_test_df[['game_id', 'nickname', 'casual_only_player']], on='game_id')
    casual_player_means = submission_df[submission_df['casual_only_player'] == 1].groupby(
        'nickname')['rating'].transform('mean')
    submission_df.loc[submission_df['casual_only_player']
                      == 1, 'rating'] = casual_player_means
    submission_df = submission_df[['game_id', 'rating']]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'submission_{model_name}_tuned.csv'
    submission_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved {model_name} predictions to {output_path}")


def run_hyperparameter_tuning(models_to_run: List[str], models_that_need_scaling: List[str], X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame, n_trials: int = N_TRIALS) -> Dict[str, optuna.study.Study]:
    """
    Runs hyperparameter tuning for multiple models using Optuna.
    Args:
        models_to_run (List[str]): List of model names to tune
        models_that_need_scaling (List[str]): List of model names that require feature scaling
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        n_trials (int): Number of trials for each model
    Returns:
        Dict[str, optuna.study.Study]: A dictionary mapping model names to their respective Optuna studies
    """
    studies = {}
    for model_name in models_to_run:
        print(f"\n--- Tuning {model_name} ---")
        needs_scaling = model_name in models_that_need_scaling

        study = optuna.create_study(direction='minimize')
        obj_func = partial(
            objective,
            model_name=model_name,
            needs_scaling=needs_scaling,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        study.optimize(obj_func, n_trials=n_trials, show_progress_bar=True)

        print(
            f"Finished tuning {model_name}. Best trial RMSE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        studies[model_name] = study
    with open(DATA_DIR / 'models' / f'tuned_studies{n_trials}_trials.pkl', 'wb') as f:
        pickle.dump(studies, f)
    return studies


def evaluate_and_generate_submissions(
        studies: Dict[str, optuna.study.Study],
        models_that_need_scaling: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        true_test_df: pd.DataFrame,
        test_game_ids: pd.Series,
        all_train_x: pd.DataFrame,
        all_train_y: pd.Series
) -> List[Dict[str, Any]]:
    """
    Evaluates the best models from the studies on the validation set and generates submission files.
    Args:
        studies (Dict[str, optuna.study.Study]): A dictionary mapping model names to their respective Optuna studies
        models_that_need_scaling (List[str]): List of model names that require feature scaling
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        true_test_df (pd.DataFrame): The test features dataframe
        test_game_ids (pd.Series): Series of game IDs for the test set
        all_train_x (pd.DataFrame): All training features (train + val)
        all_train_y (pd.Series): All training target (train + val)
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each model
    """
    results = []
    _, axes = plt.subplots(3, 2, figsize=(18, 15))
    flat_axes = axes.flatten()

    for i, (model_name, study) in enumerate(studies.items()):
        print(f"\n--- Evaluating {model_name} ---")
        needs_scaling = model_name in models_that_need_scaling

        start_time = time.time()
        best_model_for_eval = get_model_and_params(
            model_name, study.best_trial)
        best_pipeline_for_eval = create_pipeline(
            best_model_for_eval, needs_scaling)

        best_pipeline_for_eval.fit(X_train, y_train)

        duration = time.time() - start_time
        y_pred_val = best_pipeline_for_eval.predict(X_val)
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_val, y_pred_val)),
            'MAE': mean_absolute_error(y_val, y_pred_val),
        }

        results.append({
            'Model': model_name,
            'Val RMSE': metrics['RMSE'],
            'Val MAE': metrics['MAE'],
            'Fit Time (s)': duration
        })
        print(
            f"Finished {model_name} in {duration:.2f}s. Validation RMSE: {metrics['RMSE']:.4f}")

        plot_model_results(flat_axes[i], y_val,
                           y_pred_val, model_name, metrics)

        print(f"Re-training {model_name} on full dataset for submission...")
        start_time_full = time.time()

        final_model = get_model_and_params(model_name, study.best_trial)
        final_pipeline = create_pipeline(final_model, needs_scaling)
        final_pipeline.fit(all_train_x, all_train_y)

        full_train_duration = time.time() - start_time_full
        print(f"Full re-train finished in {full_train_duration:.2f}s.")

        generate_submission_file(
            final_pipeline, true_test_df, test_game_ids, model_name, SUBMISSION_DIR)

    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].axis('off')

    return results


def evaluate_and_save_results(
        studies: Dict[str, optuna.study.Study],
        models_that_need_scaling: List[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
        true_test_df: pd.DataFrame,
        test_game_ids: pd.Series,
        all_train_x: pd.DataFrame,
        all_train_y: pd.Series,
        n_trials: int = N_TRIALS
):
    """
    Evaluates the best models from the studies on the validation set and generates submission files.
    Args:
        studies (Dict[str, optuna.study.Study]): A dictionary mapping model names to their respective Optuna studies
        models_that_need_scaling (List[str]): List of model names that require feature scaling
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        true_test_df (pd.DataFrame): The test features dataframe
        test_game_ids (pd.Series): Series of game IDs for the test set
        all_train_x (pd.DataFrame): All training features (train + val)
        all_train_y (pd.Series): All training target (train + val)
        n_trials (int): Number of trials used in tuning (for filename purposes)
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing evaluation results for each model
    """
    results = evaluate_and_generate_submissions(
        studies, models_that_need_scaling,
        X_train, y_train, X_val, y_val,
        true_test_df, test_game_ids,
        all_train_x, all_train_y
    )
    results_df = pd.DataFrame(results).sort_values(
        by='Val RMSE', ascending=True)

    print("\n--- Model Comparison Results (Tuned) ---")
    print(results_df.to_string(index=False, float_format="%.4f"))
    results_df.to_csv(
        DATA_DIR / f'model_tuning_results_{n_trials}_trials.csv', index=False, float_format="%.4f")
    print("\nDisplaying prediction histograms...")
    plt.tight_layout()
    plt.savefig(DATA_DIR / 'model_tuning_results.png')
    plt.show()


def main():
    """
    Main function to run the full tuning, training, and evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run model tuning and evaluation pipeline.")
    parser.add_argument('--n_trials', type=int, default=N_TRIALS,
                        help='Number of trials for hyperparameter tuning.')
    parser.add_argument('--load_studies', action='store_true',
                        help='Load existing tuned studies from disk.')
    parser.add_argument('--run_fe', action='store_true',
                        help='Run feature engineering pipeline even if files exist.')
    args = parser.parse_args()

    print("Loading and preprocessing data...")
    train_fe_path = DATA_DIR / 'processed' / 'train_fe_no_bots.csv'
    test_fe_path = DATA_DIR / 'processed' / 'test_fe_no_bots.csv'
    train_df, true_test_df = pre_process(
        train_fe_path, test_fe_path, force_run_fe=args.run_fe)

    train_df = remove_casual_only_players(train_df)

    train_groups = train_df['nickname']
    cols_to_drop = ['nickname', 'first']
    train_x, train_y, true_test_df, test_game_ids = clean_data(
        train_df, true_test_df, cols_to_drop)

    print(f"Splitting data into train/val using GroupShuffleSplit...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    train_idx, val_idx = next(gss.split(train_x, train_y, groups=train_groups))

    X_train, X_val = train_x.iloc[train_idx], train_x.iloc[val_idx]
    y_train, y_val = train_y.iloc[train_idx], train_y.iloc[val_idx]
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    models_to_run = ['Ridge', 'XGBRegressor',
                     'LGBMRegressor', 'CatBoostRegressor']
    models_that_need_scaling = ['Ridge', 'KNeighborsRegressor']
    if not args.load_studies:
        print(
            f"\n--- Starting Model Tuning (Running {args.n_trials} Trials) ---")

        studies = run_hyperparameter_tuning(
            models_to_run, models_that_need_scaling, X_train, y_train, X_val, y_val, n_trials=args.n_trials)
    else:
        print("\n--- Loading Existing Tuned Studies ---")
        with open(DATA_DIR / 'models' / f'tuned_studies{args.n_trials}_trials.pkl', 'rb') as f:
            studies = pickle.load(f)
    evaluate_and_save_results(
        studies, models_that_need_scaling, X_train, y_train, X_val, y_val, true_test_df, test_game_ids, train_x, train_y, n_trials=args.n_trials)


if __name__ == "__main__":
    main()

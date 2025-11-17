import time
from pathlib import Path
from typing import Tuple, Any, List
from functools import partial
import pickle
import argparse

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .config import DATA_DIR, SEED, N_TRIALS, MODELS_NEEDING_SCALING, CASUAL_THRESHOLD
from .feature_engineering import pre_process, clean_data
from .util import remove_casual_only_players, create_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        return XGBRegressor(**params, random_state=SEED, n_jobs=-1, enable_categorical=False)

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
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
        cat_features_list = [
            'time_control_name', 'game_end_reason', 'rating_mode',
            'lexicon', 'game_variant', 'opponent_bot'
        ]
        return CatBoostRegressor(**params, random_state=SEED, verbose=0, thread_count=-1, cat_features=cat_features_list)

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
    raise ValueError(f"Model {model_name} not recognized for tuning.")


def objective(trial: optuna.trial.Trial, model_name: str, needs_scaling: bool, X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int = 5) -> float:
    """
    The Optuna objective function.
    Performs GroupKFold cross-validation and returns the mean validation RMSE.
    Args:
        trial (optuna.trial.Trial): The Optuna trial object
        model_name (str): Name of the model to tune
        needs_scaling (bool): Whether the model needs feature scaling
        X (pd.DataFrame): Full training features
        y (pd.Series): Full training target
        groups (pd.Series): Series of groups (nickname) for CV
    Returns:
        float: The mean validation RMSE across all folds
    """
    model = get_model_and_params(model_name, trial)
    pipeline = create_pipeline(model, needs_scaling)

    gkf = GroupKFold(n_splits=folds)
    rmses = []
    maes = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        rmses.append(rmse)
        maes.append(mae)

    trial.set_user_attr("mean_mae", np.mean(maes))

    return np.mean(rmses)


def generate_submission_file(pipeline: Pipeline, test_df: pd.DataFrame, test_game_ids: pd.Series, model_name: str, output_dir: Path):
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
    test_predictions = pipeline.predict(test_df)
    submission_df = pd.DataFrame(
        {'game_id': test_game_ids, 'rating': test_predictions})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'submission_{model_name}_tuned.csv'
    submission_df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Saved {model_name} predictions to {output_path}")


def setup_args() -> argparse.Namespace:
    """
    Sets up command-line arguments for the script.
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run model tuning and evaluation pipeline.")
    parser.add_argument('--n_trials', type=int, default=N_TRIALS,
                        help='Number of trials for hyperparameter tuning.')
    parser.add_argument('--run_fe', action='store_true',
                        help='Run feature engineering pipeline even if files exist.')
    parser.add_argument('--fe_mode', type=str, default='full',
                        choices=['full', 'lite', 'lite_cat', 'cat'],
                        help='Base mode for feature engineering: "full", "lite", or "cat".')
    parser.add_argument('--from_pkl', action='store_true',
                        help='Load models from pickle files instead of tuning.')
    parser.add_argument('--pkl_path', type=str, default='CatBoostRegressor_tuning_study 98.pkl',
                        help='Path to the pickle file for loading cached model.')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of folds for GroupKFold cross-validation.')
    return parser.parse_args()


def load_and_prep_data(fe_mode: str, force_run_fe: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses data for a specific feature engineering mode.
    Args:
        fe_mode (str): The feature engineering mode ('full', 'lite', 'cat').
        force_run_fe (bool): Whether to force-run the FE pipeline.
    Returns:
        A tuple of (train_df, test_df) after preprocessing and cleaning.
    """
    print(f"\n--- Loading and prepping data for mode: {fe_mode} ---")

    if fe_mode == 'cat' or fe_mode == 'lite_cat':
        train_path = DATA_DIR / 'processed' / 'train_fe_cat_no_bots.csv'
        test_path = DATA_DIR / 'processed' / 'test_fe_cat_no_bots.csv'
    else:
        train_path = DATA_DIR / 'processed' / 'train_fe_no_bots.csv'
        test_path = DATA_DIR / 'processed' / 'test_fe_no_bots.csv'

    train_df, test_df = pre_process(
        train_path, test_path, force_run_fe=force_run_fe, mode=fe_mode
    )

    train_df_cleaned = remove_casual_only_players(train_df)

    return train_df_cleaned, test_df


def tune_train_and_save(model_name: str,
                        args: argparse.Namespace,
                        cols_to_drop: List[str],
                        submission_dir: Path) -> dict:
    """
    Runs the full Optuna tuning, retraining, and submission pipeline for a single model.

    Args:
        model_name (str): The name of the model to run.
        args (argparse.Namespace): The script's command-line arguments.
        cols_to_drop (List[str]): List of columns to drop before training.
        submission_dir (Path): Path to save the submission file.

    Returns:
        dict: A dictionary of results for this model.
    """
    print(f"\n--- Starting workflow for {model_name} ---")

    models_that_need_scaling = ['Ridge', 'KNeighborsRegressor']

    if model_name == 'CatBoostRegressor' and args.fe_mode != 'lite_cat':
        current_fe_mode = 'cat'
    else:
        current_fe_mode = args.fe_mode

    train_df, true_test_df = load_and_prep_data(
        fe_mode=current_fe_mode, force_run_fe=args.run_fe
    )

    train_x, train_y, test_df_processed, test_game_ids, train_groups = clean_data(
        train_df, true_test_df, cols_to_drop
    )

    print(f"Tuning {model_name} with {args.n_trials} trials...")
    needs_scaling = model_name in models_that_need_scaling
    study = optuna.create_study(direction='minimize', pruner=MedianPruner(
        n_startup_trials=10, n_warmup_steps=3))
    # n_startup_trials is the number of trials before pruning starts
    # n_warmup_steps is the number of steps before pruning starts
    obj_func = partial(
        objective,
        model_name=model_name,
        needs_scaling=needs_scaling,
        X=train_x,
        y=train_y,
        groups=train_groups,
        folds=args.folds
    )

    study.optimize(obj_func, n_trials=args.n_trials,
                   show_progress_bar=True, n_jobs=-1)

    print(
        f"Finished tuning {model_name}. Best trial CV RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    study_path = DATA_DIR / 'models' / f'{model_name}_tuning_study.pkl'
    study_path.parent.mkdir(parents=True, exist_ok=True)
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)

    print(f"Retraining {model_name} on full dataset with best params...")
    start_time = time.time()
    best_model_final = get_model_and_params(
        model_name, study.best_trial)
    best_pipeline_final = create_pipeline(
        best_model_final, needs_scaling)

    best_pipeline_final.fit(train_x, train_y)
    duration = time.time() - start_time

    best_cv_rmse = study.best_value
    best_cv_mae = study.best_trial.user_attrs.get("mean_mae", np.nan)

    print(
        f"Finished {model_name} retraining in {duration:.2f}s. Best CV RMSE: {best_cv_rmse:.4f}")

    generate_submission_file(
        best_pipeline_final, test_df_processed, test_game_ids, model_name, submission_dir)

    return {
        'Model': model_name,
        'Val RMSE (CV)': best_cv_rmse,
        'Val MAE (CV)': best_cv_mae,
        'Fit Time (s)': duration
    }


def run_tuning_workflow(args: argparse.Namespace):
    """
    Runs the full tuning, training, and submission pipeline for all models.
    Args:
        args (argparse.Namespace): The script's command-line arguments.
    """
    models_to_run = ['CatBoostRegressor']
    submission_dir = DATA_DIR / 'submissions'
    cols_to_drop = ['nickname', 'first']
    results = []

    print(
        f"\n--- Starting Model Tuning (Running {args.n_trials} trials each with 5-Fold GroupKFold CV) ---")

    for model_name in models_to_run:
        model_result = tune_train_and_save(
            model_name, args, cols_to_drop, submission_dir
        )
        results.append(model_result)

    results_df = pd.DataFrame(results).sort_values(
        by='Val RMSE (CV)', ascending=True)

    print("\n--- Model Comparison Results (Tuned) ---")
    print(results_df.to_string(index=False, float_format="%.4f"))


def run_cached_workflow(args: argparse.Namespace):
    """
    Runs the prediction pipeline from a cached (pickled) Optuna study.
    Args:
        args (argparse.Namespace): The script's command-line arguments.
    Raises:
        FileNotFoundError: If the specified pickle file does not exist.
    """
    print("Running from cached model pipelines...")

    train_df, true_test_df = load_and_prep_data(
        fe_mode='cat' if 'CatBoostRegressor' in args.pkl_path else args.fe_mode,
        force_run_fe=args.run_fe
    )

    cols_to_drop = ['nickname', 'first']
    train_x, train_y, test_df_processed, test_game_ids, train_groups = clean_data(
        train_df, true_test_df, cols_to_drop
    )

    submission_dir = DATA_DIR / 'submissions'
    pkl_model = args.pkl_path
    pkl_path = DATA_DIR / 'models' / pkl_model
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file {pkl_path} not found.")
    run_from_cached(pkl_path, test_df_processed, test_game_ids, submission_dir,
                    train_x, train_y)


def run_from_cached(pkl_path: Path, true_test_df: pd.DataFrame, test_game_ids: pd.Series,
                    submission_dir: Path, train_x: pd.DataFrame, train_y: pd.Series):
    """
    Runs model training and submission generation from a cached Optuna study pickle file.
    Args:
        pkl_path (Path): Path to the Optuna study pickle file
        true_test_df (pd.DataFrame): test features dataframe
        test_game_ids (pd.Series): Series of game IDs for the test set
        submission_dir (Path): Directory to save the submission file
        train_x (pd.DataFrame): Training features
        train_y (pd.Series): Training target
    """
    print(f"Loading Optuna study from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        study = pickle.load(f)
    model_name = pkl_path.stem.split('_')[0]
    needs_scaling = model_name in MODELS_NEEDING_SCALING

    print("Re-creating best model from study...")

    best_model = get_model_and_params(
        model_name, study.best_trial
    )

    pipeline = create_pipeline(best_model, needs_scaling)

    print("Training model on full dataset")
    start_time = time.time()
    pipeline.fit(train_x, train_y)
    print(f"Finished training in {time.time() - start_time:.2f}s.")

    model_name_for_file = pkl_path.stem.replace(
        '_pipeline', '').replace('_tuning_study', '')
    generate_submission_file(
        pipeline, true_test_df, test_game_ids, model_name_for_file, submission_dir)


def main():
    """
    Main function to run the full tuning, training, and evaluation pipeline.
    """
    args = setup_args()

    if args.from_pkl:
        run_cached_workflow(args)
    else:
        run_tuning_workflow(args)


if __name__ == "__main__":
    main()

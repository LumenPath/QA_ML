import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, RandomizedSearchCV
import os
import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import joblib
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional, Union
import random
from scipy.stats import randint, uniform

# ===== Global Parameter Settings (Easy to Adjust) =====
# Hyperparameter random search switch
ENABLE_HYPERPARAMETER_SEARCH = True  # Set to True to enable random search; False uses default parameters

# Random search configuration parameters
RANDOM_SEARCH_ITERATIONS = 400  # Number of random search iterations
RANDOM_SEARCH_CV_FOLDS = 5  # Number of internal cross-validation folds for random search

# Sampling method configuration
SAMPLING_METHOD = "equal_width"  # Sampling method selection: "random" or "equal_width"; default is random sampling

# Input file settings
INPUT_FILE = 'XXX.csv'  # Input filename

# Data column index configuration (soft-coded)
ID_COL_INDEX = 0  # ID column index
TARGET_COL_INDEX = 1  # Target variable column index
FEATURE_START_INDEX = 2  # Feature start column index
FEATURE_END_INDEX = 16  # Feature end column index (inclusive)

# Equal-width sampling configuration
BINS_COUNT = 5  # Number of bins for equal-width sampling

# Test set ratio configuration
TEST_SIZE = 0.1  # Test set ratio

# Random seed
RANDOM_STATE = 42

# Cross-validation configuration
CV_FOLDS = 5  # Number of cross-validation folds

# Figure output quality settings
OUTPUT_DPI = 300  # Figure output DPI
EXPORT_CSV = True  # Whether to export CSV prediction results

# Set English font - Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']  # Removed SimHei Chinese font
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX math font
plt.rcParams['axes.unicode_minus'] = False  # Avoid rendering minus signs as boxes
plt.rcParams['figure.dpi'] = OUTPUT_DPI  # Set DPI to academic publication standard

# Custom color scheme
SCATTER_COLORS = ['#3498db', '#e74c3c']  # Blue and red for scatter plots
BAR_COLORS = ['#1a5276', '#922b21', '#6c3483', '#1e8449']  # Darker colors for bar charts
LINE_COLORS = ['#2c3e50', '#c0392b', '#16a085', '#f39c12']  # Colors for line plots
HEATMAP_CMAP = LinearSegmentedColormap.from_list('custom_blues',
                                                 ['#f7fbff', '#6baed6', '#08519c'])  # Blue gradient

# Set logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
    ]
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: str) -> None:
    """
    Set up logging to output to both file and console

    Parameters:
    - output_dir: Output directory for the log file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_log_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add file handler to the root logger
    logger.addHandler(file_handler)

    logger.info(f"Logging has been set up, log file: {log_file}")


def save_configuration_parameters(output_dir: str) -> None:
    """
    Save all parameter configurations for the current run to a text file for reproducibility

    Parameters:
    - output_dir: Output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, "run_parameters.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run time: {timestamp}\n\n")
        f.write("==== Hyperparameter Random Search Configuration ====\n")
        f.write(f"Whether to enable hyperparameter random search: {ENABLE_HYPERPARAMETER_SEARCH}\n")
        if ENABLE_HYPERPARAMETER_SEARCH:
            f.write(f"Number of random search iterations: {RANDOM_SEARCH_ITERATIONS}\n")
            f.write(f"Number of random search cross-validation folds: {RANDOM_SEARCH_CV_FOLDS}\n\n")

        f.write("==== Sampling Method Configuration ====\n")
        f.write(f"Sampling method: {SAMPLING_METHOD}\n\n")

        f.write("==== Input File Configuration ====\n")
        f.write(f"Input file: {INPUT_FILE}\n\n")

        f.write("==== Data Column Index Configuration ====\n")
        f.write(f"ID column index: {ID_COL_INDEX}\n")
        f.write(f"Target variable column index: {TARGET_COL_INDEX}\n")
        f.write(f"Feature start index: {FEATURE_START_INDEX}\n")
        f.write(f"Feature end index: {FEATURE_END_INDEX}\n\n")

        if SAMPLING_METHOD == "equal_width":
            f.write("==== Equal-Width Binning Sampling Parameters ====\n")
            f.write(f"Number of bins: {BINS_COUNT}\n\n")

        f.write("==== LightGBM Parameters ====\n")
        if not ENABLE_HYPERPARAMETER_SEARCH:
            f.write(f"Objective function: regression\n")
            f.write(f"Evaluation metric: rmse\n")
            f.write(f"Boosting type: gbdt\n")
            f.write(f"Number of leaves: 10\n")
            f.write(f"Learning rate: 0.04\n")
            f.write(f"Feature sampling ratio: 1.0\n")
            f.write(f"Sample sampling ratio: 0.8\n")
            f.write(f"Sampling frequency: 5\n\n")
        else:
            f.write(f"Use random search to optimize hyperparameters\n\n")

        f.write("==== Sampling Method Parameters ====\n")
        f.write(f"Test set ratio: {TEST_SIZE}\n")
        f.write(f"Number of cross-validation folds: {CV_FOLDS}\n\n")

        f.write("==== Other Configuration ====\n")
        f.write(f"Random seed: {RANDOM_STATE}\n\n")

    logger.info(f"Run parameter configuration has been saved to: {os.path.join(output_dir, 'run_parameters.txt')}")


def create_output_dir() -> str:
    """
    Create output directory

    Returns:
    - Output directory path
    """
    # Create main folder
    main_dir = "LightGBM_Regression_Results"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        logger.info(f"Created main output directory: {main_dir}")

    # Create timestamp-named subfolder
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(main_dir, f"Run_Results_{current_time}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created current run directory: {output_dir}")

    return output_dir


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean data:
    - Check and handle missing values

    Parameters:
    - file_path: Data file path

    Returns:
    - Cleaned dataframe
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Number of samples before data cleaning: {len(df)}")

        # Check and handle missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.info(f"Missing values found, missing counts by column:\n{missing_counts[missing_counts > 0]}")

            df_cleaned = df.dropna()
            logger.info(f"Number of samples after handling missing values: {len(df_cleaned)}")
            logger.info(f"Total number of removed samples: {len(df) - len(df_cleaned)}")
        else:
            df_cleaned = df.copy()
            logger.info("No missing values found in the data")

        return df_cleaned

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"The file is empty or has an incorrect format: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error parsing file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unknown error occurred while loading data: {str(e)}")
        raise


def random_sampling(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                    random_state: int = RANDOM_STATE) -> Tuple:
    """
    Perform simple random sampling on the data

    Parameters:
    - X: Feature data
    - y: Target variable
    - test_size: Test set ratio
    - random_state: Random seed

    Returns:
    - Features and labels for the training and test sets
    """
    try:
        # Use sklearn's train_test_split function for random sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Random sampling completed, training set size: {len(X_train)}, test set size: {len(X_test)}")

        # Analyze target variable distribution in the training and test sets
        logger.info("\nTarget variable distribution:")
        logger.info(
            f"Training set target variable: min={y_train.min():.3f}, max={y_train.max():.3f}, mean={y_train.mean():.3f}, std={y_train.std():.3f}")
        logger.info(
            f"Test set target variable: min={y_test.min():.3f}, max={y_test.max():.3f}, mean={y_test.mean():.3f}, std={y_test.std():.3f}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Random sampling failed: {str(e)}")
        raise


def stratified_sampling_equal_width(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                                    bins: int = 6, random_state: int = RANDOM_STATE) -> Tuple:
    """
    Perform stratified sampling based on target values (equal-width binning)

    Parameters:
    - X: Feature data
    - y: Target variable
    - test_size: Test set ratio
    - bins: Number of bins, i.e., how many intervals to divide the target variable range into
    - random_state: Random seed

    Returns:
    - Features and labels for the training and test sets
    """
    try:
        # First calculate the boundaries of equal-width bins - only calculated once
        bin_edges = pd.cut(y, bins=bins, retbins=True)[1]

        # Stratify by target value range (equal-width binning)
        y_bins = pd.cut(y, bins=bins, labels=False)

        # Use the stratified bins as labels for sampling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            stratify=y_bins, random_state=random_state
        )

        logger.info(f"Equal-width binning stratified sampling completed, training set size: {len(X_train)}, test set size: {len(X_test)}")

        # Analyze the sample distribution in each bin
        logger.info("\nSample distribution by interval:")

        for i in range(len(bin_edges) - 1):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]

            # Special handling for the last interval to ensure the maximum value is included
            is_last_bin = (i == len(bin_edges) - 2)

            if is_last_bin:
                train_count = ((y_train >= bin_start) & (y_train <= bin_end)).sum()
                test_count = ((y_test >= bin_start) & (y_test <= bin_end)).sum()
            else:
                train_count = ((y_train >= bin_start) & (y_train < bin_end)).sum()
                test_count = ((y_test >= bin_start) & (y_test < bin_end)).sum()

            total_count = train_count + test_count

            if total_count > 0:
                logger.info(f"Interval {i + 1} [{bin_start:.3f}-{bin_end:.3f}]: "
                            f"Total {total_count} samples, "
                            f"Training set {train_count} ({train_count / total_count * 100:.1f}%), "
                            f"Test set {test_count} ({test_count / total_count * 100:.1f}%)")
            else:
                logger.info(f"Interval {i + 1} [{bin_start:.3f}-{bin_end:.3f}]: no samples")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Equal-width binning failed: {str(e)}")
        logger.error("Samples may be insufficient or unevenly distributed. Try reducing the number of bins or using another sampling method.")
        raise


def hyperparameter_random_search(X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
    """
    Execute LightGBM hyperparameter random search

    Parameters:
    - X_train: Training set features
    - y_train: Training set target variable

    Returns:
    - Dictionary of optimal hyperparameters
    """
    # Define hyperparameter search space
    param_grid = {
        'num_leaves': randint(8, 20),
        'learning_rate': uniform(0.01, 0.2),
        'n_estimators': randint(20, 500),
        'min_child_samples': randint(5, 50),
        'max_depth': randint(5, 20),
        'colsample_bytree': uniform(0.7, 0.3),  # uniform(low, scale) generates random values in [low, low+scale], not in [low, high]
        'subsample': uniform(0.6, 0.4),
        'subsample_freq': [0, 1, 5],
        'reg_alpha': uniform(0, 10),
        'reg_lambda': uniform(0, 10),
    }

    # Create LightGBM regressor
    lgb_reg = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        boosting_type='gbdt',
        verbose=-1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    # Set up random search
    random_search = RandomizedSearchCV(
        estimator=lgb_reg,
        param_distributions=param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=RANDOM_SEARCH_CV_FOLDS,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Start random search
    logger.info(f"Starting hyperparameter random search ({RANDOM_SEARCH_ITERATIONS} iterations, {RANDOM_SEARCH_CV_FOLDS}-fold cross-validation)...")
    random_search.fit(X_train, y_train)

    # Get best parameters
    best_params = random_search.best_params_
    best_score = np.sqrt(-random_search.best_score_)  # Convert negative MSE to RMSE

    logger.info(f"Hyperparameter random search completed")
    logger.info(f"Best RMSE: {best_score:.5f}")
    logger.info(f"Best parameters: {best_params}")

    # Convert best parameters into a format usable by lgb.train
    lgb_best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': best_params['num_leaves'],
        'learning_rate': best_params['learning_rate'],
        'feature_fraction': best_params['colsample_bytree'],
        'bagging_fraction': best_params['subsample'],
        'bagging_freq': best_params['subsample_freq'],
        'min_child_samples': best_params['min_child_samples'],
        'max_depth': best_params['max_depth'],
        'lambda_l1': best_params['reg_alpha'],
        'lambda_l2': best_params['reg_lambda'],
        'verbose': -1,
        'random_state': RANDOM_STATE
    }

    # Save search results
    search_results = pd.DataFrame(random_search.cv_results_)
    return lgb_best_params, search_results


def create_lgb_model(X_train=None, y_train=None):
    """
    Create LightGBM regression model
    If hyperparameter search is enabled, execute random search to find the best parameters
    Otherwise, use default parameters

    Parameters:
    - X_train: Training data, only used when hyperparameter search is enabled
    - y_train: Training labels, only used when hyperparameter search is enabled

    Returns:
    - Configured LightGBM parameter dictionary
    - If random search is executed, also returns the search results DataFrame
    """
    if ENABLE_HYPERPARAMETER_SEARCH and X_train is not None and y_train is not None:
        logger.info("Hyperparameter random search enabled...")
        best_params, search_results = hyperparameter_random_search(X_train, y_train)
        logger.info(f"Creating LightGBM model using the best hyperparameters")
        return best_params, search_results
    else:
        # Use default parameters
        lgb_params = {
            'objective': 'regression',  # Regression task
            'metric': 'rmse',  # Evaluation metric
            'boosting_type': 'gbdt',  # Gradient boosting decision tree
            'num_leaves': 10,  # Number of leaves
            'learning_rate': 0.04,  # Learning rate
            'feature_fraction': 1,  # Feature ratio used in each iteration
            'bagging_fraction': 1,  # Randomly select 80% of data in each iteration
            'bagging_freq': 5,  # Execute bagging every 5 iterations
            'verbose': -1,  # Do not output training information
            'n_jobs': -1,  # Use all CPU cores
            'random_state': RANDOM_STATE  # Random seed
        }

        logger.info("Creating LightGBM regression model")
        logger.info(f"Parameter settings: num_leaves={lgb_params['num_leaves']}, "
                    f"learning_rate={lgb_params['learning_rate']}")

        return lgb_params, None


def plot_hyperparameter_search_results(search_results: pd.DataFrame, output_dir: str):
    """
    Visualize hyperparameter search results

    Parameters:
    - search_results: Results DataFrame from RandomizedSearchCV
    - output_dir: Output directory
    """
    try:
        # Extract RMSE score (convert negative MSE to RMSE)
        search_results['rmse'] = np.sqrt(-search_results['mean_test_score'])

        # Sort by RMSE
        sorted_results = search_results.sort_values('rmse')

        # Plot RMSE of the top 10 best models
        plt.figure(figsize=(12, 6))
        top_n = min(10, len(sorted_results))
        plt.bar(range(top_n), sorted_results['rmse'].head(top_n), color=BAR_COLORS[0])
        plt.xlabel('Model Rank', fontsize=14, family='Times New Roman')
        plt.ylabel('RMSE (lower is better)', fontsize=14, family='Times New Roman')
        plt.title('Top 10 Models from Random Search', fontsize=16, family='Times New Roman')
        plt.xticks(range(top_n), [f"#{i + 1}" for i in range(top_n)], family='Times New Roman')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save chart
        plt.tight_layout()
        filename = os.path.join(output_dir, "Hyperparameter_Search_Results_Top10_Models.png")
        plt.savefig(filename, dpi=OUTPUT_DPI)
        plt.close()
        logger.info(f"Hyperparameter search top 10 model chart has been saved: {filename}")

        # Analyze the effects of main hyperparameters on RMSE
        important_params = ['num_leaves', 'learning_rate', 'max_depth', 'min_child_samples']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, param in enumerate(important_params):
            if param in search_results.columns:
                ax = axes[i]
                # Combine parameter values and RMSE into a DataFrame
                param_df = pd.DataFrame({
                    'param_value': search_results[param],
                    'rmse': search_results['rmse']
                })

                # Select visualization method according to parameter type
                if param_df['param_value'].dtype in [np.float64, np.float32]:
                    # For continuous parameters, use scatter plot
                    scatter = ax.scatter(param_df['param_value'], param_df['rmse'],
                                         alpha=0.7, c=LINE_COLORS[i % len(LINE_COLORS)], s=50)

                    # Try to add a trend line
                    try:
                        z = np.polyfit(param_df['param_value'], param_df['rmse'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(param_df['param_value'].min(), param_df['param_value'].max(), 100)
                        ax.plot(x_range, p(x_range), '--', color=BAR_COLORS[i % len(BAR_COLORS)], linewidth=2)
                    except:
                        pass  # If fitting fails, continue to the next step
                else:
                    # For categorical parameters, use boxplot
                    boxplot = ax.boxplot([param_df[param_df['param_value'] == v]['rmse'].values
                                          for v in sorted(param_df['param_value'].unique())],
                                         patch_artist=True,
                                         labels=[str(v) for v in sorted(param_df['param_value'].unique())])

                    # Set boxplot colors
                    for box in boxplot['boxes']:
                        box.set(facecolor=BAR_COLORS[i % len(BAR_COLORS)], alpha=0.7)

                # Set axis labels and title
                ax.set_xlabel(param, fontsize=14, family='Times New Roman')
                ax.set_ylabel('RMSE', fontsize=14, family='Times New Roman')
                ax.set_title(f'Effect of {param} on Model Performance', fontsize=16, family='Times New Roman')
                ax.grid(True, linestyle='--', alpha=0.6)

                # Ensure all fonts are Times New Roman
                for text in ax.get_xticklabels() + ax.get_yticklabels():
                    text.set_fontname('Times New Roman')

        # Adjust layout and save
        fig.tight_layout()
        filename = os.path.join(output_dir, "Hyperparameter_Search_Results_Parameter_Analysis.png")
        fig.savefig(filename, dpi=OUTPUT_DPI)
        plt.close(fig)
        logger.info(f"Hyperparameter analysis chart has been saved: {filename}")

        # Export hyperparameter search results to CSV
        csv_filename = os.path.join(output_dir, "Hyperparameter_Search_Results.csv")
        sorted_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Hyperparameter search results have been saved to CSV: {csv_filename}")

    except Exception as e:
        logger.error(f"Error plotting hyperparameter search results: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def train_and_evaluate_lgb(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           cv_folds: int = 5, method_name: str = "LightGBM Regression Model",
                           output_dir: str = None) -> Dict:
    """
    Train and evaluate the LightGBM regression model
    - Use n-fold cross-validation
    - Calculate R², MAE, RMSE
    - Return the trained model and evaluation metrics

    Parameters:
    - X_train: Training set features
    - X_test: Test set features
    - y_train: Training set target variable
    - y_test: Test set target variable
    - cv_folds: Number of cross-validation folds
    - method_name: Method name (for logging output)
    - output_dir: Output directory (used to save hyperparameter search results)

    Returns:
    - Dictionary containing the model, evaluation metrics, and prediction results
    """
    # Create LightGBM parameters
    if ENABLE_HYPERPARAMETER_SEARCH:
        lgb_params, search_results = create_lgb_model(X_train, y_train)
        if output_dir is not None and search_results is not None:
            plot_hyperparameter_search_results(search_results, output_dir)
    else:
        lgb_params, _ = create_lgb_model()

    logger.info(f"\nStarting {cv_folds}-fold cross-validation [{method_name}]...")

    # Use KFold to manually implement cross-validation - using the method from the second code
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_r2_scores = []
    cv_mae_scores = []
    cv_rmse_scores = []

    # Train and evaluate each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        # Get training and validation data for the current fold
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_fold_train, y_fold_train)
        lgb_val = lgb.Dataset(X_fold_val, y_fold_val, reference=lgb_train)

        # Train model
        fold_model = lgb.train(
            lgb_params,
            lgb_train,
            num_boost_round=1000,  # Maximum number of iterations
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),  # Early stopping
                lgb.log_evaluation(period=0)  # Do not print evaluation results
            ]
        )

        # Predict and evaluate
        y_fold_pred = fold_model.predict(X_fold_val, num_iteration=fold_model.best_iteration)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))

        cv_r2_scores.append(fold_r2)
        cv_mae_scores.append(fold_mae)
        cv_rmse_scores.append(fold_rmse)

        logger.info(f"  Fold {fold}: R² = {fold_r2:.4f}, MAE = {fold_mae:.4f}, RMSE = {fold_rmse:.4f}, "
                    f"best iteration: {fold_model.best_iteration}")

    # Output average cross-validation results
    logger.info(f"\n{method_name} - average cross-validation results:")
    logger.info(f"Average validation R²: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")
    logger.info(f"Average validation MAE: {np.mean(cv_mae_scores):.4f} (±{np.std(cv_mae_scores):.4f})")
    logger.info(f"Average validation RMSE: {np.mean(cv_rmse_scores):.4f} (±{np.std(cv_rmse_scores):.4f})")

    # Train final model on the full training set
    logger.info(f"\nTraining final {method_name} model on the full training set...")
    lgb_train_final = lgb.Dataset(X_train, y_train)
    lgb_test_final = lgb.Dataset(X_test, y_test, reference=lgb_train_final)

    final_model = lgb.train(
        lgb_params,
        lgb_train_final,
        num_boost_round=1000,
        valid_sets=[lgb_train_final, lgb_test_final],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    logger.info(f"Final model best iteration: {final_model.best_iteration}")

    # Calculate training set metrics
    y_train_pred = final_model.predict(X_train, num_iteration=final_model.best_iteration)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    logger.info(f"\n{method_name} - training set evaluation results:")
    logger.info(f"R²: {train_r2:.4f}")
    logger.info(f"MAE: {train_mae:.4f}")
    logger.info(f"RMSE: {train_rmse:.4f}")

    # Evaluate on the test set
    y_test_pred = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info(f"\n{method_name} - test set evaluation results:")
    logger.info(f"R²: {test_r2:.4f}")
    logger.info(f"MAE: {test_mae:.4f}")
    logger.info(f"RMSE: {test_rmse:.4f}")

    # Get feature importance
    feature_importance = final_model.feature_importance(importance_type='split')

    # Create results dictionary
    results = {
        'model': final_model,
        'metrics': {
            'train': {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse},
            'val': {
                'r2': np.mean(cv_r2_scores),
                'mae': np.mean(cv_mae_scores),
                'rmse': np.mean(cv_rmse_scores)
            },
            'test': {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse}
        },
        'pred': {
            'train': y_train_pred,
            'test': y_test_pred
        },
        'feature_importance': feature_importance,
        'feature_names': X_train.columns.tolist(),
        'best_rounds': final_model.best_iteration
    }

    # Add information on whether hyperparameter search was used
    if ENABLE_HYPERPARAMETER_SEARCH:
        results['hyperparameter_search'] = True
        results['hyperparameters'] = lgb_params
    else:
        results['hyperparameter_search'] = False

    return results


def export_feature_importance_to_csv(feature_importance: np.ndarray, feature_names: List[str],
                                     method_name: str, output_dir: str) -> str:
    """
    Export feature importance to a CSV file for subsequent plotting and analysis

    Parameters:
    - feature_importance: Feature importance array
    - feature_names: List of feature names
    - method_name: Method name
    - output_dir: Output directory

    Returns:
    - Saved filename
    """
    try:
        # Ensure feature_importance and feature_names have the same length
        if len(feature_importance) != len(feature_names):
            logger.warning(
                f"Feature importance array length ({len(feature_importance)}) does not match feature name array length ({len(feature_names)})")
            # If they do not match, use the shorter length
            min_length = min(len(feature_importance), len(feature_names))
            feature_importance = feature_importance[:min_length]
            feature_names = feature_names[:min_length]

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature Name': feature_names,
            'Importance Score': feature_importance
        })

        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance Score', ascending=False)

        # Add ranking column
        importance_df['Rank'] = range(1, len(importance_df) + 1)

        # Adjust column order
        importance_df = importance_df[['Rank', 'Feature Name', 'Importance Score']]

        # Filename
        method_name_clean = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        filename = os.path.join(output_dir, f"Feature_Importance_{method_name_clean}.csv")

        # Save to CSV
        importance_df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Feature importance has been saved to: {filename}")

        return filename
    except Exception as e:
        logger.error(f"Error exporting feature importance CSV: {str(e)}")
        # Create a simple default CSV
        try:
            dummy_df = pd.DataFrame({
                'Feature Name': ["Feature" + str(i + 1) for i in range(len(feature_names))],
                'Importance Score': [1.0] * len(feature_names)
            })
            method_name_clean = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            filename = os.path.join(output_dir, f"Feature_Importance_{method_name_clean}_default.csv")
            dummy_df.to_csv(filename, index=False, encoding='utf-8-sig')
            return filename
        except:
            return os.path.join(output_dir, "Feature_Importance_error.csv")


def plot_feature_importance(feature_importance: np.ndarray, feature_names: List[str],
                            title: str, filename: str = None) -> None:
    """
    Plot feature importance bar chart

    Parameters:
    - feature_importance: Feature importance array
    - feature_names: List of feature names
    - title: Figure title
    - filename: Saved filename
    """
    try:
        # Ensure feature_importance and feature_names have the same length
        if len(feature_importance) != len(feature_names):
            logger.warning(
                f"Feature importance array length ({len(feature_importance)}) does not match feature name array length ({len(feature_names)})")
            # If they do not match, use the shorter length
            min_length = min(len(feature_importance), len(feature_names))
            feature_importance = feature_importance[:min_length]
            feature_names = feature_names[:min_length]

        # Create a new plotting session to ensure no residual settings
        plt.close('all')
        plt.rcdefaults()

        # Create sorted index for feature importance
        sorted_idx = np.argsort(feature_importance)

        # Select the top 15 most important features (or all features if fewer than 15)
        n_features = min(15, len(feature_names))
        top_indices = sorted_idx[-n_features:]

        # Extract corresponding feature names and importance values
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]

        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Set font and background
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Create color mapping
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.1, 0.9, len(top_features)))

        # Plot horizontal bar chart
        bars = ax.barh(range(len(top_features)), top_importance, color=colors,
                       height=0.7, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add numeric labels to each bar
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.4f}', va='center', fontsize=10, fontweight='bold',
                    family='Times New Roman')

        # Set Y-axis labels (feature names)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=12, family='Times New Roman')

        # Set X-axis label and title
        ax.set_xlabel('Feature Importance', fontsize=14, family='Times New Roman')
        ax.set_title(title, fontsize=16, pad=20, family='Times New Roman')

        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Set border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(1.0)

        # Adjust layout
        plt.tight_layout()

        # Save image
        if filename:
            fig.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight')
            logger.info(f"Feature importance plot has been saved: {filename}")

        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        # If an error occurs, create a simple default chart
        try:
            plt.close('all')
            plt.figure(figsize=(8, 6))
            plt.title('Feature Importance (Error occurred)')
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            if filename:
                plt.savefig(filename, dpi=OUTPUT_DPI)
            plt.close()
        except:
            pass


def plot_academic_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, r2: float, mae: float = None,
                          rmse: float = None, filename: str = None) -> None:
    """
    Plot an academic-quality scatter plot showing the relationship between true and predicted values - revised version
    1. Display 5 numeric tick labels on the axes, excluding the origin and endpoints
    2. Ensure all data points are within the range
    3. Use Times New Roman font for all text

    Parameters:
    - y_true: True label values
    - y_pred: Predicted label values
    - title: Figure title
    - r2: R² value
    - mae: MAE value
    - rmse: RMSE value
    - filename: Saved filename
    """
    try:
        # Create a new plotting session to ensure no residual settings
        plt.close('all')
        plt.rcdefaults()

        # Create canvas and axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set font
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = OUTPUT_DPI

        # Set background color
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Calculate regression line coefficients
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)

        # Set dynamic axis range
        data_min = min(min(y_true), min(y_pred))
        data_max = max(max(y_true), max(y_pred))

        # Add some padding to ensure all points are within the range
        range_padding = (data_max - data_min) * 0.05

        # Set axis range - ensure the minimum value is not less than 0.9
        min_val = max(0.9, data_min - range_padding)
        # Ensure the maximum value is large enough to include all data points
        max_val = data_max + range_padding

        # Set axis ranges
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # Set 7 major ticks, but only display labels for the middle 5
        num_ticks = 7  # 7 points in total; remove first and last, display 5 labels
        ticks = np.linspace(min_val, max_val, num_ticks)

        # Create labels, but set first and last to empty strings (not displayed)
        tick_labels = []
        for i, tick in enumerate(ticks):
            if i == 0 or i == num_ticks - 1:
                tick_labels.append('')  # Do not display labels at the first and last ticks
            else:
                tick_labels.append(f'{tick:.1f}')  # Display labels for the middle 5 ticks

        # Set ticks and labels
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # Apply font settings
        font_props = {'fontsize': 14, 'fontfamily': 'Times New Roman'}
        ax.set_xticklabels(tick_labels, **font_props)
        ax.set_yticklabels(tick_labels, **font_props)

        # 1. First plot the perfect prediction line (black solid line)
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1.5,
                label='Perfect Prediction (1:1)', zorder=5)

        # 2. Then plot the regression line (red dashed line)
        ax.plot([min_val, max_val], [p(min_val), p(max_val)], '--', color='#e74c3c',
                linewidth=2.5, label=f'Regression Line (y = {z[0]:.5f}x + {z[1]:.5f})', zorder=6)

        # 3. Finally plot the scatter plot
        errors = np.abs(y_true - y_pred)
        normalized_errors = errors / errors.max() if errors.max() > 0 else errors
        scatter = ax.scatter(y_true, y_pred, s=70, c=normalized_errors, cmap=plt.cm.viridis_r,
                             alpha=0.7, edgecolor='#333333', linewidth=0.5, zorder=10)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Absolute Error', fontsize=14, family='Times New Roman')
        cbar.ax.tick_params(labelsize=12)

        # Ensure colorbar tick labels also use Times New Roman
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        # Set labels and title
        ax.set_xlabel('Actual Values', fontsize=16, family='Times New Roman')
        ax.set_ylabel('Predicted Values', fontsize=16, family='Times New Roman')
        ax.set_title(f'{title} ($R^2$ = {r2:.5f})', fontsize=18, pad=20, family='Times New Roman')

        # Add performance metric text box
        if mae is None:
            mae = 0.0
        if rmse is None:
            rmse = 0.0

        textstr = f'$R^2 = {r2:.5f}$\nMAE = {mae:.5f}\nRMSE = {rmse:.5f}'

        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#666666')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=16,
                verticalalignment='top', bbox=props, family='Times New Roman')

        # Set legend
        legend = ax.legend(loc='lower right', fontsize=14, frameon=True,
                           facecolor='white', edgecolor='#666666', framealpha=0.9)

        # Ensure legend text also uses Times New Roman
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')

        # Thicken axes
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(2.0)

        # Tick settings
        ax.tick_params(which='major', direction='in', length=8, width=1.5)
        ax.minorticks_off()
        ax.grid(False)

        # Adjust layout while ensuring the axis range is not adjusted
        plt.tight_layout()

        # Confirm axis range again
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        # Save image
        if filename:
            fig.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight')
            logger.info(f"Scatter plot has been saved: {filename}")

        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting scatter plot: {str(e)}")
        # If an error occurs, create a simple default chart
        try:
            plt.close('all')
            plt.figure(figsize=(8, 6))
            plt.title('Scatter Plot (Error occurred)')
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            if filename:
                plt.savefig(filename, dpi=OUTPUT_DPI)
            plt.close()
        except:
            pass


def plot_side_by_side_scatter(train_true: np.ndarray, train_pred: np.ndarray,
                              test_true: np.ndarray, test_pred: np.ndarray,
                              title: str, train_r2: float, test_r2: float,
                              train_mae: float = None, train_rmse: float = None,
                              test_mae: float = None, test_rmse: float = None,
                              filename: str = None) -> None:
    """
    Plot side-by-side scatter plots for the training set and test set - revised version
    1. Display 5 numeric tick labels on the axes, excluding the origin and endpoints
    2. Ensure all data points are within the range
    3. Use Times New Roman font for all text

    Parameters:
    - train_true: Training set true values
    - train_pred: Training set predicted values
    - test_true: Test set true values
    - test_pred: Test set predicted values
    - title: Figure title
    - train_r2: Training set R²
    - test_r2: Test set R²
    - train_mae: Training set MAE
    - train_rmse: Training set RMSE
    - test_mae: Test set MAE
    - test_rmse: Test set RMSE
    - filename: Saved filename
    """
    try:
        # Create a new plotting session to ensure no residual settings
        plt.close('all')
        plt.rcdefaults()

        # Create canvas and subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

        # Set font
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = OUTPUT_DPI

        # Custom font properties
        font_props = {
            'fontsize': 14,
            'fontfamily': 'Times New Roman'
        }

        # Set background color
        ax1.set_facecolor('white')
        ax2.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Calculate global data range - dynamically adjusted
        all_true = np.concatenate([train_true, test_true])
        all_pred = np.concatenate([train_pred, test_pred])

        # Calculate min and max values of the data
        data_min = min(np.min(all_true), np.min(all_pred))
        data_max = max(np.max(all_true), np.max(all_pred))

        # Add padding to ensure all points are within the range
        range_padding = (data_max - data_min) * 0.05

        # Global minimum - ensure it is not less than 0.9
        global_min = max(0.9, data_min - range_padding)
        # Global maximum - ensure all data points are included
        global_max = data_max + range_padding

        # Set axis ranges
        ax1.set_xlim(global_min, global_max)
        ax1.set_ylim(global_min, global_max)
        ax2.set_xlim(global_min, global_max)

        # ===== Left: Training Set =====
        # Calculate regression line coefficients for the training set
        z_train = np.polyfit(train_true, train_pred, 1)
        p_train = np.poly1d(z_train)

        # 1. First plot the perfect prediction line (black solid line)
        ax1.plot([global_min, global_max], [global_min, global_max], 'k-', linewidth=1.5,
                 alpha=0.7, label='Perfect Prediction (1:1)', zorder=5)

        # 2. Then plot the regression line (red dashed line)
        ax1.plot([global_min, global_max], [p_train(global_min), p_train(global_max)],
                 linestyle='--', color='#c0392b', linewidth=2,
                 label=f'Regression Line (y = {z_train[0]:.5f}x + {z_train[1]:.5f})', zorder=6)

        # 3. Finally plot the scatter plot
        scatter_train = ax1.scatter(train_true, train_pred, c='#1a5276', alpha=0.7, s=60,
                                    edgecolor='#0e3b58', linewidth=0.5, zorder=10)

        # Set training set title and axis labels
        ax1.set_title(f'Training Set ($R^2$ = {train_r2:.5f})', fontsize=16, pad=15, family='Times New Roman')
        ax1.set_xlabel('Actual Values', fontsize=15, family='Times New Roman')
        ax1.set_ylabel('Predicted Values', fontsize=15, family='Times New Roman')

        # Training set legend
        legend1 = ax1.legend(loc='lower right', fontsize=13, frameon=True)
        for text in legend1.get_texts():
            text.set_fontname('Times New Roman')

        # Add training set statistics
        if train_mae is not None and train_rmse is not None:
            textstr = f'n = {len(train_true)}\nR² = {train_r2:.5f}\nMAE = {train_mae:.5f}\nRMSE = {train_rmse:.5f}'
        else:
            textstr = f'n = {len(train_true)}\nR² = {train_r2:.5f}'

        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#666666')
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props, family='Times New Roman')

        # ===== Right: Test Set =====
        # Calculate regression line coefficients for the test set
        z_test = np.polyfit(test_true, test_pred, 1)
        p_test = np.poly1d(z_test)

        # 1. First plot the perfect prediction line (black solid line)
        ax2.plot([global_min, global_max], [global_min, global_max], 'k-', linewidth=1.5,
                 alpha=0.7, label='Perfect Prediction (1:1)', zorder=5)

        # 2. Then plot the regression line (blue dashed line)
        ax2.plot([global_min, global_max], [p_test(global_min), p_test(global_max)],
                 linestyle='--', color='#1a5276', linewidth=2,
                 label=f'Regression Line (y = {z_test[0]:.5f}x + {z_test[1]:.5f})', zorder=6)

        # 3. Finally plot the scatter plot
        scatter_test = ax2.scatter(test_true, test_pred, c='#922b21', alpha=0.7, s=60,
                                   edgecolor='#7b241c', linewidth=0.5, zorder=10)

        # Set test set title and axis labels
        ax2.set_title(f'Test Set ($R^2$ = {test_r2:.5f})', fontsize=16, pad=15, family='Times New Roman')
        ax2.set_xlabel('Actual Values', fontsize=15, family='Times New Roman')

        # Test set legend
        legend2 = ax2.legend(loc='lower right', fontsize=13, frameon=True)
        for text in legend2.get_texts():
            text.set_fontname('Times New Roman')

        # Add test set statistics
        if test_mae is not None and test_rmse is not None:
            textstr = f'n = {len(test_true)}\nR² = {test_r2:.5f}\nMAE = {test_mae:.5f}\nRMSE = {test_rmse:.5f}'
        else:
            textstr = f'n = {len(test_true)}\nR² = {test_r2:.5f}'

        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                 verticalalignment='top', bbox=props, family='Times New Roman')

        # Set ticks - 7 ticks, but only display labels for the middle 5
        num_ticks = 7
        ticks = np.linspace(global_min, global_max, num_ticks)

        # Create labels, hiding the first and last
        tick_labels = []
        for i, tick in enumerate(ticks):
            if i == 0 or i == num_ticks - 1:
                tick_labels.append('')  # Do not display first and last labels
            else:
                tick_labels.append(f'{tick:.1f}')

        # Set training set ticks and labels
        ax1.set_xticks(ticks)
        ax1.set_yticks(ticks)
        ax1.set_xticklabels(tick_labels, **font_props)
        ax1.set_yticklabels(tick_labels, **font_props)

        # Set test set ticks
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(tick_labels, **font_props)

        # Add grid lines to the training set
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.tick_params(which='major', length=6, width=1.0, direction='out')
        ax1.tick_params(which='minor', length=3, width=0.5, direction='out')

        # Remove grid lines from the test set and enhance tick marks
        ax2.grid(False)
        ax2.tick_params(which='major', length=8, width=1.5, direction='out')
        ax2.tick_params(which='minor', length=5, width=1.0, direction='out')

        # Beautify axes
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(1.2)

        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_color('#333333')
            spine.set_linewidth(2.0)  # Thicken test set axis lines

        # Add overall title
        fig.suptitle(title, fontsize=18, y=0.98, family='Times New Roman')

        # Adjust layout
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        # Confirm axis ranges again
        ax1.set_xlim(global_min, global_max)
        ax1.set_ylim(global_min, global_max)
        ax2.set_xlim(global_min, global_max)

        # Save image
        if filename:
            fig.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight')
            logger.info(f"Side-by-side scatter plot has been saved: {filename}")

        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting side-by-side scatter plot: {str(e)}")
        # If an error occurs, create a simple default chart
        try:
            plt.close('all')
            plt.figure(figsize=(8, 6))
            plt.title('Side-by-Side Scatter Plot (Error occurred)')
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            if filename:
                plt.savefig(filename, dpi=OUTPUT_DPI)
            plt.close()
        except:
            pass


def export_predictions_to_csv(y_true: np.ndarray, y_pred: np.ndarray, method_name: str, data_type: str,
                              output_dir: str) -> str:
    """
    Export true values and predicted values to a CSV file

    Parameters:
    - y_true: True label values
    - y_pred: Predicted label values
    - method_name: Method name
    - data_type: Data type (training set/test set)
    - output_dir: Output directory

    Returns:
    - Saved filename
    """
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'Actual Value': y_true,
            'Predicted Value': y_pred,
            'Error': y_pred - y_true,
            'Absolute Error': np.abs(y_pred - y_true),
            'Squared Error': (y_pred - y_true) ** 2
        })

        # Filename
        method_name_clean = method_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")

        # Map data type to English
        data_type_cn = "Training_Set" if data_type == "training" else "Test_Set"

        filename = os.path.join(output_dir, f"Prediction_Results_{method_name_clean}_{data_type_cn}.csv")

        # Save to CSV
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Prediction results have been saved to: {filename}")

        return filename
    except Exception as e:
        logger.error(f"Error exporting prediction results CSV: {str(e)}")
        return os.path.join(output_dir, f"Prediction_Results_{method_name}_{data_type}_error.csv")


def main():
    """Main function - execute the complete LightGBM regression feature sampling workflow"""
    try:
        # Create output directory
        output_dir = create_output_dir()

        # Set up logging
        setup_logging(output_dir)

        # Save run configuration parameters
        save_configuration_parameters(output_dir)

        # Record hyperparameter search status
        if ENABLE_HYPERPARAMETER_SEARCH:
            logger.info("Hyperparameter random search is enabled")
            logger.info(f"Number of random search iterations: {RANDOM_SEARCH_ITERATIONS}")
            logger.info(f"Number of random search cross-validation folds: {RANDOM_SEARCH_CV_FOLDS}")
        else:
            logger.info("Hyperparameter random search is disabled; default parameters are used")

        # Load and clean data
        try:
            df_cleaned = load_and_clean_data(INPUT_FILE)
        except FileNotFoundError:
            logger.error(f"Error: File not found: '{INPUT_FILE}'")
            logger.error("Please make sure the data file is in the correct path.")
            return
        except Exception as e:
            logger.error(f"Data loading error: {e}")
            return

        # Prepare features and target variable
        X = df_cleaned.iloc[:, FEATURE_START_INDEX:FEATURE_END_INDEX + 1]
        y = df_cleaned.iloc[:, TARGET_COL_INDEX]  # Corrected here, using TARGET_COL_INDEX

        logger.info(f"Dataset dimensions: {X.shape}, number of features: {X.shape[1]}")
        logger.info(f"Feature columns: {list(X.columns)}")
        logger.info(f"Target variable: {df_cleaned.columns[TARGET_COL_INDEX]}")

        # Perform sampling according to the selected sampling method
        logger.info("\n" + "=" * 50)

        if SAMPLING_METHOD == "equal_width":
            logger.info(f"Using equal-width binning stratified sampling (bins={BINS_COUNT})")
            logger.info("=" * 50)
            try:
                X_train, X_test, y_train, y_test = stratified_sampling_equal_width(
                    X, y, test_size=TEST_SIZE, bins=BINS_COUNT, random_state=RANDOM_STATE
                )
            except Exception as e:
                logger.error(f"Equal-width binning sampling failed: {str(e)}")
                logger.info("Trying random sampling as a fallback method...")
                X_train, X_test, y_train, y_test = random_sampling(
                    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
                )
        else:  # Use random sampling by default
            logger.info(f"Using random sampling (test_size={TEST_SIZE})")
            logger.info("=" * 50)
            X_train, X_test, y_train, y_test = random_sampling(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        # Train LightGBM regression model
        sampling_method_name = "Equal_Width_Binning" if SAMPLING_METHOD == "equal_width" else "Random_Sampling"

        # If hyperparameter search is enabled, add an identifier to the method name
        if ENABLE_HYPERPARAMETER_SEARCH:
            method_name = f"LightGBM Regression ({sampling_method_name} + Hyperparameter Optimization)"
        else:
            method_name = f"LightGBM Regression ({sampling_method_name})"

        # English method name - used for chart titles
        if ENABLE_HYPERPARAMETER_SEARCH:
            eng_method_name = f"LightGBM Regression ({SAMPLING_METHOD} + Hyperparameter Tuning)"
        else:
            eng_method_name = f"LightGBM Regression ({SAMPLING_METHOD})"

        lgb_results = train_and_evaluate_lgb(
            X_train, X_test, y_train, y_test,
            cv_folds=CV_FOLDS, method_name=method_name,
            output_dir=output_dir
        )

        # Export prediction results
        if EXPORT_CSV:
            export_predictions_to_csv(
                y_train, lgb_results['pred']['train'],
                f"LGB_{sampling_method_name}", "training", output_dir
            )
            export_predictions_to_csv(
                y_test, lgb_results['pred']['test'],
                f"LGB_{sampling_method_name}", "test", output_dir
            )

        # Export feature importance to CSV
        export_feature_importance_to_csv(
            lgb_results['feature_importance'],
            lgb_results['feature_names'],
            f"LGB_{sampling_method_name}",
            output_dir
        )

        # Save model - use multiple methods to ensure successful saving
        try:
            # Method 1: Try using LightGBM's save_model method (using English filename)
            model_filename = os.path.join(output_dir, f"LightGBM_model_{sampling_method_name}.txt")
            lgb_results['model'].save_model(model_filename)
            logger.info(f"Model has been saved to: {model_filename}")
        except Exception as e:
            logger.warning(f"Failed to save model using LightGBM default method: {str(e)}")
            try:
                # Method 2: Save model using joblib
                model_filename = os.path.join(output_dir, f"LightGBM_model_{sampling_method_name}.joblib")
                joblib.dump(lgb_results['model'], model_filename)
                logger.info(f"Model has been saved using joblib to: {model_filename}")
            except Exception as e2:
                logger.error(f"Saving model using joblib also failed: {str(e2)}")
                try:
                    # Method 3: Try using a pure ASCII path
                    ascii_path = os.path.join(os.path.dirname(output_dir), "model_output")
                    if not os.path.exists(ascii_path):
                        os.makedirs(ascii_path)
                    model_filename = os.path.join(ascii_path, f"lightgbm_model.txt")
                    lgb_results['model'].save_model(model_filename)
                    logger.info(f"Model has been saved to fallback path: {model_filename}")
                except Exception as e3:
                    logger.error(f"All model saving methods failed: {str(e3)}")

        # Plot feature importance
        plot_feature_importance(
            lgb_results['feature_importance'],
            lgb_results['feature_names'],
            f"Feature Importance - LightGBM ({SAMPLING_METHOD})",
            filename=os.path.join(output_dir, f"Feature_Importance_{sampling_method_name}.png")
        )

        # Plot scatter plots - pass MAE and RMSE parameters
        plot_academic_scatter(
            y_train, lgb_results['pred']['train'],
            f"LightGBM ({SAMPLING_METHOD}) - Training Set",
            lgb_results['metrics']['train']['r2'],
            lgb_results['metrics']['train']['mae'],
            lgb_results['metrics']['train']['rmse'],
            filename=os.path.join(output_dir, f"Scatter_Plot_Training_Set_{sampling_method_name}.png")
        )

        plot_academic_scatter(
            y_test, lgb_results['pred']['test'],
            f"LightGBM ({SAMPLING_METHOD}) - Test Set",
            lgb_results['metrics']['test']['r2'],
            lgb_results['metrics']['test']['mae'],
            lgb_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Scatter_Plot_Test_Set_{sampling_method_name}.png")
        )

        # Plot side-by-side scatter plot
        plot_side_by_side_scatter(
            y_train, lgb_results['pred']['train'],
            y_test, lgb_results['pred']['test'],
            f"LightGBM Regression ({SAMPLING_METHOD})",
            lgb_results['metrics']['train']['r2'],
            lgb_results['metrics']['test']['r2'],
            lgb_results['metrics']['train']['mae'],
            lgb_results['metrics']['train']['rmse'],
            lgb_results['metrics']['test']['mae'],
            lgb_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Training_Test_Comparison_{sampling_method_name}.png")
        )

        # Create performance summary table
        summary_df = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE'],
            'Training Set': [
                lgb_results['metrics']['train']['r2'],
                lgb_results['metrics']['train']['mae'],
                lgb_results['metrics']['train']['rmse']
            ],
            'Validation Set': [
                lgb_results['metrics']['val']['r2'],
                lgb_results['metrics']['val']['mae'],
                lgb_results['metrics']['val']['rmse']
            ],
            'Test Set': [
                lgb_results['metrics']['test']['r2'],
                lgb_results['metrics']['test']['mae'],
                lgb_results['metrics']['test']['rmse']
            ]
        })

        # Save performance summary table
        summary_filename = os.path.join(output_dir, "Performance_Metrics_Summary.csv")
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Performance metrics summary has been saved to: {summary_filename}")

        # Print summary table
        logger.info("\nPerformance metrics summary:")
        logger.info(summary_df.to_string(index=False))

        logger.info("\n" + "=" * 50)
        if ENABLE_HYPERPARAMETER_SEARCH:
            logger.info(f"LightGBM regression model (with hyperparameter optimization) - {sampling_method_name} program execution completed!")
        else:
            logger.info(f"LightGBM regression model - {sampling_method_name} program execution completed!")
        logger.info("=" * 50)
        logger.info(f"All results have been saved to: {output_dir}")

        # Ensure all figures are closed
        plt.close('all')

    except Exception as e:
        logger.error(f"Error during main program execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Ensure all figures are closed
        plt.close('all')


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import joblib
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional, Union

# ===== Global Parameter Settings (Easy to Adjust) =====
# Sampling method configuration
SAMPLING_METHOD = "random"  # Sampling method selection: "random" or "equal_width"; default is random sampling

# Input file settings
INPUT_FILE = 'XXX.csv'  # Input filename

# Data column index configuration (soft-coded)
ID_COL_INDEX = 0  # ID column index
TARGET_COL_INDEX = 1  # Target variable column index
FEATURE_START_INDEX = 2  # Feature start column index
FEATURE_END_INDEX = 16  # Feature end index (inclusive)

# Equal-width sampling configuration
BINS_COUNT = 5  # Number of bins for equal-width sampling

# Test set ratio configuration
TEST_SIZE = 0.1  # Test set ratio

# Hyperparameter search control
ENABLE_RANDOM_SEARCH = True  # Control whether to execute random search optimization  True  False

# SVM parameter configuration - used as default parameters and will be overridden by random search
SVM_KERNEL = 'rbf'  # Kernel function type: 'linear', 'poly', 'rbf', 'sigmoid'
SVM_C = 1.0  # Regularization parameter
SVM_EPSILON = 0.1  # Epsilon parameter
SVM_GAMMA = 'scale'  # kernel coefficient: 'scale', 'auto' or float
SVM_DEGREE = 5  # Degree of the polynomial kernel (only used when kernel='poly')

# Feature processing configuration
FEATURE_STANDARDIZATION = True  # Whether to standardize features
USE_ROBUST_SCALER = None  # Whether to use RobustScaler instead of StandardScaler (more robust to outliers)

# Hyperparameter search configuration
RANDOM_SEARCH_ITER = 100  # Number of random search iterations
RANDOM_SEARCH_CV = 3  # Number of internal cross-validation folds for random search

# Cross-validation configuration
CV_FOLDS = 5  # Number of cross-validation folds

# Computing resource configuration
NUM_CPU_CORES = -1  # Number of CPU cores to use; -1 means using all available cores

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
    with open(os.path.join(output_dir, "run_parameter_configuration.txt"), "w", encoding="utf-8") as f:
        f.write(f"Run time: {timestamp}\n\n")
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
            f.write("==== Equal-Width Sampling Parameters ====\n")
            f.write(f"Number of bins: {BINS_COUNT}\n\n")

        f.write("==== Support Vector Machine Regression Parameters ====\n")
        f.write(f"Feature standardization: {FEATURE_STANDARDIZATION}\n")
        f.write(f"Use RobustScaler: {USE_ROBUST_SCALER}\n")
        f.write(f"Kernel function type: {SVM_KERNEL}\n")
        f.write(f"Regularization parameter C: {SVM_C}\n")
        f.write(f"Epsilon: {SVM_EPSILON}\n")
        f.write(f"Gamma: {SVM_GAMMA}\n")
        if SVM_KERNEL == 'poly':
            f.write(f"Polynomial degree: {SVM_DEGREE}\n\n")

        f.write("==== Hyperparameter Search Configuration ====\n")
        f.write(f"Whether to execute random search: {ENABLE_RANDOM_SEARCH}\n")
        f.write(f"Number of random search iterations: {RANDOM_SEARCH_ITER}\n")
        f.write(f"Number of random search cross-validation folds: {RANDOM_SEARCH_CV}\n\n")

        f.write("==== Sampling Method Parameters ====\n")
        f.write(f"Test set ratio: {TEST_SIZE}\n")
        f.write(f"Number of cross-validation folds: {CV_FOLDS}\n\n")

        f.write("==== CPU Configuration ====\n")
        f.write(f"Number of CPU cores: {NUM_CPU_CORES}\n\n")

    logger.info(f"Run parameter configuration has been saved to: {os.path.join(output_dir, 'run_parameter_configuration.txt')}")


def create_output_dir() -> str:
    """
    Create output directory

    Returns:
    - Output directory path
    """
    # Create main folder
    main_dir = "SVM_Sampling_Academic_Results"
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
                    random_state: int = 42) -> Tuple:
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
                                    bins: int = 6, random_state: int = 42) -> Tuple:
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


def standardize_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Standardize features

    Parameters:
    - X_train: Training set features
    - X_test: Test set features

    Returns:
    - Standardized training set features
    - Standardized test set features
    - Scaler used
    """
    if not FEATURE_STANDARDIZATION:
        logger.info("Feature standardization is disabled, skipping standardization step")
        return X_train, X_test, None

    # Select an appropriate scaler
    if USE_ROBUST_SCALER:
        scaler = RobustScaler()  # Use RobustScaler, more robust to outliers
        logger.info("Using RobustScaler for feature standardization (more robust to outliers)")
    else:
        scaler = StandardScaler()  # Use standard StandardScaler
        logger.info("Using StandardScaler for feature standardization")

    # Apply standardization
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrame to preserve column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Output statistics before and after standardization
    logger.info(
        f"Training set statistics before standardization: \n{X_train.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)}")
    logger.info(
        f"Training set statistics after standardization: \n{X_train_scaled_df.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(3)}")

    return X_train_scaled_df, X_test_scaled_df, scaler


def optimize_svm_params(X_train: pd.DataFrame, y_train: pd.Series,
                        n_iter: int = RANDOM_SEARCH_ITER,
                        cv: int = RANDOM_SEARCH_CV) -> Dict:
    """
    Optimize SVM model hyperparameters using random search

    Parameters:
    - X_train: Training set features
    - y_train: Training set target variable
    - n_iter: Number of random search iterations
    - cv: Number of cross-validation folds

    Returns:
    - Best parameter dictionary
    """
    logger.info(f"\nStarting hyperparameter random search (iterations: {n_iter}, CV folds: {cv})...")

    # Define hyperparameter search space
    param_dist = {
        'kernel': ['linear', 'rbf', 'sigmoid'], #'poly',
        'C': np.logspace(-3, 3, 7),  # Log-uniform distribution [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        'epsilon': np.logspace(-3, 0, 4),  # [0.001, 0.01, 0.1, 1.0]
        'gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5)),  # ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
        'degree': [2, 3, 4]  # Degree of the polynomial kernel
    }

    # Create base SVR model
    base_model = SVR()

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=NUM_CPU_CORES,
        verbose=1,
        random_state=42,
        return_train_score=True
    )

    # Execute random search
    random_search.fit(X_train, y_train)

    # Get best parameters and score
    best_params = random_search.best_params_
    best_score = -random_search.best_score_  # Convert negative MSE to MSE
    best_rmse = np.sqrt(best_score)

    logger.info(f"\nHyperparameter random search completed!")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best cross-validation RMSE: {best_rmse:.4f}")

    # Record all attempted parameters and results
    cv_results = pd.DataFrame(random_search.cv_results_)
    top_n = 5  # Show the top 5 results

    logger.info(f"\nTop {top_n} hyperparameter combinations:")
    for i in range(min(top_n, len(cv_results))):
        params = {k: cv_results.iloc[i][f'param_{k}'] for k in param_dist.keys()}
        rmse = np.sqrt(-cv_results.iloc[i]['mean_test_score'])
        logger.info(f"Rank {i + 1}: RMSE={rmse:.4f}, parameters={params}")

    return best_params


def train_and_evaluate_svm(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           best_params: Dict = None,
                           cv_folds: int = 5, method_name: str = "Support Vector Regression Model") -> Dict:
    """
    Train and evaluate the Support Vector Machine regression model
    - Use n-fold cross-validation
    - Calculate R², MAE, RMSE
    - Return the trained model and evaluation metrics

    Parameters:
    - X_train: Training set features
    - X_test: Test set features
    - y_train: Training set target variable
    - y_test: Test set target variable
    - best_params: Best parameter dictionary (if None, default parameters are used)
    - cv_folds: Number of cross-validation folds
    - method_name: Method name (for logging output)

    Returns:
    - Dictionary containing the model, evaluation metrics, and prediction results
    """
    # If no best parameters are provided, use default parameters
    if best_params is None:
        svm_model = SVR(
            kernel=SVM_KERNEL,
            C=SVM_C,
            epsilon=SVM_EPSILON,
            gamma=SVM_GAMMA,
            degree=SVM_DEGREE if SVM_KERNEL == 'poly' else 3
        )
        logger.info(f"Using default SVM parameters: kernel={SVM_KERNEL}, C={SVM_C}, epsilon={SVM_EPSILON}, "
                    f"gamma={SVM_GAMMA}, degree={SVM_DEGREE if SVM_KERNEL == 'poly' else 3}")
    else:
        svm_model = SVR(**best_params)
        logger.info(f"Using optimized SVM parameters: {best_params}")

    logger.info(f"\nStarting {cv_folds}-fold cross-validation [{method_name}]...")

    # Use sklearn's built-in cross-validation functionality
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_results = cross_validate(
        svm_model, X_train, y_train,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=False,
        n_jobs=NUM_CPU_CORES
    )

    # Extract cross-validation results
    cv_r2_scores = cv_results['test_r2']
    cv_mae_scores = -cv_results['test_neg_mean_absolute_error']  # Convert to positive values
    cv_rmse_scores = -cv_results['test_neg_root_mean_squared_error']  # Convert to positive values

    # Print results for each fold
    for fold in range(cv_folds):
        logger.info(f"  Fold {fold + 1}: R² = {cv_r2_scores[fold]:.4f}, "
                    f"MAE = {cv_mae_scores[fold]:.4f}, RMSE = {cv_rmse_scores[fold]:.4f}")

    # Output average cross-validation results
    logger.info(f"\n{method_name} - average cross-validation results:")
    logger.info(f"Average validation R²: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")
    logger.info(f"Average validation MAE: {np.mean(cv_mae_scores):.4f} (±{np.std(cv_mae_scores):.4f})")
    logger.info(f"Average validation RMSE: {np.mean(cv_rmse_scores):.4f} (±{np.std(cv_rmse_scores):.4f})")

    # Train final model on the full training set
    if best_params is None:
        final_model = SVR(
            kernel=SVM_KERNEL,
            C=SVM_C,
            epsilon=SVM_EPSILON,
            gamma=SVM_GAMMA,
            degree=SVM_DEGREE if SVM_KERNEL == 'poly' else 3
        )
    else:
        final_model = SVR(**best_params)

    final_model.fit(X_train, y_train)

    # Calculate training set metrics
    y_train_pred = final_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    logger.info(f"\n{method_name} - training set evaluation results:")
    logger.info(f"R²: {train_r2:.4f}")
    logger.info(f"MAE: {train_mae:.4f}")
    logger.info(f"RMSE: {train_rmse:.4f}")

    # Evaluate on the test set
    y_test_pred = final_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info(f"\n{method_name} - test set evaluation results:")
    logger.info(f"R²: {test_r2:.4f}")
    logger.info(f"MAE: {test_mae:.4f}")
    logger.info(f"RMSE: {test_rmse:.4f}")

    # Record final SVM parameters
    logger.info(f"\nFinal SVM parameters:")
    for param, value in final_model.get_params().items():
        logger.info(f"{param}: {value}")

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
        'true': {
            'train': y_train,
            'test': y_test
        }
    }

    return results


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
            tick_labels.append('')  # Do not display first and last labels
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

    # Convert data type name
    if data_type.lower() == "training":
        data_type_zh = "Training_Set"
    elif data_type.lower() == "test":
        data_type_zh = "Test_Set"
    else:
        data_type_zh = data_type

    filename = os.path.join(output_dir, f"Prediction_Results_{method_name_clean}_{data_type_zh}.csv")

    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    logger.info(f"Prediction results have been saved to: {filename}")

    return filename


def main():
    """Main function - execute the complete Support Vector Machine regression feature sampling workflow"""
    try:
        # Create output directory
        output_dir = create_output_dir()

        # Set up logging
        setup_logging(output_dir)

        # Save run configuration parameters
        save_configuration_parameters(output_dir)

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
                    X, y, test_size=TEST_SIZE, bins=BINS_COUNT, random_state=42
                )
            except Exception as e:
                logger.error(f"Equal-width binning sampling failed: {str(e)}")
                logger.info("Trying random sampling as a fallback method...")
                X_train, X_test, y_train, y_test = random_sampling(
                    X, y, test_size=TEST_SIZE, random_state=42
                )
        else:  # Use random sampling by default
            logger.info(f"Using random sampling (test_size={TEST_SIZE})")
            logger.info("=" * 50)
            X_train, X_test, y_train, y_test = random_sampling(
                X, y, test_size=TEST_SIZE, random_state=42
            )

        # Standardize features
        logger.info("\n" + "=" * 50)
        logger.info("Feature standardization")
        X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
        logger.info("=" * 50)

        # Hyperparameter optimization
        best_params = None
        if ENABLE_RANDOM_SEARCH:
            logger.info("\n" + "=" * 50)
            logger.info("Starting SVM hyperparameter random search optimization")
            best_params = optimize_svm_params(
                X_train_scaled, y_train,
                n_iter=RANDOM_SEARCH_ITER,
                cv=RANDOM_SEARCH_CV
            )
            logger.info("=" * 50)
        else:
            logger.info("\n" + "=" * 50)
            logger.info("Random search is disabled, using default hyperparameters")
            best_params = None
            logger.info("=" * 50)

        # Train Support Vector Machine regression model
        kernel_name = best_params.get('kernel', SVM_KERNEL) if best_params else SVM_KERNEL
        sampling_method_name = "Equal Width Binning" if SAMPLING_METHOD == "equal_width" else "Random Sampling"  # English sampling method name
        method_name = f"Support Vector Regression ({sampling_method_name}, kernel={kernel_name})"  # English method name

        # Set sampling method name for output files
        sampling_method_zh = "Equal_Width_Binning" if SAMPLING_METHOD == "equal_width" else "Random_Sampling"

        if FEATURE_STANDARDIZATION:
            scaler_name = "RobustScaler" if USE_ROBUST_SCALER else "StandardScaler"
            method_name += f", with {scaler_name}"

        svm_results = train_and_evaluate_svm(
            X_train_scaled, X_test_scaled, y_train, y_test,
            best_params=best_params,
            cv_folds=CV_FOLDS, method_name=method_name
        )

        # Export prediction results
        if EXPORT_CSV:
            export_predictions_to_csv(
                y_train, svm_results['pred']['train'],
                f"SVM_{SAMPLING_METHOD}_{kernel_name}", "training", output_dir
            )
            export_predictions_to_csv(
                y_test, svm_results['pred']['test'],
                f"SVM_{SAMPLING_METHOD}_{kernel_name}", "test", output_dir
            )

        # Save model and scaler
        model_filename = os.path.join(output_dir, f"Support_Vector_Machine_Regression_Model_{sampling_method_zh}_{kernel_name}.pkl")
        if scaler is not None:
            scaler_filename = os.path.join(output_dir, f"Feature_Scaler_{sampling_method_zh}.pkl")
            joblib.dump(scaler, scaler_filename)
            logger.info(f"Scaler has been saved to: {scaler_filename}")

        joblib.dump(svm_results['model'], model_filename)
        logger.info(f"Model has been saved to: {model_filename}")

        # Plot scatter plots - pass MAE and RMSE parameters
        plot_academic_scatter(
            y_train, svm_results['pred']['train'],
            f"SVR ({sampling_method_name}, kernel={kernel_name}) - Training Set",
            svm_results['metrics']['train']['r2'],
            svm_results['metrics']['train']['mae'],
            svm_results['metrics']['train']['rmse'],
            filename=os.path.join(output_dir, f"Support_Vector_Machine_Regression_{sampling_method_zh}_Training_Set_Scatter.png")
        )

        plot_academic_scatter(
            y_test, svm_results['pred']['test'],
            f"SVR ({sampling_method_name}) - Test Set",
            svm_results['metrics']['test']['r2'],
            svm_results['metrics']['test']['mae'],
            svm_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Support_Vector_Machine_Regression_{sampling_method_zh}_Test_Set_Scatter.png")
        )

        # Plot side-by-side scatter plot
        plot_side_by_side_scatter(
            y_train, svm_results['pred']['train'],
            y_test, svm_results['pred']['test'],
            f"Support Vector Regression ({sampling_method_name}, kernel={kernel_name})",
            svm_results['metrics']['train']['r2'],
            svm_results['metrics']['test']['r2'],
            svm_results['metrics']['train']['mae'],
            svm_results['metrics']['train']['rmse'],
            svm_results['metrics']['test']['mae'],
            svm_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Support_Vector_Machine_Regression_{sampling_method_zh}_Training_Test_Comparison.png")
        )

        # Create performance summary table
        summary_df = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE'],
            'Training Set': [
                svm_results['metrics']['train']['r2'],
                svm_results['metrics']['train']['mae'],
                svm_results['metrics']['train']['rmse']
            ],
            'Validation Set': [
                svm_results['metrics']['val']['r2'],
                svm_results['metrics']['val']['mae'],
                svm_results['metrics']['val']['rmse']
            ],
            'Test Set': [
                svm_results['metrics']['test']['r2'],
                svm_results['metrics']['test']['mae'],
                svm_results['metrics']['test']['rmse']
            ]
        })

        # Save performance summary table
        summary_filename = os.path.join(output_dir, "Performance_Metrics_Summary.csv")
        summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        logger.info(f"Performance metrics summary has been saved to: {summary_filename}")

        # Save best parameter table
        if best_params:
            params_df = pd.DataFrame([best_params])
            params_filename = os.path.join(output_dir, "Best_Parameters.csv")
            params_df.to_csv(params_filename, index=False, encoding='utf-8-sig')
            logger.info(f"Best parameters have been saved to: {params_filename}")
        else:
            # Save default parameters
            default_params = {
                'kernel': SVM_KERNEL,
                'C': SVM_C,
                'epsilon': SVM_EPSILON,
                'gamma': SVM_GAMMA,
                'degree': SVM_DEGREE if SVM_KERNEL == 'poly' else 3
            }
            params_df = pd.DataFrame([default_params])
            params_filename = os.path.join(output_dir, "Default_Parameters.csv")
            params_df.to_csv(params_filename, index=False, encoding='utf-8-sig')
            logger.info(f"Default parameters have been saved to: {params_filename}")

        # Print summary table
        logger.info("\nPerformance metrics summary:")
        logger.info(summary_df.to_string(index=False))

        logger.info("\n" + "=" * 50)
        logger.info(f"Support Vector Machine regression model - {sampling_method_zh} program execution completed!")
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

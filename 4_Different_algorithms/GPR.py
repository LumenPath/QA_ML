import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, ConstantKernel as C, \
    WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import os
import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import joblib
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional, Union
from scipy.stats import loguniform, uniform

# ===== Global Parameter Settings (Easy to Adjust) =====
# Hyperparameter random search switch
ENABLE_RANDOM_SEARCH = False  # Set to True to enable hyperparameter random search; False uses default parameters

# Random search configuration (only valid when ENABLE_RANDOM_SEARCH=True)
RANDOM_SEARCH_ITERATIONS = 50  # Number of random search iterations
RANDOM_SEARCH_CV_FOLDS = 3  # Number of cross-validation folds in random search

# Sampling method configuration
SAMPLING_METHOD = "random"  # Sampling method selection: "random" or "equal_width"; default is random sampling

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

# Gaussian Process Regression parameter configuration - optimized for data with 304 samples and 14 features
FEATURE_STANDARDIZATION = True  # Whether to standardize features
USE_ROBUST_SCALER = True  # Whether to use RobustScaler instead of StandardScaler (more robust to outliers)
GPR_KERNEL_TYPE = 'rational_quadratic'  # Kernel type: 'rbf', 'matern', 'rational_quadratic', 'exp_sine_squared', 'rbf_white'
GPR_ALPHA = 0.005  # Noise parameter alpha (smaller values can reduce the risk of overfitting)
GPR_NORMALIZE_Y = False  # Whether to normalize the target variable
GPR_N_RESTARTS_OPTIMIZER = 0  # Number of optimizer restarts (suitable for small- to medium-sized datasets)

# RBF kernel parameters
GPR_RBF_LENGTH_SCALE = 1.0  # Initial length scale for the RBF kernel
GPR_RBF_LENGTH_SCALE_BOUNDS = (1e-2, 1e2)  # Bounds for the length scale

# Parameters only for the Matern kernel
GPR_MATERN_NU = 2.5  # Smoothness parameter for the Matern kernel (0.5, 1.5, 2.5)

# Noise kernel parameters (for the rbf_white combined kernel)
GPR_NOISE_LEVEL = 0.1  # Initial white-noise level
GPR_NOISE_LEVEL_BOUNDS = (1e-5, 1.0)  # Noise-level bounds

# Cross-validation configuration
CV_FOLDS = 5  # Number of cross-validation folds

# Computing resource configuration
NUM_CPU_CORES = -1  # Number of CPU cores to use; -1 means using all available cores

# Figure output quality settings
OUTPUT_DPI = 600  # Figure output DPI
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

        f.write("==== Hyperparameter Random Search Configuration ====\n")
        f.write(f"Whether to enable hyperparameter random search: {ENABLE_RANDOM_SEARCH}\n")
        if ENABLE_RANDOM_SEARCH:
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
            f.write("==== Equal-Width Sampling Parameters ====\n")
            f.write(f"Number of bins: {BINS_COUNT}\n\n")

        f.write("==== Gaussian Process Regression Parameters ====\n")
        f.write(f"Feature standardization: {FEATURE_STANDARDIZATION}\n")
        f.write(f"Use RobustScaler: {USE_ROBUST_SCALER}\n")

        if not ENABLE_RANDOM_SEARCH:
            f.write(f"Kernel type: {GPR_KERNEL_TYPE}\n")
            f.write(f"Noise parameter alpha: {GPR_ALPHA}\n")
            f.write(f"Normalize target variable: {GPR_NORMALIZE_Y}\n")
            f.write(f"Number of optimizer restarts: {GPR_N_RESTARTS_OPTIMIZER}\n")
            if GPR_KERNEL_TYPE == 'rbf' or GPR_KERNEL_TYPE == 'rbf_white':
                f.write(f"Initial length scale for RBF kernel: {GPR_RBF_LENGTH_SCALE}\n")
                f.write(f"Length-scale bounds for RBF kernel: {GPR_RBF_LENGTH_SCALE_BOUNDS}\n")
            if GPR_KERNEL_TYPE == 'rbf_white':
                f.write(f"Initial white-noise level: {GPR_NOISE_LEVEL}\n")
                f.write(f"White-noise level bounds: {GPR_NOISE_LEVEL_BOUNDS}\n")
            if GPR_KERNEL_TYPE == 'matern':
                f.write(f"Smoothness parameter Nu for Matern kernel: {GPR_MATERN_NU}\n\n")

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
    main_dir = "Gaussian_Process_Regression_Model_Results"
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


def create_gpr_kernel():
    """
    Create the kernel function for Gaussian Process Regression according to the configuration

    Optimized selection for a dataset with 304 samples and 14 features

    Returns:
    - Configured kernel function
    """
    # Create different types of kernel functions
    if GPR_KERNEL_TYPE == 'rbf':
        # Simple RBF kernel - suitable for most small- to medium-sized datasets
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=GPR_RBF_LENGTH_SCALE,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS
        )
    elif GPR_KERNEL_TYPE == 'rbf_white':
        # RBF + WhiteKernel combination - explicitly models noise and is more robust for noisy data
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=GPR_RBF_LENGTH_SCALE,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS
        ) + WhiteKernel(
            noise_level=GPR_NOISE_LEVEL,
            noise_level_bounds=GPR_NOISE_LEVEL_BOUNDS
        )
    elif GPR_KERNEL_TYPE == 'matern':
        # Matern kernel - more flexible than RBF and better suited for non-smooth data
        kernel = C(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=GPR_RBF_LENGTH_SCALE,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS,
            nu=GPR_MATERN_NU
        )
    elif GPR_KERNEL_TYPE == 'rational_quadratic':
        # RationalQuadratic kernel - can be viewed as a sum of RBF kernels with different length scales
        kernel = C(1.0, constant_value_bounds="fixed") * RationalQuadratic(
            length_scale=GPR_RBF_LENGTH_SCALE,
            alpha=0.1,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS
        )
    elif GPR_KERNEL_TYPE == 'exp_sine_squared':
        # ExpSineSquared kernel - suitable for periodic data
        kernel = C(1.0, constant_value_bounds="fixed") * ExpSineSquared(
            length_scale=GPR_RBF_LENGTH_SCALE,
            periodicity=1.0,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS
        )
    else:
        # Use RBF + WhiteKernel combined kernel by default
        logger.warning(f"Unknown kernel type '{GPR_KERNEL_TYPE}', using the default RBF+White kernel")
        kernel = C(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=GPR_RBF_LENGTH_SCALE,
            length_scale_bounds=GPR_RBF_LENGTH_SCALE_BOUNDS
        ) + WhiteKernel(
            noise_level=GPR_NOISE_LEVEL,
            noise_level_bounds=GPR_NOISE_LEVEL_BOUNDS
        )

    return kernel


def create_gpr_pipeline():
    """
    Create the Pipeline for Gaussian Process Regression, including feature standardization and kernel function

    Returns:
    - Configured Pipeline
    """
    # Create kernel function
    kernel = create_gpr_kernel()

    # Create Gaussian Process Regression model
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GPR_ALPHA,
        normalize_y=GPR_NORMALIZE_Y,
        n_restarts_optimizer=GPR_N_RESTARTS_OPTIMIZER,
        random_state=42
    )

    # Whether to add feature standardization
    if FEATURE_STANDARDIZATION:
        # Select the appropriate scaler
        if USE_ROBUST_SCALER:
            scaler = RobustScaler()  # Use RobustScaler, which is more robust to outliers
            logger.info("Using RobustScaler for feature standardization (more robust to outliers)")
        else:
            scaler = StandardScaler()  # Use standard StandardScaler
            logger.info("Using StandardScaler for feature standardization")

        # Build Pipeline with standardization
        pipeline = Pipeline([
            ('scaler', scaler),
            ('gpr', gpr)
        ])
    else:
        # Do not use standardization; directly return the GPR model
        pipeline = gpr
        logger.info("Created Gaussian Process Regression model (without feature standardization)")

    return pipeline


def create_gpr_random_search_pipeline():
    """
    Create a Gaussian Process Regression Pipeline for random search

    Returns:
    - Configured Pipeline for random search
    """
    # Create base GPR model (parameters will be set by RandomizedSearchCV)
    gpr = GaussianProcessRegressor(random_state=42)

    # Whether to add feature standardization
    if FEATURE_STANDARDIZATION:
        # Select the appropriate scaler
        if USE_ROBUST_SCALER:
            scaler = RobustScaler()
            logger.info("Random search uses RobustScaler for feature standardization")
        else:
            scaler = StandardScaler()
            logger.info("Random search uses StandardScaler for feature standardization")

        # Build Pipeline with standardization
        pipeline = Pipeline([
            ('scaler', scaler),
            ('gpr', gpr)
        ])
    else:
        # Do not use standardization; directly return the GPR model
        pipeline = gpr
        logger.info("Random search uses Gaussian Process Regression model (without feature standardization)")

    return pipeline


def create_param_grid_for_random_search():
    """
    Create parameter grid for random search, modified to avoid overfitting

    Returns:
    - Parameter grid dictionary
    """
    # Basic parameters - adjust alpha lower bound to avoid extremely small values
    param_grid = {
        'normalize_y': [True, False],  # Prefer True to reduce overfitting
        'alpha': loguniform(1e-6, 1.0),  # Increase alpha lower bound, 1e-10 -> 1e-6
        'n_restarts_optimizer': [0, 3, 5, 10]
    }

    # Kernel parameters
    # Create different types of kernel functions, each with its own hyperparameter range
    kernel_options = []

    # RBF kernel - limit upper bound of length_scale to prevent overfitting
    rbf_length_scale = loguniform(1e-2, 1e2)  # Lower upper bound, 1e3 -> 1e2
    kernel_options.append(('rbf', C(1.0, constant_value_bounds="fixed") *
                           RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))))

    # RBF + White kernel - add mandatory white-noise kernel to reduce overfitting
    rbf_white_length_scale = loguniform(1e-2, 1e2)
    rbf_white_noise_level = loguniform(1e-4, 1.0)  # Increase lower bound, 1e-10 -> 1e-4
    kernel_options.append(('rbf_white', C(1.0, constant_value_bounds="fixed") *
                           RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                           WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 1.0))))

    # Matern kernel (multiple nu values) - increase minimum nu to improve smoothness
    for nu in [1.5, 2.5]:  # Remove nu=0.5, the least smooth option, which is prone to overfitting
        matern_length_scale = loguniform(1e-2, 1e2)  # Limit upper bound
        kernel_options.append((f'matern_{nu}', C(1.0, constant_value_bounds="fixed") *
                               Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=nu)))

    # RationalQuadratic kernel
    rq_length_scale = loguniform(1e-2, 1e2)
    rq_alpha = loguniform(1e-2, 10.0)
    kernel_options.append(('rational_quadratic', C(1.0, constant_value_bounds="fixed") *
                           RationalQuadratic(length_scale=1.0, alpha=1.0,
                                             length_scale_bounds=(1e-2, 1e2))))

    # Add kernel combinations with fixed noise - forcibly add white noise
    kernel_options.append(('matern_white', C(1.0, constant_value_bounds="fixed") *
                           Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) +
                           WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-3, 1.0))))

    # If using Pipeline, add prefix to parameter names
    if FEATURE_STANDARDIZATION:
        param_grid = {'gpr__' + k: v for k, v in param_grid.items()}
        param_grid['gpr__kernel'] = [k[1] for k in kernel_options]
    else:
        param_grid['kernel'] = [k[1] for k in kernel_options]

    return param_grid


def extract_kernel_type(kernel_str: str) -> str:
    """
    Extract kernel type from the kernel string

    Parameters:
    - kernel_str: String representation of the kernel function

    Returns:
    - Short description of kernel type
    """
    if 'RBF' in kernel_str and 'WhiteKernel' in kernel_str:
        return 'RBF+White'
    elif 'RBF' in kernel_str:
        return 'RBF'
    elif 'Matern' in kernel_str:
        nu_value = '?'
        if 'nu=0.5' in kernel_str:
            nu_value = '0.5'
        elif 'nu=1.5' in kernel_str:
            nu_value = '1.5'
        elif 'nu=2.5' in kernel_str:
            nu_value = '2.5'
        return f'Matern(nu={nu_value})'
    elif 'RationalQuadratic' in kernel_str:
        return 'RationalQuadratic'
    elif 'ExpSineSquared' in kernel_str:
        return 'ExpSineSquared'
    else:
        return 'Other'


def random_search_hyperparameters(X_train, y_train):
    """
    Perform hyperparameter random search for the Gaussian Process Regression model with additional anti-overfitting measures

    Parameters:
    - X_train: Training set features
    - y_train: Training set labels

    Returns:
    - Best model and random search results
    """
    logger.info("\n" + "=" * 50)
    logger.info("Starting hyperparameter random search...")
    logger.info("=" * 50)

    # Create Pipeline for random search
    pipeline = create_gpr_random_search_pipeline()

    # Create parameter grid - use the modified parameter space
    param_grid = create_param_grid_for_random_search()

    # Create random search object - increasing cross-validation folds can reduce overfitting
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=RANDOM_SEARCH_ITERATIONS,
        cv=max(RANDOM_SEARCH_CV_FOLDS, 5),  # Ensure at least 5-fold cross-validation
        scoring='neg_mean_squared_error',
        n_jobs=NUM_CPU_CORES,
        verbose=2,
        random_state=42
    )

    # Execute random search
    random_search.fit(X_train, y_train)

    # Record best parameters
    logger.info("\nBest hyperparameters:")
    for param, value in random_search.best_params_.items():
        if param.endswith('kernel'):
            logger.info(f"{param}: {str(value)}")
        else:
            logger.info(f"{param}: {value}")

    logger.info(f"Best MSE score: {-random_search.best_score_:.6f}")
    logger.info(f"Best RMSE score: {np.sqrt(-random_search.best_score_):.6f}")

    # Get some additional parameters of the best model to confirm that the configuration is reasonable
    best_model = random_search.best_estimator_
    if hasattr(best_model, 'named_steps') and 'gpr' in best_model.named_steps:
        gpr = best_model.named_steps['gpr']
    else:
        gpr = best_model

    # Check and record white-noise level
    kernel_params = gpr.kernel_.get_params()
    has_white_noise = False
    for param_name, param_value in kernel_params.items():
        if 'WhiteKernel' in str(param_value.__class__):
            has_white_noise = True
            logger.info(f"Model includes a white-noise component: {param_value}")

    if not has_white_noise:
        logger.warning("The best model does not include a white-noise component, which may lead to overfitting")

    # Return best model and random search results
    return random_search.best_estimator_, random_search


def plot_random_search_results(random_search_results, output_dir: str) -> None:
    """
    Plot visualization charts for random search results

    Parameters:
    - random_search_results: Results from RandomizedSearchCV
    - output_dir: Output directory path
    """
    if not ENABLE_RANDOM_SEARCH or random_search_results is None:
        return

    # Extract all parameter configurations and corresponding scores
    results = pd.DataFrame(random_search_results.cv_results_)

    # 1. Plot hyperparameter importance chart
    plt.figure(figsize=(12, 8))
    # Convert scores (negative MSE to positive RMSE for easier interpretation)
    results['rmse'] = np.sqrt(-results['mean_test_score'])

    # Find several of the most important parameters (parameters with variation)
    param_names = [name for name in results.columns if name.startswith('param_')]
    important_params = []

    for param in param_names:
        # Skip the kernel parameter; it is an object and cannot directly use nunique()
        if 'kernel' in param:
            continue

        try:
            # Try converting parameter values to strings to avoid unhashable type issues
            param_values = results[param].astype(str)
            if len(set(param_values)) > 1:  # Use set instead of nunique()
                param_short = param.replace('param_', '').replace('gpr__', '')
                important_params.append((param, param_short))
        except:
            # Ignore parameters that cannot be processed
            logger.warning(f"Skipping parameter {param} - this type cannot be processed")

    # If there is a kernel parameter, it needs special handling - record it but do not compare using nunique
    kernel_param = None
    for param in param_names:
        if 'kernel' in param:
            kernel_param = param
            break

    # Plot scatter plots for the top N most important continuous parameters
    n_plots = min(4, len(important_params))
    if n_plots == 0:
        logger.info("No continuous parameters available for visualization")
        # Create a simple RMSE distribution plot instead
        plt.figure(figsize=(10, 6))
        plt.hist(results['rmse'], bins=10, alpha=0.7, color='#3498db')
        plt.xlabel('RMSE', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of RMSE values across trials', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'RMSE_Distribution_Plot.png'), dpi=OUTPUT_DPI)
        plt.close()
    else:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

        if n_plots == 1:
            axes = [axes]  # Ensure axes is a list

        for i, (param, param_short) in enumerate(important_params[:n_plots]):
            ax = axes[i]
            try:
                # Try converting parameter values to numeric type
                param_values = results[param].astype(float)

                # Fix division-by-zero issue: check whether the minimum value is 0
                if param_values.min() == 0:
                    # If the minimum value is 0, use linear scale
                    ax.plot(param_values, results['rmse'], 'o', alpha=0.6)
                else:
                    # Use logarithmic axis for log-scale parameters
                    if param_values.max() / param_values.min() > 100:
                        ax.semilogx(param_values, results['rmse'], 'o', alpha=0.6)
                    else:
                        ax.plot(param_values, results['rmse'], 'o', alpha=0.6)
            except:
                # If it cannot be converted to numeric, treat it as categorical
                # First convert values to strings to ensure they are comparable
                str_values = results[param].astype(str)
                unique_values = sorted(list(set(str_values)))
                value_to_index = {val: idx for idx, val in enumerate(unique_values)}

                # Map string values to indices
                indices = [value_to_index[val] for val in str_values]

                # Calculate average RMSE by group
                groups = {}
                for idx, rmse in zip(indices, results['rmse']):
                    if idx not in groups:
                        groups[idx] = []
                    groups[idx].append(rmse)

                avg_rmse = [np.mean(groups[idx]) for idx in sorted(groups.keys())]

                # Plot bar chart
                ax.bar(range(len(unique_values)), avg_rmse, alpha=0.7)
                ax.set_xticks(range(len(unique_values)))
                ax.set_xticklabels([val[:10] + '...' if len(val) > 10 else val for val in unique_values])

            ax.set_xlabel(param_short, fontsize=14)
            ax.set_ylabel('RMSE', fontsize=14)
            ax.set_title(f'RMSE vs {param_short}', fontsize=16)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, 'Hyperparameter_Importance_Analysis.png'), dpi=OUTPUT_DPI)
        plt.close(fig)

    # 2. Plot comparison between the best and worst models
    best_idx = results['rank_test_score'].argmin()
    worst_idx = results['rank_test_score'].argmax()

    best_params = {}
    worst_params = {}

    for param in param_names:
        param_short = param.replace('param_', '').replace('gpr__', '')
        if 'kernel' in param:
            # For the kernel parameter, only store the type name
            try:
                best_kernel = str(results.iloc[best_idx][param])
                worst_kernel = str(results.iloc[worst_idx][param])

                # Extract kernel type name
                best_kernel_type = extract_kernel_type(best_kernel)
                worst_kernel_type = extract_kernel_type(worst_kernel)

                best_params[param_short] = best_kernel_type
                worst_params[param_short] = worst_kernel_type
            except:
                logger.warning(f"Unable to process kernel parameter comparison: {param}")
        else:
            try:
                best_params[param_short] = results.iloc[best_idx][param]
                worst_params[param_short] = results.iloc[worst_idx][param]
            except:
                # There may be parameters that cannot be processed
                logger.warning(f"Unable to process parameter comparison: {param}")

    # Create parameter comparison table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    table_data = []
    table_columns = ['Parameter', 'Best Value', 'Worst Value']

    # Get all comparable parameters
    comparable_params = set(best_params.keys()) & set(worst_params.keys())

    for param in comparable_params:
        # Format values (if numeric)
        best_val = best_params[param]
        worst_val = worst_params[param]

        try:
            if isinstance(best_val, (int, float)) and not isinstance(best_val, bool):
                if float(best_val) < 0.01 or float(best_val) > 100:
                    best_val = f"{float(best_val):.2e}"
                else:
                    best_val = f"{float(best_val):.4f}"

            if isinstance(worst_val, (int, float)) and not isinstance(worst_val, bool):
                if float(worst_val) < 0.01 or float(worst_val) > 100:
                    worst_val = f"{float(worst_val):.2e}"
                else:
                    worst_val = f"{float(worst_val):.4f}"
        except:
            # If not numeric, keep as is
            pass

        table_data.append([param, str(best_val), str(worst_val)])

    # Add performance metric
    table_data.append(['RMSE', f"{np.sqrt(-results.iloc[best_idx]['mean_test_score']):.4f}",
                       f"{np.sqrt(-results.iloc[worst_idx]['mean_test_score']):.4f}"])

    # Create table
    table = ax.table(cellText=table_data, colLabels=table_columns, cellLoc='center',
                     loc='center', colWidths=[0.3, 0.35, 0.35])

    # Set table style
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Set title
    plt.title('Best vs Worst Hyperparameters Comparison', fontsize=16, pad=20)
    plt.tight_layout()

    # Save chart
    plt.savefig(os.path.join(output_dir, 'Best_vs_Worst_Hyperparameter_Comparison.png'), dpi=OUTPUT_DPI, bbox_inches='tight')
    plt.close(fig)

    # 3. Save complete random search results to CSV
    csv_path = os.path.join(output_dir, 'Random_Search_Results.csv')

    # Create an exportable results DataFrame
    export_results = pd.DataFrame()
    export_results['iteration'] = range(1, len(results) + 1)
    export_results['rmse'] = np.sqrt(-results['mean_test_score'])
    export_results['rank'] = results['rank_test_score']

    # Add parameter columns, but the kernel function requires special handling
    for param in param_names:
        param_short = param.replace('param_', '').replace('gpr__', '')
        if 'kernel' not in param:
            export_results[param_short] = results[param]
        else:
            # For kernel functions, only save the type name
            kernel_types = []
            for k in results[param]:
                kernel_types.append(extract_kernel_type(str(k)))
            export_results['kernel_type'] = kernel_types

    # Save to CSV
    export_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"Random search results have been saved to: {csv_path}")


def train_and_evaluate_gpr(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           cv_folds: int = 5, method_name: str = "Gaussian Process Regression Model") -> Dict:
    """
    Train and evaluate the Gaussian Process Regression model
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

    Returns:
    - Dictionary containing the model, evaluation metrics, and prediction results
    """
    # Determine which model training method to use (random search or default parameters)
    if ENABLE_RANDOM_SEARCH:
        logger.info(f"\nUsing hyperparameter random search to optimize the Gaussian Process Regression model...")
        # Perform random search
        final_pipeline, random_search_results = random_search_hyperparameters(X_train, y_train)

        # Get parameter information for logging
        if hasattr(final_pipeline, 'named_steps') and 'gpr' in final_pipeline.named_steps:
            gpr_model = final_pipeline.named_steps['gpr']
            logger.info(f"\nOptimal kernel parameters:")
            logger.info(f"{gpr_model.kernel_}")
        else:
            # Direct GPR object instead of Pipeline
            logger.info(f"\nOptimal kernel parameters:")
            logger.info(f"{final_pipeline.kernel_}")

        # Update method name for result evaluation
        method_name += " (with Random Search)"
    else:
        # Use default parameters to create GPR Pipeline
        final_pipeline = create_gpr_pipeline()
        logger.info(f"\nStarting {cv_folds}-fold cross-validation [{method_name}]...")

    # Use sklearn's built-in cross-validation function to evaluate the model
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_results = cross_validate(
        final_pipeline, X_train, y_train,
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

    # If random search is not used, retrain the model on the full training set
    if not ENABLE_RANDOM_SEARCH:
        final_pipeline.fit(X_train, y_train)

    # Calculate training set metrics
    y_train_pred = final_pipeline.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    logger.info(f"\n{method_name} - training set evaluation results:")
    logger.info(f"R²: {train_r2:.4f}")
    logger.info(f"MAE: {train_mae:.4f}")
    logger.info(f"RMSE: {train_rmse:.4f}")

    # Evaluate on the test set
    y_test_pred = final_pipeline.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info(f"\n{method_name} - test set evaluation results:")
    logger.info(f"R²: {test_r2:.4f}")
    logger.info(f"MAE: {test_mae:.4f}")
    logger.info(f"RMSE: {test_rmse:.4f}")

    # Create results dictionary
    results = {
        'model': final_pipeline,
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

    # If random search was used, add random search results
    if ENABLE_RANDOM_SEARCH:
        results['random_search'] = random_search_results

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

    # Convert data type to output name
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
    """Main function - execute the complete Gaussian Process Regression feature sampling workflow"""
    try:
        # Create output directory
        output_dir = create_output_dir()

        # Set up logging
        setup_logging(output_dir)

        # Save run configuration parameters
        save_configuration_parameters(output_dir)

        # Display hyperparameter random search status
        if ENABLE_RANDOM_SEARCH:
            logger.info("\n" + "=" * 80)
            logger.info(f"Hyperparameter random search is enabled and will execute {RANDOM_SEARCH_ITERATIONS} iterations")
            logger.info("=" * 80)
        else:
            logger.info("\n" + "=" * 80)
            logger.info("Hyperparameter random search is not enabled; default parameters will be used")
            logger.info("=" * 80)

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

        # Train Gaussian Process Regression model
        if ENABLE_RANDOM_SEARCH:
            # When using random search, do not predetermine the kernel type
            kernel_name = "Optimized"
        else:
            kernel_name = "RBF+WhiteKernel" if GPR_KERNEL_TYPE == "rbf_white" else GPR_KERNEL_TYPE.upper()

        sampling_method_name = "Equal Width Binning" if SAMPLING_METHOD == "equal_width" else "Random Sampling"  # English sampling method name

        if ENABLE_RANDOM_SEARCH:
            method_name = f"Gaussian Process Regression ({sampling_method_name}, Random Search)"  # English method name
        else:
            method_name = f"Gaussian Process Regression ({sampling_method_name}, kernel={kernel_name})"  # English method name

        # Set sampling method name for output files
        sampling_method_zh = "Equal_Width_Binning" if SAMPLING_METHOD == "equal_width" else "Random_Sampling"

        if FEATURE_STANDARDIZATION and not ENABLE_RANDOM_SEARCH:
            scaler_name = "RobustScaler" if USE_ROBUST_SCALER else "StandardScaler"
            method_name += f", with {scaler_name}"

        # Train model
        gpr_results = train_and_evaluate_gpr(
            X_train, X_test, y_train, y_test,
            cv_folds=CV_FOLDS, method_name=method_name
        )

        # If random search was used, plot random search results
        if ENABLE_RANDOM_SEARCH and 'random_search' in gpr_results:
            plot_random_search_results(gpr_results['random_search'], output_dir)

        # Export prediction results
        if EXPORT_CSV:
            method_suffix = "RandomSearch" if ENABLE_RANDOM_SEARCH else f"{GPR_KERNEL_TYPE}"
            export_predictions_to_csv(
                y_train, gpr_results['pred']['train'],
                f"GPR_{SAMPLING_METHOD}_{method_suffix}", "training", output_dir
            )
            export_predictions_to_csv(
                y_test, gpr_results['pred']['test'],
                f"GPR_{SAMPLING_METHOD}_{method_suffix}", "test", output_dir
            )

        # Save model
        model_suffix = "Random_Search" if ENABLE_RANDOM_SEARCH else GPR_KERNEL_TYPE
        model_filename = os.path.join(output_dir, f"Gaussian_Process_Regression_Model_{sampling_method_zh}_{model_suffix}.pkl")
        joblib.dump(gpr_results['model'], model_filename)
        logger.info(f"Model has been saved to: {model_filename}")

        # Plot scatter plots - pass MAE and RMSE parameters
        if ENABLE_RANDOM_SEARCH:
            plot_title_prefix = f"GPR ({sampling_method_name}, Random Search)"
        else:
            plot_title_prefix = f"GPR ({sampling_method_name}, kernel={kernel_name})"

        plot_academic_scatter(
            y_train, gpr_results['pred']['train'],
            f"{plot_title_prefix} - Training Set",
            gpr_results['metrics']['train']['r2'],
            gpr_results['metrics']['train']['mae'],
            gpr_results['metrics']['train']['rmse'],
            filename=os.path.join(output_dir, f"Gaussian_Process_Regression_{sampling_method_zh}_Training_Set_Scatter.png")
        )

        plot_academic_scatter(
            y_test, gpr_results['pred']['test'],
            f"{plot_title_prefix} - Test Set",
            gpr_results['metrics']['test']['r2'],
            gpr_results['metrics']['test']['mae'],
            gpr_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Gaussian_Process_Regression_{sampling_method_zh}_Test_Set_Scatter.png")
        )

        # Plot side-by-side scatter plot
        plot_side_by_side_scatter(
            y_train, gpr_results['pred']['train'],
            y_test, gpr_results['pred']['test'],
            plot_title_prefix,
            gpr_results['metrics']['train']['r2'],
            gpr_results['metrics']['test']['r2'],
            gpr_results['metrics']['train']['mae'],
            gpr_results['metrics']['train']['rmse'],
            gpr_results['metrics']['test']['mae'],
            gpr_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"Gaussian_Process_Regression_{sampling_method_zh}_Training_Test_Comparison.png")
        )

        # Create performance summary table
        summary_df = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE'],
            'Training Set': [
                gpr_results['metrics']['train']['r2'],
                gpr_results['metrics']['train']['mae'],
                gpr_results['metrics']['train']['rmse']
            ],
            'Validation Set': [
                gpr_results['metrics']['val']['r2'],
                gpr_results['metrics']['val']['mae'],
                gpr_results['metrics']['val']['rmse']
            ],
            'Test Set': [
                gpr_results['metrics']['test']['r2'],
                gpr_results['metrics']['test']['mae'],
                gpr_results['metrics']['test']['rmse']
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

        if ENABLE_RANDOM_SEARCH:
            logger.info(f"Gaussian Process Regression model - {sampling_method_zh} - hyperparameter random search program execution completed!")
        else:
            logger.info(f"Gaussian Process Regression model - {sampling_method_zh} program execution completed!")

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

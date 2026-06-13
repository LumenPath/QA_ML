import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import joblib
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional, Union
import ast
import random

# Import PyTorch-related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ===== Reproducibility Settings =====
def set_all_seeds(seed=42):
    """
    Set all random seeds to ensure reproducible results

    Parameters:
        seed: Integer random seed
    """
    # Python built-in random module
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)

    # CUDA random seed (if GPU is used)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the CUDA backend to deterministic mode (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch 1.8+, force the use of deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except:
            print("Warning: Unable to set deterministic algorithms; the PyTorch version may be incompatible")
    elif hasattr(torch, 'set_deterministic'):  # PyTorch 1.7 and earlier versions
        try:
            torch.set_deterministic(True)
        except:
            print("Warning: Unable to set deterministic mode; the PyTorch version may be incompatible")


def worker_init_fn(worker_id):
    """
    Worker initialization function for DataLoader to ensure reproducibility in multi-threaded data loading
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===== Global Parameter Settings (Easy to Adjust) =====
# Random search configuration
ENABLE_RANDOM_SEARCH = False  # Whether to enable random hyperparameter search False True
RANDOM_SEARCH_TRIALS = 300  # Number of random search trials

# Sampling method configuration
SAMPLING_METHOD = "random"  # Sampling method selection: "random" or "equal_width"

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

# Feature processing configuration
FEATURE_STANDARDIZATION = True  # Whether to standardize features
USE_ROBUST_SCALER = None  # Whether to use RobustScaler instead of StandardScaler (more robust to outliers)

# Convolutional neural network parameter configuration - optimized for data with 309 samples and 14 features
CNN_BATCH_SIZE = 64  # Batch size
CNN_EPOCHS = 200  # Number of training epochs
CNN_LEARNING_RATE = 0.001  # Learning rate
CNN_DROPOUT_RATE = 0.3  # Dropout rate
CNN_L2_REG = 1e-05  # L2 regularization coefficient
CNN_EARLY_STOPPING = 15  # Early stopping patience

# Network structure configuration - changed to use a list to define the fully connected layer structure
CNN_CONV1_FILTERS = 256  # Number of filters in the first convolutional layer
CNN_CONV2_FILTERS = 512  # Number of filters in the second convolutional layer
CNN_FC_LAYERS = [512, 1024, 2048, 1024, 512]  # Fully connected layer structure [hidden layer 1, hidden layer 2, ...]

# Cross-validation configuration
CV_FOLDS = 5  # Number of cross-validation folds

# Computing resource configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU acceleration
NUM_WORKERS = 0  # Number of data loader worker processes

# Figure output quality settings
OUTPUT_DPI = 600  # Figure output DPI
EXPORT_CSV = True  # Whether to export CSV prediction results

# Set English font - Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
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

        f.write("==== Random Search Configuration ====\n")
        f.write(f"Enable random search: {ENABLE_RANDOM_SEARCH}\n")
        if ENABLE_RANDOM_SEARCH:
            f.write(f"Number of random search trials: {RANDOM_SEARCH_TRIALS}\n")
        f.write("\n")

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

        f.write("==== CNN Model Parameters ====\n")
        f.write(f"Feature standardization: {FEATURE_STANDARDIZATION}\n")
        f.write(f"Use RobustScaler: {USE_ROBUST_SCALER}\n")
        f.write(f"Batch size: {CNN_BATCH_SIZE}\n")
        f.write(f"Training epochs: {CNN_EPOCHS}\n")
        f.write(f"Learning rate: {CNN_LEARNING_RATE}\n")
        f.write(f"Dropout rate: {CNN_DROPOUT_RATE}\n")
        f.write(f"L2 regularization coefficient: {CNN_L2_REG}\n")
        f.write(f"Early stopping patience: {CNN_EARLY_STOPPING}\n\n")

        f.write("==== Network Structure ====\n")
        f.write(f"Number of filters in the first convolutional layer: {CNN_CONV1_FILTERS}\n")
        f.write(f"Number of filters in the second convolutional layer: {CNN_CONV2_FILTERS}\n")
        f.write(f"Fully connected layer structure: {CNN_FC_LAYERS}\n\n")

        f.write("==== Sampling Method Parameters ====\n")
        f.write(f"Test set ratio: {TEST_SIZE}\n")
        f.write(f"Number of cross-validation folds: {CV_FOLDS}\n\n")

        f.write("==== Computing Resources ====\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Number of data loader worker processes: {NUM_WORKERS}\n\n")

        f.write("==== Reproducibility Settings ====\n")
        f.write("Random seeds have been set to ensure reproducibility\n\n")

    logger.info(f"Run parameter configuration has been saved to: {os.path.join(output_dir, 'run_parameter_configuration.txt')}")


def create_output_dir() -> str:
    """
    Create output directory

    Returns:
    - Output directory path
    """
    # Create main folder
    main_dir = "CNN_Regression_Model_Results"
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


# Define convolutional neural network model
class CNNRegressor(nn.Module):
    """
    Lightweight convolutional neural network for small-sample regression problems

    For regression problems with 14 features and 309 samples, the data is transformed for convolutional network processing
    Features are reshaped into a small matrix, and shallow convolution is used to extract features
    Supports configurable fully connected layer structure
    """

    def __init__(self, num_features: int = 9, dropout_rate: float = 0.3, fc_layers: List[int] = None):
        super(CNNRegressor, self).__init__()

        # Use the provided fc_layers or the default value
        if fc_layers is None:
            fc_layers = CNN_FC_LAYERS

        # Calculate how to reshape features (as close to square as possible)
        # For 9 features
        self.reshape_dim = (3, 3)

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, CNN_CONV1_FILTERS, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(CNN_CONV1_FILTERS),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(CNN_CONV1_FILTERS, CNN_CONV2_FILTERS, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(CNN_CONV2_FILTERS),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Dynamically calculate flattened feature dimension
        # Create a sample input and pass it through convolutional layers to determine the dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.reshape_dim[0], self.reshape_dim[1])
            dummy_output = self.conv2(self.conv1(dummy_input))
            self.flat_features = dummy_output.numel()
            logger.info(f"Flattened feature dimension after CNN convolution: {self.flat_features}")

        # Build dynamic fully connected layers
        fc_layers_sizes = [self.flat_features] + fc_layers + [1]  # Add input layer and output layer

        # Build fully connected layer sequence
        fc_modules = []
        for i in range(len(fc_layers_sizes) - 1):
            fc_modules.append(nn.Linear(fc_layers_sizes[i], fc_layers_sizes[i + 1]))

            # Do not add BN and ReLU to the last layer
            if i < len(fc_layers_sizes) - 2:
                fc_modules.append(nn.BatchNorm1d(fc_layers_sizes[i + 1]))
                fc_modules.append(nn.ReLU())
                fc_modules.append(nn.Dropout(dropout_rate))

        # Create fully connected layer sequence
        self.fc = nn.Sequential(*fc_modules)

        # Record network structure
        logger.info(f"CNN network structure:")
        logger.info(f"- Input features: {num_features}, reshaped to: {self.reshape_dim}")
        logger.info(f"- First convolutional layer: {CNN_CONV1_FILTERS} filters, 2x2 kernel")
        logger.info(f"- Second convolutional layer: {CNN_CONV2_FILTERS} filters, 2x2 kernel")
        logger.info(f"- Fully connected layer structure: {fc_layers_sizes}")

    def forward(self, x):
        # Shape of input x: [batch_size, num_features]
        batch_size = x.size(0)

        # Reshape to [batch_size, 1, reshape_dim[0], reshape_dim[1]]
        x = x.view(batch_size, 1, self.reshape_dim[0], self.reshape_dim[1])

        # Apply convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = self.fc(x)

        # Output
        return x.squeeze(1)


class EarlyStopping:
    """
    Early stopping mechanism that stops training when validation loss no longer improves
    """

    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info('Early stopping triggered, stopping training')
        return self.early_stop


def train_cnn_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_cnn(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Collect predictions and targets
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    # Calculate overall loss
    epoch_loss = running_loss / len(dataloader.dataset)

    # Combine predictions and targets from batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate evaluation metrics
    r2 = r2_score(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return epoch_loss, r2, mae, rmse, all_preds, all_targets


def random_search_hyperparameters(X_train, y_train, X_val, y_val, n_trials=10):
    """
    Perform random search to find the best hyperparameters

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features
    - y_val: Validation labels
    - n_trials: Number of random trials

    Returns:
    - Dictionary of best hyperparameters
    """
    logger.info(f"Starting random hyperparameter search, total trials: {n_trials}...")

    # Set random seed
    np.random.seed(42)

    # Define hyperparameter search space
    param_distributions = {
        'batch_size': [8, 16, 32, 64],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'weight_decay': [1e-6, 1e-5, 1e-4],
        'conv1_filters': [64, 128, 256],
        'conv2_filters': [64, 128, 256],
        'fc_layers': [
            [512, 1024, 512],
            [256, 512, 256],
            [128, 256, 128],
            [256, 256, 256],
            [512, 512, 512],
            [512, 1024, 1024, 512],
            [512, 1024, 1024, 1024, 512],
            [512, 1024, 2048, 1024, 512]
        ]
    }

    best_val_loss = float('inf')
    best_params = None
    results = []

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
    y_train_tensor = torch.FloatTensor(y_train.values if isinstance(y_train, pd.Series) else y_train)
    X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
    y_val_tensor = torch.FloatTensor(y_val.values if isinstance(y_val, pd.Series) else y_val)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    for trial in range(n_trials):
        # Randomly select hyperparameters
        params = {
            'batch_size': int(np.random.choice(param_distributions['batch_size'])),
            'learning_rate': float(np.random.choice(param_distributions['learning_rate'])),
            'dropout_rate': float(np.random.choice(param_distributions['dropout_rate'])),
            'weight_decay': float(np.random.choice(param_distributions['weight_decay'])),
            'conv1_filters': int(np.random.choice(param_distributions['conv1_filters'])),
            'conv2_filters': int(np.random.choice(param_distributions['conv2_filters'])),
            'fc_layers': param_distributions['fc_layers'][
                int(np.random.choice(len(param_distributions['fc_layers'])))
            ]
        }

        logger.info(f"Trial {trial + 1}/{n_trials}: {params}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(params['batch_size']),
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE.type == 'cuda' else False,
            worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if DEVICE.type == 'cuda' else False,
            worker_init_fn=worker_init_fn
        )

        # Set temporary global variables
        global CNN_CONV1_FILTERS, CNN_CONV2_FILTERS
        orig_conv1_filters = CNN_CONV1_FILTERS
        orig_conv2_filters = CNN_CONV2_FILTERS

        CNN_CONV1_FILTERS = params['conv1_filters']
        CNN_CONV2_FILTERS = params['conv2_filters']

        # Create model
        num_features = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[1]
        model = CNNRegressor(
            num_features=num_features,
            dropout_rate=params['dropout_rate'],
            fc_layers=params['fc_layers']
        ).to(DEVICE)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        # Early stopping
        early_stopping = EarlyStopping(patience=5, verbose=False)

        # Train model
        for epoch in range(30):  # Limit to 30 epochs to speed up search
            # Train one epoch
            train_cnn_epoch(model, train_loader, criterion, optimizer, DEVICE)

            # Validate
            val_loss, val_r2, val_mae, val_rmse, _, _ = validate_cnn(
                model, val_loader, criterion, DEVICE
            )

            # Check whether to stop early
            if early_stopping(val_loss):
                break

        # Record results
        results.append({
            'params': params,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'epochs': epoch + 1
        })

        logger.info(f"  Result: validation loss={val_loss:.6f}, R²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f}")

        # Update best parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            logger.info(f"  New best parameters! validation loss={val_loss:.6f}")

        # Restore global variables
        CNN_CONV1_FILTERS = orig_conv1_filters
        CNN_CONV2_FILTERS = orig_conv2_filters

    # Sort and display all results
    sorted_results = sorted(results, key=lambda x: x['val_loss'])

    logger.info("\nRandom search hyperparameter results summary (sorted by validation loss):")
    for i, result in enumerate(sorted_results[:5]):  # Display top 5 results
        logger.info(f"Rank {i + 1}: validation loss={result['val_loss']:.6f}, R²={result['val_r2']:.4f}")
        logger.info(f"  Parameters: {result['params']}")

    logger.info(f"\nBest hyperparameters: {best_params}")

    return best_params


def train_cnn_model(X_train, y_train, X_val=None, y_val=None, batch_size=CNN_BATCH_SIZE,
                    epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE, weight_decay=CNN_L2_REG,
                    dropout_rate=CNN_DROPOUT_RATE, fc_layers=None,
                    device=DEVICE, patience=CNN_EARLY_STOPPING):
    """
    Train CNN regression model

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_val: Validation features (if not provided, the training set is used for validation)
    - y_val: Validation labels
    - batch_size: Batch size
    - epochs: Number of training epochs
    - lr: Learning rate
    - weight_decay: L2 regularization coefficient
    - dropout_rate: Dropout rate
    - fc_layers: Fully connected layer structure
    - device: Training device (CPU or GPU)
    - patience: Early stopping patience

    Returns:
    - Trained model and training history
    """
    # If fc_layers is not provided, use the global setting
    if fc_layers is None:
        fc_layers = CNN_FC_LAYERS

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values if isinstance(X_train, pd.DataFrame) else X_train)
    y_train_tensor = torch.FloatTensor(y_train.values if isinstance(y_train, pd.Series) else y_train)

    # Create training dataset and data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        worker_init_fn=worker_init_fn  # Add worker initialization function to ensure reproducibility
    )

    # If a validation set is provided, create validation data loader
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val.values if isinstance(X_val, pd.DataFrame) else X_val)
        y_val_tensor = torch.FloatTensor(y_val.values if isinstance(y_val, pd.Series) else y_val)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type == 'cuda' else False,
            worker_init_fn=worker_init_fn  # Add worker initialization function to ensure reproducibility
        )
    else:
        # If there is no validation set, use the training set
        val_loader = train_loader

    # Create model
    num_features = X_train.shape[1] if isinstance(X_train, pd.DataFrame) else X_train.shape[1]
    model = CNNRegressor(num_features=num_features, dropout_rate=dropout_rate, fc_layers=fc_layers).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler - disable verbose warning
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # Early stopping mechanism
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Save the best model state
    best_model_state = None
    best_val_loss = float('inf')

    # Training history record
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_mae': [],
        'val_rmse': []
    }

    logger.info(f"Starting CNN model training, total epochs: {epochs}")

    for epoch in range(epochs):
        # Train one epoch
        train_loss = train_cnn_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_r2, val_mae, val_rmse, _, _ = validate_cnn(
            model, val_loader, criterion, device
        )

        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            logger.info(f"Learning rate adjusted from {old_lr} to {new_lr}")

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"training loss: {train_loss:.6f}, validation loss: {val_loss:.6f}, "
                        f"validation R²: {val_r2:.4f}, validation MAE: {val_mae:.4f}, validation RMSE: {val_rmse:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Check whether to stop early
        if early_stopping(val_loss):
            logger.info(f"Early stopping training after epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, history


def train_and_evaluate_cnn(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           cv_folds: int = 5, method_name: str = "CNN Regression Model") -> Dict:
    """
    Train CNN regression model and perform cross-validation evaluation

    Parameters:
    - X_train: Training features
    - X_test: Test features
    - y_train: Training labels
    - y_test: Test labels
    - cv_folds: Number of cross-validation folds
    - method_name: Method name (for logging)

    Returns:
    - Dictionary containing the model, evaluation metrics, and prediction results
    """
    # Data preprocessing - standardize features
    if FEATURE_STANDARDIZATION:
        if USE_ROBUST_SCALER:
            scaler = RobustScaler()
            logger.info("Using RobustScaler for feature standardization (more robust to outliers)")
        else:
            scaler = StandardScaler()
            logger.info("Using StandardScaler for feature standardization")

        # Fit and apply to training and test sets
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        # No standardization
        X_train_scaled = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_scaled = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        logger.info("No feature standardization performed")

    # Convert to numpy arrays if not already
    y_train_values = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test

    # Cross-validation
    logger.info(f"\nStarting {cv_folds}-fold cross-validation [{method_name}]...")

    # Ensure KFold randomness is reproducible
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Store evaluation metrics for each fold
    cv_r2_scores = []
    cv_mae_scores = []
    cv_rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        # Split into training and validation sets
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train_values[train_idx], y_train_values[val_idx]

        # Train model
        logger.info(f"\nStarting fold {fold + 1} cross-validation...")
        fold_model, _ = train_cnn_model(
            X_fold_train, y_fold_train, X_fold_val, y_fold_val,
            batch_size=CNN_BATCH_SIZE, epochs=CNN_EPOCHS,
            lr=CNN_LEARNING_RATE, weight_decay=CNN_L2_REG,
            dropout_rate=CNN_DROPOUT_RATE, fc_layers=CNN_FC_LAYERS,
            device=DEVICE, patience=CNN_EARLY_STOPPING
        )

        # Convert features to PyTorch tensor
        X_fold_val_tensor = torch.FloatTensor(X_fold_val).to(DEVICE)

        # Get predictions
        fold_model.eval()
        with torch.no_grad():
            y_fold_pred = fold_model(X_fold_val_tensor).cpu().numpy()

        # Calculate evaluation metrics
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))

        # Record results
        cv_r2_scores.append(fold_r2)
        cv_mae_scores.append(fold_mae)
        cv_rmse_scores.append(fold_rmse)

        # Output results for this fold
        logger.info(f"  Fold {fold + 1}: R² = {fold_r2:.4f}, "
                    f"MAE = {fold_mae:.4f}, RMSE = {fold_rmse:.4f}")

    # Output average cross-validation results
    logger.info(f"\n{method_name} - average cross-validation results:")
    logger.info(f"Average validation R²: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})")
    logger.info(f"Average validation MAE: {np.mean(cv_mae_scores):.4f} (±{np.std(cv_mae_scores):.4f})")
    logger.info(f"Average validation RMSE: {np.mean(cv_rmse_scores):.4f} (±{np.std(cv_rmse_scores):.4f})")

    # Train final model on the full training set
    logger.info("\nTraining final model on the full training set...")
    final_model, train_history = train_cnn_model(
        X_train_scaled, y_train_values, batch_size=CNN_BATCH_SIZE,
        epochs=CNN_EPOCHS, lr=CNN_LEARNING_RATE, weight_decay=CNN_L2_REG,
        dropout_rate=CNN_DROPOUT_RATE, fc_layers=CNN_FC_LAYERS,
        device=DEVICE, patience=CNN_EARLY_STOPPING
    )

    # Evaluate final model performance on the training set
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(DEVICE)
    y_train_tensor = torch.FloatTensor(y_train_values).to(DEVICE)

    train_criterion = nn.MSELoss()
    _, train_r2, train_mae, train_rmse, y_train_pred, _ = validate_cnn(
        final_model, DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=CNN_BATCH_SIZE,
            shuffle=False,
            worker_init_fn=worker_init_fn
        ),
        train_criterion, DEVICE
    )

    logger.info(f"\n{method_name} - training set evaluation results:")
    logger.info(f"R²: {train_r2:.4f}")
    logger.info(f"MAE: {train_mae:.4f}")
    logger.info(f"RMSE: {train_rmse:.4f}")

    # Evaluate final model performance on the test set
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
    y_test_tensor = torch.FloatTensor(y_test_values).to(DEVICE)

    test_criterion = nn.MSELoss()
    _, test_r2, test_mae, test_rmse, y_test_pred, _ = validate_cnn(
        final_model, DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=CNN_BATCH_SIZE,
            shuffle=False,
            worker_init_fn=worker_init_fn
        ),
        test_criterion, DEVICE
    )

    logger.info(f"\n{method_name} - test set evaluation results:")
    logger.info(f"R²: {test_r2:.4f}")
    logger.info(f"MAE: {test_mae:.4f}")
    logger.info(f"RMSE: {test_rmse:.4f}")

    # Create results dictionary
    results = {
        'model': final_model,
        'scaler': scaler if FEATURE_STANDARDIZATION else None,
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
            'train': y_train_values,
            'test': y_test_values
        },
        'history': train_history
    }

    return results


def plot_training_history(history: Dict, output_dir: str, title: str = "CNN Model Training History"):
    """
    Plot training history curves

    Parameters:
    - history: Training history dictionary
    - output_dir: Output directory
    - title: Figure title
    """
    plt.close('all')
    plt.rcdefaults()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Set font
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Plot loss curves
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Model Loss', fontsize=16, family='Times New Roman')
    ax1.set_xlabel('Epochs', fontsize=14, family='Times New Roman')
    ax1.set_ylabel('Loss (MSE)', fontsize=14, family='Times New Roman')
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Set font for axis tick labels
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Plot evaluation metrics
    ax2.plot(epochs, history['val_r2'], 'g-', linewidth=2, label='R²')
    ax2.plot(epochs, history['val_mae'], 'c-', linewidth=2, label='MAE')
    ax2.plot(epochs, history['val_rmse'], 'y-', linewidth=2, label='RMSE')
    ax2.set_title('Validation Metrics', fontsize=16, family='Times New Roman')
    ax2.set_xlabel('Epochs', fontsize=14, family='Times New Roman')
    ax2.set_ylabel('Metric Value', fontsize=14, family='Times New Roman')
    ax2.legend(fontsize=12, loc='center right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Set font for axis tick labels
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontname('Times New Roman')

    # Set overall title
    fig.suptitle(title, fontsize=18, family='Times New Roman', y=0.98)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    # Save image
    plt.savefig(os.path.join(output_dir, "CNN_Model_Training_History.png"), dpi=OUTPUT_DPI, bbox_inches='tight')
    logger.info(f"Training history plot has been saved to: {os.path.join(output_dir, 'CNN_Model_Training_History.png')}")

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


def save_pytorch_model(model, scaler, output_dir: str, model_name: str = "CNN_Regression_Model"):
    """
    Save PyTorch model and scaler

    Parameters:
    - model: PyTorch model
    - scaler: Scaler
    - output_dir: Output directory
    - model_name: Model name
    """
    # Save model
    model_path = os.path.join(output_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"PyTorch model has been saved to: {model_path}")

    # If there is a scaler, save it
    if scaler is not None:
        scaler_path = os.path.join(output_dir, f"{model_name}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler has been saved to: {scaler_path}")


def plot_academic_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, r2: float, mae: float = None,
                          rmse: float = None, filename: str = None) -> None:
    """
    Plot an academic-quality scatter plot showing the relationship between true and predicted values

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
    Plot side-by-side scatter plots for the training set and test set

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


def main():
    """Main function - execute the complete CNN regression feature sampling workflow"""
    try:
        # Set random seed to ensure reproducibility
        set_all_seeds(42)

        # Create output directory
        output_dir = create_output_dir()

        # Set up logging
        setup_logging(output_dir)

        # Record reproducibility settings
        logger.info("Random seed has been set to 42 to ensure reproducibility")

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

        # Preprocess data
        if FEATURE_STANDARDIZATION:
            if USE_ROBUST_SCALER:
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
            scaler = None

        # If random search is enabled, find the best hyperparameters
        best_params = None
        if ENABLE_RANDOM_SEARCH:
            logger.info("\n" + "=" * 50)
            logger.info("Starting random hyperparameter search...")
            logger.info("=" * 50)

            # Split a portion from the training set as the validation set for hyperparameter search
            X_search_train, X_search_val, y_search_train, y_search_val = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42
            )

            # Perform random search
            best_params = random_search_hyperparameters(
                X_search_train, y_search_train,
                X_search_val, y_search_val,
                n_trials=RANDOM_SEARCH_TRIALS
            )

            # Update global parameters
            global CNN_BATCH_SIZE, CNN_LEARNING_RATE, CNN_DROPOUT_RATE, CNN_L2_REG
            global CNN_CONV1_FILTERS, CNN_CONV2_FILTERS, CNN_FC_LAYERS

            CNN_BATCH_SIZE = best_params['batch_size']
            CNN_LEARNING_RATE = best_params['learning_rate']
            CNN_DROPOUT_RATE = best_params['dropout_rate']
            CNN_L2_REG = best_params['weight_decay']
            CNN_CONV1_FILTERS = best_params['conv1_filters']
            CNN_CONV2_FILTERS = best_params['conv2_filters']
            CNN_FC_LAYERS = best_params['fc_layers']

            logger.info("\nUpdated network configuration after hyperparameter update:")
            logger.info(f"Batch size: {CNN_BATCH_SIZE}")
            logger.info(f"Learning rate: {CNN_LEARNING_RATE}")
            logger.info(f"Dropout rate: {CNN_DROPOUT_RATE}")
            logger.info(f"L2 regularization coefficient: {CNN_L2_REG}")
            logger.info(f"Number of filters in the first convolutional layer: {CNN_CONV1_FILTERS}")
            logger.info(f"Number of filters in the second convolutional layer: {CNN_CONV2_FILTERS}")
            logger.info(f"Fully connected layer structure: {CNN_FC_LAYERS}")

            # Save hyperparameters to file
            with open(os.path.join(output_dir, "Best_Hyperparameters.txt"), "w", encoding="utf-8") as f:
                f.write("Best hyperparameters found by random search:\n\n")
                for param_name, param_value in best_params.items():
                    f.write(f"{param_name}: {param_value}\n")

            logger.info(f"Best hyperparameters have been saved to: {os.path.join(output_dir, 'Best_Hyperparameters.txt')}")

        # Train CNN regression model
        sampling_method_name = "Equal Width Binning" if SAMPLING_METHOD == "equal_width" else "Random Sampling"  # English sampling method name
        method_name = f"CNN Regression ({sampling_method_name})"  # English method name

        # Set sampling method name for output files
        sampling_method_zh = "Equal_Width_Binning" if SAMPLING_METHOD == "equal_width" else "Random_Sampling"

        if FEATURE_STANDARDIZATION:
            scaler_name = "RobustScaler" if USE_ROBUST_SCALER else "StandardScaler"
            method_name += f", with {scaler_name}"

        logger.info(f"\nStarting training of CNN regression model ({sampling_method_name})...")

        cnn_results = train_and_evaluate_cnn(
            X_train, X_test, y_train, y_test,
            cv_folds=CV_FOLDS, method_name=method_name
        )

        # Plot training history
        plot_training_history(
            cnn_results['history'],
            output_dir,
            title=f"CNN Regression Model Training ({sampling_method_name})"
        )

        # Export prediction results
        if EXPORT_CSV:
            export_predictions_to_csv(
                cnn_results['true']['train'], cnn_results['pred']['train'],
                f"CNN_{SAMPLING_METHOD}", "training", output_dir
            )
            export_predictions_to_csv(
                cnn_results['true']['test'], cnn_results['pred']['test'],
                f"CNN_{SAMPLING_METHOD}", "test", output_dir
            )

        # Save model
        save_pytorch_model(
            cnn_results['model'],
            cnn_results['scaler'],
            output_dir,
            model_name=f"CNN_Regression_Model_{sampling_method_zh}"
        )

        # Plot scatter plots - pass MAE and RMSE parameters
        plot_academic_scatter(
            cnn_results['true']['train'], cnn_results['pred']['train'],
            f"CNN Regression ({sampling_method_name}) - Training Set",
            cnn_results['metrics']['train']['r2'],
            cnn_results['metrics']['train']['mae'],
            cnn_results['metrics']['train']['rmse'],
            filename=os.path.join(output_dir, f"CNN_Regression_{sampling_method_zh}_Training_Set_Scatter.png")
        )

        plot_academic_scatter(
            cnn_results['true']['test'], cnn_results['pred']['test'],
            f"CNN Regression ({sampling_method_name}) - Test Set",
            cnn_results['metrics']['test']['r2'],
            cnn_results['metrics']['test']['mae'],
            cnn_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"CNN_Regression_{sampling_method_zh}_Test_Set_Scatter.png")
        )

        # Plot side-by-side scatter plot
        plot_side_by_side_scatter(
            cnn_results['true']['train'], cnn_results['pred']['train'],
            cnn_results['true']['test'], cnn_results['pred']['test'],
            f"CNN Regression ({sampling_method_name})",
            cnn_results['metrics']['train']['r2'],
            cnn_results['metrics']['test']['r2'],
            cnn_results['metrics']['train']['mae'],
            cnn_results['metrics']['train']['rmse'],
            cnn_results['metrics']['test']['mae'],
            cnn_results['metrics']['test']['rmse'],
            filename=os.path.join(output_dir, f"CNN_Regression_{sampling_method_zh}_Training_Test_Comparison.png")
        )

        # Create performance summary table
        summary_df = pd.DataFrame({
            'Metric': ['R²', 'MAE', 'RMSE'],
            'Training Set': [
                cnn_results['metrics']['train']['r2'],
                cnn_results['metrics']['train']['mae'],
                cnn_results['metrics']['train']['rmse']
            ],
            'Validation Set': [
                cnn_results['metrics']['val']['r2'],
                cnn_results['metrics']['val']['mae'],
                cnn_results['metrics']['val']['rmse']
            ],
            'Test Set': [
                cnn_results['metrics']['test']['r2'],
                cnn_results['metrics']['test']['mae'],
                cnn_results['metrics']['test']['rmse']
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
        logger.info(f"CNN regression model - {sampling_method_zh} program execution completed!")
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

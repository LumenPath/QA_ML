import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate
import os
import datetime
import seaborn as sns
import joblib
import warnings
import traceback

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ===== Global Parameter Settings (Easy to Adjust) =====
# --- File Path Configuration ---
TRAIN_FILE = 'final_ssl_train.parquet'  # Training set filename
TEST_FILE = 'test.parquet'  # Test set filename
MODEL_FILE = 'best_ssl_model.joblib'  # Pretrained model filename for performance comparison
PREDICTION_FILE = 'XXX.csv'  # Data file to be predicted

# --- Training Parameters ---
RF_N_ESTIMATORS = 49  # Number of trees in the random forest
RF_N_JOBS = -1  # Number of CPU cores to use (-1 means using all available cores)
CV_FOLDS = 5  # Number of cross-validation folds
RANDOM_STATE = 42  # Random seed to ensure reproducible results

# --- Data Column Name Configuration ---
TARGET_COLUMN_NAME = 'target'  # Column name of the target variable (value to be predicted)

# --- Output Settings ---
OUTPUT_DPI = 600  # Figure output DPI

# Set English font - Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX math font
plt.rcParams['axes.unicode_minus'] = False  # Avoid rendering minus signs as boxes


# Create output directory
def create_output_dir():
    main_dir = "RF_Model_Prediction_and_Ranking_Results"
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
        print(f"Created main output directory: {main_dir}")

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(main_dir, f"Run_Results_{current_time}")
    os.makedirs(output_dir)
    print(f"Created current run directory: {output_dir}")
    return output_dir


# Data loading function (supports .parquet and .csv)
def load_data(file_path):
    """
    Load data according to the file extension.
    """
    print(f"Loading data: {file_path}")
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Please use .csv or .parquet")
    print(f"Successfully loaded {len(df)} samples.")
    return df


# Model training and evaluation function
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    rf_model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=RF_N_JOBS)

    print(f"\nStarting {CV_FOLDS}-fold cross-validation...")
    scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_results = cross_validate(rf_model, X_train, y_train, cv=CV_FOLDS, scoring=scoring, n_jobs=RF_N_JOBS)

    print(f"Average cross-validation results:")
    print(f"  Average validation R²: {np.mean(cv_results['test_r2']):.9f} (±{np.std(cv_results['test_r2']):.9f})")

    print("\nTraining final model on the full training set...")
    final_model = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=RF_N_JOBS)
    final_model.fit(X_train, y_train)

    # Evaluate on the training set and test set
    y_train_pred = final_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_test_pred = final_model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print("\n--- Newly Trained Model ---")
    print(f"Training set evaluation results: R² = {train_r2:.9f}, MAE = {train_mae:.9f}, RMSE = {train_rmse:.9f}")
    print(f"Test set evaluation results: R² = {test_r2:.9f}, MAE = {test_mae:.9f}, RMSE = {test_rmse:.9f}")

    results = {'model': final_model, 'train_r2': train_r2, 'test_r2': test_r2,
               'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred}
    return results


# Function for evaluating a preloaded model
def evaluate_pretrained_model(model, X_train, y_train, X_test, y_test):
    print("\n--- Loaded Comparison Model ---")
    # Evaluate on the training set and test set
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Training set evaluation results: R² = {train_r2:.9f}, MAE = {train_mae:.9f}, RMSE = {train_rmse:.9f}")
    print(f"Test set evaluation results: R² = {test_r2:.9f}, MAE = {test_mae:.9f}, RMSE = {test_rmse:.9f}")


# Academic scatter plot function
def plot_academic_scatter(y_true, y_pred, title, r2, filename=None):
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    plt.scatter(y_true, y_pred, s=50, alpha=0.6)
    min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction (1:1)')
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'{title} ($R^2$ = {r2:.9f})', fontsize=16, pad=20)
    plt.legend(loc='lower right', fontsize=12, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=OUTPUT_DPI, bbox_inches='tight')
        print(f"Scatter plot has been saved: {filename}")
    plt.close()


# Function for predicting new data
def predict_new_data(model, input_filepath, output_filepath, feature_names):
    print("\n" + "=" * 50)
    print("Stage 2: Start prediction using the newly trained model...")
    print("=" * 50)
    try:
        df_to_predict_raw = load_data(input_filepath)

        if df_to_predict_raw.columns[0].lower() in ['unnamed: 0', 'index', 'serial number']:
            original_info = df_to_predict_raw.iloc[:, [0, 1]].copy()
            original_info.columns = ['Serial Number', 'Filename']
        else:
            original_info = df_to_predict_raw.iloc[:, [0, 1]].copy()
            original_info.columns = ['Serial Number', 'Filename']

        X_predict = df_to_predict_raw[feature_names]

        if X_predict.isnull().sum().any():
            print("Warning: Missing values found in the data to be predicted; they will be filled with 0.")
            X_predict = X_predict.fillna(0)

        print(f"Successfully loaded {len(X_predict)} rows of data for prediction.")
        predictions = model.predict(X_predict)

        output_df = original_info
        output_df['Predicted Value'] = predictions
        output_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"Prediction completed, results have been saved to: {output_filepath}")
        return output_filepath

    except FileNotFoundError:
        print(f"Error: Prediction input file '{input_filepath}' not found.")
        return None
    except KeyError:
        print("Error: Columns in the prediction file do not match the features used for model training.")
        missing_cols = set(feature_names) - set(df_to_predict_raw.columns)
        print(f"Missing feature columns: {list(missing_cols)}")
        return None
    except Exception as e:
        print(f"Error occurred during prediction: {e}")
        traceback.print_exc()
        return None


# Analysis and ranking function
def analyze_and_rank_predictions(prediction_filepath, output_dir):
    print("\n" + "=" * 50)
    print("Stage 3: Start analyzing and ranking prediction results...")
    print("=" * 50)
    try:
        df = pd.read_csv(prediction_filepath)
        print("Step 1: Group by structure and keep the entry with the minimum predicted value for each structure...")
        df['structure_group'] = df['Filename'].str.split('_pro').str[0]

        min_value_indices = df.groupby('structure_group')['Predicted Value'].idxmin()
        filtered_df = df.loc[min_value_indices].copy()

        filtered_output_path = os.path.join(output_dir, 'filtered_predictions.csv')
        filtered_df[['Serial Number', 'Filename', 'Predicted Value']].to_csv(filtered_output_path, index=False, encoding='utf-8-sig')
        print(f"Filtering completed, retained {len(filtered_df)} structures, results saved to: {filtered_output_path}")

        print("\nStep 2: Sort the filtered structures by predicted value from high to low...")
        ranked_df = filtered_df.sort_values(by='Predicted Value', ascending=False).copy()
        ranked_df['Rank'] = np.arange(1, len(ranked_df) + 1)
        final_ranked_df = ranked_df[['Rank', 'Serial Number', 'Filename', 'Predicted Value']]

        ranked_output_path = os.path.join(output_dir, 'ranked_predictions.csv')
        final_ranked_df.to_csv(ranked_output_path, index=False, encoding='utf-8-sig')
        print(f"Sorting and ranking completed, results have been saved to: {ranked_output_path}")

        print("\n" + "-" * 20 + " Top 10 Structure Information " + "-" * 20)
        print(final_ranked_df.head(10).to_string(index=False))
        print("-" * (44 + len("Top 10 Structure Information")))

    except FileNotFoundError:
        print(f"Error: Prediction result file '{prediction_filepath}' not found for analysis.")
    except Exception as e:
        print(f"Error occurred during analysis and ranking: {e}")
        traceback.print_exc()


# Main function
def main():
    output_dir = create_output_dir()

    try:
        # --- Stage 1: Model Training and Comparison ---
        print("=" * 50)
        print("Stage 1: Model Training and Comparison")
        print("=" * 50)

        # Load data
        train_df = load_data(TRAIN_FILE)
        test_df = load_data(TEST_FILE)

        # Clean and prepare data
        train_df.dropna(inplace=True)
        print(f"After removing missing values, remaining training samples: {len(train_df)}")

        if TARGET_COLUMN_NAME not in train_df.columns:
            raise KeyError(f"Target variable column '{TARGET_COLUMN_NAME}' is not in the training file.")

        numeric_train_df = train_df.select_dtypes(include=np.number)
        numeric_test_df = test_df.select_dtypes(include=np.number)

        non_numeric_cols = train_df.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_cols:
            print(f"Warning: The following non-numeric columns were found and ignored: {non_numeric_cols}")

        y_train = numeric_train_df[TARGET_COLUMN_NAME]
        X_train = numeric_train_df.drop(columns=[TARGET_COLUMN_NAME])
        y_test = numeric_test_df[TARGET_COLUMN_NAME]
        X_test = numeric_test_df.drop(columns=[TARGET_COLUMN_NAME])

        feature_columns = X_train.columns.tolist()
        X_test = X_test[feature_columns]

        # Train and evaluate the new model
        model_results = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        newly_trained_model = model_results['model']

        # Save the newly trained model
        model_save_path = os.path.join(output_dir, "trained_model.joblib")
        joblib.dump(newly_trained_model, model_save_path)
        print(f"\nNewly trained model has been saved to: {model_save_path}")

        # Load and evaluate the old model used for comparison
        if os.path.exists(MODEL_FILE):
            print(f"\nLoading {MODEL_FILE} for performance comparison...")
            loaded_model = joblib.load(MODEL_FILE)
            evaluate_pretrained_model(loaded_model, X_train, y_train, X_test, y_test)
        else:
            print(f"\nComparison model file '{MODEL_FILE}' was not found in the current directory, skipping comparison.")

        # Visualize the performance of the new model
        plot_academic_scatter(y_train, model_results['y_train_pred'], "Newly Trained Model - Training Set",
                              model_results['train_r2'], os.path.join(output_dir, "RF_New_Model_Training_Set_Scatter.png"))
        plot_academic_scatter(y_test, model_results['y_test_pred'], "Newly Trained Model - Test Set",
                              model_results['test_r2'], os.path.join(output_dir, "RF_New_Model_Test_Set_Scatter.png"))
        print("\nModel training and evaluation completed.")

        # --- Stages 2 & 3: Prediction and Analysis ---
        prediction_output_file = os.path.join(output_dir, 'Prediction_Results_ranked.csv')
        prediction_result_path = predict_new_data(newly_trained_model, PREDICTION_FILE, prediction_output_file,
                                                  feature_columns)

        if prediction_result_path:
            analyze_and_rank_predictions(prediction_result_path, output_dir)

        print(f"\nAll workflows have been completed. All results have been saved to directory: {output_dir}")

    except (FileNotFoundError, KeyError) as e:
        print(f"\nError: {e}")
        print("Program terminated. Please check file paths and column name configuration.")
    except Exception as e:
        print(f"\nUnexpected error occurred during program execution: {e}")
        traceback.print_exc()
    finally:
        plt.close('all')


if __name__ == "__main__":
    main()

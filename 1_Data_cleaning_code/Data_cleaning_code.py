import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend to prevent Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import os
import datetime
import joblib
import warnings
import itertools

warnings.filterwarnings('ignore')

# --- Configuration & Global Parameters ---

# --- Method Switches ---
USE_TARGET_RANGE_FILTER = True
USE_MODIFIED_Z_SCORE = True
USE_BOXPLOT_METHOD = True

# --- File and Column Settings ---
INPUT_FILE = 'XXX.csv'
TARGET_RANGE_MIN = 0.8
TARGET_RANGE_MAX = 2.0
ID_COLUMN = 0
TARGET_COLUMN = 1
FEATURE_START_COLUMN = 2
FEATURE_END_COLUMN = 26

# --- Parameter Ranges for Optimization ---
MODIFIED_Z_THRESHOLDS = [0.5, 0.8, 1.0, 1.1, 1.2, 1.25, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 3.0]
IQR_FACTORS = [1, 1.5, 2.0, 2.2, 2.4]

# --- Model & Plotting Settings ---
CROSS_VAL_FOLDS = 5

# --- Plotting Style Configuration ---
BASE_FONT_SIZE = 18
TITLE_FONT_SIZE = BASE_FONT_SIZE + 2
LABEL_FONT_SIZE = BASE_FONT_SIZE
TICK_LABEL_SIZE = BASE_FONT_SIZE

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = LABEL_FONT_SIZE
plt.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
plt.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
plt.rcParams['legend.fontsize'] = LABEL_FONT_SIZE
plt.rcParams['xtick.labelsize'] = TICK_LABEL_SIZE
plt.rcParams['ytick.labelsize'] = TICK_LABEL_SIZE
plt.rcParams['axes.unicode_minus'] = False

sns.set_theme(style="whitegrid", palette="colorblind")

# --- Output Directory Setup ---
filter_method_name = "Z-Score_Boxplot_Optimized" if USE_MODIFIED_Z_SCORE and USE_BOXPLOT_METHOD else \
    "Z-Score_Optimized" if USE_MODIFIED_Z_SCORE else \
        "Boxplot_Optimized" if USE_BOXPLOT_METHOD else "Base_Model"

main_dir = "Machine_Learning_Data_Cleaning_English_Charts_Version"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(main_dir, f"Results_{filter_method_name}_{current_time}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")


def load_and_clean_data(file_path, threshold_min=TARGET_RANGE_MIN, threshold_max=TARGET_RANGE_MAX):
    """Loads and cleans data."""
    print("--- 1. Loading and Cleaning Data ---")
    df = pd.read_csv(file_path)
    df_original = df.copy()
    df_cleaned = df.copy()
    if USE_TARGET_RANGE_FILTER:
        df_cleaned = df_cleaned[
            (df_cleaned.iloc[:, TARGET_COLUMN] >= threshold_min) &
            (df_cleaned.iloc[:, TARGET_COLUMN] <= threshold_max)
            ]
    if df_cleaned.isnull().sum().any():
        df_cleaned = df_cleaned.dropna()
    print(f"Samples after cleaning: {len(df_cleaned)}\n")
    return df_cleaned, df_original


def analyze_data(X, y, output_dir):
    """Performs initial data analysis and visualization."""
    print("--- 2. Initial Data Analysis and Visualization ---")
    plt.figure(figsize=(10, 6))
    sns.histplot(y, bins=30, kde=True)
    plt.axvline(y.mean(), color='r', linestyle='--', label=f'Mean: {y.mean():.4f}')
    plt.axvline(y.median(), color='g', linestyle='-.', label=f'Median: {y.median():.4f}')
    plt.xlabel('Target Variable Value');
    plt.ylabel('Frequency');
    plt.title('Distribution of Target Variable')
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_Target_Variable_Distribution.png"), dpi=300);
    plt.close()

    data_corr = X.copy()
    data_corr['Target'] = y
    correlation = data_corr.corr()
    target_corr = correlation['Target'].drop('Target').abs().sort_values(ascending=False)
    top_features = list(target_corr.head(15).index) + ['Target']
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation.loc[top_features, top_features], annot=True, cmap='coolwarm',
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, annot_kws={"size": 10})
    plt.title('Correlation Heatmap (Top 15 Features vs. Target)');
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_Correlation_Heatmap.png"), dpi=300);
    plt.close()
    print("Saved initial analysis plots.\n")


def get_cross_validation_residuals(X, y, n_splits=CROSS_VAL_FOLDS, random_state=42):
    """Calculates prediction residuals using K-fold cross-validation with progress prints."""
    print("--- 3. Calculating Cross-Validation Residuals ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_predictions, cv_true = np.zeros(X.shape[0]), np.zeros(X.shape[0])

    # (RESTORED) Progress printing for each fold
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"  - Processing fold {fold_idx + 1}/{n_splits}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        cv_predictions[test_idx] = model.predict(X_test)
        cv_true[test_idx] = y_test

    cv_r2 = r2_score(cv_true, cv_predictions)
    cv_mae = mean_absolute_error(cv_true, cv_predictions)
    plt.figure(figsize=(8, 8))
    plt.scatter(cv_true, cv_predictions, alpha=0.6, edgecolors='w', s=50)
    plt.plot([cv_true.min(), cv_true.max()], [cv_true.min(), cv_true.max()], 'r--', lw=2, label='y = x line')
    plt.xlabel('Actual Values');
    plt.ylabel('Predicted Values')
    plt.title(f'Cross-Validation: Predicted vs. Actual\n(R²={cv_r2:.4f}, MAE={cv_mae:.4f})')
    plt.legend();
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_CV_Predicted_vs_Actual.png"), dpi=300);
    plt.close()
    print("Saved predicted vs. actual plot.\n")
    return cv_true - cv_predictions, np.abs(cv_true - cv_predictions)


def analyze_residuals(residuals, abs_errors, output_dir):
    """Analyzes the distribution of residuals."""
    print("--- 4. Analyzing Residuals ---")
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True, color='blue',
                 line_kws={'color': 'red', 'lw': 2, 'label': 'Density Curve'})
    plt.axvline(0, color='black', linestyle='--', label='Zero Residuals')
    plt.xlabel('Residual Value');
    plt.ylabel('Frequency');
    plt.title('Distribution of Cross-Validation Residuals')
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_Residuals_Distribution.png"), dpi=300);
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(abs_errors, bins=30, kde=True, color='green',
                 line_kws={'color': 'orange', 'lw': 2, 'label': 'Density Curve'})
    plt.axvline(abs_errors.mean(), color='r', linestyle='--', label=f'Mean: {abs_errors.mean():.4f}')
    plt.xlabel('Absolute Error');
    plt.ylabel('Frequency');
    plt.title('Distribution of Absolute Errors')
    plt.legend();
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_Absolute_Error_Distribution.png"), dpi=300);
    plt.close()

    fig = plt.figure(figsize=(8, 8));
    ax = fig.add_subplot(111)
    sm.qqplot(residuals, line='45', fit=True, ax=ax)
    ax.set_title('Normal Q-Q Plot of Residuals');
    ax.set_xlabel('Theoretical Quantiles');
    ax.set_ylabel('Sample Quantiles')
    plt.tight_layout();
    plt.savefig(os.path.join(output_dir, "06_Residuals_QQ_Plot.png"), dpi=300);
    plt.close()
    print("Saved residual analysis plots.\n")


def modified_z_score_detection(residuals, threshold):
    median_residual = np.median(residuals)
    abs_deviation = np.abs(residuals - median_residual)
    mad = np.median(abs_deviation)
    if mad == 0: mad = 1e-10
    modified_z_scores = 0.6745 * abs_deviation / mad
    return modified_z_scores > threshold, modified_z_scores


def boxplot_detection(abs_errors, iqr_factor):
    Q1, Q3 = np.percentile(abs_errors, [25, 75])
    upper_bound = Q3 + iqr_factor * (Q3 - Q1)
    return abs_errors > upper_bound, upper_bound


def evaluate_parameter_combinations(X, y, cv_residuals, cv_abs_errors, output_dir, random_state=42):
    """Evaluates model performance for different outlier detection parameter combinations."""
    print("--- 5. Evaluating Parameter Combinations ---")
    performance_results = []
    base_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    cv_results = cross_validate(base_model, X, y, cv=CROSS_VAL_FOLDS,
                                scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'])
    original_r2, original_mae, original_rmse = cv_results['test_r2'].mean(), -cv_results[
        'test_neg_mean_absolute_error'].mean(), -cv_results['test_neg_root_mean_squared_error'].mean()

    param_combinations = list(itertools.product(MODIFIED_Z_THRESHOLDS, IQR_FACTORS))
    total_combinations = len(param_combinations)

    for idx, (z_threshold, iqr_factor) in enumerate(param_combinations):
        # (RESTORED) Progress printing for each parameter combination
        print(
            f"\nEvaluating parameter combination {idx + 1}/{total_combinations}: Z-Threshold={z_threshold}, IQR-Factor={iqr_factor}")

        z_out, _ = modified_z_score_detection(cv_residuals, z_threshold) if USE_MODIFIED_Z_SCORE else (
        np.zeros_like(cv_residuals, dtype=bool), None)
        box_out, _ = boxplot_detection(cv_abs_errors, iqr_factor) if USE_BOXPLOT_METHOD else (
        np.zeros_like(cv_abs_errors, dtype=bool), None)
        outliers = np.logical_or(z_out, box_out)

        print(f"  - Outliers detected: {outliers.sum()} ({outliers.sum() / len(y) * 100:.2f}%)")
        if outliers.sum() / len(y) > 0.5 or len(y) - outliers.sum() < 2 * CROSS_VAL_FOLDS:
            print("  - Skipping: Too many outliers removed or too few samples left.")
            continue

        X_f, y_f = X.iloc[~outliers], y.iloc[~outliers]
        try:
            filt_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
            cv_f = cross_validate(filt_model, X_f, y_f, cv=CROSS_VAL_FOLDS,
                                  scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'])
            r2_f, mae_f, rmse_f = cv_f['test_r2'].mean(), -cv_f['test_neg_mean_absolute_error'].mean(), -cv_f[
                'test_neg_root_mean_squared_error'].mean()
            r2_imp = ((r2_f - original_r2) / abs(original_r2)) * 100 if original_r2 != 0 else 0
            mae_imp = ((original_mae - mae_f) / original_mae) * 100
            rmse_imp = ((original_rmse - rmse_f) / original_rmse) * 100

            performance_results.append({
                'Z-Threshold': z_threshold, 'IQR-Factor': iqr_factor, 'Outlier Count': outliers.sum(),
                'Outlier Percentage': outliers.sum() / len(y) * 100,
                'R2': r2_f, 'MAE': mae_f, 'RMSE': rmse_f,
                'R2 Improvement': r2_imp, 'MAE Improvement': mae_imp, 'RMSE Improvement': rmse_imp,
                'Overall Score': (r2_imp + mae_imp + rmse_imp) / 3
            })
        except Exception as e:
            print(f"  - Error during evaluation: {e}")
            continue

    results_df = pd.DataFrame(performance_results)
    if results_df.empty: return results_df, None, original_r2, original_mae, original_rmse

    results_df.to_csv(os.path.join(output_dir, "parameter_optimization_results.csv"), index=False)
    pivot = results_df.pivot(index='Z-Threshold', columns='IQR-Factor', values='Overall Score')
    plt.figure(figsize=(10, 8));
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Overall Performance Improvement Score (%)');
    plt.xlabel('Boxplot IQR Factor');
    plt.ylabel('Modified Z-Score Threshold')
    plt.tight_layout();
    plt.savefig(os.path.join(output_dir, "07_Heatmap_Performance_Score.png"), dpi=300);
    plt.close()

    best_params = results_df.loc[results_df['Overall Score'].idxmax()]
    print("\nSaved parameter evaluation results.\n")
    return results_df, best_params, original_r2, original_mae, original_rmse


def generate_decision_report(results_df, best_params, output_dir):
    """(RESTORED) Generates a markdown decision report."""
    print("--- 6. Generating Decision Report ---")
    if results_df.empty or best_params is None: return

    with open(os.path.join(output_dir, "Decision_Report.md"), "w", encoding="utf-8") as f:
        f.write("# Outlier Detection Parameter Optimization Report\n\n")
        f.write(f"**Report Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Optimal Z-Score Threshold:** `{best_params['Z-Threshold']}`\n")
        f.write(f"- **Optimal IQR Factor:** `{best_params['IQR-Factor']}`\n")
        f.write(
            f"This combination identified **{int(best_params['Outlier Count'])} outliers ({best_params['Outlier Percentage']:.2f}%)** and provided the best overall performance improvement:\n")
        f.write(f"- **R² Improvement:** `{best_params['R2 Improvement']:.2f}%`\n")
        f.write(f"- **MAE Improvement:** `{best_params['MAE Improvement']:.2f}%`\n")
        f.write(f"- **RMSE Improvement:** `{best_params['RMSE Improvement']:.2f}%`\n\n")
        f.write("## 2. Recommendations\n\n")
        f.write(
            "Based on the results, we recommend applying this optimal parameter set for outlier removal before final model training.\n\n")
        if best_params['Outlier Percentage'] > 15:
            f.write(
                "**Caution:** The percentage of outliers is high. Manual inspection of the identified outliers is recommended to ensure they are not valid data points from an underrepresented class.\n")
    print("Decision report saved.\n")


def evaluate_best_model(X, y, best_params, output_dir, random_state=42):
    """Compares the final model against the original model."""
    print("--- 7. Final Model Evaluation ---")
    if best_params is None: return None

    # Recalculate residuals on the full dataset for outlier identification
    full_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    full_model.fit(X, y)
    predictions = full_model.predict(X)
    residuals = y - predictions
    abs_errors = np.abs(residuals)

    z_out, z_scores = modified_z_score_detection(residuals, best_params['Z-Threshold']) if USE_MODIFIED_Z_SCORE else (
    np.zeros_like(residuals, dtype=bool), None)
    box_out, upper_bound = boxplot_detection(abs_errors, best_params['IQR-Factor']) if USE_BOXPLOT_METHOD else (
    np.zeros_like(abs_errors, dtype=bool), 0)
    outliers = np.logical_or(z_out, box_out)
    X_f, y_f = X.iloc[~outliers], y.iloc[~outliers]

    orig_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    filt_model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)

    print("Generating learning curves with confidence intervals...")
    train_sizes = np.linspace(0.1, 1.0, 10)

    _, train_scores_orig, test_scores_orig = learning_curve(orig_model, X, y, train_sizes=train_sizes,
                                                            cv=CROSS_VAL_FOLDS, scoring='neg_mean_absolute_error',
                                                            n_jobs=-1)
    _, train_scores_filt, test_scores_filt = learning_curve(filt_model, X_f, y_f, train_sizes=train_sizes,
                                                            cv=CROSS_VAL_FOLDS, scoring='neg_mean_absolute_error',
                                                            n_jobs=-1)

    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, -np.mean(train_scores_orig, axis=1), 'o-', color='r', label='Training (Original)')
    plt.fill_between(train_sizes, -np.mean(train_scores_orig, axis=1) - np.std(train_scores_orig, axis=1),
                     -np.mean(train_scores_orig, axis=1) + np.std(train_scores_orig, axis=1), alpha=0.1, color='r')
    plt.plot(train_sizes, -np.mean(test_scores_orig, axis=1), 'o-', color='darkred',
             label='Cross-Validation (Original)')
    plt.fill_between(train_sizes, -np.mean(test_scores_orig, axis=1) - np.std(test_scores_orig, axis=1),
                     -np.mean(test_scores_orig, axis=1) + np.std(test_scores_orig, axis=1), alpha=0.1, color='darkred')
    plt.plot(train_sizes, -np.mean(train_scores_filt, axis=1), 's-', color='b', label='Training (Filtered)')
    plt.fill_between(train_sizes, -np.mean(train_scores_filt, axis=1) - np.std(train_scores_filt, axis=1),
                     -np.mean(train_scores_filt, axis=1) + np.std(train_scores_filt, axis=1), alpha=0.1, color='b')
    plt.plot(train_sizes, -np.mean(test_scores_filt, axis=1), 's-', color='darkblue',
             label='Cross-Validation (Filtered)')
    plt.fill_between(train_sizes, -np.mean(test_scores_filt, axis=1) - np.std(test_scores_filt, axis=1),
                     -np.mean(test_scores_filt, axis=1) + np.std(test_scores_filt, axis=1), alpha=0.1, color='darkblue')

    plt.xlabel('Number of Training Samples');
    plt.ylabel('Mean Absolute Error (MAE)');
    plt.title('Learning Curves: Original vs. Filtered Model')
    plt.legend(loc='best');
    plt.grid(True);
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "12_Learning_Curves_Comparison.png"), dpi=300);
    plt.close()

    orig_model.fit(X, y);
    filt_model.fit(X_f, y_f)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 14), sharey=True, sharex=True)
    ax1.scatter(y, orig_model.predict(X), alpha=0.6, label='Normal');
    ax1.scatter(y[outliers], orig_model.predict(X)[outliers], color='red', marker='x', label='Outliers')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--');
    ax1.set_title(
        f"Original Model (R²={r2_score(y, orig_model.predict(X)):.4f}, MAE={mean_absolute_error(y, orig_model.predict(X)):.4f})");
    ax1.legend()
    ax2.scatter(y_f, filt_model.predict(X_f), alpha=0.6);
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    ax2.set_title(
        f"Filtered Model (R²={r2_score(y_f, filt_model.predict(X_f)):.4f}, MAE={mean_absolute_error(y_f, filt_model.predict(X_f)):.4f})");
    ax2.set_xlabel('Actual Value');
    ax1.set_ylabel('Predicted Value');
    ax2.set_ylabel('Predicted Value')
    plt.tight_layout();
    plt.savefig(os.path.join(output_dir, "13_Model_Prediction_Comparison.png"), dpi=300);
    plt.close()

    if USE_MODIFIED_Z_SCORE and z_scores is not None:
        plt.figure(figsize=(10, 6));
        sns.histplot(z_scores, bins=30, kde=True, color='purple', line_kws={'color': 'orange', 'lw': 2})
        plt.axvline(best_params['Z-Threshold'], color='r', linestyle='--',
                    label=f"Threshold: {best_params['Z-Threshold']}")
        plt.xlabel('Modified Z-Score');
        plt.ylabel('Frequency');
        plt.title('Distribution of Modified Z-Scores');
        plt.legend();
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "14_Best_Z-Score_Distribution.png"), dpi=300);
        plt.close()

    if USE_BOXPLOT_METHOD:
        plt.figure(figsize=(10, 6));
        sns.boxplot(x=abs_errors);
        plt.axvline(upper_bound, color='r', linestyle='--', label=f'Upper Bound: {upper_bound:.4f}')
        plt.xlabel('Absolute Error');
        plt.title('Boxplot of Absolute Errors');
        plt.legend();
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "15_Best_Boxplot.png"), dpi=300);
        plt.close()

    print("Saved final evaluation plots.\n")
    return {'original': {'model': orig_model}, 'filtered': {'model': filt_model}, 'outliers': outliers}


def analyze_feature_importance(model, X, output_dir, prefix=""):
    """Analyzes and visualizes model feature importances."""
    print(f"--- 8. Analyzing Feature Importance for {prefix} Model ---")
    df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance',
                                                                                                    ascending=False)
    plt.figure(figsize=(10, 8));
    sns.barplot(x='importance', y='feature', data=df.head(20), palette='viridis')
    plt.xlabel('Feature Importance');
    plt.ylabel('Feature');
    plt.title(f'{prefix} Feature Importance (Top 20)')
    plt.tight_layout();
    plt.savefig(os.path.join(output_dir, f"{prefix}_Feature_Importance.png"), dpi=300);
    plt.close()
    print(f"Saved {prefix} feature importance plot.\n")


def save_results(df_cleaned, df_original, outliers, best_params, performance_metrics, output_dir, filename):
    """(RESTORED) Saves the final data, models, and a detailed summary.txt file."""
    print("--- 9. Saving Final Results ---")

    # Save datasets
    df_valid = df_cleaned.iloc[~outliers]
    df_output_cleaned = df_original[df_original.iloc[:, ID_COLUMN].isin(set(df_valid.iloc[:, ID_COLUMN]))]
    df_output_cleaned.to_csv(os.path.join(output_dir, f"{filename}_cleaned.csv"), index=False)
    if outliers.sum() > 0:
        df_cleaned.iloc[outliers].to_csv(os.path.join(output_dir, f"{filename}_outliers.csv"), index=False)

    # Save models
    joblib.dump(performance_metrics['original']['model'], os.path.join(output_dir, "original_model.pkl"))
    joblib.dump(performance_metrics['filtered']['model'], os.path.join(output_dir, "filtered_model.pkl"))

    # Save detailed summary.txt
    with open(os.path.join(output_dir, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("Outlier Detection and Removal Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("Configuration Parameters:\n")
        f.write(f"- Use Target Range Filter: {USE_TARGET_RANGE_FILTER}\n")
        if USE_TARGET_RANGE_FILTER:
            f.write(f"- Target Value Range: [{TARGET_RANGE_MIN}, {TARGET_RANGE_MAX}]\n")
        f.write(f"- Use Modified Z-Score Method: {USE_MODIFIED_Z_SCORE}\n")
        if USE_MODIFIED_Z_SCORE:
            f.write(f"- Optimal Modified Z-Score Threshold: {best_params['Z-Threshold']}\n")
        f.write(f"- Use Boxplot Method: {USE_BOXPLOT_METHOD}\n")
        if USE_BOXPLOT_METHOD:
            f.write(f"- Optimal Boxplot IQR Factor: {best_params['IQR-Factor']}\n")
        f.write(f"- Cross-Validation Folds: {CROSS_VAL_FOLDS}\n\n")
        f.write("Data Information:\n")
        f.write(f"- Original Samples: {len(df_original)}\n")
        f.write(f"- Samples After Pre-processing: {len(df_cleaned)}\n")
        f.write(f"- Detected Outliers: {outliers.sum()}\n")
        f.write(f"- Outlier Percentage: {outliers.sum() / len(df_cleaned) * 100:.2f}%\n")
        f.write(f"- Final Samples Retained: {len(df_valid)}\n\n")
        f.write("Performance Comparison (based on cross-validation):\n")
        f.write(f"- Original Model R²: {performance_metrics['original']['r2']:.4f}\n")
        f.write(f"- Filtered Model R²: {best_params['R2']:.4f}\n")
        f.write(f"- R² Improvement: {best_params['R2 Improvement']:+.2f}%\n\n")
        f.write(f"- Original Model MAE: {performance_metrics['original']['mae']:.4f}\n")
        f.write(f"- Filtered Model MAE: {best_params['MAE']:.4f}\n")
        f.write(f"- MAE Improvement: {best_params['MAE Improvement']:+.2f}%\n\n")
        f.write(f"- Original Model RMSE: {performance_metrics['original']['rmse']:.4f}\n")
        f.write(f"- Filtered Model RMSE: {best_params['RMSE']:.4f}\n")
        f.write(f"- RMSE Improvement: {best_params['RMSE Improvement']:+.2f}%\n\n")
        f.write("=" * 50 + "\n")
        f.write(f"Run ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Saved final data, models, and summary report.")


def main():
    """Main execution workflow."""
    try:
        df_cleaned, df_original = load_and_clean_data(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: '{INPUT_FILE}'.");
        return

    X, y = df_cleaned.iloc[:, FEATURE_START_COLUMN:FEATURE_END_COLUMN + 1], df_cleaned.iloc[:, TARGET_COLUMN]
    analyze_data(X, y, output_dir)
    cv_residuals, cv_abs_errors = get_cross_validation_residuals(X, y)
    analyze_residuals(cv_residuals, cv_abs_errors, output_dir)

    if not (USE_MODIFIED_Z_SCORE or USE_BOXPLOT_METHOD):
        print("No outlier detection methods enabled. Halting.");
        return

    params_results, best_params, orig_r2, orig_mae, orig_rmse = evaluate_parameter_combinations(X, y, cv_residuals,
                                                                                                cv_abs_errors,
                                                                                                output_dir)

    if best_params is not None and not params_results.empty:
        generate_decision_report(params_results, best_params, output_dir)
        final_models = evaluate_best_model(X, y, best_params, output_dir)
        if final_models:
            analyze_feature_importance(final_models['original']['model'], X, output_dir, prefix="Original_Model")
            analyze_feature_importance(final_models['filtered']['model'], X.iloc[~final_models['outliers']], output_dir,
                                       prefix="Filtered_Model")

            # Prepare performance dict for the summary file
            performance_metrics = {
                'original': {'model': final_models['original']['model'], 'r2': orig_r2, 'mae': orig_mae,
                             'rmse': orig_rmse},
                'filtered': {'model': final_models['filtered']['model']}
            }

            filename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
            save_results(df_cleaned, df_original, final_models['outliers'], best_params, performance_metrics,
                         output_dir, filename)
    else:
        print("Could not find an optimal parameter set.")

    print("=" * 50 + f"\nProcessing complete. All results are in: {output_dir}\n" + "=" * 50)


if __name__ == "__main__":
    main()

"""
High-precision data storage module - used to save and load semi-supervised learning data, ensuring full precision retention
- Data loading and visualization based on Parquet files
- Retains HDF5 output functionality (for compatibility)
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap


def save_data_with_high_precision(X_train, y_train, X_test, y_test, feature_names,
                                  best_model, all_pseudo_X=None, all_pseudo_y=None,
                                  best_pseudo_X=None, best_pseudo_y=None,
                                  ssl_model=None, data_dir=None):
    """
    Save training and test data in a high-precision format, supporting future model training and visualization

    Parameters:
    ------
    X_train, y_train: Original training set
    X_test, y_test: Test set
    feature_names: List of feature names
    best_model: Trained best model
    all_pseudo_X, all_pseudo_y: All pseudo-labeled samples
    best_pseudo_X, best_pseudo_y: Pseudo-labeled samples used by the best model
    ssl_model: Semi-supervised learning model object
    data_dir: Data saving directory

    Returns:
    ------
    dict: Dictionary containing all output file paths
    """
    if data_dir is None:
        print("No data directory provided, skipping save operation")
        return {}

    # Ensure the directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Define output filenames
    results = {}

    # Base paths
    base_path = os.path.join(data_dir, 'ssl_model_data')
    parquet_dir = os.path.join(data_dir, 'parquet_data')

    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)

    # Save the path of parquet_dir
    results['parquet_dir'] = parquet_dir

    # 1. Prepare dataframes

    # Training set
    train_df = pd.DataFrame()
    if isinstance(X_train, pd.DataFrame):
        # If X is already a DataFrame
        for feature in feature_names:
            train_df[feature] = X_train[feature].values
    else:
        # If X is a numpy array
        for i, feature in enumerate(feature_names):
            train_df[feature] = X_train[:, i]

    # Add target variable
    train_df['target'] = y_train if not isinstance(y_train, pd.Series) else y_train.values
    train_df['sample_type'] = 'original'
    train_df['id'] = [f'train_{i}' for i in range(len(y_train))]

    # Test set
    test_df = pd.DataFrame()
    if isinstance(X_test, pd.DataFrame):
        for feature in feature_names:
            test_df[feature] = X_test[feature].values
    else:
        for i, feature in enumerate(feature_names):
            test_df[feature] = X_test[:, i]

    test_df['target'] = y_test if not isinstance(y_test, pd.Series) else y_test.values
    test_df['sample_type'] = 'test'
    test_df['id'] = [f'test_{i}' for i in range(len(y_test))]

    # Directly use the provided best pseudo-labeled samples without attempting reconstruction
    if best_pseudo_X is not None and best_pseudo_y is not None and len(best_pseudo_X) > 0:
        print(f"Using provided best pseudo-labeled samples ({len(best_pseudo_X)} samples)")
        best_pseudo_df = pd.DataFrame()

        # Add features
        for i, feature in enumerate(feature_names):
            if isinstance(best_pseudo_X[0], np.ndarray):
                best_pseudo_df[feature] = [x[i] for x in best_pseudo_X]
            elif isinstance(best_pseudo_X[0], pd.Series):
                best_pseudo_df[feature] = [x[feature] if feature in x else x.iloc[i] for x in best_pseudo_X]
            elif isinstance(best_pseudo_X[0], pd.DataFrame):
                best_pseudo_df[feature] = [x[feature].iloc[0] if feature in x else 0 for x in best_pseudo_X]
            else:
                try:
                    best_pseudo_df[feature] = [x[i] if isinstance(x, (list, tuple, np.ndarray)) else
                                               getattr(x, feature, 0) for x in best_pseudo_X]
                except:
                    best_pseudo_df[feature] = [0] * len(best_pseudo_y)

        # Add target variable
        if isinstance(best_pseudo_y, (list, tuple)):
            best_pseudo_df['target'] = best_pseudo_y
        else:
            best_pseudo_df['target'] = list(best_pseudo_y)

        best_pseudo_df['sample_type'] = 'best_pseudo'
        best_pseudo_df['id'] = [f'best_pseudo_{i}' for i in range(len(best_pseudo_y))]

        # Final training set = original training set + best pseudo-labeled samples
        final_train_df = pd.concat([train_df, best_pseudo_df], ignore_index=True)
        print(f"final_ssl_train.parquet will contain the original training set ({len(train_df)} samples) + best pseudo-labeled samples ({len(best_pseudo_df)} samples)")
    else:
        final_train_df = train_df
        print(f"final_ssl_train.parquet will contain only the original training set ({len(train_df)} samples), with no pseudo-labels")

    # 2. Save as Parquet files - retain full precision
    original_train_file = os.path.join(parquet_dir, 'original_train.parquet')
    test_file = os.path.join(parquet_dir, 'test.parquet')
    final_train_file = os.path.join(parquet_dir, 'final_ssl_train.parquet')

    # Save feature-name metadata to a Parquet file
    metadata_file = os.path.join(parquet_dir, 'metadata.parquet')
    metadata_df = pd.DataFrame({
        'feature_name': feature_names,
        'index': list(range(len(feature_names)))
    })
    metadata_df['train_samples'] = len(train_df)
    metadata_df['test_samples'] = len(test_df)
    metadata_df['pseudo_samples'] = len(final_train_df) - len(train_df)
    metadata_df.to_parquet(metadata_file, index=False)

    # Save performance metrics to a Parquet file
    metrics_file = os.path.join(parquet_dir, 'metrics.parquet')

    train_df.to_parquet(original_train_file, index=False)
    test_df.to_parquet(test_file, index=False)
    final_train_df.to_parquet(final_train_file, index=False)

    results['train_parquet'] = original_train_file
    results['test_parquet'] = test_file
    results['final_train_parquet'] = final_train_file
    results['metadata_parquet'] = metadata_file

    # 3. Save as HDF5 - organized into a single file (only for archiving, not for loading)
    hdf_file = os.path.join(data_dir, 'ssl_model_data.h5')
    with pd.HDFStore(hdf_file) as store:
        store['original_train'] = train_df
        store['test'] = test_df
        store['final_train'] = final_train_df

        # Save feature names and other metadata
        store['metadata'] = pd.Series({
            'features': ','.join(feature_names),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'pseudo_samples': len(final_train_df) - len(train_df)
        })

    results['hdf_file'] = hdf_file

    # 4. Save model
    model_file = os.path.join(data_dir, 'best_ssl_model.joblib')
    joblib.dump(best_model, model_file)
    results['model_file'] = model_file

    # 5. Calculate performance metrics
    y_train_pred = best_model.predict(train_df[feature_names])
    y_test_pred = best_model.predict(test_df[feature_names])
    y_final_train_pred = best_model.predict(final_train_df[feature_names])

    # Calculate metrics
    metrics = {
        'original_train': {
            'r2': r2_score(train_df['target'], y_train_pred),
            'mae': mean_absolute_error(train_df['target'], y_train_pred),
            'rmse': np.sqrt(mean_squared_error(train_df['target'], y_train_pred))
        },
        'test': {
            'r2': r2_score(test_df['target'], y_test_pred),
            'mae': mean_absolute_error(test_df['target'], y_test_pred),
            'rmse': np.sqrt(mean_squared_error(test_df['target'], y_test_pred))
        },
        'final_train': {
            'r2': r2_score(final_train_df['target'], y_final_train_pred),
            'mae': mean_absolute_error(final_train_df['target'], y_final_train_pred),
            'rmse': np.sqrt(mean_squared_error(final_train_df['target'], y_final_train_pred))
        }
    }

    # Save metrics to a Parquet file
    metrics_rows = []
    for dataset, dataset_metrics in metrics.items():
        for metric_name, value in dataset_metrics.items():
            metrics_rows.append({
                'dataset': dataset,
                'metric': metric_name,
                'value': value
            })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_parquet(metrics_file, index=False)
    results['metrics_parquet'] = metrics_file

    # Save prediction results to Parquet
    pred_file = os.path.join(parquet_dir, 'predictions.parquet')

    train_pred_df = pd.DataFrame({
        'dataset': 'original_train',
        'id': train_df['id'],
        'true': train_df['target'],
        'pred': y_train_pred,
        'error': y_train_pred - train_df['target'],
    })

    test_pred_df = pd.DataFrame({
        'dataset': 'test',
        'id': test_df['id'],
        'true': test_df['target'],
        'pred': y_test_pred,
        'error': y_test_pred - test_df['target'],
    })

    final_train_pred_df = pd.DataFrame({
        'dataset': 'final_train',
        'id': final_train_df['id'],
        'true': final_train_df['target'],
        'pred': y_final_train_pred,
        'error': y_final_train_pred - final_train_df['target'],
    })

    pred_df = pd.concat([train_pred_df, test_pred_df, final_train_pred_df], ignore_index=True)
    pred_df.to_parquet(pred_file, index=False)
    results['predictions_parquet'] = pred_file

    # Save feature importance to Parquet
    importance_file = os.path.join(parquet_dir, 'feature_importance.parquet')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_,
        'std': np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    })
    importance_df.to_parquet(importance_file, index=False)
    results['importance_parquet'] = importance_file

    # Save training parameters to Parquet
    params_file = os.path.join(parquet_dir, 'model_params.parquet')
    model_params = best_model.get_params()
    params_df = pd.DataFrame([{k: str(v) for k, v in model_params.items()}])
    params_df.to_parquet(params_file, index=False)
    results['params_parquet'] = params_file

    print(f"\nAll data has been saved with full precision retained:")
    print(f"- Parquet file directory (primary data format): {parquet_dir}")
    print(f"  - Original training set: {os.path.basename(original_train_file)}")
    print(f"  - Test set: {os.path.basename(test_file)}")
    print(f"  - Final training set: {os.path.basename(final_train_file)}")
    print(f"  - Metadata: {os.path.basename(metadata_file)}")
    print(f"  - Performance metrics: {os.path.basename(metrics_file)}")
    print(f"  - Prediction results: {os.path.basename(pred_file)}")
    print(f"  - Feature importance: {os.path.basename(importance_file)}")
    print(f"  - Model parameters: {os.path.basename(params_file)}")
    print(f"- HDF5 file (archive): {hdf_file}")
    print(f"- Model file: {model_file}")

    print("\nTo load the data and train the model, please use the train_and_evaluate_from_parquet function.")

    return results


def generate_visualizations_from_parquet(parquet_dir, model_file, output_dir, feature_names=None):
    """
    Generate high-precision academic-style visualizations from Parquet files - modified version, using the combined training set

    Parameters:
    ------
    parquet_dir: Parquet file directory
    model_file: Path to the trained model file
    output_dir: Output directory
    feature_names: List of feature names; if None, inferred from the data
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model
    model = joblib.load(model_file)

    # Load Parquet files
    try:
        # Directly load the final training set (containing original training set + best pseudo-labeled samples)
        final_train_df = pd.read_parquet(os.path.join(parquet_dir, 'final_ssl_train.parquet'))
        test_df = pd.read_parquet(os.path.join(parquet_dir, 'test.parquet'))

        # Try to load metadata and metrics
        try:
            metadata_df = pd.read_parquet(os.path.join(parquet_dir, 'metadata.parquet'))
            metrics_df = pd.read_parquet(os.path.join(parquet_dir, 'metrics.parquet'))
            predictions_df = pd.read_parquet(os.path.join(parquet_dir, 'predictions.parquet'))
            importance_df = pd.read_parquet(os.path.join(parquet_dir, 'feature_importance.parquet'))

            # Get feature names from metadata
            if feature_names is None and 'feature_name' in metadata_df.columns:
                feature_names = metadata_df['feature_name'].tolist()

        except Exception as e:
            print(f"Warning: Unable to load auxiliary Parquet files; information will be inferred from the data: {e}")

    except Exception as e:
        print(f"Error: Unable to load Parquet data files: {e}")
        return None

    # If feature names are not provided, infer feature names
    if feature_names is None:
        # Exclude non-feature columns
        non_feature_cols = {'target', 'sample_type', 'id'}
        feature_names = [col for col in final_train_df.columns if col not in non_feature_cols]

    # Extract features and labels from each dataset
    X_final_train = final_train_df[feature_names]
    y_final_train = final_train_df['target']

    X_test = test_df[feature_names]
    y_test = test_df['target']

    # Use the model for prediction
    y_final_train_pred = model.predict(X_final_train)
    y_test_pred = model.predict(X_test)

    # Calculate performance metrics
    final_train_r2 = r2_score(y_final_train, y_final_train_pred)
    final_train_mae = mean_absolute_error(y_final_train, y_final_train_pred)
    final_train_rmse = np.sqrt(mean_squared_error(y_final_train, y_final_train_pred))

    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Set an academic style that differs from the second code block
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'Times', 'Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math font
    plt.rcParams['axes.unicode_minus'] = False  # Avoid rendering minus signs as boxes
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # Use a slightly different color scheme
    high_precision_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    scatter_cmap = plt.cm.plasma

    # Set image output DPI
    output_dpi = 600

    # Generate scatter plot for the final training set
    plt.figure(figsize=(10, 8))

    # Create normalized error color mapping
    errors = np.abs(y_final_train - y_final_train_pred)
    norm_errors = errors / errors.max() if errors.max() > 0 else errors

    sc = plt.scatter(y_final_train, y_final_train_pred, c=norm_errors,
                     cmap=scatter_cmap, alpha=0.8, s=80, edgecolor='k', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Normalized Absolute Error', fontsize=12, fontweight='bold')

    # Calculate regression line
    z = np.polyfit(y_final_train, y_final_train_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(y_final_train), max(y_final_train), 100)
    plt.plot(x_range, p(x_range), '--', color=high_precision_colors[1],
             linewidth=2.5, label=f'Regression Line\ny = {z[0]:.5f}x + {z[1]:.5f}')

    # Add 1:1 ideal line
    min_val = min(min(y_final_train), min(y_final_train_pred))
    max_val = max(max(y_final_train), max(y_final_train_pred))
    buffer = (max_val - min_val) * 0.05
    plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer],
             '-', color='k', linewidth=1.5, label='Perfect Prediction (1:1)')

    # Get the number of pseudo-labeled samples
    pseudo_samples_count = 0
    if 'sample_type' in final_train_df.columns:
        pseudo_samples_count = final_train_df[final_train_df['sample_type'] == 'best_pseudo'].shape[0]

    # Set title and labels
    if pseudo_samples_count > 0:
        train_title = f'High-Precision Combined Training Set (Original + {pseudo_samples_count} Pseudo-labels)\nR² = {final_train_r2:.5f}, MAE = {final_train_mae:.5f}'
    else:
        train_title = f'High-Precision Training Set\nR² = {final_train_r2:.5f}, MAE = {final_train_mae:.5f}'

    plt.title(train_title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Observed Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')

    # Add statistics text box
    stats_text = (f'n = {len(y_final_train)}\n'
                  f'R² = {final_train_r2:.5f}\n'
                  f'MAE = {final_train_mae:.5f}\n'
                  f'RMSE = {final_train_rmse:.5f}')

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray')
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save image
    train_scatter_file = os.path.join(output_dir, 'high_precision_train_scatter.png')
    plt.savefig(train_scatter_file, dpi=output_dpi, bbox_inches='tight')
    plt.close()

    # Generate test set scatter plot (similar style)
    plt.figure(figsize=(10, 8))

    # Create normalized error color mapping
    errors = np.abs(y_test - y_test_pred)
    norm_errors = errors / errors.max() if errors.max() > 0 else errors

    sc = plt.scatter(y_test, y_test_pred, c=norm_errors,
                     cmap=scatter_cmap, alpha=0.8, s=80, edgecolor='k', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Normalized Absolute Error', fontsize=12, fontweight='bold')

    # Calculate regression line
    z = np.polyfit(y_test, y_test_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(y_test), max(y_test), 100)
    plt.plot(x_range, p(x_range), '--', color=high_precision_colors[1],
             linewidth=2.5, label=f'Regression Line\ny = {z[0]:.5f}x + {z[1]:.5f}')

    # Add 1:1 ideal line
    min_val = min(min(y_test), min(y_test_pred))
    max_val = max(max(y_test), max(y_test_pred))
    buffer = (max_val - min_val) * 0.05
    plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer],
             '-', color='k', linewidth=1.5, label='Perfect Prediction (1:1)')

    # Set title and labels
    plt.title(f'High-Precision Test Set Predictions\nR² = {test_r2:.5f}, MAE = {test_mae:.5f}',
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Observed Values', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')

    # Add statistics text box
    stats_text = (f'n = {len(y_test)}\n'
                  f'R² = {test_r2:.5f}\n'
                  f'MAE = {test_mae:.5f}\n'
                  f'RMSE = {test_rmse:.5f}')

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save image
    test_scatter_file = os.path.join(output_dir, 'high_precision_test_scatter.png')
    plt.savefig(test_scatter_file, dpi=output_dpi, bbox_inches='tight')
    plt.close()

    # Generate feature importance plot
    plt.figure(figsize=(12, 10))

    # Get feature importance
    importances = model.feature_importances_

    # Get feature importance from each tree to calculate standard deviation
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'std': std
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Set custom color mapping
    cmap = plt.cm.viridis

    # Draw horizontal bar chart
    y_pos = np.arange(len(importance_df))
    plt.barh(y_pos, importance_df['importance'], xerr=importance_df['std'],
             align='center', alpha=0.8, color=plt.cm.viridis(np.linspace(0, 0.8, len(importance_df))),
             capsize=5, height=0.7, error_kw={'ecolor': 'black', 'capthick': 1.5})

    plt.yticks(y_pos, importance_df['feature'], fontsize=10)
    plt.xlabel('Feature Importance', fontsize=14, fontweight='bold')
    plt.title('High-Precision Model Feature Importance Ranking', fontsize=16, fontweight='bold', pad=15)

    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save image
    importance_file = os.path.join(output_dir, 'high_precision_feature_importance.png')
    plt.savefig(importance_file, dpi=output_dpi, bbox_inches='tight')
    plt.close()

    # Generate side-by-side comparison plot (final training set and test set)
    plt.figure(figsize=(16, 7))

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Calculate global data range
    all_true = np.concatenate([y_final_train, y_test])
    all_pred = np.concatenate([y_final_train_pred, y_test_pred])
    min_val = min(np.min(all_true), np.min(all_pred))
    max_val = max(np.max(all_true), np.max(all_pred))
    buffer = (max_val - min_val) * 0.05

    # Set shared axis range
    global_min = min_val - buffer
    global_max = max_val + buffer

    # Left: final training set
    errors1 = np.abs(y_final_train - y_final_train_pred)
    norm_errors1 = errors1 / errors1.max() if errors1.max() > 0 else errors1

    sc1 = ax1.scatter(y_final_train, y_final_train_pred, c=norm_errors1,
                      cmap=scatter_cmap, alpha=0.8, s=60, edgecolor='k', linewidth=0.5)

    # Add regression line
    z1 = np.polyfit(y_final_train, y_final_train_pred, 1)
    p1 = np.poly1d(z1)
    x_range = np.linspace(global_min, global_max, 100)
    ax1.plot(x_range, p1(x_range), '--', color=high_precision_colors[1],
             linewidth=2, label=f'Regression Line\ny = {z1[0]:.5f}x + {z1[1]:.5f}')

    # Add 1:1 line
    ax1.plot([global_min, global_max], [global_min, global_max],
             '-', color='k', linewidth=1.5, label='Perfect Prediction (1:1)')

    # Set left title and labels
    if pseudo_samples_count > 0:
        train_title = f'Combined Training Set\n(Original + {pseudo_samples_count} Pseudo-labels)\nR² = {final_train_r2:.5f}'
    else:
        train_title = f'Training Set\nR² = {final_train_r2:.5f}'

    ax1.set_title(train_title, fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Observed Values', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')

    # Add statistics
    stats_text1 = (f'n = {len(y_final_train)}\n'
                   f'R² = {final_train_r2:.5f}\n'
                   f'MAE = {final_train_mae:.5f}')

    ax1.text(0.05, 0.95, stats_text1, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Right: test set
    errors2 = np.abs(y_test - y_test_pred)
    norm_errors2 = errors2 / errors2.max() if errors2.max() > 0 else errors2

    sc2 = ax2.scatter(y_test, y_test_pred, c=norm_errors2,
                      cmap=scatter_cmap, alpha=0.8, s=60, edgecolor='k', linewidth=0.5)

    # Add regression line
    z2 = np.polyfit(y_test, y_test_pred, 1)
    p2 = np.poly1d(z2)
    ax2.plot(x_range, p2(x_range), '--', color=high_precision_colors[0],
             linewidth=2, label=f'Regression Line\ny = {z2[0]:.5f}x + {z2[1]:.5f}')

    # Add 1:1 line
    ax2.plot([global_min, global_max], [global_min, global_max],
             '-', color='k', linewidth=1.5, label='Perfect Prediction (1:1)')

    # Set right title and labels
    ax2.set_title(f'Test Set\nR² = {test_r2:.5f}', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Observed Values', fontsize=13, fontweight='bold')

    # Add statistics
    stats_text2 = (f'n = {len(y_test)}\n'
                   f'R² = {test_r2:.5f}\n'
                   f'MAE = {test_mae:.5f}')

    ax2.text(0.05, 0.95, stats_text2, transform=ax2.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Set the same axis range
    ax1.set_xlim(global_min, global_max)
    ax1.set_ylim(global_min, global_max)
    ax2.set_xlim(global_min, global_max)

    # Add minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # Add overall title
    fig.suptitle('High-Precision Model Evaluation', fontsize=16, fontweight='bold', y=0.98)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save image
    side_by_side_file = os.path.join(output_dir, 'high_precision_side_by_side.png')
    plt.savefig(side_by_side_file, dpi=output_dpi, bbox_inches='tight')
    plt.close()

    print(f"\nGenerated high-precision academic-style visualizations from Parquet data:")
    print(f"- Final training set scatter plot: {train_scatter_file}")
    print(f"- Test set scatter plot: {test_scatter_file}")
    print(f"- Feature importance plot: {importance_file}")
    print(f"- Side-by-side comparison plot: {side_by_side_file}")

    return {
        'train_scatter': train_scatter_file,
        'test_scatter': test_scatter_file,
        'feature_importance': importance_file,
        'side_by_side': side_by_side_file
    }


def load_data_from_parquet(parquet_dir):
    """
    Load datasets from a Parquet file directory

    Parameters:
    ------
    parquet_dir: Path to the Parquet file directory

    Returns:
    ------
    Dictionary containing training

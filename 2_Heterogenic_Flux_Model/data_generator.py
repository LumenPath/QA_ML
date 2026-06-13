import pandas as pd
import numpy as np
import os
import time
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from scipy.stats import ks_2samp
from scipy import linalg


class ImprovedDataGenerator:
    """
    Improved data generator - optimized for performance, speed, and data quality
    Optimized for multiple features
    Uses SVD decomposition and Gaussian process regression to generate high-quality data
    """

    def __init__(self, input_data, output_dir, n_generate, id_col_index=0, target_col_index=1,
                 feature_start_index=2, feature_end_index=None, id_column_name='materials',
                 target_column_name='property', random_state=42):
        """
        Initialize the data generator
        """
        self.original_data = input_data
        self.output_dir = output_dir
        self.n_generate = n_generate
        self.random_state = random_state
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.id_col_index = id_col_index
        self.target_col_index = target_col_index
        self.feature_start_index = feature_start_index
        self.feature_end_index = feature_end_index
        self.id_column_name = id_column_name
        self.target_column_name = target_column_name

        # Set the random seed to ensure reproducibility
        np.random.seed(self.random_state)

    def compute_stable_correlation_matrix(self):
        """Compute a stable correlation matrix"""
        # Compute the initial correlation matrix
        if isinstance(self.original_X, pd.DataFrame):
            corr_matrix = np.corrcoef(self.original_X.values.T)
        else:
            corr_matrix = np.corrcoef(self.original_X.T)

        # Add a small diagonal perturbation
        eps = 1e-6
        corr_matrix += eps * np.eye(corr_matrix.shape[0])

        # Ensure symmetry
        corr_matrix = (corr_matrix + corr_matrix.T) / 2

        try:
            # Use eigenvalue decomposition to ensure positive definiteness
            eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-6)  # Ensure eigenvalues are positive
            stabilized_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            return stabilized_matrix
        except Exception as e:
            print(f"Correlation matrix stabilization error: {e}")
            return np.eye(corr_matrix.shape[0])

    def load_data(self):
        """Load and preprocess data"""
        try:
            # Extract features and target
            self.original_index = self.original_data.iloc[:, self.id_col_index]
            self.original_y = self.original_data.iloc[:, self.target_col_index]

            # Use the specified indexes to obtain feature columns
            if self.feature_end_index is None:
                self.feature_end_index = len(self.original_data.columns) - 1

            self.feature_names = self.original_data.columns[
                                 self.feature_start_index:self.feature_end_index + 1].tolist()
            self.original_X = self.original_data[self.feature_names]

            # Data standardization
            self.X_scaled = self.scaler_X.fit_transform(self.original_X)
            self.y_scaled = self.scaler_y.fit_transform(self.original_y.values.reshape(-1, 1))

            # Compute the correlation matrix
            self.correlation_matrix = self.compute_stable_correlation_matrix()

            print(f"Data loaded successfully, shape: {self.original_X.shape}")
            print(f"Number of features used: {len(self.feature_names)}")
            print("Selected features:", self.feature_names)

            return self

        except Exception as e:
            print(f"Data loading error: {e}")
            raise

    def build_models(self):
        """Build feature and target models"""
        try:
            print("Starting model construction...")

            # 1. Create a KDE model for each feature
            print("Building KDE models for feature distributions...")
            self.kde_models = []
            for i in range(self.X_scaled.shape[1]):
                # Use Scott's rule to calculate the optimal bandwidth
                bandwidth = 1.06 * np.std(self.X_scaled[:, i]) * (len(self.X_scaled) ** (-1 / 5))
                bandwidth = max(bandwidth, 0.01)  # Ensure the bandwidth is not too small
                kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
                kde.fit(self.X_scaled[:, i].reshape(-1, 1))
                self.kde_models.append(kde)
            print(f"KDE model construction completed, with {len(self.kde_models)} feature models in total")

            # 2. Compute the singular value decomposition of feature correlations
            print("Computing SVD decomposition of feature correlations...")
            U, s, Vt = np.linalg.svd(self.correlation_matrix)
            # Ensure all singular values are positive
            s = np.maximum(s, 1e-6)
            # Compute the Cholesky factor L such that L * L.T = the correlation matrix
            self.L = U @ np.diag(np.sqrt(s))
            print("SVD decomposition completed")

            # 3. Build a GPR model to predict target values
            print("Building the Gaussian process regression model...")
            n_features = self.X_scaled.shape[1]
            # Build a composite kernel function
            kernel = 1.0 * Matern(length_scale=[1.0] * n_features, nu=2.5) + \
                     WhiteKernel(noise_level=0.1) + \
                     1.0 * RBF(length_scale=[1.0] * n_features)

            self.gpr = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=5,  # Reduce the number of restarts to improve speed
                random_state=self.random_state,
                normalize_y=True,
                alpha=1e-6
            )

            # Train the GPR model
            self.gpr.fit(self.X_scaled, self.y_scaled)
            print("Gaussian process regression model construction completed")

            # 4. Build a nearest-neighbor model for local density evaluation
            print("Building the nearest-neighbor model...")
            self.n_neighbors = min(10, len(self.X_scaled) // 2)
            self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
            self.nn.fit(self.X_scaled)
            print("Nearest-neighbor model construction completed")

            print("All models built successfully")
            return self

        except Exception as e:
            print(f"Model construction error: {e}")
            raise

    def generate_samples(self):
        """Generate new samples - combine the advantages of two schemes"""
        try:
            print(f"Starting to generate {self.n_generate} samples...")

            # Initial random seed
            base_seed = self.random_state

            # 1. Initialize the feature array
            n_features = self.X_scaled.shape[1]
            generated_features_scaled = np.zeros((self.n_generate, n_features))

            # 2. Use batch processing to generate samples and improve performance
            batch_size = 1000  # Number of samples generated per batch
            remaining = self.n_generate
            current_idx = 0

            while remaining > 0:
                current_batch = min(batch_size, remaining)
                print(f"Generating batch: {current_batch} samples, completed: {current_idx}/{self.n_generate}")

                # Use a different random seed for the current batch
                batch_seed = base_seed + current_idx
                np.random.seed(batch_seed)

                # Sample each feature independently
                independent_samples = np.zeros((current_batch, n_features))
                for i in range(n_features):
                    # Use a different random seed for each feature
                    feature_seed = batch_seed + i * 1000
                    # Draw samples from the KDE model
                    samples = self.kde_models[i].sample(current_batch, random_state=feature_seed)
                    independent_samples[:, i] = samples.flatten()

                    # Add a small random perturbation to increase diversity - use a dynamic noise level
                    noise_level = 0.01 * np.std(samples)
                    independent_samples[:, i] += np.random.normal(0, noise_level, size=current_batch)

                # Apply the correlation structure and add random perturbations to each sample
                for j in range(current_batch):
                    # Use a different random seed for each sample
                    sample_seed = batch_seed + j * 100
                    np.random.seed(sample_seed)

                    # Basic feature generation
                    generated_features_scaled[current_idx + j] = np.dot(self.L, independent_samples[j])

                    # Add a small random perturbation
                    noise = np.random.normal(0, 0.02, size=n_features)
                    generated_features_scaled[current_idx + j] += noise

                current_idx += current_batch
                remaining -= current_batch

            # 3. Apply local structure optimization, increasing the proportion to 50%
            optimize_percent = 0.5
            optimize_indices = np.random.choice(
                self.n_generate,
                size=min(int(self.n_generate * optimize_percent), 1000),
                replace=False
            )

            print(f"Optimizing the local structure of {len(optimize_indices)} samples...")

            for idx in optimize_indices:
                # Set a different random seed for each optimized sample
                np.random.seed(base_seed + idx * 200)

                # Find nearest neighbors
                distances, nn_indices = self.nn.kneighbors([generated_features_scaled[idx]])

                # Get neighbor data
                neighbors = self.X_scaled[nn_indices[0]]

                # Compute the local mean and covariance
                local_mean = np.mean(neighbors, axis=0)
                local_cov = np.cov(neighbors.T) + 1e-6 * np.eye(n_features)

                # Use Cholesky decomposition to apply the local structure
                try:
                    L_local = np.linalg.cholesky(local_cov)
                    # Use random perturbation
                    noise = np.random.randn(n_features)
                    # Apply the local structure while retaining part of the originally generated features - use a random mixing coefficient
                    alpha = np.random.uniform(0.5, 0.9)  # Random mixing coefficient
                    generated_features_scaled[idx] = alpha * generated_features_scaled[idx] + \
                                                     (1 - alpha) * (local_mean + L_local @ noise)

                    # Add additional unique perturbation
                    extra_noise = np.random.normal(0, 0.03, size=n_features)
                    generated_features_scaled[idx] += extra_noise

                except np.linalg.LinAlgError:
                    # If decomposition fails, add random perturbation
                    generated_features_scaled[idx] += np.random.normal(0, 0.05, n_features)
                    continue

            # 4. Ensure feature diversity
            self._ensure_feature_diversity(generated_features_scaled)

            # 5. Inverse-transform standardized features
            print("Inverse-transforming standardized features...")
            self.generated_features = self.scaler_X.inverse_transform(generated_features_scaled)

            # 6. Generate target values - use a different random seed
            print("Generating target values...")
            np.random.seed(base_seed + 5000)

            y_mean, y_std = self.gpr.predict(generated_features_scaled, return_std=True)

            # Generate target values using the predicted mean and standard deviation, adding an appropriate amount of noise
            y_noise = np.random.normal(0, 1, size=y_mean.shape) * y_std * 0.1
            y_samples = y_mean.reshape(-1, 1) + y_noise.reshape(-1, 1)

            # Inverse-transform standardized target values
            self.generated_targets = self.scaler_y.inverse_transform(y_samples)

            print(f"Successfully generated {self.n_generate} new samples")
            return self

        except Exception as e:
            print(f"Sample generation error: {e}")
            raise

    def _ensure_feature_diversity(self, features_array):
        """Ensure feature diversity and handle duplicate values"""
        print("Checking and ensuring feature diversity...")
        n_samples, n_features = features_array.shape

        # For each feature, check whether duplicate values exist
        for i in range(n_features):
            # Get all values of the current feature
            feature_values = features_array[:, i]

            # Compute the ratio of unique values
            unique_ratio = len(np.unique(feature_values)) / len(feature_values)

            # If the ratio of unique values is below the threshold, add random perturbation
            if unique_ratio < 0.95:  # 95% of values should be unique
                print(f"Feature {i} has a low unique-value ratio ({unique_ratio:.2f}); adding random perturbation...")

                # Set the random seed
                np.random.seed(self.random_state + i * 200)

                # Compute the standard deviation of this feature
                feature_std = np.std(feature_values)

                # Add a small random perturbation, 1% of the standard deviation
                features_array[:, i] += np.random.normal(0, feature_std * 0.01, n_samples)

        # Check the uniqueness of overall sample rows
        print("Checking the uniqueness of sample rows...")
        duplicated_rows = 0

        for i in range(n_samples):
            # Limit the comparison range to improve performance
            for j in range(i + 1, min(i + 1000, n_samples)):
                if np.allclose(features_array[i], features_array[j], rtol=1e-5, atol=1e-5):
                    # Add random perturbation to one of the rows
                    np.random.seed(self.random_state + j * 300)
                    features_array[j] += np.random.normal(0, 0.01, n_features)
                    duplicated_rows += 1

        if duplicated_rows > 0:
            print(f"Processed {duplicated_rows} duplicate sample rows")
        else:
            print("No duplicate sample rows found")

        return features_array

    def validate_generation(self):
        """Validate the quality of the generated data"""
        try:
            print("Starting validation of generated data quality...")
            validation_metrics = {}

            # 1. Feature distribution validation
            print("Validating feature distributions...")
            feature_p_values = []
            for i, feature_name in enumerate(self.feature_names):
                if isinstance(self.original_X, pd.DataFrame):
                    original_feature = self.original_X[feature_name].values
                else:
                    original_feature = self.original_X[:, i]

                _, p_value = ks_2samp(
                    original_feature,
                    self.generated_features[:, i]
                )
                validation_metrics[f'feature_{feature_name}_ks_pvalue'] = p_value
                feature_p_values.append(p_value)

            # Compute the average and minimum values of feature p-values
            validation_metrics['feature_avg_ks_pvalue'] = np.mean(feature_p_values)
            validation_metrics['feature_min_ks_pvalue'] = np.min(feature_p_values)

            # 2. Target distribution validation
            print("Validating target variable distribution...")
            if isinstance(self.original_y, pd.Series):
                original_y = self.original_y.values
            else:
                original_y = self.original_y

            _, p_value = ks_2samp(
                original_y,
                self.generated_targets.flatten()
            )
            validation_metrics['target_ks_pvalue'] = p_value

            # 3. Correlation structure validation
            print("Validating correlation structure...")
            generated_corr = np.corrcoef(self.generated_features.T)
            validation_metrics['correlation_diff'] = np.mean(
                np.abs(self.correlation_matrix - generated_corr)
            )

            # 4. Data range validation
            for i, feature_name in enumerate(self.feature_names):
                if isinstance(self.original_X, pd.DataFrame):
                    original_range = (self.original_X[feature_name].min(), self.original_X[feature_name].max())
                else:
                    original_range = (np.min(self.original_X[:, i]), np.max(self.original_X[:, i]))

                generated_range = (np.min(self.generated_features[:, i]), np.max(self.generated_features[:, i]))

                range_diff_min = abs(original_range[0] - generated_range[0]) / max(abs(original_range[0]), 1e-10)
                range_diff_max = abs(original_range[1] - generated_range[1]) / max(abs(original_range[1]), 1e-10)

                validation_metrics[f'feature_{feature_name}_range_diff'] = (range_diff_min + range_diff_max) / 2

            # Print validation results
            print("\nValidation metrics:")
            for metric, value in validation_metrics.items():
                print(f"{metric}: {value}")

            return validation_metrics

        except Exception as e:
            print(f"Validation process error: {e}")
            raise

    def save_results(self):
        """Save the generated data using the system default precision"""
        try:
            print("Saving generated data...")

            # Create a data frame containing only the generated data
            generated_df = pd.DataFrame({
                self.id_column_name: [f"sample{i}" for i in range(len(self.original_y) + 1,
                                                                  len(self.original_y) + len(
                                                                      self.generated_targets) + 1)]
            })

            # Add the target variable using the system default precision
            generated_df[self.target_column_name] = self.generated_targets.flatten()

            # Add feature columns using the system default precision
            for i, feature_name in enumerate(self.feature_names):
                generated_df[feature_name] = self.generated_features[:, i]

            # Build the output file name
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            generated_filename = os.path.join(
                self.output_dir,
                f"generated_data_{len(self.feature_names)}f_{self.n_generate}s_{timestamp}.csv"
            )

            # Save the file using the system default precision
            generated_df.to_csv(generated_filename, index=False, lineterminator='\n')

            print(f"\nGenerated data has been saved to:\n{os.path.basename(generated_filename)}")
            print(f"Note: Generated data has been saved using the system default precision")

            return generated_filename, generated_df

        except Exception as e:
            print(f"Result saving error: {e}")
            raise

    def get_generated_data(self):
        """Get the generated data for semi-supervised learning"""
        return self.generated_features, self.generated_targets.flatten()

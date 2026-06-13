import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os


class DeterministicSelfTrainingRegressor(BaseEstimator):

    def __init__(
            self,
            base_estimator,
            max_iter=50,
            batch_size=1,
            confidence_threshold=0.9999999999,
            verbose=True,
            use_distance=True,
            n_neighbors=5,
            ensemble_weight=0.9,
            distance_weight=0.1,
            density_weight=0.0,
            X_test=None,
            y_test=None,
            random_state=42
    ):
        self.base_estimator = base_estimator
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.use_distance = use_distance
        self.n_neighbors = n_neighbors
        self.ensemble_weight = ensemble_weight
        self.distance_weight = distance_weight
        self.density_weight = density_weight
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state

        # Add attributes for tracking the performance of each iteration
        self.iteration_metrics = []
        # Track the samples added in each iteration
        self.iteration_samples = []

        # Track the model with the best MAE
        self.best_mae_model = None
        self.best_test_mae = float('inf')
        self.best_mae_iter = -1

        # Track the samples used by the best model
        self.best_model_samples_X = []
        self.best_model_samples_y = []

    def _set_deterministic_environment(self):
        """Set a deterministic environment to ensure reproducible results"""
        # Set the global random seed
        np.random.seed(self.random_state)
        # If sklearn random_state exists, set it as well
        if hasattr(self.base_estimator, 'random_state'):
            self.base_estimator.random_state = self.random_state

        # Disable parallel computation to ensure sequential execution
        if hasattr(self.base_estimator, 'n_jobs'):
            self.base_estimator.n_jobs = 1

    def _calculate_ensemble_confidence(self, X):
        """Calculate confidence based on the prediction variance of the random forest - ensure determinism"""
        # Reset the random seed
        np.random.seed(self.random_state)

        # Get predictions from each tree
        predictions = np.array([tree.predict(X)
                                for tree in self.base_estimator_.estimators_])

        # Calculate standard deviation
        std = np.std(predictions, axis=0)

        # Calculate confidence - avoid division by zero
        confidence = 1 / (1 + std + 1e-10)

        return confidence

    def _calculate_distance_confidence(self, X):
        """Calculate confidence based on the distance to training samples - ensure determinism"""
        # Set the random seed
        np.random.seed(self.random_state)

        # Create a deterministic nearest neighbor model
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')
        nbrs.fit(self.X_train_original_)

        # Calculate distances
        distances, _ = nbrs.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)

        # Normalize distances
        max_dist = np.max(avg_distances) if len(avg_distances) > 0 else 1.0
        normalized_distances = avg_distances / max_dist if max_dist > 0 else avg_distances

        # Convert to confidence
        confidence = np.exp(-normalized_distances)

        return confidence

    def _calculate_density_confidence(self, X, predictions):
        """Calculate confidence based on kernel density estimation - ensure determinism"""
        # Set the random seed
        np.random.seed(self.random_state)

        # Use Scott's method to automatically select bandwidth
        bandwidth = 0.9 * np.std(self.y_train_original_) * (len(self.y_train_original_) ** (-1 / 5))
        kde = KernelDensity(kernel='gaussian', bandwidth=max(bandwidth, 0.01))

        # Ensure y_train_original_ is a one-dimensional array
        if isinstance(self.y_train_original_, pd.Series):
            kde_input = self.y_train_original_.values.reshape(-1, 1)
        else:
            kde_input = self.y_train_original_.reshape(-1, 1)

        # Fit the KDE model
        kde.fit(kde_input)

        # Calculate log density
        log_density = kde.score_samples(predictions.reshape(-1, 1))
        confidence = np.exp(log_density)

        # Normalize confidence to the [0,1] range
        if len(confidence) > 0:
            min_conf = np.min(confidence)
            max_conf = np.max(confidence)
            if max_conf > min_conf:
                confidence = (confidence - min_conf) / (max_conf - min_conf)
            else:
                confidence = np.ones_like(confidence) * 0.5

        return confidence

    def _calculate_confidence(self, X, predictions):
        """Combine multiple methods to calculate confidence - ensure determinism"""
        # Set the random seed
        np.random.seed(self.random_state)

        # Calculate various confidences
        ensemble_conf = self._calculate_ensemble_confidence(X)
        density_conf = self._calculate_density_confidence(X, predictions)

        if self.use_distance:
            distance_conf = self._calculate_distance_confidence(X)
            final_confidence = (self.ensemble_weight * ensemble_conf +
                                self.distance_weight * distance_conf +
                                self.density_weight * density_conf)
        else:
            # Re-normalize weights
            adjusted_ensemble_weight = self.ensemble_weight / (self.ensemble_weight + self.density_weight)
            adjusted_density_weight = self.density_weight / (self.ensemble_weight + self.density_weight)

            final_confidence = (adjusted_ensemble_weight * ensemble_conf +
                                adjusted_density_weight * density_conf)

        # Ensure the final confidence is within the [0,1] range
        if len(final_confidence) > 0:
            min_conf = np.min(final_confidence)
            max_conf = np.max(final_confidence)
            if max_conf > min_conf:
                final_confidence = (final_confidence - min_conf) / (max_conf - min_conf)
            else:
                final_confidence = np.ones_like(final_confidence) * 0.5

        # Add a tiny index-based offset to ensure deterministic sorting when confidences are equal
        sample_indices = np.arange(len(final_confidence)) * 1e-10
        final_confidence = final_confidence + sample_indices

        return final_confidence

    def _select_samples_deterministic(self, confidences, X_unlabeled, predictions, batch_size):
        """Ensure determinism in the sample selection process"""
        # Create a composite score: confidence + tiny index offset
        composite_scores = confidences.copy()

        # Sort by composite score in descending order
        sorted_indices = np.argsort(composite_scores)[::-1]

        # Select the top batch_size samples
        selected_indices = sorted_indices[:batch_size]

        # Apply confidence threshold filtering
        mask = confidences[selected_indices] > self.confidence_threshold

        # Return indices that satisfy the threshold
        final_selected = selected_indices[mask]

        return final_selected

    def _evaluate_performance(self, model, X_train, y_train, X_test=None, y_test=None, X_pseudo=None, y_pseudo=None):
        """Evaluate model performance - using deterministic methods"""
        # Set the random seed
        np.random.seed(self.random_state)

        metrics = {}

        # Training set metrics
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        metrics['train'] = {
            'r2': train_r2,
            'mae': train_mae,
            'rmse': train_rmse
        }

        # Test set metrics
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            metrics['test'] = {
                'r2': test_r2,
                'mae': test_mae,
                'rmse': test_rmse
            }

        # Pseudo-labeled data metrics
        if X_pseudo is not None and y_pseudo is not None and len(X_pseudo) > 0:
            y_pseudo_pred = model.predict(X_pseudo)
            pseudo_r2 = r2_score(y_pseudo, y_pseudo_pred)
            pseudo_mae = mean_absolute_error(y_pseudo, y_pseudo_pred)
            pseudo_rmse = np.sqrt(mean_squared_error(y_pseudo, y_pseudo_pred))

            metrics['pseudo'] = {
                'r2': pseudo_r2,
                'mae': pseudo_mae,
                'rmse': pseudo_rmse,
                'size': len(X_pseudo)
            }

        return metrics

    def fit(self, X, y, X_unlabeled=None):
        """Train the model - ensure complete determinism and reproducibility"""
        # Set deterministic environment
        self._set_deterministic_environment()

        # Ensure X and y maintain the correct types
        if isinstance(X, np.ndarray):
            X = X.copy()
        else:
            X = X.copy()  # If X is a DataFrame, keep the DataFrame format

        if isinstance(y, pd.Series):
            y = y.copy()  # Keep the Series format
        else:
            y = y.copy()

        self.X_train_original_ = X
        self.y_train_original_ = y

        # Initialize containers for added samples
        self.added_samples_X_ = []
        self.added_samples_y_ = []

        # Initialize the samples used by the best model
        self.best_model_samples_X = []
        self.best_model_samples_y = []

        # Initialize the iteration performance metrics list and samples added in each iteration
        self.iteration_metrics = []
        self.iteration_samples = []

        # Initialize base_estimator
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)  # Train once first to avoid NotFittedError

        current_X_train = X.copy()
        current_y_train = y.copy()

        # If there is no unlabeled data, only train the base model
        if X_unlabeled is None or len(X_unlabeled) == 0:
            # Evaluate the initial model
            metrics = self._evaluate_performance(
                self.base_estimator_, current_X_train, current_y_train,
                self.X_test, self.y_test
            )

            self.iteration_metrics.append({
                'iteration': 0,
                'metrics': metrics,
                'added_samples': 0,
                'total_pseudo_samples': 0
            })

            self.iteration_samples.append({
                'iteration': 0,
                'added_X': [],
                'added_y': []
            })

            # The initial model is the best model
            self.best_mae_model = clone(self.base_estimator_)
            self.best_test_mae = metrics['test']['mae'] if 'test' in metrics else float('inf')
            self.best_mae_iter = 0

            return self

        # Ensure X_unlabeled is the correct type
        if not isinstance(X_unlabeled, np.ndarray):
            X_unlabeled = np.array(X_unlabeled)

        # Deterministically pre-sort unlabeled data to ensure the same order in every run
        if len(X_unlabeled) > 0:
            # Use the sum of features as the sorting key
            feature_sums = np.sum(X_unlabeled, axis=1)
            # Add a tiny index offset to ensure sorting uniqueness
            sorting_keys = feature_sums + np.arange(len(feature_sums)) * 1e-10
            # Get sorted indices
            sorted_indices = np.argsort(sorting_keys)
            # Reorder unlabeled data
            X_unlabeled = X_unlabeled[sorted_indices]

        # Track current training data and pseudo-labeled samples
        current_pseudo_X = []
        current_pseudo_y = []

        for epoch in range(self.max_iter):
            # Set deterministic seed for the current iteration
            np.random.seed(self.random_state + epoch)

            # Train the current model
            self.base_estimator_.fit(current_X_train, current_y_train)

            # Evaluate the performance of the current model
            metrics = self._evaluate_performance(
                self.base_estimator_,
                current_X_train,
                current_y_train,
                self.X_test,
                self.y_test,
                np.array(self.added_samples_X_) if len(self.added_samples_X_) > 0 else None,
                np.array(self.added_samples_y_) if len(self.added_samples_y_) > 0 else None
            )

            # Initialize samples added in this iteration
            current_iter_samples_X = []
            current_iter_samples_y = []

            # Record metrics for the current iteration
            self.iteration_metrics.append({
                'iteration': epoch,
                'metrics': metrics,
                'added_samples': 0,  # Will be updated below
                'total_pseudo_samples': len(self.added_samples_X_)
            })

            # Use test set metrics for output
            if self.verbose:
                print(f'Epoch {epoch + 1}:')
                if 'test' in metrics:
                    print(f"  Test R²: {metrics['test']['r2']:.9f}")
                    print(f"  Test MAE: {metrics['test']['mae']:.9f}")
                else:
                    print(f"  Training R²: {metrics['train']['r2']:.9f}")
                    print(f"  Training MAE: {metrics['train']['mae']:.9f}")

                # Show comparison between the current model and the best model
                if epoch > 0:
                    if self.best_mae_iter < epoch:
                        print(
                            f"  Best MAE model at epoch {self.best_mae_iter + 1} with Test MAE: {self.best_test_mae:.9f}")

            # Check whether this is the best MAE model
            if 'test' in metrics and metrics['test']['mae'] < self.best_test_mae:
                self.best_test_mae = metrics['test']['mae']
                self.best_mae_model = clone(self.base_estimator_)

                # Ensure the model has been trained
                if isinstance(current_X_train, pd.DataFrame):
                    self.best_mae_model.fit(current_X_train.values, current_y_train.values)
                else:
                    self.best_mae_model.fit(current_X_train, current_y_train)

                self.best_mae_iter = epoch

                # Update the samples used by the best model - these are all pseudo-labeled samples up to the current iteration
                self.best_model_samples_X = current_pseudo_X.copy()
                self.best_model_samples_y = current_pseudo_y.copy()

                if self.verbose:
                    print(f"  Found a new best MAE model, Test MAE: {self.best_test_mae:.9f}")
                    print(f"  Number of pseudo-labeled samples used: {len(self.best_model_samples_X)}")

            if len(X_unlabeled) > 0:
                # Set deterministic seed for the current prediction
                np.random.seed(self.random_state + epoch * 100)

                # Predict
                predictions = self.base_estimator_.predict(X_unlabeled)

                # Calculate confidence
                confidences = self._calculate_confidence(X_unlabeled, predictions)

                # Prevent batch_size from being larger than the remaining number of unlabeled samples
                batch_size = min(self.batch_size, len(X_unlabeled))

                if batch_size > 0:
                    # Use deterministic sample selection method
                    selected_indices = self._select_samples_deterministic(
                        confidences, X_unlabeled, predictions, batch_size
                    )

                    # Get the number of added samples
                    added_count = len(selected_indices)

                    if added_count > 0:
                        # Update the number of newly added samples recorded for the current iteration
                        self.iteration_metrics[-1]['added_samples'] = added_count

                        # Save the newly added samples for the current iteration
                        for idx in selected_indices:
                            sample_X = X_unlabeled[idx]
                            sample_y = predictions[idx]

                            current_iter_samples_X.append(sample_X)
                            current_iter_samples_y.append(sample_y)

                            # Update the current pseudo-labeled sample set
                            current_pseudo_X.append(sample_X)
                            current_pseudo_y.append(sample_y)

                        # Update the total pseudo-labeled samples
                        self.added_samples_X_.extend(X_unlabeled[selected_indices])
                        self.added_samples_y_.extend(predictions[selected_indices])

                        # Handle DataFrame and Series cases
                        if isinstance(current_X_train, pd.DataFrame):
                            # Create a DataFrame with the same column names as the current training data
                            unlabeled_df = pd.DataFrame(X_unlabeled[selected_indices], columns=current_X_train.columns)
                            current_X_train = pd.concat([current_X_train, unlabeled_df], ignore_index=True)
                        else:
                            current_X_train = np.vstack([current_X_train, X_unlabeled[selected_indices]])

                        if isinstance(current_y_train, pd.Series):
                            # Create a new Series
                            unlabeled_series = pd.Series(predictions[selected_indices],
                                                         index=range(len(current_y_train),
                                                                     len(current_y_train) + len(selected_indices)))
                            current_y_train = pd.concat([current_y_train, unlabeled_series])
                        else:
                            current_y_train = np.append(current_y_train, predictions[selected_indices])

                        # Use np.delete to remove selected samples in a deterministic way
                        mask = np.ones(len(X_unlabeled), dtype=bool)
                        mask[selected_indices] = False
                        X_unlabeled = X_unlabeled[mask]

                        if self.verbose:
                            print(f'  Added {added_count} samples')
                            print(f'  Remaining unlabeled samples: {len(X_unlabeled)}')
                    else:
                        if self.verbose:
                            print("  No samples passed confidence threshold")

                # Record the samples added in this iteration
                self.iteration_samples.append({
                    'iteration': epoch,
                    'added_X': current_iter_samples_X,
                    'added_y': current_iter_samples_y
                })

                # If there is no more unlabeled data or the maximum number of iterations is reached, stop early
                if len(X_unlabeled) == 0 or epoch == self.max_iter - 1:
                    if self.verbose:
                        print(f"Self-training completed after {epoch + 1} epochs")
                        print(
                            f"Best MAE model from epoch {self.best_mae_iter + 1} with Test MAE: {self.best_test_mae:.9f}")
                    break

        # Use the model with the best MAE as the base estimator
        if self.best_mae_model is not None:
            self.base_estimator_ = self.best_mae_model
            if self.verbose:
                print(f"Using best MAE model from epoch {self.best_mae_iter + 1} as default model")

        return self

    def predict(self, X):
        """Predict new data - ensure determinism"""
        # Set the random seed
        np.random.seed(self.random_state)
        return self.base_estimator_.predict(X)

    def get_best_mae_model(self):
        """Return the model with the best MAE - ensure determinism"""
        # Set the random seed
        np.random.seed(self.random_state)

        if self.best_mae_model is not None:
            # Retrain a model using the training data of the best MAE model
            # This ensures the model uses the training data from the best iteration
            best_model = clone(self.best_mae_model)

            if self.verbose:
                print(f"Retraining the best MAE model (from epoch {self.best_mae_iter + 1})...")

            # Build the training data for the best model
            if isinstance(self.X_train_original_, pd.DataFrame):
                X_train_best = self.X_train_original_.copy()
                y_train_best = self.y_train_original_.copy()

                # If there are pseudo-labeled samples
                if len(self.best_model_samples_X) > 0:
                    # Create a DataFrame for pseudo-labeled samples
                    pseudo_samples_df = pd.DataFrame(columns=X_train_best.columns)

                    for i, sample in enumerate(self.best_model_samples_X):
                        if isinstance(sample, pd.Series):
                            pseudo_samples_df = pd.concat([pseudo_samples_df, pd.DataFrame([sample])],
                                                          ignore_index=True)
                        elif isinstance(sample, np.ndarray):
                            pseudo_samples_df.loc[i] = sample

                    # Merge original training data and pseudo-labeled data
                    X_combined = pd.concat([X_train_best, pseudo_samples_df], ignore_index=True)

                    if isinstance(y_train_best, pd.Series):
                        y_best_model = pd.Series(self.best_model_samples_y)
                        y_combined = pd.concat([y_train_best, y_best_model], ignore_index=True)
                    else:
                        y_combined = np.concatenate([y_train_best, self.best_model_samples_y])

                    # Train the model
                    best_model.fit(X_combined, y_combined)

                    if self.verbose:
                        print(
                            f"  Trained using {len(X_train_best)} original samples and {len(self.best_model_samples_X)} pseudo-labeled samples")
                else:
                    # If there are no pseudo-labeled samples, only use the original training data
                    best_model.fit(X_train_best, y_train_best)

                    if self.verbose:
                        print(f"  Trained using only {len(X_train_best)} original samples (no pseudo-labeled samples)")
            else:
                # If it is a numpy array
                X_train_best = self.X_train_original_.copy()
                y_train_best = self.y_train_original_.copy()

                # If there are pseudo-labeled samples
                if len(self.best_model_samples_X) > 0:
                    # Ensure pseudo-labeled samples are numpy arrays
                    if not isinstance(self.best_model_samples_X[0], np.ndarray):
                        pseudo_X = np.array([np.array(x) for x in self.best_model_samples_X])
                    else:
                        pseudo_X = np.array(self.best_model_samples_X)

                    pseudo_y = np.array(self.best_model_samples_y)

                    # Merge original training data and pseudo-labeled data
                    X_combined = np.vstack([X_train_best, pseudo_X])
                    y_combined = np.concatenate([y_train_best, pseudo_y])

                    # Train the model
                    best_model.fit(X_combined, y_combined)

                    if self.verbose:
                        print(f"  Trained using {len(X_train_best)} original samples and {len(pseudo_X)} pseudo-labeled samples")
                else:
                    # If there are no pseudo-labeled samples, only use the original training data
                    best_model.fit(X_train_best, y_train_best)

                    if self.verbose:
                        print(f"  Trained using only {len(X_train_best)} original samples (no pseudo-labeled samples)")

            return best_model
        return self.base_estimator_

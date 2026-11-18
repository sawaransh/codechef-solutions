import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class RidgeRegressionGradientDescent:
    """
    Implement Ridge Regression using
    manual Gradient Descent optimization.
    """

    def __init__(self):
        """
        Load the correlated dataset
        and initialize required attributes.
        """
        self.dataset = pd.read_csv("ridge_regression_correlated.csv")
        self.learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
        self.regularization_values = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]
        self.iterations = 1000

        self.X = None
        self.y = None

        self.X_train = None
        self.X_validation = None
        self.X_test = None

        self.y_train = None
        self.y_validation = None
        self.y_test = None

        self.scaler = None

        self.training_results = {}
        self.evaluation_results = {}

    def prepare_data(self):
        """
        Separate the input features
        and target variable.
        """
        self.X = self.dataset.drop(columns=["target"]).values.astype(float)
        self.y = self.dataset["target"].values.astype(float)

    def split_data(self):
        """
        Divide the dataset into training,
        validation, and testing portions.
        """
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X, self.y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        self.X_train = X_train
        self.y_train = y_train
        self.X_validation = X_val
        self.y_validation = y_val
        self.X_test = X_test
        self.y_test = y_test

    def scale_features(self):
        """
        Standardize all features using
        statistics learned from training data.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_validation = self.scaler.transform(self.X_validation)
        self.X_test = self.scaler.transform(self.X_test)

    def initialize_beta(self, X):
        """
        Create the initial coefficient vector
        including support for an intercept.
        """
        n_features = X.shape[1]
        return np.zeros(n_features + 1)

    def predict(self, X, beta):
        """
        Generate predictions using
        the supplied coefficients.
        """
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_with_intercept @ beta

    def ridge_cost(self, X, y, beta, regularization):
        """
        Calculate the Ridge Regression
        cost without penalizing the intercept.
        """
        m = len(y)
        predictions = self.predict(X, beta)
        errors = predictions - y
        mse_term = (1 / (2 * m)) * np.sum(errors ** 2)
        reg_term = (regularization / (2 * m)) * np.sum(beta[1:] ** 2)
        return float(mse_term + reg_term)

    def gradient_descent(
        self,
        X,
        y,
        learning_rate,
        regularization,
        iterations
    ):
        """
        Optimize Ridge Regression coefficients
        using manual Gradient Descent.
        """
        m = len(y)
        beta = self.initialize_beta(X)
        X_with_intercept = np.c_[np.ones(m), X]

        for _ in range(iterations):
            predictions = X_with_intercept @ beta
            errors = predictions - y

            gradients = (1 / m) * (X_with_intercept.T @ errors)
            gradients[1:] += (regularization / m) * beta[1:]

            beta = beta - learning_rate * gradients

            if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e100:
                return None

        return beta

    def calculate_r2(self, y_true, y_pred):
        """
        Calculate the R2 score manually.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - (ss_res / ss_tot))

    def train_parameter_combinations(self):
        """
        Train Ridge models for every
        configured parameter combination.
        """
        self.training_results = {}
        for lr in self.learning_rates:
            for reg in self.regularization_values:
                beta = self.gradient_descent(
                    self.X_train, self.y_train, lr, reg, self.iterations
                )
                self.training_results[(lr, reg)] = beta
        return self.training_results

    def evaluate_parameter_combinations(self):
        """
        Evaluate every valid model
        using validation data.
        """
        self.evaluation_results = {}
        for params, beta in self.training_results.items():
            if beta is None:
                self.evaluation_results[params] = {
                    "cost": np.nan,
                    "r2_score": np.nan,
                }
                continue

            _, reg = params
            val_cost = self.ridge_cost(
                self.X_validation, self.y_validation, beta, reg
            )
            y_val_pred = self.predict(self.X_validation, beta)
            val_r2 = self.calculate_r2(self.y_validation, y_val_pred)

            if np.isfinite(val_cost) and np.isfinite(val_r2):
                self.evaluation_results[params] = {
                    "cost": val_cost,
                    "r2_score": val_r2,
                }
            else:
                self.evaluation_results[params] = {
                    "cost": np.nan,
                    "r2_score": np.nan,
                }
        return self.evaluation_results

    def get_best_parameters(self):
        """
        Select the best parameter combination
        using validation performance.
        """
        best_params = None
        best_r2 = -np.inf
        best_cost = np.inf

        for params, metrics in self.evaluation_results.items():
            if metrics is None:
                continue

            r2 = metrics["r2_score"]
            cost = metrics["cost"]

            if not (np.isfinite(r2) and np.isfinite(cost)):
                continue

            if np.isclose(r2, best_r2, rtol=1e-9, atol=1e-9):
                if cost < best_cost:
                    best_r2 = r2
                    best_cost = cost
                    best_params = params
            elif r2 > best_r2:
                best_r2 = r2
                best_cost = cost
                best_params = params

        return best_params

    def get_best_beta(self):
        """
        Return the coefficient vector
        belonging to the selected parameters.
        """
        best_params = self.get_best_parameters()
        if best_params is None:
            return None
        return self.training_results[best_params]

    def final_evaluation(self):
        """
        Evaluate the selected model
        on the independent test set.
        """
        best_beta = self.get_best_beta()
        if best_beta is None:
            return None
        y_test_pred = self.predict(self.X_test, best_beta)
        return self.calculate_r2(self.y_test, y_test_pred)

    def get_dataset(self):
        """
        Return the loaded dataset.
        """
        return self.dataset


if __name__ == "__main__":

    model = RidgeRegressionGradientDescent()

    model.prepare_data()
    model.split_data()
    model.scale_features()
    model.train_parameter_combinations()
    model.evaluate_parameter_combinations()

    print("Best Parameters:")
    print(model.get_best_parameters())

    print("\nBest Beta:")
    print(model.get_best_beta())

    print("\nFinal Test R2 Score:")
    print(model.final_evaluation())
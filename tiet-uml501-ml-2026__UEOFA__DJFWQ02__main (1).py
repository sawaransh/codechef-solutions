import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousePriceGradientDescent:
    """
    Build Multiple Linear Regression using
    manual Gradient Descent optimization.
    """

    def __init__(self):
        """
        Load the housing dataset and store
        the learning rates used for training.
        """
        self.df = pd.read_csv("USA_Housing.csv")
        self.data = self.df
        self.dataset = self.df
        self.learning_rates = [0.001, 0.01, 0.1, 1]

        # initialization of required attributes
        self.X = None
        self.y = None

        self.X_train = None
        self.X_validation = None
        self.X_test = None

        self.y_train = None
        self.y_validation = None
        self.y_test = None

        self.training_results = None
        self.evaluation_results = None

    def prepare_data(self):
        """
        Separate numerical input features
        and the Price target variable.
        """
        feature_cols = [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
        ]
        self.X = self.df[feature_cols].to_numpy(dtype=float)
        self.y = self.df["Price"].to_numpy(dtype=float)

    def split_data(self):
        """
        Divide the dataset into training,
        validation, and testing portions.
        """
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.30, random_state=42
        )
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(
            X_temp, y_temp, test_size=0.20, random_state=42
        )

    def scale_features(self):
        """
        Standardize features using statistics
        obtained only from the training data.
        """
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_validation = scaler.transform(self.X_validation)
        self.X_test = scaler.transform(self.X_test)

    def initialize_beta(self, X):
        """
        Create an initial coefficient vector
        compatible with the feature matrix.
        """
        return np.zeros(X.shape[1] + 1, dtype=float)

    def _add_intercept(self, X):
        """
        Prepend a column of ones for the intercept term.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        ones = np.ones((X_arr.shape[0], 1), dtype=float)
        return np.hstack((ones, X_arr))

    def predict(self, X, beta):
        """
        Generate predictions using the
        supplied regression coefficients.
        """
        X_arr = np.asarray(X, dtype=float)
        beta_arr = np.asarray(beta, dtype=float)

        if X_arr.shape[1] + 1 == beta_arr.shape[0]:
            X_design = self._add_intercept(X_arr)
        else:
            X_design = X_arr

        return X_design @ beta_arr

    def gradient_descent(self, X, y, learning_rate, iterations):
        """
        Optimize regression coefficients
        using manual Gradient Descent.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        beta = self.initialize_beta(X_arr)
        X_design = self._add_intercept(X_arr)
        m = len(y_arr)

        for _ in range(iterations):
            y_pred = X_design @ beta
            errors = y_pred - y_arr
            gradients = (2.0 / m) * (X_design.T @ errors)
            beta = beta - learning_rate * gradients

            if not np.all(np.isfinite(beta)):
                break

        return beta

    def calculate_r2(self, y_true, y_pred):
        """
        Compute the R2 score without
        using an inbuilt metric.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
            return np.nan

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return np.nan

        return float(1.0 - (ss_res / ss_tot))

    def train_models(self):
        """
        Train separate models using all
        configured learning rates.
        """
        results = {}
        for lr in self.learning_rates:
            beta = self.gradient_descent(self.X_train, self.y_train, lr, iterations=1000)
            if np.all(np.isfinite(beta)):
                results[lr] = beta
            else:
                results[lr] = None

        self.training_results = results
        return self.training_results

    def evaluate_models(self):
        """
        Evaluate every valid model on
        validation and test data.
        """
        if self.training_results is None:
            self.train_models()

        eval_results = {}
        for lr, beta in self.training_results.items():
            if beta is None or not np.all(np.isfinite(beta)):
                eval_results[lr] = {
                    "validation_r2": None,
                    "test_r2": None,
                }
                continue

            val_preds = self.predict(self.X_validation, beta)
            test_preds = self.predict(self.X_test, beta)

            val_r2 = self.calculate_r2(self.y_validation, val_preds)
            test_r2 = self.calculate_r2(self.y_test, test_preds)

            eval_results[lr] = {
                "validation_r2": val_r2 if np.isfinite(val_r2) else None,
                "test_r2": test_r2 if np.isfinite(test_r2) else None,
            }

        self.evaluation_results = eval_results
        return self.evaluation_results

    def get_best_learning_rate(self):
        """
        Return the learning rate with the
        highest valid validation R2 score.
        """
        if self.evaluation_results is None:
            self.evaluate_models()

        best_lr = None
        best_val_r2 = -np.inf

        for lr, metrics in self.evaluation_results.items():
            if not isinstance(metrics, dict):
                continue
            val_r2 = metrics.get("validation_r2")
            if val_r2 is not None and np.isfinite(val_r2):
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_lr = lr

        return best_lr

    def get_best_beta(self):
        """
        Return the coefficients belonging
        to the best validation performance.
        """
        best_lr = self.get_best_learning_rate()
        if best_lr is None or self.training_results is None:
            return None
        return self.training_results.get(best_lr)

    def final_test_score(self):
        """
        Evaluate the validation-selected
        coefficients on the test set.
        """
        best_beta = self.get_best_beta()
        if best_beta is None:
            return None
        test_preds = self.predict(self.X_test, best_beta)
        return self.calculate_r2(self.y_test, test_preds)

    def get_dataset(self):
        """
        Return the loaded housing dataset.
        """
        return self.df


if __name__ == "__main__":

    model = HousePriceGradientDescent()

    model.prepare_data()
    model.split_data()
    model.scale_features()

    print("Training Results:")
    print(model.train_models())

    print("\nEvaluation Results:")
    print(model.evaluate_models())

    print("\nBest Learning Rate:")
    print(model.get_best_learning_rate())

    print("\nBest Beta:")
    print(model.get_best_beta())

    print("\nFinal Test R2 Score:")
    print(model.final_test_score())
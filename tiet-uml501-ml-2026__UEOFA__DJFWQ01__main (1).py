
import pandas as pd
import numpy as np


class HousePriceCrossValidation:
    """
    Multiple Linear Regression using manual Least Squares
    and 5-fold cross-validation.
    """

    def __init__(self):
        """Load and store the housing dataset."""
        self.data = pd.read_csv("USA_Housing.csv")

        self.X = None
        self.y = None
        self.X_scaled = None

        self.mean = None
        self.std = None

        self.folds = None
        self.validation_results = []

        self.best_beta = None
        self.final_r2 = None

    def prepare_data(self):
        """Separate numerical features and target."""
        feature_columns = [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population"
        ]

        self.X = self.data[feature_columns].to_numpy(dtype=float)
        self.y = self.data["Price"].to_numpy(dtype=float)

        return self.X, self.y

    def scale_features(self):
        """Standardize numerical input features."""
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)

        self.std[self.std == 0] = 1

        self.X_scaled = (self.X - self.mean) / self.std

        return self.X_scaled

    def create_folds(self):
        """Divide all samples into exactly five folds."""
        indices = np.arange(len(self.X_scaled))

        np.random.seed(42)
        np.random.shuffle(indices)

        self.folds = np.array_split(indices, 5)

        return self.folds

    def fit_least_squares(self, X, y):
        """Fit regression coefficients using the pseudoinverse."""
        X_with_intercept = np.column_stack(
            (np.ones(X.shape[0]), X)
        )

        beta = np.linalg.pinv(X_with_intercept) @ y

        return beta

    def predict(self, X, beta):
        """Generate predictions using the coefficients."""
        X_with_intercept = np.column_stack(
            (np.ones(X.shape[0]), X)
        )

        return X_with_intercept @ beta

    def calculate_r2(self, y_true, y_pred):
        """Calculate R-squared manually."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    def run_cross_validation(self):
        """Perform exactly five cross-validation iterations."""
        self.validation_results = []

        for i in range(5):
            test_indices = self.folds[i]

            train_indices = np.concatenate([
                self.folds[j]
                for j in range(5)
                if j != i
            ])

            X_train = self.X_scaled[train_indices]
            y_train = self.y[train_indices]

            X_test = self.X_scaled[test_indices]
            y_test = self.y[test_indices]

            beta = self.fit_least_squares(X_train, y_train)

            predictions = self.predict(X_test, beta)

            r2 = self.calculate_r2(y_test, predictions)

            self.validation_results.append({
                "fold": i + 1,
                "beta": beta,
                "r2_score": r2
            })

        return self.validation_results

    def get_best_beta(self):
        """Return coefficients from the fold with the highest R2."""
        if not self.validation_results:
            return None

        best_result = max(
            self.validation_results,
            key=lambda result: result["r2_score"]
        )

        self.best_beta = best_result["beta"]

        return self.best_beta

    def final_evaluation(self):
        """Evaluate the best coefficients on a 70/30 split."""
        if self.best_beta is None:
            self.get_best_beta()

        indices = np.arange(len(self.X_scaled))

        np.random.seed(42)
        np.random.shuffle(indices)

        split_index = int(0.70 * len(indices))

        train_indices = indices[:split_index]
        test_indices = indices[split_index:]

        X_test = self.X_scaled[test_indices]
        y_test = self.y[test_indices]

        predictions = self.predict(X_test, self.best_beta)

        self.final_r2 = self.calculate_r2(y_test, predictions)

        return self.final_r2

    def get_dataset(self):
        """Return the loaded dataset."""
        return self.data


if __name__ == "__main__":

    model = HousePriceCrossValidation()

    model.prepare_data()
    model.scale_features()
    model.create_folds()

    print("Cross Validation Results:")
    print(model.run_cross_validation())

    print("\nBest Beta:")
    print(model.get_best_beta())

    print("\nFinal R2 Score:")
    print(model.final_evaluation())

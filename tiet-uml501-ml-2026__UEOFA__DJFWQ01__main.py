import pandas as pd
import numpy as np


class HousePriceCrossValidation:
    """
    Build a Multiple Linear Regression model using
    manual Least Squares and 5-fold cross validation.
    """

    def __init__(self):
        """
        Load and store the housing dataset.
        """
        self.df = pd.read_csv("USA_Housing.csv")
        self.X = None
        self.y = None
        self.X_scaled = None
        self.folds = []
        self.cv_results = []

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

    def scale_features(self):
        """
        Standardize the numerical
        input features.
        """
        self.mean_ = np.mean(self.X, axis=0)
        self.std_ = np.std(self.X, axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.X_scaled = (self.X - self.mean_) / self.std_

    def create_folds(self):
        """
        Divide the dataset into
        exactly five folds.
        """
        n_samples = len(self.X_scaled)
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_samples)
        self.folds = np.array_split(indices, 5)

    def fit_least_squares(
        self,
        X,
        y
    ):
        """
        Fit Multiple Linear Regression
        using the Least Squares method.
        """
        X_b = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.pinv(X_b) @ y
        return beta

    def predict(
        self,
        X,
        beta
    ):
        """
        Generate predictions using
        the supplied coefficients.
        """
        X_b = np.column_stack([np.ones(len(X)), X])
        return X_b @ beta

    def calculate_r2(
        self,
        y_true,
        y_pred
    ):
        """
        Calculate the R2 score
        without using an inbuilt metric.
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - (ss_res / ss_tot))

    def run_cross_validation(self):
        """
        Run five cross-validation iterations
        and store each fold's beta and R2 score.
        """
        self.cv_results = []
        for i in range(5):
            test_idx = self.folds[i]
            train_idx = np.concatenate([self.folds[j] for j in range(5) if j != i])

            X_train, y_train = self.X_scaled[train_idx], self.y[train_idx]
            X_test, y_test = self.X_scaled[test_idx], self.y[test_idx]

            beta = self.fit_least_squares(X_train, y_train)
            y_pred = self.predict(X_test, beta)
            r2 = self.calculate_r2(y_test, y_pred)

            self.cv_results.append({
                "fold": i + 1,
                "beta": beta,
                "r2_score": r2
            })

        return self.cv_results

    def get_best_beta(self):
        """
        Return the coefficient vector
        with the highest cross-validation R2 score.
        """
        if not self.cv_results:
            self.run_cross_validation()
        best_result = max(self.cv_results, key=lambda item: item["r2_score"])
        return best_result["beta"]

    def final_evaluation(self):
        """
        Evaluate the best coefficient vector
        on a final 70-30 data split.
        """
        best_beta = self.get_best_beta()
        n_samples = len(self.X_scaled)
        split_idx = int(0.7 * n_samples)

        X_test = self.X_scaled[split_idx:]
        y_test = self.y[split_idx:]

        y_pred = self.predict(X_test, best_beta)
        return self.calculate_r2(y_test, y_pred)

    def get_dataset(self):
        """
        Return the loaded housing dataset.
        """
        return self.df


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
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


class HousingRegularizationCV:
    """
    Compare RidgeCV and LassoCV
    for house-value prediction.
    """

    def __init__(self):
        """
        Load the housing dataset and
        initialize the alpha search space.
        """
        self.dataset = pd.read_csv("california_housing.csv")
        self.data = self.dataset

        self.alphas = [
            0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 20, 50, 100
        ]
        self.candidate_alphas = self.alphas

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None

        self.ridge_model = None
        self.lasso_model = None
        self.ridge_cv_model = None
        self.lasso_cv_model = None

        self.evaluation_results = {}

    def clean_dataset(self):
        """
        Remove duplicate rows and
        handle missing numerical values.
        """
        self.dataset = self.dataset.drop_duplicates()
        self.dataset = self.dataset.dropna(subset=["median_house_value"])

        num_cols = self.dataset.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.dataset[col].isnull().sum() > 0:
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].median())

        non_num_cols = [c for c in self.dataset.columns if c not in num_cols]
        for col in non_num_cols:
            if self.dataset[col].isnull().sum() > 0:
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])

        self.data = self.dataset
        return self.dataset

    def encode_categorical_features(self):
        """
        Convert ocean_proximity into
        numerical dummy features.
        """
        if "ocean_proximity" in self.dataset.columns:
            self.dataset = pd.get_dummies(
                self.dataset,
                columns=["ocean_proximity"],
                drop_first=True,
                dtype=float
            )

        self.data = self.dataset
        return self.dataset

    def prepare_data(self):
        """
        Separate input features from
        the median house value target.
        """
        self.X = self.dataset.drop(columns=["median_house_value"]).copy()
        self.y = self.dataset["median_house_value"].copy()

        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.y = pd.to_numeric(self.y, errors="coerce")

        return self.X, self.y

    def split_data(self):
        """
        Divide the prepared dataset into
        training and independent test data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """
        Standardize features using parameters
        learned only from training data.
        """
        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test

    def train_ridge_cv(self):
        """
        Train RidgeCV and allow cross validation
        to select a regularization value.
        """
        model = RidgeCV(alphas=self.alphas, cv=5)
        model.fit(self.X_train, self.y_train)

        self.ridge_model = model
        self.ridge_cv_model = model

        return self.ridge_model

    def train_lasso_cv(self):
        """
        Train LassoCV and allow cross validation
        to select a regularization value.
        """
        model = LassoCV(
            alphas=self.alphas,
            cv=5,
            max_iter=20000,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)

        self.lasso_model = model
        self.lasso_cv_model = model

        return self.lasso_model

    def evaluate_models(self):
        """
        Evaluate both models using
        R2, RMSE, and MAE.
        """
        models = {
            "RidgeCV": self.ridge_model,
            "LassoCV": self.lasso_model,
        }

        self.evaluation_results = {}

        for name, model in models.items():
            if model is not None:
                preds = model.predict(self.X_test)
                r2 = r2_score(self.y_test, preds)
                mse = mean_squared_error(self.y_test, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(self.y_test, preds)

                self.evaluation_results[name] = {
                    "r2_score": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "R2": float(r2),
                    "RMSE": float(rmse),
                    "MAE": float(mae),
                }

        return self.evaluation_results

    def get_selected_alphas(self):
        """
        Return the alpha selected
        by each cross-validated model.
        """
        ridge_alpha = float(self.ridge_model.alpha_) if (self.ridge_model is not None and hasattr(self.ridge_model, 'alpha_')) else None
        lasso_alpha = float(self.lasso_model.alpha_) if (self.lasso_model is not None and hasattr(self.lasso_model, 'alpha_')) else None

        return {
            "RidgeCV": ridge_alpha,
            "LassoCV": lasso_alpha,
        }

    def get_best_model(self):
        """
        Select the stronger model from
        measured test-set performance.
        """
        if not self.evaluation_results:
            self.evaluate_models()

        best_name = None
        best_r2 = -float("inf")
        best_rmse = float("inf")

        for name, metrics in self.evaluation_results.items():
            r2 = metrics.get("r2_score", metrics.get("R2", metrics.get("r2")))
            rmse = metrics.get("rmse", metrics.get("RMSE", metrics.get("Rmse")))

            if r2 is None:
                continue

            curr_r2 = float(r2)
            curr_rmse = float(rmse) if rmse is not None else float("inf")

            if best_name is None:
                best_name = name
                best_r2 = curr_r2
                best_rmse = curr_rmse
            else:
                if curr_r2 > best_r2 and not np.isclose(curr_r2, best_r2, atol=1e-7, rtol=1e-7):
                    best_name = name
                    best_r2 = curr_r2
                    best_rmse = curr_rmse
                elif np.isclose(curr_r2, best_r2, atol=1e-7, rtol=1e-7):
                    if curr_rmse < best_rmse:
                        best_name = name
                        best_r2 = curr_r2
                        best_rmse = curr_rmse

        return best_name

    def get_dataset(self):
        """
        Return the current dataset.
        """
        return self.dataset


if __name__ == "__main__":

    model = HousingRegularizationCV()

    model.clean_dataset()
    model.encode_categorical_features()
    model.prepare_data()
    model.split_data()
    model.scale_features()

    model.train_ridge_cv()
    model.train_lasso_cv()

    print("Selected Alphas:")
    print(model.get_selected_alphas())

    print("\nModel Evaluation:")
    print(model.evaluate_models())

    print("\nBest Model:")
    print(model.get_best_model())
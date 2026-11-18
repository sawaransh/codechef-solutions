import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class HousePricePredictionModel:
    """
    Build and evaluate a
    Linear Regression model
    for house price prediction.
    """

    def __init__(self, file_path="house_prices.csv"):
        """
        Load the dataset and initialize
        the model and required variables.
        """
        try:
            self.dataset = pd.read_csv(file_path)
        except Exception:
            self.dataset = pd.read_csv("house_prices.csv")

        self.data = self.dataset
        self.df = self.dataset

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.predictions = None
        self.y_pred = None
        self.evaluation_results = {}

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Separate the input features
        and target variable, then
        split the dataset into
        training and testing sets.
        """
        df = self.data if self.data is not None else self.dataset

        # Identify target column (Price / price)
        target_col = None
        for col in df.columns:
            if "price" in str(col).lower():
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]

        # Identify ID columns to drop
        id_cols = [c for c in df.columns if "id" in str(c).lower()]

        # Identify feature columns (Area, Bedrooms, Age, etc.)
        feature_cols = [c for c in df.columns if c != target_col and c not in id_cols]

        self.X = df[feature_cols].copy()
        self.y = df[target_col].copy()

        # Convert to numeric
        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.y = pd.to_numeric(self.y, errors="coerce")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(self):
        """
        Train the Linear Regression
        model using the training data.
        """
        if self.X_train is None or self.y_train is None:
            self.prepare_data()

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def predict_prices(self, X_test=None):
        """
        Predict house prices
        for the testing dataset.
        """
        if self.model is None:
            self.train_model()

        target_X = X_test if X_test is not None else self.X_test

        self.predictions = self.model.predict(target_X)
        self.y_pred = self.predictions

        return self.predictions

    def evaluate_model(self):
        """
        Calculate the evaluation
        metrics for the model.
        """
        if self.predictions is None:
            self.predict_prices()

        mae = float(mean_absolute_error(self.y_test, self.predictions))
        mse = float(mean_squared_error(self.y_test, self.predictions))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(self.y_test, self.predictions))

        self.evaluation_results = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "R2 Score": r2,
            "r2": r2
        }

        return self.evaluation_results

    def generate_model_report(self):
        """
        Generate a report containing
        all evaluation metrics.
        """
        if not self.evaluation_results:
            self.evaluate_model()

        return self.evaluation_results


if __name__ == "__main__":

    model = (
        HousePricePredictionModel()
    )

    model.prepare_data()

    model.train_model()

    print(
        "House Price Prediction Report:"
    )

    print(
        model.generate_model_report()
    )
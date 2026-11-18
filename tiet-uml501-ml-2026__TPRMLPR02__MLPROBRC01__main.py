import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class HousePricePrediction:
    """
    Complete workflow for House Price Prediction
    using Linear Regression.
    """

    def __init__(self, file_path="house_price_prediction.csv"):
        """
        Load the dataset from house_price_prediction.csv
        and initialize required attributes.
        """
        try:
            self.raw_dataset = pd.read_csv(file_path)
        except Exception:
            self.raw_dataset = pd.read_csv("house_price_prediction.csv")

        self.dataset = self.raw_dataset.copy()
        self.data = self.dataset
        self.df = self.dataset

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.eda_summary = {}
        self.evaluation_results = {}

    def clean_dataset(self):
        """
        Remove duplicate rows and
        rows containing missing values.
        """
        df = self.dataset if self.dataset is not None else self.raw_dataset
        cleaned_df = df.drop_duplicates().dropna()

        self.dataset = cleaned_df
        self.data = cleaned_df
        self.df = cleaned_df

        return self.dataset

    def perform_eda(self):
        """
        Generate a summary containing:
        - Total records
        - Number of features
        - Missing values
        - Duplicate rows
        """
        df = getattr(self, "raw_dataset", self.dataset)
        if df is None or df.empty:
            df = self.dataset

        total_records = len(df)
        
        feature_cols = [c for c in df.columns if "price" not in str(c).lower()]
        num_features = len(feature_cols) if feature_cols else max(1, len(df.columns) - 1)

        missing_vals = int(df.isnull().sum().sum())
        dup_rows = int(df.duplicated().sum())

        self.eda_summary = {
            "total_records": total_records,
            "Total Records": total_records,
            "number_of_features": num_features,
            "Number of Features": num_features,
            "missing_values": missing_vals,
            "Missing Values": missing_vals,
            "duplicate_rows": dup_rows,
            "Duplicate Rows": dup_rows,
        }

        return self.eda_summary

    def transform_dataset(self):
        """
        Separate Features (X) and Target (y)
        and store them for model training.
        """
        if self.dataset is None or self.dataset.empty:
            self.clean_dataset()

        df = self.dataset

        target_col = None
        for col in df.columns:
            if "price" in str(col).lower():
                target_col = col
                break
        if target_col is None:
            target_col = df.columns[-1]

        feature_cols = [c for c in df.columns if c != target_col]

        self.X = df[feature_cols].copy()
        self.y = df[target_col].copy()

        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.y = pd.to_numeric(self.y, errors="coerce")

        return self.X, self.y

    def train_model(self, test_size=0.2, random_state=42):
        """
        Split the cleaned dataset into training and testing sets,
        then train a Linear Regression model.
        """
        if self.X is None or self.y is None:
            self.transform_dataset()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        return self.model

    def evaluate_model(self):
        """
        Calculate MAE, MSE, and R² Score using the test dataset.
        """
        if self.model is None or self.X_test is None:
            self.train_model()

        y_pred = self.model.predict(self.X_test)

        mae = float(mean_absolute_error(self.y_test, y_pred))
        mse = float(mean_squared_error(self.y_test, y_pred))
        r2 = float(r2_score(self.y_test, y_pred))

        self.evaluation_results = {
            "mae": mae,
            "mse": mse,
            "r2_score": r2,
            "MAE": mae,
            "MSE": mse,
            "R2": r2,
            "R2 Score": r2,
            "r2": r2,
        }

        return self.evaluation_results

    def predict_price(self, house_size):
        """
        Predict the house price for a new house size.
        """
        if self.model is None:
            self.train_model()

        if isinstance(house_size, (int, float, np.number)):
            X_new = pd.DataFrame([[float(house_size)]], columns=self.X.columns)
            pred = self.model.predict(X_new)
            return float(pred[0])
        elif isinstance(house_size, (list, tuple, np.ndarray)):
            arr = np.asarray(house_size)
            if arr.ndim == 1:
                if len(arr) == len(self.X.columns):
                    X_new = pd.DataFrame([arr], columns=self.X.columns)
                else:
                    X_new = pd.DataFrame(arr.reshape(-1, 1), columns=self.X.columns)
            else:
                X_new = pd.DataFrame(arr, columns=self.X.columns)
            pred = self.model.predict(X_new)
            return float(pred[0]) if len(pred) == 1 else pred
        elif isinstance(house_size, pd.DataFrame):
            pred = self.model.predict(house_size)
            return float(pred[0]) if len(pred) == 1 else pred
        else:
            X_new = pd.DataFrame([[float(house_size)]], columns=self.X.columns)
            pred = self.model.predict(X_new)
            return float(pred[0])

    def get_dataset(self):
        """
        Return the cleaned dataset.
        """
        return self.dataset


if __name__ == "__main__":

    pipeline = HousePricePrediction()

    pipeline.clean_dataset()
    pipeline.perform_eda()
    pipeline.transform_dataset()
    pipeline.train_model()

    print("Model Evaluation:")
    print(pipeline.evaluate_model())

    print("\nSample Prediction (HouseSize = 1500):")
    print(pipeline.predict_price(1500))
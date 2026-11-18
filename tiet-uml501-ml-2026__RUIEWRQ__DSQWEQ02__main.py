import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


class HittersSalaryRegression:
    """
    Compare Linear, Ridge, and LASSO
    regression for salary prediction.
    """

    def __init__(self):
        """
        Load the Hitters dataset and
        initialize required attributes.
        """
        self.dataset = pd.read_csv("Hitters.csv")
        self.data = self.dataset
        self.alpha = 0.5748
        self.random_state = 42

        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None

        self.linear_model = None
        self.ridge_model = None
        self.lasso_model = None

        self.evaluation_results = {}

    def clean_dataset(self):
        """
        Clean the dataset before preprocessing
        """
        # Remove duplicate rows
        self.dataset = self.dataset.drop_duplicates()

        # Remove rows where Salary is missing
        self.dataset = self.dataset.dropna(subset=["Salary"])

        # Handle any remaining missing values in features
        num_cols = self.dataset.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.dataset[col].isnull().sum() > 0:
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].median())

        cat_cols = self.dataset.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            if self.dataset[col].isnull().sum() > 0:
                self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])

        self.data = self.dataset
        return self.dataset

    def handle_outliers(self):
        """
        Limit extreme values in numerical
        input features using IQR boundaries.
        """
        numeric_features = [
            col for col in self.dataset.select_dtypes(include=[np.number]).columns
            if col != "Salary"
        ]

        for col in numeric_features:
            Q1 = self.dataset[col].quantile(0.25)
            Q3 = self.dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            self.dataset[col] = np.clip(self.dataset[col], lower_bound, upper_bound)

        self.data = self.dataset
        return self.dataset

    def encode_categorical_features(self):
        """
        Convert categorical predictors
        into numerical dummy features.
        """
        categorical_cols = ["League", "Division", "NewLeague"]
        existing_cols = [c for c in categorical_cols if c in self.dataset.columns]

        if existing_cols:
            self.dataset = pd.get_dummies(
                self.dataset,
                columns=existing_cols,
                drop_first=True,
                dtype=float
            )

        self.data = self.dataset
        return self.dataset

    def prepare_data(self):
        """
        Separate input features
        and the Salary target.
        """
        self.X = self.dataset.drop(columns=["Salary"]).copy()
        self.y = self.dataset["Salary"].copy()

        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.y = pd.to_numeric(self.y, errors="coerce")

        return self.X, self.y

    def split_data(self):
        """
        Divide the dataset into
        70% training and 30% testing data.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.30,
            random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def scale_features(self):
        """
        Scale features using parameters
        learned from training data only.
        """
        self.scaler = StandardScaler()

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        return self.X_train, self.X_test

    def train_linear_model(self):
        """
        Train and store the
        Linear Regression model.
        """
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train, self.y_train)

        return self.linear_model

    def train_ridge_model(self):
        """
        Train Ridge Regression using
        the configured regularization value.
        """
        self.ridge_model = Ridge(alpha=self.alpha, random_state=42)
        self.ridge_model.fit(self.X_train, self.y_train)

        return self.ridge_model

    def train_lasso_model(self):
        """
        Train LASSO Regression using
        the configured regularization value.
        """
        self.lasso_model = Lasso(alpha=self.alpha, random_state=42)
        self.lasso_model.fit(self.X_train, self.y_train)

        return self.lasso_model

    def evaluate_models(self):
        """
        Evaluate every model using
        R2 score and RMSE.
        """
        models = {
            "Linear Regression": self.linear_model,
            "Ridge Regression": self.ridge_model,
            "LASSO Regression": self.lasso_model,
        }

        self.evaluation_results = {}

        for name, model in models.items():
            if model is not None:
                preds = model.predict(self.X_test)
                r2 = r2_score(self.y_test, preds)
                mse = mean_squared_error(self.y_test, preds)
                rmse = np.sqrt(mse)

                self.evaluation_results[name] = {
                    "r2_score": float(r2),
                    "rmse": float(rmse),
                    "R2": float(r2),
                    "RMSE": float(rmse),
                }

        return self.evaluation_results

    def get_best_model(self):
        """
        Select the strongest model
        from the measured test performance.
        """
        if not self.evaluation_results:
            self.evaluate_models()

        best_name = None
        best_r2 = float("-inf")
        best_rmse = float("inf")

        for name, metrics in self.evaluation_results.items():
            r2 = metrics.get("r2_score", metrics.get("R2"))
            rmse = metrics.get("rmse", metrics.get("RMSE"))

            if r2 > best_r2:
                best_r2 = r2
                best_rmse = rmse
                best_name = name
            elif np.isclose(r2, best_r2, rtol=1e-9, atol=1e-12):
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_name = name

        return best_name

    def get_dataset(self):
        """
        Return the current dataset.
        """
        return self.dataset


if __name__ == "__main__":

    model = HittersSalaryRegression()

    model.clean_dataset()
    model.handle_outliers()
    model.encode_categorical_features()
    model.prepare_data()
    model.split_data()
    model.scale_features()

    model.train_linear_model()
    model.train_ridge_model()
    model.train_lasso_model()

    print("Model Evaluation:")
    print(model.evaluate_models())

    print("\nBest Model:")
    print(model.get_best_model())
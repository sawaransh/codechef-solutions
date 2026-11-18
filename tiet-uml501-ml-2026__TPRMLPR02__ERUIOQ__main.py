import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV
)
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class CropYieldAnomalyRegression:
    """
    Estimate crop-yield anomalies using
    multiple advanced regression models.
    """

    def __init__(self):
        """
        Load the crop-yield dataset and
        initialize the regression pipeline.
        """
        self.dataset = pd.read_csv("crop_yield.csv")
        self.continuous_features = [
            "Area",
            "Annual_Rainfall",
            "Fertilizer",
            "Pesticide",
        ]
        self.alphas = np.logspace(-3, 3, 50)
        self.feature_names = []
        self.trend_models = {}
        self.global_trend = None

        self.train_data = None
        self.test_data = None
        self.train_df = None
        self.test_df = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.scaler = None
        self.linear_model = None
        self.polynomial = None
        self.poly = None
        self.polynomial_model = None
        self.poly_model = None
        self.ridge_model = None
        self.lasso_model = None
        self.evaluation_results = {}

    def clean_dataset(self):
        """
        Clean duplicate, missing, and
        invalid dataset observations.
        """
        df = self.dataset.copy()
        df = df.drop_duplicates()

        for col in ["Crop", "Season", "State"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = np.nan

        numeric_cols = [
            "Crop_Year",
            "Area",
            "Production",
            "Annual_Rainfall",
            "Fertilizer",
            "Pesticide",
            "Yield",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Crop", "State", "Crop_Year", "Yield"])
        df["Crop_Year"] = df["Crop_Year"].astype(int)

        df = df[(df["Yield"] >= 0) & (df["Area"].isna() | (df["Area"] > 0))]

        for col in ["Area", "Annual_Rainfall", "Fertilizer", "Pesticide"]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if "Season" in df.columns:
            df["Season"] = df["Season"].fillna(df["Season"].mode()[0])

        self.dataset = df.reset_index(drop=True)

    def split_temporal_data(self):
        """
        Divide observations into earlier
        training years and later test years.
        """
        self.dataset = self.dataset.sort_values("Crop_Year").reset_index(drop=True)
        unique_years = np.sort(self.dataset["Crop_Year"].unique())
        split_idx = int(np.floor(0.8 * len(unique_years)))
        split_idx = max(1, min(split_idx, len(unique_years) - 1))
        cutoff_year = unique_years[split_idx - 1]

        self.train_data = self.dataset[self.dataset["Crop_Year"] <= cutoff_year].copy().reset_index(drop=True)
        self.test_data = self.dataset[self.dataset["Crop_Year"] > cutoff_year].copy().reset_index(drop=True)
        self.train_df = self.train_data
        self.test_df = self.test_data

    def create_yield_anomaly(self):
        """
        Detrend yield within State-Crop groups
        using historical training data only.
        """
        fallback_lr = LinearRegression()
        fallback_lr.fit(self.train_data[["Crop_Year"]].values, self.train_data["Yield"].values)
        self.global_trend = float(fallback_lr.coef_[0])

        self.trend_models = {}
        for (state, crop), grp in self.train_data.groupby(["State", "Crop"]):
            if len(grp) >= 2 and grp["Crop_Year"].nunique() >= 2:
                lr = LinearRegression()
                lr.fit(grp[["Crop_Year"]].values, grp["Yield"].values)
                self.trend_models[(state, crop)] = lr

        def compute_anomaly(df):
            df = df.copy()
            expected = fallback_lr.predict(df[["Crop_Year"]].values)
            for (state, crop), idx in df.groupby(["State", "Crop"]).groups.items():
                grp_model = self.trend_models.get((state, crop))
                if grp_model is not None:
                    expected[idx] = grp_model.predict(df.loc[idx, ["Crop_Year"]].values)
            df["Expected_Yield"] = expected
            df["Yield_Anomaly"] = df["Yield"].values - expected
            return df

        self.train_data = compute_anomaly(self.train_data)
        self.test_data = compute_anomaly(self.test_data)
        self.train_df = self.train_data
        self.test_df = self.test_data
        self.dataset = pd.concat([self.train_data, self.test_data], ignore_index=True)

    def prepare_features(self):
        """
        Build aligned numerical feature
        matrices for anomaly prediction.
        """
        self.y_train = self.train_data["Yield_Anomaly"].values.astype(float)
        self.y_test = self.test_data["Yield_Anomaly"].values.astype(float)

        drop_cols = ["Crop_Year", "Yield", "Production", "Yield_Anomaly", "Expected_Yield"]
        X_train_raw = self.train_data.drop(columns=[c for c in drop_cols if c in self.train_data.columns])
        X_test_raw = self.test_data.drop(columns=[c for c in drop_cols if c in self.test_data.columns])

        cat_cols = [c for c in ["Crop", "Season", "State"] if c in X_train_raw.columns]

        X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True, dtype=float)
        X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True, dtype=float)
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)

        other_cols = [c for c in X_train_enc.columns if c not in self.continuous_features]
        ordered_cols = self.continuous_features + other_cols
        X_train_enc = X_train_enc[ordered_cols]
        X_test_enc = X_test_enc[ordered_cols]

        self.feature_names = list(ordered_cols)
        self.X_train = X_train_enc.values.astype(float)
        self.X_test = X_test_enc.values.astype(float)

    def scale_features(self):
        """
        Standardize features using values
        learned only from training data.
        """
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_linear_model(self):
        """
        Train Multiple Linear Regression
        on the anomaly target.
        """
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_train, self.y_train)

    def train_polynomial_model(self):
        """
        Generate degree-2 polynomial features
        only from the continuous predictors.
        """
        n_cont = len(self.continuous_features)
        self.polynomial = PolynomialFeatures(degree=2, include_bias=False)
        self.poly = self.polynomial

        cont_train = self.polynomial.fit_transform(self.X_train[:, :n_cont])
        cont_test = self.polynomial.transform(self.X_test[:, :n_cont])

        self.X_train_poly = np.hstack([cont_train, self.X_train[:, n_cont:]])
        self.X_test_poly = np.hstack([cont_test, self.X_test[:, n_cont:]])

        self.polynomial_model = LinearRegression()
        self.polynomial_model.fit(self.X_train_poly, self.y_train)
        self.poly_model = self.polynomial_model

    def train_ridge_model(self):
        """
        Train cross-validated Ridge Regression
        using the scaled training data.
        """
        self.ridge_model = RidgeCV(alphas=self.alphas, cv=5)
        self.ridge_model.fit(self.X_train, self.y_train)

    def train_lasso_model(self):
        """
        Train cross-validated LASSO Regression
        using the scaled training data.
        """
        self.lasso_model = LassoCV(alphas=self.alphas, cv=5, max_iter=5000, random_state=42)
        self.lasso_model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        """
        Evaluate all trained models using
        RMSE, MAE, R2, and correlation.
        """
        predictions = {
            "Linear Regression": self.linear_model.predict(self.X_test),
            "Polynomial Regression": self.polynomial_model.predict(self.X_test_poly),
            "Ridge Regression": self.ridge_model.predict(self.X_test),
            "LASSO Regression": self.lasso_model.predict(self.X_test),
        }

        results = {}
        for name, y_pred in predictions.items():
            rmse = float(np.sqrt(mean_squared_error(self.y_test, y_pred)))
            mae = float(mean_absolute_error(self.y_test, y_pred))
            r2 = float(r2_score(self.y_test, y_pred))
            if np.std(y_pred) == 0 or np.std(self.y_test) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(self.y_test, y_pred)[0, 1])

            results[name] = {
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "correlation": corr,
            }

        self.evaluation_results = results
        return results

    def get_best_model(self):
        """
        Select the strongest model using
        measured test-set performance.
        """
        if not self.evaluation_results:
            self.evaluate_models()

        best_name = max(
            self.evaluation_results.keys(),
            key=lambda name: (
                round(self.evaluation_results[name]["r2_score"], 6),
                -self.evaluation_results[name]["rmse"],
            ),
        )
        return best_name

    def get_selected_alphas(self):
        """
        Return the regularization values
        selected by Ridge and LASSO.
        """
        return {
            "Ridge Regression": float(self.ridge_model.alpha_),
            "LASSO Regression": float(self.lasso_model.alpha_),
        }

    def get_dataset(self):
        """
        Return the current processed dataset.
        """
        return self.dataset


if __name__ == "__main__":

    model = CropYieldAnomalyRegression()

    model.clean_dataset()
    model.split_temporal_data()
    model.create_yield_anomaly()
    model.prepare_features()
    model.scale_features()

    model.train_linear_model()
    model.train_polynomial_model()
    model.train_ridge_model()
    model.train_lasso_model()

    print("Selected Alphas:")
    print(model.get_selected_alphas())

    print("\nModel Evaluation:")
    print(model.evaluate_models())

    print("\nBest Model:")
    print(model.get_best_model())
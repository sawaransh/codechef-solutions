import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class WeatherDecisionTree:
    """Train and tune Decision Trees for next-day rain prediction."""

    def __init__(self):
        # TODO: Load weather_large.csv
        self.df = pd.read_csv("weather_large.csv")

        self.parameter_sets = [
            {
                "criterion": "gini",
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": None
            },
            {
                "criterion": "entropy",
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": None
            },
            {
                "criterion": "gini",
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 4,
                "max_features": "sqrt"
            },
            {
                "criterion": "entropy",
                "max_depth": None,
                "min_samples_split": 15,
                "min_samples_leaf": 5,
                "max_features": "sqrt"
            },
            {
                "criterion": "log_loss",
                "max_depth": 10,
                "min_samples_split": 8,
                "min_samples_leaf": 3,
                "max_features": None
            }
        ]

        self.X_train = None
        self.X_validation = None
        self.X_test = None
        self.y_train = None
        self.y_validation = None
        self.y_test = None

        self.feature_columns = None
        self.numeric_columns = None
        self.categorical_columns = None
        self.numeric_medians = {}
        self.categorical_modes = {}

        self.models = []
        self.validation_results = None
        self.best_model = None
        self.best_parameters = None

    def prepare_data(self):
        """Clean data, create date features, encode target, and split."""
        # TODO
        df = self.df.drop_duplicates().dropna(subset=["RainTomorrow"]).copy(); 
        df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0}); 
        dt = pd.to_datetime(df["Date"], errors="coerce"); df["Year"] = dt.dt.year; df["Month"] = dt.dt.month; 
        df = df.drop(columns=["Date"]); X = df.drop(columns=["RainTomorrow"]); y = df["RainTomorrow"]; 
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42); 
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42); 
        self.X_train, self.X_validation, self.X_test = X_train, X_val, X_test; self.y_train, self.y_validation, self.y_test = y_train, y_val, y_test

    def preprocess_features(self):
        """Impute and encode features using training information only."""
        # TODO
        self.numeric_columns = self.X_train.select_dtypes(include="number").columns.tolist(); 
        self.categorical_columns = [c for c in self.X_train.columns if c not in self.numeric_columns];     
        self.numeric_medians = self.X_train[self.numeric_columns].median().to_dict(); self.categorical_modes = {c: self.X_train[c].mode().iloc[0] 
        for c in self.categorical_columns}; fill = {**self.numeric_medians, **self.categorical_modes}; 
        enc = lambda d: pd.get_dummies(d.fillna(fill), columns=self.categorical_columns, dtype=int); self.X_train = enc(self.X_train);
        self.feature_columns = self.X_train.columns.tolist(); self.X_validation = enc(self.X_validation).reindex(columns=self.feature_columns, fill_value=0);
        self.X_test = enc(self.X_test).reindex(columns=self.feature_columns, fill_value=0)
        self.X_train = self.X_train.to_numpy(dtype=float)
        self.X_validation = self.X_validation.to_numpy(dtype=float)
        self.X_test = self.X_test.to_numpy(dtype=float)

    def train_parameter_models(self):
        """Train one Decision Tree for every parameter configuration."""
        # TODO
        self.models = [DecisionTreeClassifier(**p, random_state=42).fit(self.X_train, self.y_train) for p in self.parameter_sets]

    def evaluate_validation_models(self):
        """Evaluate every tree on validation data."""
        # TODO
        preds = [m.predict(self.X_validation) for m in self.models]; self.validation_results = [{"parameters": p, "accuracy": accuracy_score(self.y_validation, pr), 
        "f1_score": f1_score(self.y_validation, pr)} for p, pr in zip(self.parameter_sets, preds)]; return self.validation_results

    def select_best_model(self):
        """Select the strongest tree using validation performance."""
        # TODO
        results = self.validation_results or self.evaluate_validation_models(); 
        best = max(range(len(self.models)), key=lambda i: (results[i]["f1_score"], results[i]["accuracy"]));
        self.best_model = self.models[best]; self.best_parameters = self.parameter_sets[best]; return self.best_model

    def evaluate_test_set(self):
        """Evaluate the selected model on untouched test data."""
        # TODO
        pred = self.best_model.predict(self.X_test); return {"accuracy": accuracy_score(self.y_test, pred), "precision": precision_score(self.y_test, pred), 
        "recall": recall_score(self.y_test, pred), "f1_score": f1_score(self.y_test, pred), "confusion_matrix": confusion_matrix(self.y_test, pred)}

    def get_best_parameters(self):
        return self.best_parameters

    def get_dataset(self):
        # TODO: Return the DataFrame loaded in __init__()
        return self.df

if __name__ == "__main__":
    model = WeatherDecisionTree()

    model.prepare_data()
    model.preprocess_features()
    model.train_parameter_models()

    print("Validation Results:")
    print(model.evaluate_validation_models())

    model.select_best_model()

    print("\nBest Parameters:")
    print(model.get_best_parameters())

    print("\nTest Evaluation:")
    print(model.evaluate_test_set())
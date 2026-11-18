import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class DigitsKNNGridSearch:
    """Tune K for KNN using GridSearchCV."""

    def __init__(self):
        # Load digits_knn.csv
        self.df = pd.read_csv("digits_knn.csv")

        self.k_values = [1, 3, 5, 7, 9, 11, 15, 21]

        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.pipeline = None
        self.grid_search = None

    def prepare_data(self):
        """Prepare features, target, and train-test split."""
        self.X = self.df.drop(columns=["target"])
        self.y = self.df["target"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )

    def create_pipeline(self):
        """Create StandardScaler and KNN pipeline."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])

    def run_grid_search(self):
        """Tune K using GridSearchCV."""
        param_grid = {
            "knn__n_neighbors": self.k_values
        }

        self.grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy"
        )
        self.grid_search.fit(self.X_train, self.y_train)

    def get_best_k(self):
        """Return the K selected by GridSearchCV."""
        return int(self.grid_search.best_params_["knn__n_neighbors"])

    def get_cv_results(self):
        """Return mean cross-validation score for each K."""
        params = self.grid_search.cv_results_["param_knn__n_neighbors"]
        mean_scores = self.grid_search.cv_results_["mean_test_score"]
        return {int(k): float(score) for k, score in zip(params, mean_scores)}

    def evaluate_test_model(self):
        """Evaluate the selected KNN model on test data."""
        y_pred = self.grid_search.best_estimator_.predict(self.X_test)

        return {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred, average="macro")),
            "recall": float(recall_score(self.y_test, y_pred, average="macro")),
            "f1_score": float(f1_score(self.y_test, y_pred, average="macro")),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred)
        }

    def evaluate_best_model(self):
        """Evaluate the selected KNN model on test data."""
        return self.evaluate_test_model()

    def predict(self, X):
        """Predict labels using the best estimator."""
        return self.grid_search.best_estimator_.predict(X)

    def get_dataset(self):
        """Return the loaded DataFrame."""
        return self.df

    def get_data_summary(self):
        """Return the loaded DataFrame."""
        return self.df


if __name__ == "__main__":
    model = DigitsKNNGridSearch()

    model.prepare_data()
    model.create_pipeline()
    model.run_grid_search()

    print("Best K:")
    print(model.get_best_k())

    print("\nCross-Validation Results:")
    print(model.get_cv_results())

    print("\nTest Evaluation:")
    print(model.evaluate_test_model())
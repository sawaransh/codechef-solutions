import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)


class HyperparameterTuning:

    def __init__(self):
        """
        Load dataset.
        """

        print("Loading dataset...")

        self.data = pd.read_csv("dataset.csv")

    def prepare_data(self):
        """
        Prepare dataset.
        """

        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def perform_grid_search(self, X_train, y_train):
        """
        Perform Grid Search.
        """

        print("\nPerforming Grid Search...")

        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 3, 5],
            "min_samples_split": [2, 4]
        }

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("\nBest Grid Search Parameters:")
        print(grid_search.best_params_)

        return grid_search.best_estimator_

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance.
        """

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\nConfusion Matrix:")
        print(cm)

        print("\nAccuracy:")
        print(accuracy)

        print("\nPrecision:")
        print(precision)

        print("\nRecall:")
        print(recall)

        print("\nF1 Score:")
        print(f1)

        print("\nROC AUC:")
        print(roc_auc)

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig("roc_curve.png")
        plt.close()

    def perform_random_search(self, X_train, y_train):
        """
        Perform Random Search.
        """

        print("\nPerforming Random Search...")

        rf = RandomForestClassifier(random_state=42)

        param_dist = {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [None, 3, 5, 7],
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2]
        }

        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1
        )

        random_search.fit(X_train, y_train)

        print("\nBest Random Search Parameters:")
        print(random_search.best_params_)

        return random_search.best_estimator_

    def cross_validation(self, model, X_train, y_train):
        """
        Compute cross validation score.
        """

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")

        print("\nCross Validation Score:")
        print(scores.mean())


def run_hyperparameter_tuning():

    model = HyperparameterTuning()

    X_train, X_test, y_train, y_test = model.prepare_data()

    grid_model = model.perform_grid_search(X_train, y_train)

    model.evaluate_model(grid_model, X_test, y_test)

    random_model = model.perform_random_search(X_train, y_train)

    model.cross_validation(random_model, X_train, y_train)


if __name__ == "__main__":
    run_hyperparameter_tuning()